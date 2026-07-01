"""Tests for the DiaTrend raw-workbook parser."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_processing.diatrend.parser import (
    BOLUS_BASE_COLS,
    BOLUS_EXTENDED_EXTRA_COLS,
    DiaTrendParseError,
    parse_subject,
)
from data_processing.diatrend.tests.fixtures.generate_fixtures import (
    FixturePlan,
    subject_id,
)


def _path_for(plan: FixturePlan, fixtures_dir: Path) -> Path:
    return fixtures_dir / f"{subject_id(plan)}.xlsx"


def _write_minimal_workbook(
    path: Path,
    sheets: dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def test_all_fixtures_parse(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        assert result.subject_id == subject_id(plan)


def test_cohort_inferred_from_bolus_schema(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        assert result.cohort == plan.cohort, (
            f"{subject_id(plan)}: parser inferred cohort {result.cohort} "
            f"but plan says {plan.cohort}"
        )
        if plan.cohort == 2:
            assert result.bolus_schema == "extended"
            assert "insulinOnBoard" in result.bolus.columns
        else:
            assert result.bolus_schema == "base"
            assert "insulinOnBoard" not in result.bolus.columns


def test_reverse_chronology_detected_and_corrected(
    diatrend_fixtures_dir, fixture_plans
):
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        assert result.was_reversed == plan.reversed_order, (
            f"{subject_id(plan)}: was_reversed mismatch"
        )
        assert result.cgm["date"].is_monotonic_increasing
        assert result.bolus["date"].is_monotonic_increasing
        if result.basal is not None and not result.basal.empty:
            assert result.basal["date"].is_monotonic_increasing


def test_date_columns_are_datetime(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        assert pd.api.types.is_datetime64_any_dtype(result.cgm["date"])
        assert pd.api.types.is_datetime64_any_dtype(result.bolus["date"])
        if result.basal is not None and not result.basal.empty:
            assert pd.api.types.is_datetime64_any_dtype(result.basal["date"])


def test_basal_present_iff_cohort_one(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        has_basal_data = result.basal is not None and not result.basal.empty
        if plan.cohort == 1:
            assert has_basal_data, f"{subject_id(plan)}: cohort 1 missing basal data"
        else:
            assert not has_basal_data, (
                f"{subject_id(plan)}: cohort 2 should not carry basal data"
            )


def test_off_grid_bolus_timestamps_preserved(diatrend_fixtures_dir, fixture_plans):
    seen_off_grid = False
    for plan in fixture_plans:
        result = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        seconds = result.bolus["date"].dt.second
        if (seconds != 0).any():
            seen_off_grid = True
            break
    assert seen_off_grid, (
        "Generator should produce sub-minute timestamps so the parser's "
        "no-snap-on-load behaviour is observable in the test suite."
    )


def test_missing_cgm_sheet_raises(tmp_path):
    path = tmp_path / "S9001.xlsx"
    _write_minimal_workbook(
        path,
        {
            "Bolus": pd.DataFrame(
                columns=list(BOLUS_BASE_COLS),
            )
        },
    )
    with pytest.raises(DiaTrendParseError, match="missing required sheet 'CGM'"):
        parse_subject(path)


def test_missing_bolus_sheet_raises(tmp_path):
    path = tmp_path / "S9002.xlsx"
    _write_minimal_workbook(
        path,
        {"CGM": pd.DataFrame({"date": pd.to_datetime([]), "mg/dl": []})},
    )
    with pytest.raises(DiaTrendParseError, match="missing required sheet 'Bolus'"):
        parse_subject(path)


def test_partial_extended_bolus_schema_raises(tmp_path):
    path = tmp_path / "S9003.xlsx"
    cols = list(BOLUS_BASE_COLS) + ["insulinOnBoard"]
    bolus_frame = pd.DataFrame(columns=cols)
    _write_minimal_workbook(
        path,
        {
            "CGM": pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "mg/dl": [120.0]}),
            "Bolus": bolus_frame,
        },
    )
    with pytest.raises(DiaTrendParseError, match="partial extended schema"):
        parse_subject(path)


def test_cohort_basal_inconsistency_raises(tmp_path):
    path = tmp_path / "S9004.xlsx"
    base_cols = list(BOLUS_BASE_COLS)
    extended_cols = base_cols + list(BOLUS_EXTENDED_EXTRA_COLS)
    extended_bolus = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01 08:00:00"),
                "normal": 5.0,
                "carbInput": 50.0,
                "insulinCarbRatio": 10.0,
                "bgInput": 120.0,
                "recommended.carb": 5.0,
                "recommended.net": 5.0,
                "recommended.correction": 0.0,
                "insulinSensitivityFactor": 50.0,
                "targetBloodGlucose": 120.0,
                "insulinOnBoard": 1.0,
            }
        ],
        columns=extended_cols,
    )
    basal_with_data = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01 00:00:00"),
                "duration": 3_600_000,
                "rate": 0.8,
            }
        ]
    )
    _write_minimal_workbook(
        path,
        {
            "CGM": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-01 08:00:00"]),
                    "mg/dl": [120.0],
                }
            ),
            "Bolus": extended_bolus,
            "Basal": basal_with_data,
        },
    )
    with pytest.raises(DiaTrendParseError, match="cohort 2.* coexists with"):
        parse_subject(path)


def test_cohort_one_without_basal_raises(tmp_path):
    path = tmp_path / "S9005.xlsx"
    base_bolus = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01 08:00:00"),
                "normal": 5.0,
                "carbInput": 50.0,
                "insulinCarbRatio": 10.0,
                "bgInput": 120.0,
                "recommended.carb": 5.0,
                "recommended.net": 5.0,
            }
        ],
        columns=list(BOLUS_BASE_COLS),
    )
    _write_minimal_workbook(
        path,
        {
            "CGM": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-01 08:00:00"]),
                    "mg/dl": [120.0],
                }
            ),
            "Bolus": base_bolus,
        },
    )
    with pytest.raises(DiaTrendParseError, match="cohort 1.* no Basal sheet data"):
        parse_subject(path)


def test_generator_is_deterministic(tmp_path):
    from data_processing.diatrend.tests.fixtures.generate_fixtures import generate_all

    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_all(a, seed=42)
    generate_all(b, seed=42)
    files_a = sorted(p.name for p in a.iterdir())
    files_b = sorted(p.name for p in b.iterdir())
    assert files_a == files_b
    for fname in files_a:
        ra = parse_subject(a / fname)
        rb = parse_subject(b / fname)
        pd.testing.assert_frame_equal(ra.cgm, rb.cgm)
        pd.testing.assert_frame_equal(ra.bolus, rb.bolus)
        if ra.basal is None:
            assert rb.basal is None
        else:
            pd.testing.assert_frame_equal(ra.basal, rb.basal)
