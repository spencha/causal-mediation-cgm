"""Tests for the BOB kernel + cohort 2 IOB validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing.diatrend.bob_kernel import (
    IOBKernelParams,
    ValidationStats,
    bob_at_meal_times,
    calibrate_dia,
    compute_bob,
    iob_fraction,
    validate_kernel,
)
from data_processing.diatrend.parser import (
    BOLUS_BASE_COLS,
    BOLUS_EXTENDED_COLS,
    SubjectData,
    parse_subject,
)
from data_processing.diatrend.tests.fixtures.generate_fixtures import (
    FixturePlan,
    subject_id,
)


def _path_for(plan: FixturePlan, fixtures_dir: Path) -> Path:
    return fixtures_dir / f"{subject_id(plan)}.xlsx"


def _make_extended_bolus_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(BOLUS_EXTENDED_COLS)).sort_values(
        "date"
    ).reset_index(drop=True)


def _bolus_row(t: pd.Timestamp, *, carbs: float, normal: float, iob: float | None) -> dict:
    return {
        "date": t,
        "normal": normal,
        "carbInput": carbs,
        "insulinCarbRatio": 10.0,
        "bgInput": 130.0,
        "recommended.carb": carbs / 10.0 if carbs else 0.0,
        "recommended.net": normal,
        "recommended.correction": 0.0,
        "insulinSensitivityFactor": 50.0,
        "targetBloodGlucose": 120.0,
        "insulinOnBoard": iob if iob is not None else float("nan"),
    }


def test_kernel_starts_at_one_and_ends_at_zero():
    params = IOBKernelParams()
    assert iob_fraction(np.array([0.0]), params)[0] == pytest.approx(1.0)
    assert iob_fraction(np.array([params.dia_min]), params)[0] == 0.0
    assert iob_fraction(np.array([params.dia_min + 1.0]), params)[0] == 0.0


def test_kernel_is_monotone_decreasing():
    params = IOBKernelParams()
    t = np.linspace(0.0, params.dia_min, 49)
    f = iob_fraction(t, params)
    diffs = np.diff(f)
    assert (diffs <= 1e-9).all(), "kernel must be monotone decreasing on [0, DIA]"


def test_kernel_rejects_invalid_params():
    with pytest.raises(ValueError, match="weight"):
        IOBKernelParams(weight=0.0)
    with pytest.raises(ValueError, match="time constants"):
        IOBKernelParams(fast_tau_min=-1.0)
    with pytest.raises(ValueError, match="fast_tau_min must be"):
        IOBKernelParams(fast_tau_min=200.0, slow_tau_min=100.0)
    with pytest.raises(ValueError, match="dia_min"):
        IOBKernelParams(dia_min=0.0)


def test_compute_bob_includes_only_prior_boluses_by_default():
    params = IOBKernelParams()
    base = pd.Timestamp("2020-01-01 12:00:00")
    bolus_times = np.array(
        [base, base + pd.Timedelta(minutes=30)], dtype="datetime64[ns]"
    )
    doses = np.array([5.0, 3.0])
    query = np.array([base + pd.Timedelta(minutes=60)], dtype="datetime64[ns]")
    bob = compute_bob(bolus_times, doses, query, params)
    # Query is 60 min after the first bolus (5 U) and 30 min after the
    # second (3 U). Both contribute via the kernel; both elapsed values
    # are strictly positive so default ``inclusive=False`` includes them.
    expected = 5.0 * iob_fraction(np.array([60.0]), params)[0] + 3.0 * iob_fraction(
        np.array([30.0]), params
    )[0]
    assert bob[0] == pytest.approx(expected)


def test_compute_bob_inclusive_flag_controls_self_contribution():
    params = IOBKernelParams()
    t = pd.Timestamp("2020-01-01 12:00:00")
    bolus_times = np.array([t], dtype="datetime64[ns]")
    doses = np.array([5.0])
    query = np.array([t], dtype="datetime64[ns]")
    assert compute_bob(bolus_times, doses, query, params, inclusive=False)[0] == 0.0
    assert (
        compute_bob(bolus_times, doses, query, params, inclusive=True)[0]
        == pytest.approx(5.0)
    )


def test_compute_bob_empty_history_returns_zeros():
    params = IOBKernelParams()
    bolus_times = np.array([], dtype="datetime64[ns]")
    doses = np.array([], dtype=float)
    query = np.array(
        [pd.Timestamp("2020-01-01 12:00:00")], dtype="datetime64[ns]"
    )
    bob = compute_bob(bolus_times, doses, query, params)
    assert bob.shape == (1,)
    assert bob[0] == 0.0


def test_compute_bob_ignores_boluses_beyond_dia():
    params = IOBKernelParams(dia_min=180.0)
    base = pd.Timestamp("2020-01-01 12:00:00")
    bolus_times = np.array([base], dtype="datetime64[ns]")
    doses = np.array([5.0])
    query = np.array(
        [base + pd.Timedelta(minutes=180), base + pd.Timedelta(minutes=181)],
        dtype="datetime64[ns]",
    )
    bob = compute_bob(bolus_times, doses, query, params)
    assert bob[0] == 0.0
    assert bob[1] == 0.0


def test_validate_kernel_perfect_agreement_when_iob_synthesized_from_kernel():
    """If the cohort-2 pump IOB column is itself generated from the
    same kernel, validate_kernel should report near-perfect agreement.
    This is the kernel-correctness end-to-end check."""
    params = IOBKernelParams()
    base = pd.Timestamp("2020-01-01 08:00:00")
    # Bolus spacing must be strictly less than DIA so prior boluses
    # contribute non-zero IOB at each event time. 2h spacing leaves the
    # previous bolus at ~18% IOB-fraction under the default kernel.
    times = [base + pd.Timedelta(hours=2 * h) for h in range(10)]
    doses = [4.0, 3.5, 5.5, 6.0, 4.0, 3.0, 5.0, 4.5, 5.0, 3.0]
    bolus_times = np.array(times, dtype="datetime64[ns]")
    dose_arr = np.array(doses, dtype=float)
    synthetic_iob = compute_bob(bolus_times, dose_arr, bolus_times, params, inclusive=False)

    rows = []
    for t, dose, iob in zip(times, doses, synthetic_iob):
        rows.append(_bolus_row(t, carbs=40.0, normal=dose, iob=float(iob)))
    bolus_df = _make_extended_bolus_frame(rows)
    subject = SubjectData(
        subject_id="S9001",
        cohort=2,
        bolus_schema="extended",
        cgm=pd.DataFrame(
            {"date": [base], "mg/dl": [150.0]}
        ),
        bolus=bolus_df,
        basal=None,
        was_reversed=False,
    )
    stats = validate_kernel(subject, params)
    assert stats is not None
    assert stats.pearson_r > 0.999
    assert stats.rmse < 1e-6


def test_validate_kernel_returns_none_for_cohort_one(diatrend_fixtures_dir, fixture_plans):
    cohort1_plan = next(p for p in fixture_plans if p.cohort == 1)
    subject = parse_subject(_path_for(cohort1_plan, diatrend_fixtures_dir))
    assert validate_kernel(subject, IOBKernelParams()) is None


def test_validate_kernel_runs_on_cohort_two_fixture(
    diatrend_fixtures_dir, fixture_plans
):
    cohort2_plan = next(p for p in fixture_plans if p.cohort == 2)
    subject = parse_subject(_path_for(cohort2_plan, diatrend_fixtures_dir))
    stats = validate_kernel(subject, IOBKernelParams())
    assert stats is not None
    assert stats.n > 1
    # Fixture IOB is uniform random noise unrelated to the kernel, so
    # correlation should be small but a finite number, and Bland-Altman
    # bias is well-defined.
    assert np.isfinite(stats.bland_altman_bias)
    assert np.isfinite(stats.rmse)


def test_bob_at_meal_times_returns_correct_columns_and_filter():
    base = pd.Timestamp("2020-01-01 08:00:00")
    rows = [
        _bolus_row(base, carbs=40.0, normal=4.0, iob=0.0),
        _bolus_row(base + pd.Timedelta(minutes=60), carbs=0.0, normal=1.5, iob=2.0),
        _bolus_row(base + pd.Timedelta(hours=4), carbs=50.0, normal=5.0, iob=0.5),
    ]
    bolus_df = _make_extended_bolus_frame(rows)
    enriched = bob_at_meal_times(bolus_df, IOBKernelParams(), require_meal=True)
    assert {"kernel_bob", "elapsed_first_bolus_min"} <= set(enriched.columns)
    assert len(enriched) == 2  # only the two meal rows survive the filter
    # First meal: no prior boluses, BOB == 0
    assert enriched["kernel_bob"].iloc[0] == 0.0


def test_calibrate_dia_picks_a_grid_value(diatrend_fixtures_dir, fixture_plans):
    subjects = [
        parse_subject(_path_for(p, diatrend_fixtures_dir))
        for p in fixture_plans
        if p.cohort == 2
    ]
    grid = (120.0, 180.0, 240.0, 300.0, 360.0)
    best, mse_by_dia = calibrate_dia(subjects, dia_grid_min=grid)
    assert best in grid
    assert set(mse_by_dia.keys()) == set(grid)
    # At least one DIA must produce a finite MSE.
    assert any(np.isfinite(m) for m in mse_by_dia.values())


def test_calibrate_dia_returns_best_when_synthetic_iob_calibrated_to_known_dia():
    """Generate cohort-2 subjects where insulinOnBoard is synthesized
    from a kernel with DIA = 240 min, plus a tiny amount of noise.
    Calibration over a grid should pick a DIA close to 240."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2020-01-01 08:00:00")
    true_params = IOBKernelParams(dia_min=240.0)
    subjects = []
    for sid_idx, n_doses in enumerate([12, 16, 10, 14]):
        times = [base + pd.Timedelta(hours=2 * i) for i in range(n_doses)]
        doses = rng.uniform(2.0, 6.0, size=n_doses)
        ts_arr = np.array(times, dtype="datetime64[ns]")
        iob_clean = compute_bob(ts_arr, doses, ts_arr, true_params, inclusive=False)
        iob_noisy = iob_clean + rng.normal(0.0, 0.05, size=n_doses)
        rows = [
            _bolus_row(t, carbs=40.0, normal=float(d), iob=float(i))
            for t, d, i in zip(times, doses, iob_noisy)
        ]
        bolus_df = _make_extended_bolus_frame(rows)
        subjects.append(
            SubjectData(
                subject_id=f"S{9001 + sid_idx}",
                cohort=2,
                bolus_schema="extended",
                cgm=pd.DataFrame({"date": [base], "mg/dl": [150.0]}),
                bolus=bolus_df,
                basal=None,
                was_reversed=False,
            )
        )
    grid = (120.0, 180.0, 240.0, 300.0, 360.0)
    best, mse_by_dia = calibrate_dia(subjects, dia_grid_min=grid)
    # With noise small relative to the kernel's signal, the optimal DIA
    # should be the true 240-min value or its immediate neighbor.
    assert best in (180.0, 240.0, 300.0)
