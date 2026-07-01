"""Tests for the DiaTrend diagnostic report writer."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from data_processing.diatrend.diagnostics import (
    DiagnosticReport,
    compute_diagnostics,
    main,
    run,
    write_json,
    write_markdown,
)
from data_processing.diatrend.episode_builder import build_episodes
from data_processing.diatrend.parser import parse_subject
from data_processing.diatrend.tests.fixtures.generate_fixtures import (
    FixturePlan,
    subject_id,
)


# A made-up ISO-format date that the leak-guard regex would match. We
# search the rendered Markdown for *no occurrences* of this shape to
# make sure the report body never embeds individual timestamps.
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _path_for(plan: FixturePlan, fixtures_dir: Path) -> Path:
    return fixtures_dir / f"{subject_id(plan)}.xlsx"


def _build_triples(fixtures_dir: Path, plans):
    triples = []
    for plan in plans:
        subject = parse_subject(_path_for(plan, fixtures_dir))
        result = build_episodes(subject)
        triples.append((subject, result, None))
    return triples


def test_compute_diagnostics_basic_shape(diatrend_fixtures_dir, fixture_plans):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    assert isinstance(report, DiagnosticReport)
    assert report.run_id == "test-run"
    assert report.n_subjects == len(fixture_plans)
    expected_c1 = sum(1 for p in fixture_plans if p.cohort == 1)
    expected_c2 = sum(1 for p in fixture_plans if p.cohort == 2)
    assert report.n_cohort_1 == expected_c1
    assert report.n_cohort_2 == expected_c2
    assert report.n_failed_parse == 0


def test_reverse_chronology_subjects_listed(diatrend_fixtures_dir, fixture_plans):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    expected_reversed = sorted(
        (subject_id(p) for p in fixture_plans if p.reversed_order),
        key=lambda s: int("".join(ch for ch in s if ch.isdigit())),
    )
    assert report.reversed_subject_ids == expected_reversed


def test_pooled_cgm_gap_is_close_to_five_minutes(
    diatrend_fixtures_dir, fixture_plans
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    # Fixtures use 5-min nominal spacing with seconds-level jitter.
    # Median should be very close to 5 min.
    assert 4.5 <= report.pooled_cgm_gap_median_min <= 5.5
    assert 0.5 <= report.pooled_cgm_5min_match_rate <= 1.0


def test_bolus_coverage_fraction_in_unit_interval(
    diatrend_fixtures_dir, fixture_plans
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    for s in report.subjects:
        if s.parse_error is not None:
            continue
        assert 0.0 <= s.bolus_coverage_frac <= 1.0 + 1e-9


def test_gap_fixture_shows_up_in_rejection_counts(
    diatrend_fixtures_dir, fixture_plans
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    total_rejections = sum(report.rejection_reason_counts.values())
    # Fixture S1005 carves a 3-hour gap surrounding a noon meal. At
    # minimum, that meal is rejected.
    assert total_rejections >= 1


def test_feature_summaries_have_expected_keys(diatrend_fixtures_dir, fixture_plans):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    for name in ("carbInput", "normal", "bgInput", "insulinOnBoard"):
        assert name in report.feature_summaries
    # carbInput's pmin must be > 0 because we only pool rows where
    # carbInput > 0 (i.e. meals, not corrections).
    assert report.feature_summaries["carbInput"].pmin > 0.0
    # IOB is cohort-2 only; pooled n must be > 0 since we have 3
    # cohort-2 fixtures in the default plan.
    assert report.feature_summaries["insulinOnBoard"].n > 0


def test_retained_feature_summaries_respect_caps(
    diatrend_fixtures_dir, fixture_plans
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    for name in (
        "treatment_carbs",
        "mediator_bolus",
        "bg_input_at_meal",
        "iob_at_meal",
    ):
        assert name in report.retained_feature_summaries
    # Retained features are measured over kept episodes, so the episode
    # caps must bound them.
    carbs = report.retained_feature_summaries["treatment_carbs"]
    if carbs.n > 0:
        assert carbs.pmax <= 200.0
    bg = report.retained_feature_summaries["bg_input_at_meal"]
    if bg.n > 0:
        assert bg.pmax <= 600.0


def test_markdown_contains_required_sections(
    diatrend_fixtures_dir, fixture_plans, tmp_path
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    md_path = tmp_path / "report.md"
    write_markdown(report, md_path)
    body = md_path.read_text()
    for header in (
        "DiaTrend diagnostic report",
        "## Overview",
        "## Per-subject parse status",
        "## CGM temporal sanity",
        "## Episode-construction rejections",
        "## Retained episodes by meal type and cohort",
        "## Feature distributions — raw bolus rows",
        "## Feature distributions — retained episodes",
    ):
        assert header in body, f"missing section: {header}"


def test_markdown_body_has_no_iso_date_outside_header(
    diatrend_fixtures_dir, fixture_plans, tmp_path
):
    """The leak-guard regex matches `\\b\\d{4}-\\d{2}-\\d{2}\\b`. The report
    must not embed individual timestamps. The only acceptable ISO date
    is the run_id in the header — and only because we pass a non-ISO
    run_id from tests, so the body should contain zero matches."""
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    md_path = tmp_path / "report.md"
    write_markdown(report, md_path)
    matches = ISO_DATE_RE.findall(md_path.read_text())
    assert matches == [], (
        f"Markdown contains ISO date literals: {matches} — "
        "could indicate individual timestamps leaking through."
    )


def test_json_round_trips(diatrend_fixtures_dir, fixture_plans, tmp_path):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    json_path = tmp_path / "report.json"
    write_json(report, json_path)
    payload = json.loads(json_path.read_text())
    assert payload["run_id"] == "test-run"
    assert payload["n_subjects"] == len(fixture_plans)
    assert set(payload["feature_summaries"].keys()) == {
        "carbInput",
        "normal",
        "bgInput",
        "insulinOnBoard",
    }
    assert set(payload["retained_feature_summaries"].keys()) == {
        "treatment_carbs",
        "mediator_bolus",
        "bg_input_at_meal",
        "iob_at_meal",
    }
    # meal_type_counts_by_cohort keys are stringified (JSON requirement).
    assert set(payload["meal_type_counts_by_cohort"].keys()) <= {"1", "2"}


def test_run_writes_both_files(diatrend_fixtures_dir, tmp_path):
    md_path, json_path = run(diatrend_fixtures_dir, tmp_path, run_id="cli-test")
    assert md_path.exists()
    assert json_path.exists()
    assert md_path.name == "cli-test.md"
    assert json_path.name == "cli-test.json"


def test_main_cli_exits_zero(diatrend_fixtures_dir, tmp_path):
    rc = main(
        [
            "--input-dir",
            str(diatrend_fixtures_dir),
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "cli-smoke",
        ]
    )
    assert rc == 0
    assert (tmp_path / "cli-smoke.md").exists()
    assert (tmp_path / "cli-smoke.json").exists()


def test_main_cli_returns_2_when_no_workbooks(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"
    rc = main(["--input-dir", str(empty), "--output-dir", str(out)])
    assert rc == 2


def test_parse_failure_recorded_not_raised(tmp_path):
    bad_path = tmp_path / "S9999.xlsx"
    bad_frame = pd.DataFrame({"foo": [1, 2, 3]})
    with pd.ExcelWriter(bad_path, engine="openpyxl") as writer:
        bad_frame.to_excel(writer, sheet_name="Bogus", index=False)
    out_dir = tmp_path / "out"
    md_path, json_path = run(tmp_path, out_dir, run_id="bad-test")
    payload = json.loads(json_path.read_text())
    assert payload["n_failed_parse"] == 1
    assert payload["subjects"][0]["parse_error"] is not None


def test_subject_ordering_is_numeric_not_lexicographic(
    diatrend_fixtures_dir, fixture_plans
):
    triples = _build_triples(diatrend_fixtures_dir, fixture_plans)
    report = compute_diagnostics(triples, run_id="test-run")
    indices = [int("".join(ch for ch in s.subject_id if ch.isdigit())) for s in report.subjects]
    assert indices == sorted(indices)
