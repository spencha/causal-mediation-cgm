"""Tests for the DiaTrend meal-centered episode builder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing.diatrend.episode_builder import (
    Episode,
    EpisodeBuildResult,
    EpisodeConfig,
    build_episodes,
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


def _make_cgm_grid(
    start: pd.Timestamp,
    *,
    days: float = 2.0,
    interval_min: int = 5,
    value: float = 150.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(0)
    n = int((days * 24 * 60) // interval_min)
    times = [start + pd.Timedelta(minutes=interval_min * i) for i in range(n)]
    glucose = np.full(n, value) + rng.normal(0.0, 5.0, size=n)
    return pd.DataFrame({"date": times, "mg/dl": glucose})


def _make_bolus_row(
    t: pd.Timestamp,
    *,
    carbs: float,
    normal: float,
    iob: float | None = None,
    bg: float = 130.0,
) -> dict:
    row = {
        "date": t,
        "normal": normal,
        "carbInput": carbs,
        "insulinCarbRatio": 10.0,
        "bgInput": bg,
        "recommended.carb": carbs / 10.0,
        "recommended.net": normal,
    }
    if iob is not None:
        row["recommended.correction"] = 0.0
        row["insulinSensitivityFactor"] = 50.0
        row["targetBloodGlucose"] = 120.0
        row["insulinOnBoard"] = iob
    return row


def _make_subject(
    cgm: pd.DataFrame, bolus_rows: list[dict], cohort: int, subject: str = "S9999"
) -> SubjectData:
    cols = list(BOLUS_EXTENDED_COLS) if cohort == 2 else list(BOLUS_BASE_COLS)
    bolus = pd.DataFrame(bolus_rows, columns=cols).sort_values("date").reset_index(drop=True)
    return SubjectData(
        subject_id=subject,
        cohort=cohort,  # type: ignore[arg-type]
        bolus_schema="extended" if cohort == 2 else "base",
        cgm=cgm.sort_values("date").reset_index(drop=True),
        bolus=bolus,
        basal=None if cohort == 2 else pd.DataFrame(
            [{"date": cgm["date"].iloc[0], "duration": 60_000, "rate": 0.8}]
        ),
        was_reversed=False,
    )


def test_each_fixture_produces_episodes(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        subject = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        result = build_episodes(subject)
        assert result.subject_id == subject.subject_id
        assert result.cohort == plan.cohort
        assert result.n_candidate_meals > 0
        assert len(result.episodes) > 0, (
            f"{subject.subject_id}: zero episodes from {result.n_candidate_meals} candidates "
            f"(rejected: {[r.reason for r in result.rejected]})"
        )


def test_episodes_have_canonical_window_shape(diatrend_fixtures_dir, fixture_plans):
    cfg = EpisodeConfig()
    expected_bins = cfg.n_bins
    for plan in fixture_plans:
        result = build_episodes(parse_subject(_path_for(plan, diatrend_fixtures_dir)))
        for ep in result.episodes:
            assert ep.cgm_grid.shape == (expected_bins,)
            assert ep.cgm_times_relative_min.shape == (expected_bins,)
            assert ep.cgm_times_relative_min[0] == -cfg.pre_min
            assert ep.cgm_times_relative_min[-1] == cfg.post_total_min - cfg.interval_min


def test_treatment_matches_source_carbs(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        subject = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        result = build_episodes(subject)
        source_carbs = subject.bolus.loc[
            subject.bolus["carbInput"] > 0, "carbInput"
        ].sum()
        episode_carbs = sum(ep.treatment_carbs for ep in result.episodes)
        # Episode carbs may be < source carbs because some meals are
        # rejected by the filters, but they should never exceed it (the
        # bolus merge sums carbs across multiple rows into one meal
        # event but does not create carbs out of thin air).
        assert episode_carbs <= source_carbs + 1e-9


def test_mediator_equals_normal_sum_in_window(diatrend_fixtures_dir, fixture_plans):
    cfg = EpisodeConfig()
    for plan in fixture_plans:
        subject = parse_subject(_path_for(plan, diatrend_fixtures_dir))
        result = build_episodes(subject)
        for ep in result.episodes:
            lo = ep.meal_time - pd.Timedelta(minutes=cfg.mediator_window_pre_min)
            hi = ep.meal_time + pd.Timedelta(minutes=cfg.mediator_window_post_min)
            expected = (
                subject.bolus.loc[
                    (subject.bolus["date"] >= lo) & (subject.bolus["date"] <= hi),
                    "normal",
                ]
                .fillna(0.0)
                .sum()
            )
            assert ep.mediator_bolus == pytest.approx(expected)


def test_iob_present_iff_cohort_two(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = build_episodes(parse_subject(_path_for(plan, diatrend_fixtures_dir)))
        for ep in result.episodes:
            if plan.cohort == 2:
                assert ep.iob_at_meal is not None, (
                    f"{ep.subject_id}: cohort 2 episode missing IOB"
                )
                assert isinstance(ep.iob_at_meal, float)
            else:
                assert ep.iob_at_meal is None, (
                    f"{ep.subject_id}: cohort 1 episode unexpectedly has IOB"
                )


def test_meal_type_inferred_from_hour(diatrend_fixtures_dir, fixture_plans):
    cfg = EpisodeConfig()
    for plan in fixture_plans:
        result = build_episodes(parse_subject(_path_for(plan, diatrend_fixtures_dir)))
        for ep in result.episodes:
            hour = ep.meal_time.hour
            if cfg.breakfast_hours[0] <= hour < cfg.breakfast_hours[1]:
                assert ep.meal_type == "breakfast"
            elif cfg.lunch_hours[0] <= hour < cfg.lunch_hours[1]:
                assert ep.meal_type == "lunch"
            elif cfg.dinner_hours[0] <= hour < cfg.dinner_hours[1]:
                assert ep.meal_type == "dinner"
            else:
                assert ep.meal_type == "snack"


def test_episodes_emerge_in_chronological_order(diatrend_fixtures_dir, fixture_plans):
    for plan in fixture_plans:
        result = build_episodes(parse_subject(_path_for(plan, diatrend_fixtures_dir)))
        times = [ep.meal_time for ep in result.episodes]
        assert times == sorted(times), (
            f"{result.subject_id}: episodes out of chronological order"
        )


def test_cgm_gap_fixture_drops_affected_meal(diatrend_fixtures_dir, fixture_plans):
    gap_plan = next(p for p in fixture_plans if p.induce_missing_premeal)
    subject = parse_subject(_path_for(gap_plan, diatrend_fixtures_dir))
    result = build_episodes(subject)
    # The fixture carves a 3-hour CGM gap surrounding the day-1 lunch
    # bolus. At minimum, the lunch episode on day 1 should be rejected
    # for either consecutive-NaN or postprandial-coverage reasons.
    rejection_reasons = " ".join(r.reason for r in result.rejected)
    assert any(
        keyword in rejection_reasons
        for keyword in ("consecutive NaN", "postprandial coverage", "last CGM bin is NaN")
    ), f"{subject.subject_id}: expected a NaN-related rejection, got: {rejection_reasons}"


def test_two_close_meals_merge_into_one_episode():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=40.0, normal=4.0, iob=1.0
        ),
        _make_bolus_row(
            start + pd.Timedelta(hours=12, minutes=10), carbs=20.0, normal=2.0, iob=4.5
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 1
    ep = result.episodes[0]
    assert ep.bolus_merged_count == 2
    assert ep.treatment_carbs == pytest.approx(60.0)
    # IOB of the merged event is taken from the FIRST row (the one that
    # triggers the meal), not the later top-up.
    assert ep.iob_at_meal == pytest.approx(1.0)
    # The mediator window covers [-120, +60] min. Both bolus rows fall in
    # it (they are 0 and +10 min from the merged meal_time), so the
    # mediator is the sum of both normals.
    assert ep.mediator_bolus == pytest.approx(6.0)


def test_bolus_merge_disabled_keeps_meals_separate():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=40.0, normal=4.0, iob=1.0
        ),
        _make_bolus_row(
            start + pd.Timedelta(hours=12, minutes=10), carbs=20.0, normal=2.0, iob=4.5
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    cfg = EpisodeConfig(bolus_merge_window_min=0.0)
    result = build_episodes(subject, cfg)
    # Without merging, the second meal (+10 min) is treated as its own
    # candidate meal event. It falls into the first meal's grace window
    # (within +60 min) so contamination does NOT trigger on the first
    # meal; however the second meal's own contamination check also
    # finds nothing later, and it has its own valid window.
    assert len(result.episodes) == 2


def test_implausible_carb_input_rejected():
    # Carb input above the 200 g cap is dropped regardless of bolus. The
    # large (20 U) bolus here means the previous combined carb/bolus-ratio
    # filter (carb > 200 & bolus < 15) would have KEPT this meal; the
    # carb-only cap rejects it.
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=250.0, normal=20.0, iob=0.0
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 0
    assert any("implausible" in r.reason for r in result.rejected)


def test_high_carb_with_high_bolus_kept_below_cap():
    # A large meal (180 g) under the 200 g cap is retained even though the
    # bolus is small — the carb-only rule does not look at the bolus.
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=180.0, normal=2.0, iob=0.0
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 1
    assert not any("implausible" in r.reason for r in result.rejected)


def test_implausible_bginput_rejected():
    # Pre-meal reference glucose above the 600 mg/dL cap is a data-entry
    # error; the episode is dropped.
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=40.0, normal=4.0, bg=1038.0
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 0
    assert any("implausible bgInput" in r.reason for r in result.rejected)


def test_high_but_plausible_bginput_kept():
    # A high-but-possible reference glucose (400 mg/dL) is retained — the
    # cap targets only impossible values.
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=40.0, normal=4.0, bg=400.0
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 1
    assert not any("implausible bgInput" in r.reason for r in result.rejected)


def test_contamination_by_second_meal_rejects_first():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_cgm_grid(start, days=2.0)
    # First meal at noon; second meal at +90 min (inside the postprandial
    # window but outside the grace and outside the bolus-merge window),
    # large enough to trip the 5 g contamination threshold.
    bolus = [
        _make_bolus_row(
            start + pd.Timedelta(hours=12), carbs=50.0, normal=5.0, iob=0.5
        ),
        _make_bolus_row(
            start + pd.Timedelta(hours=12, minutes=90), carbs=30.0, normal=3.0, iob=2.0
        ),
    ]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    # The FIRST meal is rejected for contamination.
    first_meal_rejected = any(
        r.meal_time == start + pd.Timedelta(hours=12)
        and "contamination" in r.reason
        for r in result.rejected
    )
    assert first_meal_rejected
    # The SECOND meal at +90 min has no further meal after it inside its
    # own postprandial window (within the synthetic 2-day CGM record),
    # so it survives.
    second_meal_kept = any(
        ep.meal_time == start + pd.Timedelta(hours=12, minutes=90)
        for ep in result.episodes
    )
    assert second_meal_kept


def test_postprandial_coverage_filter_rejects_sparse_episode():
    start = pd.Timestamp("2020-01-01 00:00:00")
    full = _make_cgm_grid(start, days=2.0)
    # Carve out all CGM in the +60..+240 range of a noon meal so that
    # postprandial coverage is exactly zero past the grace window.
    meal_t = start + pd.Timedelta(hours=12)
    drop_lo = meal_t + pd.Timedelta(minutes=60)
    drop_hi = meal_t + pd.Timedelta(minutes=240)
    cgm = full.loc[(full["date"] < drop_lo) | (full["date"] >= drop_hi)].reset_index(drop=True)
    bolus = [_make_bolus_row(meal_t, carbs=50.0, normal=5.0, iob=0.5)]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 0
    assert any(
        "consecutive NaN" in r.reason
        or "postprandial coverage" in r.reason
        or "last CGM bin" in r.reason
        for r in result.rejected
    )


def test_config_rejects_oversized_merge_window():
    with pytest.raises(ValueError, match="bolus_merge_window_min"):
        EpisodeConfig(bolus_merge_window_min=120.0, post_grace_min=60)


def test_short_record_drops_window_without_full_coverage():
    start = pd.Timestamp("2020-01-01 00:00:00")
    # CGM record only 90 min long; cannot cover the [-120, +240] window
    # around a meal 30 min into the record.
    short_cgm = _make_cgm_grid(start, days=90 / (24 * 60))
    meal_t = start + pd.Timedelta(minutes=30)
    bolus = [_make_bolus_row(meal_t, carbs=50.0, normal=5.0, iob=0.5)]
    subject = _make_subject(short_cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 0
    assert any("full" in r.reason and "window" in r.reason for r in result.rejected)


def _make_descending_cgm_grid(start: pd.Timestamp, *, days: float = 1.0) -> pd.DataFrame:
    """CGM trace that monotonically decreases — used to exercise the
    carb-bolus consistency filter (meal with no bolus + glucose drops)."""
    interval_min = 5
    n = int((days * 24 * 60) // interval_min)
    times = [start + pd.Timedelta(minutes=interval_min * i) for i in range(n)]
    glucose = np.linspace(220.0, 90.0, n)
    return pd.DataFrame({"date": times, "mg/dl": glucose})


def test_carb_bolus_consistency_filter_rejects_meal_with_no_bolus_and_glucose_drop():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_descending_cgm_grid(start, days=1.0)
    meal_t = start + pd.Timedelta(hours=4)  # leaves room for [-120, +240] window
    # A meal-bolus row carrying carbs but normal=0 — i.e. user announced
    # the meal carbs but did not deliver a bolus. Pair with a descending
    # CGM trace so glucose_change < 0.
    bolus = [_make_bolus_row(meal_t, carbs=50.0, normal=0.0, iob=0.0)]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 0
    assert any("carb-bolus consistency" in r.reason for r in result.rejected)


def test_carb_bolus_consistency_filter_can_be_disabled():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_descending_cgm_grid(start, days=1.0)
    meal_t = start + pd.Timedelta(hours=4)
    bolus = [_make_bolus_row(meal_t, carbs=50.0, normal=0.0, iob=0.0)]
    subject = _make_subject(cgm, bolus, cohort=2)
    cfg = EpisodeConfig(enable_carb_bolus_consistency_filter=False)
    result = build_episodes(subject, cfg)
    # Without the consistency filter, the episode is retained.
    assert len(result.episodes) == 1
    assert result.episodes[0].mediator_bolus == 0.0


def test_carb_bolus_consistency_filter_keeps_meal_with_bolus_present():
    start = pd.Timestamp("2020-01-01 00:00:00")
    cgm = _make_descending_cgm_grid(start, days=1.0)
    meal_t = start + pd.Timedelta(hours=4)
    # Same descending CGM but the meal HAS a bolus — consistency filter
    # should not fire; the episode survives.
    bolus = [_make_bolus_row(meal_t, carbs=50.0, normal=5.0, iob=0.0)]
    subject = _make_subject(cgm, bolus, cohort=2)
    result = build_episodes(subject)
    assert len(result.episodes) == 1
