"""Meal-centered episode construction for DiaTrend.

Takes a parsed ``SubjectData`` and produces one ``Episode`` per surviving
meal event. The episode is the unit the CLAE is trained on. Defaults
match the OhioT1DM convention so cross-dataset comparisons in the
manuscript are apples-to-apples: pre-meal window = 120 min, post-meal
total = 240 min, post-meal grace = 60 min (the +0..+60 region is the
mediator's tail, not part of the outcome), 5-min binning. The mediator
is the sum of bolus ``normal`` over [-120, +60] min around the meal.

DiaTrend-specific choices, documented in Section 7 of the handoff:

- Meals are inferred from bolus rows with ``carbInput > 0`` (DiaTrend
  has no meal-annotation stream of its own).
- Sequential meal events within ``bolus_merge_window_min`` of each
  other are merged into one (carbs and bolus summed). Correction
  boluses (carbInput == 0) are not merged in; their contribution
  appears via the mediator window, not the meal event itself.
- Meal type is inferred from clock time, with configurable hour
  boundaries. The manuscript must disclose that DiaTrend meal types
  are inferred rather than annotated.
- The pre-treatment IOB covariate is only populated for cohort 2
  (where the pump-reported ``insulinOnBoard`` column exists). For
  cohort 1 it is ``None``; downstream models either drop cohort 1
  from IOB-adjusted arms or substitute a kernel-derived BOB (the
  Section 8.5 robustness analysis, not built here).

Interpolation of missing CGM bins is *not* performed at episode-build
time. The OhioT1DM Python loader (``resid_ae_utils.load_windows``)
handles linear + bfill/ffill + zero-fill at training time; preserving
NaNs here keeps the episode the raw artifact and lets the diagnostic
report quantify true missingness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from data_processing.diatrend.parser import SubjectData


@dataclass(frozen=True)
class EpisodeConfig:
    """Configuration for ``build_episodes``. All fields have sensible
    DiaTrend defaults matching the OhioT1DM convention; override at
    construction time for sensitivity analyses."""

    pre_min: int = 120
    post_total_min: int = 240
    post_grace_min: int = 60
    interval_min: int = 5

    mediator_window_pre_min: int = 120
    mediator_window_post_min: int = 60

    bolus_merge_window_min: float = 15.0

    postprandial_coverage_min_frac: float = 0.80
    max_consecutive_nan_bins: int = 6
    require_last_bin_non_nan: bool = True
    contamination_carb_threshold_g: float = 5.0
    implausible_carb_threshold_g: float = 200.0
    implausible_bg_threshold_mgdl: float = 600.0

    # OhioT1DM carb-bolus consistency filter (lines 308-311 of
    # z_meal_mediation_analysis_data_*_5min.R): drop windows where the
    # observed glucose trajectory is physiologically inconsistent with
    # the recorded carb / bolus inputs. The first OhioT1DM clause
    # (carb==0 & bolus>0 & glucose_change>=0) cannot fire here because
    # DiaTrend meal candidates are pre-filtered to carbInput > 0, so
    # only the second clause is applied:
    # drop (carb>0 & bolus_in_full_window==0 & glucose_change<0).
    # `glucose_change` is the end-of-window minus start-of-window
    # glucose, matching OhioT1DM's `final_glucose - start_gluc`.
    enable_carb_bolus_consistency_filter: bool = True

    breakfast_hours: tuple[int, int] = (5, 10)
    lunch_hours: tuple[int, int] = (10, 14)
    dinner_hours: tuple[int, int] = (17, 22)

    def __post_init__(self) -> None:
        if self.bolus_merge_window_min > self.post_grace_min:
            # If merge window exceeds the grace period, a merged-in second
            # meal could land inside the contamination check region and
            # be flagged as a different-meal contaminant. Forbid it so
            # the filter semantics are unambiguous.
            raise ValueError(
                f"bolus_merge_window_min ({self.bolus_merge_window_min}) "
                f"must be <= post_grace_min ({self.post_grace_min})"
            )

    @property
    def n_bins(self) -> int:
        return (self.pre_min + self.post_total_min) // self.interval_min

    @property
    def n_bins_pre(self) -> int:
        return self.pre_min // self.interval_min

    @property
    def n_bins_post(self) -> int:
        return self.post_total_min // self.interval_min

    @property
    def n_bins_postprandial(self) -> int:
        return self.n_bins_post


@dataclass(frozen=True, eq=False)
class Episode:
    subject_id: str
    cohort: Literal[1, 2]
    meal_id: int
    meal_time: pd.Timestamp
    meal_day: int
    meal_type: str
    treatment_carbs: float
    mediator_bolus: float
    iob_at_meal: float | None
    bg_input_at_meal: float | None
    cgm_grid: np.ndarray
    cgm_times_relative_min: np.ndarray
    n_consecutive_nan_max: int
    postprandial_coverage_frac: float
    contamination_carbs_max: float
    bolus_merged_count: int


@dataclass(frozen=True)
class RejectedMeal:
    subject_id: str
    meal_time: pd.Timestamp
    reason: str


@dataclass(frozen=True, eq=False)
class EpisodeBuildResult:
    subject_id: str
    cohort: Literal[1, 2]
    episodes: list[Episode] = field(default_factory=list)
    rejected: list[RejectedMeal] = field(default_factory=list)
    n_candidate_meals: int = 0


def _merge_close_meals(
    meals: pd.DataFrame, merge_window_min: float
) -> list[dict]:
    """Group meal-bolus rows whose successive timestamps fall within
    ``merge_window_min``. Returns one merged-meal dict per group."""
    if meals.empty:
        return []
    if merge_window_min <= 0:
        return [
            {
                "date": row["date"],
                "carbInput": row["carbInput"],
                "normal_at_meal": row["normal"],
                "bgInput": row.get("bgInput"),
                "insulinOnBoard": row.get("insulinOnBoard"),
                "merged_count": 1,
            }
            for _, row in meals.iterrows()
        ]

    merged: list[dict] = []
    threshold = pd.Timedelta(minutes=merge_window_min)
    current: dict | None = None
    last_time: pd.Timestamp | None = None

    for _, row in meals.iterrows():
        t: pd.Timestamp = row["date"]
        if current is None or (last_time is not None and t - last_time > threshold):
            current = {
                "date": t,
                "carbInput": float(row["carbInput"]),
                "normal_at_meal": float(row["normal"]),
                "bgInput": _safe_float(row.get("bgInput")),
                "insulinOnBoard": _safe_float(row.get("insulinOnBoard")),
                "merged_count": 1,
            }
            merged.append(current)
        else:
            current["carbInput"] += float(row["carbInput"])
            current["normal_at_meal"] += float(row["normal"])
            current["merged_count"] += 1
        last_time = t
    return merged


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _bin_cgm(
    cgm: pd.DataFrame, meal_time: pd.Timestamp, config: EpisodeConfig
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (cgm_grid, times_relative_min, has_full_window).

    The grid has ``config.n_bins`` entries; each entry is the mean of
    CGM readings whose timestamp falls in
    ``[bin_start, bin_start + interval_min)``. NaN if the bin is empty.
    ``has_full_window`` is True iff the CGM record covers the entire
    [meal_time - pre, meal_time + post_total) range, even if some bins
    are NaN — this filters out subjects whose record is too short.
    """
    interval = pd.Timedelta(minutes=config.interval_min)
    window_start = meal_time - pd.Timedelta(minutes=config.pre_min)
    window_end = meal_time + pd.Timedelta(minutes=config.post_total_min)

    if cgm.empty:
        return (
            np.full(config.n_bins, np.nan),
            _relative_minutes(config),
            False,
        )

    cgm_min = cgm["date"].iloc[0]
    cgm_max = cgm["date"].iloc[-1]
    has_full_window = cgm_min <= window_start and cgm_max >= window_end - interval

    in_window = cgm.loc[(cgm["date"] >= window_start) & (cgm["date"] < window_end)]
    if in_window.empty:
        return (
            np.full(config.n_bins, np.nan),
            _relative_minutes(config),
            has_full_window,
        )

    bin_idx = (
        (in_window["date"] - window_start).dt.total_seconds().to_numpy()
        // (config.interval_min * 60)
    ).astype(int)
    values = in_window["mg/dl"].to_numpy(dtype=float)

    grid = np.full(config.n_bins, np.nan)
    counts = np.zeros(config.n_bins, dtype=int)
    sums = np.zeros(config.n_bins, dtype=float)
    valid = (bin_idx >= 0) & (bin_idx < config.n_bins)
    np.add.at(sums, bin_idx[valid], values[valid])
    np.add.at(counts, bin_idx[valid], 1)
    non_empty = counts > 0
    grid[non_empty] = sums[non_empty] / counts[non_empty]

    return grid, _relative_minutes(config), has_full_window


def _relative_minutes(config: EpisodeConfig) -> np.ndarray:
    return np.arange(config.n_bins) * config.interval_min - config.pre_min


def _max_consecutive_nan(grid: np.ndarray) -> int:
    max_run = 0
    run = 0
    for value in grid:
        if np.isnan(value):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _infer_meal_type(meal_time: pd.Timestamp, config: EpisodeConfig) -> str:
    hour = meal_time.hour
    if config.breakfast_hours[0] <= hour < config.breakfast_hours[1]:
        return "breakfast"
    if config.lunch_hours[0] <= hour < config.lunch_hours[1]:
        return "lunch"
    if config.dinner_hours[0] <= hour < config.dinner_hours[1]:
        return "dinner"
    return "snack"


def _sum_bolus_in_window(
    bolus: pd.DataFrame, meal_time: pd.Timestamp, config: EpisodeConfig
) -> float:
    lo = meal_time - pd.Timedelta(minutes=config.mediator_window_pre_min)
    hi = meal_time + pd.Timedelta(minutes=config.mediator_window_post_min)
    rows = bolus.loc[(bolus["date"] >= lo) & (bolus["date"] <= hi)]
    if rows.empty:
        return 0.0
    return float(rows["normal"].fillna(0.0).sum())


def _max_contamination_carbs(
    meals: pd.DataFrame, meal_time: pd.Timestamp, config: EpisodeConfig
) -> float:
    lo = meal_time + pd.Timedelta(minutes=config.post_grace_min)
    hi = meal_time + pd.Timedelta(minutes=config.post_total_min)
    later = meals.loc[
        (meals["date"] > lo) & (meals["date"] <= hi) & (meals["date"] != meal_time)
    ]
    if later.empty:
        return 0.0
    return float(later["carbInput"].fillna(0.0).max())


def build_episodes(
    subject: SubjectData, config: EpisodeConfig | None = None
) -> EpisodeBuildResult:
    """Build meal-centered episodes from one subject's parsed workbook."""
    cfg = config or EpisodeConfig()

    meals = subject.bolus.loc[subject.bolus["carbInput"] > 0].reset_index(drop=True)
    n_candidate = len(meals)

    if meals.empty:
        return EpisodeBuildResult(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            n_candidate_meals=0,
        )

    merged = _merge_close_meals(meals, cfg.bolus_merge_window_min)

    if subject.cgm.empty:
        rejected = [
            RejectedMeal(subject.subject_id, m["date"], "no CGM data") for m in merged
        ]
        return EpisodeBuildResult(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            rejected=rejected,
            n_candidate_meals=n_candidate,
        )

    first_cgm_date = subject.cgm["date"].iloc[0].normalize()
    episodes: list[Episode] = []
    rejected: list[RejectedMeal] = []

    for merged_idx, m in enumerate(merged, start=1):
        meal_time: pd.Timestamp = m["date"]
        treatment = float(m["carbInput"])
        mediator = _sum_bolus_in_window(subject.bolus, meal_time, cfg)
        contamination = _max_contamination_carbs(meals, meal_time, cfg)

        if treatment > cfg.implausible_carb_threshold_g:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    f"implausible carb input "
                    f"({treatment:.1f}g carbs > {cfg.implausible_carb_threshold_g:.0f}g cap)",
                )
            )
            continue

        # bgInput is the pre-meal reference glucose (pre-treatment context,
        # not the treatment or mediator). Values above this cap are
        # physiologically impossible data-entry errors; drop the episode.
        # bgInput may be None when the meal-bolus row carried no reference
        # glucose, in which case the filter does not fire.
        bg_input = m["bgInput"]
        if bg_input is not None and bg_input > cfg.implausible_bg_threshold_mgdl:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    f"implausible bgInput "
                    f"({bg_input:.0f} mg/dL > {cfg.implausible_bg_threshold_mgdl:.0f} mg/dL cap)",
                )
            )
            continue

        if contamination > cfg.contamination_carb_threshold_g:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    f"postprandial contamination "
                    f"(another meal with {contamination:.1f}g carbs after +{cfg.post_grace_min} min)",
                )
            )
            continue

        grid, rel_min, has_full = _bin_cgm(subject.cgm, meal_time, cfg)
        if not has_full:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    "CGM record does not span the full [-pre, +post] window",
                )
            )
            continue

        if cfg.require_last_bin_non_nan and np.isnan(grid[-1]):
            rejected.append(
                RejectedMeal(
                    subject.subject_id, meal_time, "last CGM bin is NaN"
                )
            )
            continue

        max_run = _max_consecutive_nan(grid)
        if max_run > cfg.max_consecutive_nan_bins:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    f"{max_run} consecutive NaN CGM bins "
                    f"(> {cfg.max_consecutive_nan_bins})",
                )
            )
            continue

        postprandial = grid[cfg.n_bins_pre:]
        coverage = float(np.isfinite(postprandial).mean())
        if coverage < cfg.postprandial_coverage_min_frac:
            rejected.append(
                RejectedMeal(
                    subject.subject_id,
                    meal_time,
                    f"postprandial coverage {coverage:.2f} "
                    f"< {cfg.postprandial_coverage_min_frac:.2f}",
                )
            )
            continue

        if cfg.enable_carb_bolus_consistency_filter:
            window_lo = meal_time - pd.Timedelta(minutes=cfg.pre_min)
            window_hi = meal_time + pd.Timedelta(minutes=cfg.post_total_min)
            window_bolus_total = float(
                subject.bolus.loc[
                    (subject.bolus["date"] >= window_lo)
                    & (subject.bolus["date"] <= window_hi),
                    "normal",
                ]
                .fillna(0.0)
                .sum()
            )
            glucose_change = grid[-1] - grid[0]
            if (
                np.isfinite(glucose_change)
                and window_bolus_total == 0.0
                and glucose_change < 0.0
            ):
                rejected.append(
                    RejectedMeal(
                        subject.subject_id,
                        meal_time,
                        f"carb-bolus consistency: carbs={treatment:.1f}g, "
                        f"window_bolus=0U, glucose_change={glucose_change:.0f}mg/dL "
                        f"(meal w/ no bolus and glucose decreased)",
                    )
                )
                continue

        iob = m["insulinOnBoard"] if subject.cohort == 2 else None
        bg = m["bgInput"]

        episodes.append(
            Episode(
                subject_id=subject.subject_id,
                cohort=subject.cohort,
                meal_id=len(episodes) + 1,
                meal_time=meal_time,
                meal_day=int((meal_time.normalize() - first_cgm_date).days) + 1,
                meal_type=_infer_meal_type(meal_time, cfg),
                treatment_carbs=treatment,
                mediator_bolus=mediator,
                iob_at_meal=iob,
                bg_input_at_meal=bg,
                cgm_grid=grid,
                cgm_times_relative_min=rel_min,
                n_consecutive_nan_max=max_run,
                postprandial_coverage_frac=coverage,
                contamination_carbs_max=contamination,
                bolus_merged_count=m["merged_count"],
            )
        )

    return EpisodeBuildResult(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        episodes=episodes,
        rejected=rejected,
        n_candidate_meals=n_candidate,
    )
