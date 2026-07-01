"""Generate synthetic DiaTrend-shaped Excel workbooks for testing.

The generator is the source of truth; the workbooks it writes are not
tracked in git (``*.xlsx`` is gitignored). At pytest session start, a
small collection of workbooks is written into the session's tmp_path
and discarded when the session ends. This keeps the repo free of binary
fixtures and guarantees that fixtures never drift from their generator.

The five-subject plan below exercises every DiaTrend quirk called out
in the project handoff:

- both cohorts (1: base bolus + Basal sheet; 2: extended bolus + no
  Basal), so the parser's schema dispatch is covered;
- chronological and reverse-chronological row orderings, so the
  monotonicity detector is covered (Section 6.4);
- off-grid bolus timestamps (seconds-level jitter), so the episode
  builder's snap-or-not behaviour can be exercised downstream;
- one episode with a deliberate pre-meal CGM gap, so the missingness
  filter can be exercised downstream;
- one cohort-2 fixture with an empty (but present) Basal sheet, so the
  parser's tolerance of the extra sheet is covered.

Subject identifiers are constructed at runtime from integers in the
1001..1099 range, deliberately outside the real subject-ID range
(S1..S54). They are never embedded as literals so the leak guard does
not see a ``S\\d+`` pattern in this file's source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

FIXTURE_BASE_DATE: datetime = datetime(2020, 1, 1, 8, 0, 0)
CGM_GRID_MIN: int = 5


@dataclass(frozen=True)
class FixturePlan:
    subject_index: int
    cohort: Literal[1, 2]
    days: int
    reversed_order: bool
    basal_present: bool
    basal_empty: bool
    induce_missing_premeal: bool


DEFAULT_PLAN: tuple[FixturePlan, ...] = (
    FixturePlan(1001, 1, 10, False, True,  False, False),
    FixturePlan(1002, 1,  8, True,  True,  False, False),
    FixturePlan(1003, 2, 12, False, False, False, False),
    FixturePlan(1004, 2,  7, True,  False, False, False),
    FixturePlan(1005, 2,  9, False, True,  True,  True),
)


def subject_id(plan: FixturePlan) -> str:
    return f"S{plan.subject_index}"


def _cgm(
    days: int,
    rng: np.random.Generator,
    start: datetime,
    drop_window: tuple[datetime, datetime] | None,
) -> pd.DataFrame:
    n = days * 24 * (60 // CGM_GRID_MIN)
    jitter = rng.integers(low=0, high=60, size=n)
    ts = [
        start + timedelta(minutes=CGM_GRID_MIN * i, seconds=int(jitter[i]))
        for i in range(n)
    ]
    diurnal = 30.0 * np.sin(np.arange(n) * 2 * np.pi / 288.0)
    noise = rng.normal(0.0, 20.0, size=n)
    glucose = np.clip(150.0 + diurnal + noise, 60.0, 350.0)
    frame = pd.DataFrame({"date": ts, "mg/dl": glucose})
    if drop_window is not None:
        lo, hi = drop_window
        frame = frame.loc[(frame["date"] < lo) | (frame["date"] >= hi)].reset_index(drop=True)
    return frame


def _bolus_row(
    t: datetime, rng: np.random.Generator, extended: bool, *, is_meal: bool
) -> dict:
    if is_meal:
        carbs = float(rng.integers(30, 80))
        ratio = float(rng.choice([8.0, 10.0, 12.0]))
        normal = carbs / ratio
        bg = float(rng.integers(90, 220))
    else:
        carbs = 0.0
        ratio = 10.0
        normal = float(rng.uniform(0.5, 2.5))
        bg = float(rng.integers(180, 260))
    row = {
        "date": t,
        "normal": normal,
        "carbInput": carbs,
        "insulinCarbRatio": ratio,
        "bgInput": bg,
        "recommended.carb": carbs / ratio if is_meal else 0.0,
        "recommended.net": normal,
    }
    if extended:
        row["recommended.correction"] = max(0.0, (bg - 120.0) / 50.0)
        row["insulinSensitivityFactor"] = 50.0
        row["targetBloodGlucose"] = 120.0
        row["insulinOnBoard"] = float(rng.uniform(0.0, 4.0))
    return row


def _bolus(
    days: int, rng: np.random.Generator, start: datetime, extended: bool
) -> pd.DataFrame:
    rows: list[dict] = []
    for d in range(days):
        for hour in (7, 12, 19):
            t = start + timedelta(
                days=d,
                hours=hour - start.hour,
                minutes=int(rng.integers(-10, 10)),
                seconds=int(rng.integers(0, 60)),
            )
            rows.append(_bolus_row(t, rng, extended, is_meal=True))
        if rng.random() < 0.4:
            t = start + timedelta(
                days=d,
                hours=int(rng.integers(0, 24)),
                minutes=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 60)),
            )
            rows.append(_bolus_row(t, rng, extended, is_meal=False))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _basal(days: int, rng: np.random.Generator, start: datetime) -> pd.DataFrame:
    rows: list[dict] = []
    rate = 0.8
    for d in range(days):
        for _ in range(int(rng.integers(0, 3))):
            t = start + timedelta(
                days=d,
                hours=int(rng.integers(0, 24)),
                minutes=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 60)),
            )
            rate = float(np.clip(rate + rng.normal(0.0, 0.1), 0.3, 1.5))
            rows.append(
                {
                    "date": t,
                    "duration": int(rng.integers(30, 240) * 60 * 1000),
                    "rate": rate,
                }
            )
    if not rows:
        rows.append({"date": start, "duration": 24 * 60 * 60 * 1000, "rate": rate})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _write(
    path: Path,
    cgm: pd.DataFrame,
    bolus: pd.DataFrame,
    basal: pd.DataFrame | None,
    reverse: bool,
) -> None:
    if reverse:
        cgm = cgm.iloc[::-1].reset_index(drop=True)
        bolus = bolus.iloc[::-1].reset_index(drop=True)
        if basal is not None and not basal.empty:
            basal = basal.iloc[::-1].reset_index(drop=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        cgm.to_excel(writer, sheet_name="CGM", index=False)
        bolus.to_excel(writer, sheet_name="Bolus", index=False)
        if basal is not None:
            basal.to_excel(writer, sheet_name="Basal", index=False)


def generate_one(plan: FixturePlan, output_dir: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed + plan.subject_index)
    start = FIXTURE_BASE_DATE
    drop = None
    if plan.induce_missing_premeal:
        meal_t = start + timedelta(days=1, hours=12 - start.hour)
        drop = (meal_t - timedelta(hours=2), meal_t + timedelta(hours=1))
    cgm = _cgm(plan.days, rng, start, drop_window=drop)
    bolus = _bolus(plan.days, rng, start, extended=(plan.cohort == 2))
    basal: pd.DataFrame | None = None
    if plan.basal_present:
        if plan.basal_empty:
            basal = pd.DataFrame(columns=["date", "duration", "rate"])
        else:
            basal = _basal(plan.days, rng, start)
    path = Path(output_dir) / f"{subject_id(plan)}.xlsx"
    _write(path, cgm, bolus, basal, reverse=plan.reversed_order)
    return path


def generate_all(
    output_dir: Path | str,
    seed: int = 42,
    plans: tuple[FixturePlan, ...] | None = None,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if plans is None:
        plans = DEFAULT_PLAN
    return [generate_one(plan, output_dir, seed) for plan in plans]
