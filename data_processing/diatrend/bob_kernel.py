"""Bolus-on-board kernel for DiaTrend's full-sample sensitivity analysis.

Per Section 8.5 of the handoff, the IOB-adjusted primary analysis uses
the 37 cohort-2 subjects (pump-reported ``insulinOnBoard``). The full
54-subject sensitivity arm substitutes a kernel-derived BOB so cohort 1
can be included. This module implements that kernel, the cohort-2
validation step (Pearson/Spearman, Bland-Altman, RMSE) that confirms
the kernel matches pump IOB within an acceptable tolerance, and a
DIA calibration step that minimizes within-subject MSE against the
pump series.

Kernel choice
-------------
A weighted-biexponential decay, normalized so the fraction starts at
1.0 at the moment of bolus delivery and reaches exactly 0.0 at the
duration of insulin action (DIA), with monotone decline in between.
The biexponential form is standard in T1D pharmacokinetic modeling
and gives one rapid-decay term and one slow-decay term, capturing the
shape of subcutaneous-insulin absorption tails better than a single
exponential. DIA = 240 min default, tunable.

The kernel is intentionally separated from the BOB summation so a
caller can swap to a Walsh-quadratic curve, a linear decay, or any
other monotone-decreasing function with the same signature and the
rest of the pipeline still works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from data_processing.diatrend.parser import SubjectData


@dataclass(frozen=True)
class IOBKernelParams:
    """Biexponential IOB-fraction kernel parameters.

    ``dia_min`` is the duration of insulin action: the kernel is
    truncated to 0 for elapsed >= dia_min. ``weight`` is the weight
    on the slow component (0..1); the fast component carries
    ``1 - weight``. ``fast_tau_min`` and ``slow_tau_min`` are the
    time constants of the two exponential terms in minutes.

    Defaults give a smooth monotone decline reaching ~5% at 120 min,
    ~1% at 200 min, exactly 0 at 240 min, which is approximately the
    profile published for rapid-acting analogs.
    """

    dia_min: float = 240.0
    weight: float = 0.7
    fast_tau_min: float = 30.0
    slow_tau_min: float = 120.0

    def __post_init__(self) -> None:
        if not 0.0 < self.weight < 1.0:
            raise ValueError("weight must be in (0, 1)")
        if self.fast_tau_min <= 0 or self.slow_tau_min <= 0:
            raise ValueError("time constants must be positive")
        if self.fast_tau_min >= self.slow_tau_min:
            raise ValueError(
                "fast_tau_min must be strictly less than slow_tau_min "
                "so the fast term decays faster"
            )
        if self.dia_min <= 0:
            raise ValueError("dia_min must be positive")


class IOBFractionFn(Protocol):
    def __call__(
        self, elapsed_min: np.ndarray, params: IOBKernelParams
    ) -> np.ndarray: ...


def iob_fraction(elapsed_min: np.ndarray, params: IOBKernelParams) -> np.ndarray:
    """Fraction of an insulin dose still on board after ``elapsed_min``.

    Vectorized over ``elapsed_min``. Returns values in [0, 1] with
    ``iob_fraction(0, …) == 1`` and ``iob_fraction(>= dia_min, …) == 0``.
    """
    elapsed = np.asarray(elapsed_min, dtype=float)
    out = np.zeros_like(elapsed)
    active = (elapsed >= 0) & (elapsed < params.dia_min)
    if not np.any(active):
        return out

    e = elapsed[active]
    raw_t = (1 - params.weight) * np.exp(-e / params.fast_tau_min) + params.weight * np.exp(
        -e / params.slow_tau_min
    )
    raw_dia = (1 - params.weight) * np.exp(-params.dia_min / params.fast_tau_min) + params.weight * np.exp(
        -params.dia_min / params.slow_tau_min
    )
    out[active] = (raw_t - raw_dia) / (1.0 - raw_dia)
    return out


def compute_bob(
    bolus_times: np.ndarray,
    bolus_doses: np.ndarray,
    query_times: np.ndarray,
    params: IOBKernelParams,
    *,
    fraction_fn: IOBFractionFn = iob_fraction,
    inclusive: bool = False,
) -> np.ndarray:
    """Compute BOB at each query time as the kernel-weighted sum of prior doses.

    Parameters
    ----------
    bolus_times, bolus_doses
        Bolus event times (as numpy datetime64) and corresponding doses (U).
        Both arrays must have the same length.
    query_times
        Times at which to compute BOB.
    params
        Kernel parameters.
    fraction_fn
        The IOB-fraction kernel function. Defaults to the biexponential
        in this module; swap in to test alternative kernels (Walsh,
        linear) with the same signature.
    inclusive
        If True, a bolus at exactly the query time contributes its full
        dose to BOB (elapsed = 0, fraction = 1). If False (default),
        only strictly-prior boluses contribute. The convention matters
        when validating against pump IOB: pump IOB at the moment of a
        new bolus typically REFLECTS prior insulin only, so
        ``inclusive=False`` is the right setting for that comparison.
    """
    bolus_times = np.asarray(bolus_times, dtype="datetime64[ns]")
    bolus_doses = np.asarray(bolus_doses, dtype=float)
    query_times = np.asarray(query_times, dtype="datetime64[ns]")

    if bolus_times.shape != bolus_doses.shape:
        raise ValueError("bolus_times and bolus_doses must be the same length")

    if bolus_times.size == 0:
        return np.zeros(query_times.shape[0], dtype=float)

    out = np.zeros(query_times.shape[0], dtype=float)
    for i, t_q in enumerate(query_times):
        elapsed_sec = (t_q - bolus_times).astype("timedelta64[s]").astype(float)
        elapsed_min = elapsed_sec / 60.0
        if inclusive:
            active = (elapsed_min >= 0.0) & (elapsed_min < params.dia_min)
        else:
            active = (elapsed_min > 0.0) & (elapsed_min < params.dia_min)
        if not np.any(active):
            continue
        contributions = bolus_doses[active] * fraction_fn(elapsed_min[active], params)
        out[i] = float(contributions.sum())
    return out


def bob_at_meal_times(
    bolus_df: pd.DataFrame,
    params: IOBKernelParams,
    *,
    require_meal: bool = True,
) -> pd.DataFrame:
    """Compute kernel BOB at each meal-bolus row of a subject.

    Returns the input frame with two added columns:
    ``kernel_bob`` (kernel-derived BOB before this bolus) and
    ``elapsed_first_bolus_min`` (minutes since the first bolus event
    in the subject's record; 0 for the first bolus).

    ``require_meal=True`` restricts the output to rows with
    ``carbInput > 0``; set False to compute BOB at every bolus row.
    """
    if bolus_df.empty:
        out = bolus_df.copy()
        out["kernel_bob"] = np.array([], dtype=float)
        out["elapsed_first_bolus_min"] = np.array([], dtype=float)
        return out

    times = bolus_df["date"].to_numpy()
    doses = bolus_df["normal"].fillna(0.0).to_numpy()
    bob = compute_bob(times, doses, times, params, inclusive=False)

    elapsed = (
        (bolus_df["date"] - bolus_df["date"].iloc[0])
        .dt.total_seconds()
        .to_numpy()
        / 60.0
    )
    out = bolus_df.copy()
    out["kernel_bob"] = bob
    out["elapsed_first_bolus_min"] = elapsed
    if require_meal:
        out = out.loc[out["carbInput"] > 0].reset_index(drop=True)
    return out


@dataclass(frozen=True)
class ValidationStats:
    """Per-subject (or pooled) agreement between kernel BOB and pump IOB."""

    subject_id: str
    n: int
    pearson_r: float
    spearman_r: float
    bland_altman_bias: float
    bland_altman_loa_lower: float
    bland_altman_loa_upper: float
    residual_mean: float
    residual_sd: float
    rmse: float


def _correlate(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        if np.std(rx) == 0 or np.std(ry) == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])
    raise ValueError(method)


def validate_kernel(
    subject: SubjectData, params: IOBKernelParams
) -> ValidationStats | None:
    """Compare kernel BOB to pump IOB at each cohort-2 meal-bolus row.

    Returns ``None`` for cohort 1 (no pump IOB to validate against) or
    when fewer than 2 paired observations are available.
    """
    if subject.cohort != 2:
        return None
    if "insulinOnBoard" not in subject.bolus.columns:
        return None

    enriched = bob_at_meal_times(subject.bolus, params, require_meal=False)
    paired = enriched.dropna(subset=["insulinOnBoard"])
    if len(paired) < 2:
        return None

    pump = paired["insulinOnBoard"].to_numpy(dtype=float)
    kernel = paired["kernel_bob"].to_numpy(dtype=float)
    diff = pump - kernel

    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
    if np.isnan(sd):
        loa_lo = loa_hi = float("nan")
    else:
        loa_lo = bias - 1.96 * sd
        loa_hi = bias + 1.96 * sd

    return ValidationStats(
        subject_id=subject.subject_id,
        n=int(diff.size),
        pearson_r=_correlate(pump, kernel, "pearson"),
        spearman_r=_correlate(pump, kernel, "spearman"),
        bland_altman_bias=bias,
        bland_altman_loa_lower=loa_lo,
        bland_altman_loa_upper=loa_hi,
        residual_mean=bias,
        residual_sd=sd,
        rmse=float(np.sqrt(np.mean(diff ** 2))),
    )


def calibrate_dia(
    subjects: list[SubjectData],
    *,
    base_params: IOBKernelParams = IOBKernelParams(),
    dia_grid_min: tuple[float, ...] = tuple(range(120, 361, 15)),
) -> tuple[float, dict[float, float]]:
    """Pick the DIA that minimizes the within-subject mean squared error
    between kernel BOB and pump IOB, pooled across cohort-2 subjects.

    Other kernel parameters (weight, time constants) are held at their
    values in ``base_params``; only ``dia_min`` is varied across the
    grid. Returns ``(best_dia_min, mse_by_dia)``.
    """
    mse_by_dia: dict[float, float] = {}
    for dia in dia_grid_min:
        params = IOBKernelParams(
            dia_min=float(dia),
            weight=base_params.weight,
            fast_tau_min=base_params.fast_tau_min,
            slow_tau_min=base_params.slow_tau_min,
        )
        total_sq = 0.0
        total_n = 0
        for subject in subjects:
            if subject.cohort != 2:
                continue
            stats = validate_kernel(subject, params)
            if stats is None or not np.isfinite(stats.rmse):
                continue
            total_sq += stats.rmse ** 2 * stats.n
            total_n += stats.n
        mse_by_dia[float(dia)] = total_sq / total_n if total_n > 0 else float("nan")

    finite = {d: m for d, m in mse_by_dia.items() if np.isfinite(m)}
    if not finite:
        return base_params.dia_min, mse_by_dia
    best_dia = min(finite, key=lambda d: finite[d])
    return best_dia, mse_by_dia
