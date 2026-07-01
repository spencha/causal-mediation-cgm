"""Feature distribution figure for DiaTrend.

Walks a directory of raw DiaTrend ``.xlsx`` workbooks, parses each,
builds meal-centered episodes, and writes a multi-panel PNG showing
the empirical distribution of the four features that decisions about
filter cutoffs depend on: ``carbInput``, ``normal`` (bolus dose at
the meal event), ``bgInput``, and ``insulinOnBoard``. Two views per
feature: the **raw** distribution across every meal-bolus row
(carbInput > 0) and the **retained** distribution across episodes
that survive the episode-builder filter set.

A sidecar ``.txt`` is also written with the same percentile values
that appear on the figure, so a reviewer can read the exact numbers
without sampling pixels.

The figure and sidecar are aggregated-only (histograms + percentile
markers) and safe to share — no individual timestamps or
single-subject curves appear.

Usage:
    python visualization_code/feature_distributions_diatrend.py \\
        --input-dir /path/to/DiaTrend/raw \\
        --output analysis_data/diatrend/diagnostics/feature_distributions.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_processing.diatrend.episode_builder import build_episodes  # noqa: E402
from data_processing.diatrend.parser import (  # noqa: E402
    DiaTrendParseError,
    parse_subject,
)


PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9)


@dataclass
class FeatureBundle:
    raw_carb: np.ndarray
    raw_normal: np.ndarray
    raw_bg: np.ndarray
    raw_iob: np.ndarray
    ret_carb: np.ndarray
    ret_mediator: np.ndarray
    ret_bg: np.ndarray
    ret_iob: np.ndarray


def collect_features(input_dir: Path) -> FeatureBundle:
    raw_carb: list[np.ndarray] = []
    raw_normal: list[np.ndarray] = []
    raw_bg: list[np.ndarray] = []
    raw_iob: list[np.ndarray] = []
    ret_carb: list[float] = []
    ret_mediator: list[float] = []
    ret_bg: list[float] = []
    ret_iob: list[float] = []

    workbooks = sorted(input_dir.glob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No .xlsx files in {input_dir}")

    n_parsed = 0
    n_failed = 0
    for path in workbooks:
        try:
            subject = parse_subject(path)
        except DiaTrendParseError:
            n_failed += 1
            continue
        n_parsed += 1

        meals = subject.bolus.loc[subject.bolus["carbInput"] > 0]
        if not meals.empty:
            raw_carb.append(meals["carbInput"].dropna().to_numpy(dtype=float))
            raw_normal.append(meals["normal"].dropna().to_numpy(dtype=float))
            raw_bg.append(meals["bgInput"].dropna().to_numpy(dtype=float))
            if subject.cohort == 2 and "insulinOnBoard" in meals.columns:
                raw_iob.append(
                    meals["insulinOnBoard"].dropna().to_numpy(dtype=float)
                )

        result = build_episodes(subject)
        for ep in result.episodes:
            ret_carb.append(float(ep.treatment_carbs))
            ret_mediator.append(float(ep.mediator_bolus))
            if ep.bg_input_at_meal is not None and np.isfinite(ep.bg_input_at_meal):
                ret_bg.append(float(ep.bg_input_at_meal))
            if ep.iob_at_meal is not None and np.isfinite(ep.iob_at_meal):
                ret_iob.append(float(ep.iob_at_meal))

    print(f"  Parsed {n_parsed} subjects ({n_failed} failed).")
    print(f"  Retained {len(ret_carb)} episodes.")

    def cat(lists: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(lists) if lists else np.array([], dtype=float)

    return FeatureBundle(
        raw_carb=cat(raw_carb),
        raw_normal=cat(raw_normal),
        raw_bg=cat(raw_bg),
        raw_iob=cat(raw_iob),
        ret_carb=np.asarray(ret_carb, dtype=float),
        ret_mediator=np.asarray(ret_mediator, dtype=float),
        ret_bg=np.asarray(ret_bg, dtype=float),
        ret_iob=np.asarray(ret_iob, dtype=float),
    )


def _percentile_dict(values: np.ndarray) -> dict[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {p: float("nan") for p in PERCENTILES}
    pcts = np.percentile(finite, PERCENTILES)
    return dict(zip(PERCENTILES, [float(v) for v in pcts]))


def plot_panel(
    ax,
    values: np.ndarray,
    label: str,
    *,
    bins: int = 80,
    log_y: bool = True,
    clip_quantile: float | None = 0.999,
) -> dict[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        ax.set_title(f"{label}\n(no data)", fontsize=10)
        ax.set_axis_off()
        return {p: float("nan") for p in PERCENTILES}

    plot_vals = finite
    if clip_quantile is not None and finite.size > 50:
        clip = float(np.quantile(finite, clip_quantile))
        plot_vals = finite[finite <= clip]

    ax.hist(plot_vals, bins=bins, alpha=0.6, edgecolor="black", linewidth=0.3)
    if log_y:
        ax.set_yscale("log")

    pcts = np.percentile(finite, PERCENTILES)
    cmap = plt.get_cmap("viridis")
    n = len(PERCENTILES)
    for i, (p, val) in enumerate(zip(PERCENTILES, pcts)):
        color = cmap(i / max(1, n - 1))
        ax.axvline(val, color=color, linestyle="--", alpha=0.85, linewidth=1.0)

    legend_lines = [f"p{p:g}={v:.2f}" for p, v in zip(PERCENTILES, pcts)]
    legend_lines.append(f"max={finite.max():.2f}")
    legend_lines.append(f"n={finite.size}")
    ax.text(
        0.98,
        0.97,
        "\n".join(legend_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    title = label
    if clip_quantile is not None and plot_vals.size < finite.size:
        title += f" (x-axis clipped at p{clip_quantile * 100:g} for readability)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(label.split(" — ")[0] if " — " in label else label)
    ax.set_ylabel("count (log)" if log_y else "count")
    return dict(zip(PERCENTILES, [float(v) for v in pcts]))


def write_sidecar(path: Path, summary: dict[str, dict[float, float]]) -> None:
    lines = [
        "DiaTrend feature distributions — percentile summary",
        "",
        f"Percentiles computed: {', '.join('p' + str(p) for p in PERCENTILES)}",
        "",
    ]
    for name, pcts in summary.items():
        lines.append(f"## {name}")
        for p, v in pcts.items():
            lines.append(f"  p{p:g} = {v:.3f}")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Plot DiaTrend feature distributions (raw + retained)."
    )
    p.add_argument("--input-dir", required=True, help="Directory of .xlsx workbooks.")
    p.add_argument(
        "--output",
        default="feature_distributions_diatrend.png",
        help="Output figure path (.png).",
    )
    args = p.parse_args(argv)

    input_dir = Path(args.input_dir).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading workbooks from {input_dir} ...")
    bundle = collect_features(input_dir)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))

    summary: dict[str, dict[float, float]] = {}

    summary["carbInput — raw meal-bolus rows"] = plot_panel(
        axes[0, 0], bundle.raw_carb, "carbInput (g) — raw meal-bolus rows"
    )
    summary["treatment_carbs — retained episodes"] = plot_panel(
        axes[0, 1], bundle.ret_carb, "treatment_carbs (g) — retained episodes"
    )

    summary["normal — raw meal-bolus rows"] = plot_panel(
        axes[1, 0], bundle.raw_normal, "normal (U) — raw meal-bolus rows"
    )
    summary["mediator_bolus — retained episodes"] = plot_panel(
        axes[1, 1],
        bundle.ret_mediator,
        "mediator_bolus (U) — retained episodes",
    )

    summary["bgInput — raw meal-bolus rows"] = plot_panel(
        axes[2, 0], bundle.raw_bg, "bgInput (mg/dL) — raw meal-bolus rows"
    )
    summary["bgInput at meal — retained episodes"] = plot_panel(
        axes[2, 1],
        bundle.ret_bg,
        "bgInput at meal (mg/dL) — retained episodes",
    )

    summary["insulinOnBoard — raw [cohort 2]"] = plot_panel(
        axes[3, 0],
        bundle.raw_iob,
        "insulinOnBoard (U) — raw [cohort 2 only]",
    )
    summary["iob_at_meal — retained [cohort 2]"] = plot_panel(
        axes[3, 1],
        bundle.ret_iob,
        "iob_at_meal (U) — retained episodes [cohort 2 only]",
    )

    fig.suptitle(
        "DiaTrend feature distributions: raw meal-bolus rows vs episodes "
        "retained after the OhioT1DM-matching filter set\n"
        "(dashed vertical lines = percentiles; x-axis clipped at p99.9 for readability; "
        "true max shown in the legend box)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(output, dpi=120, bbox_inches="tight")
    print(f"\nSaved figure: {output}")

    sidecar = output.with_suffix(".percentiles.txt")
    write_sidecar(sidecar, summary)
    print(f"Saved percentile summary: {sidecar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
