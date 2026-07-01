#!/usr/bin/env python3
"""Generate DiaTrend mediation figures & tables that MATCH the OhioT1DM ones.

Both datasets now share one plotting primitive (``mediation_plotting``) and this
script drives OhioT1DM's *own* figure functions (``generate_mediation_outputs``)
so DiaTrend and Ohio figures are identical by construction, not by two scripts
independently trying to look alike. The DiaTrend ``arm`` (base / demowt /
demographics / ...) plays the role of Ohio's ``covariate_mode`` (phi / pca).
The 'base' arm is the OhioT1DM-literal spec (no demographic covariates).

Consumes the grid CSV(s) produced by the quasi-Bayesian sweep
(``cma_cluster/diatrend/run_all_timepoints.R``) -- columns: arm, meal, model,
tau, offset_g, timepoint, n_episodes, n_subjects, acme[,_lo,_hi,_p], ade[...],
total[...], prop_mediated, prop_p. (The bootstrap is a SEPARATE robustness
figure; see plot_bootstrap_results / the bootstrap grid. The quasi-Bayesian grid
is what matches Ohio's inference.) Figures are always built from these result
data frames, never from parsed printed tables.

Output layout is <out>/<arm>/<inference>/, where <inference> is quasi_bayes or
bootstrap (set with --inference). For each arm it writes, under that dir:
  fig_mediation_<arm>.{png,pdf}             all-meals x model overview (npj Fig 3)
  table3_pooled_<arm>.tex                   pooled timepoint x dose (Table 3)
  table4_t<t>_off<o>_<arm>.tex              meal x model at one cell (Table 4)
  grid_tidy_<arm>.csv                       the same numbers, tidy.
  <meal>/fig_qr_panel[_offsetNg].png        LMER + QR-tau panels  (Ohio layout)
  <meal>/fig_offset_comparison_lmer.png     offsets side by side  (Ohio layout)
  <meal>/fig_effects_<model>[_offsetNg].png single-panel journal figures

So e.g. mediation_results/diatrend/figures/base/quasi_bayes/ and
        mediation_results/diatrend/figures/base/bootstrap/.

Usage:
  python visualization_code/generate_diatrend_mediation_outputs.py \
      --grid 'mediation_results/diatrend/grid_diatrend_*_base.csv' \
      --out  mediation_results/diatrend/figures --inference quasi_bayes \
      --table3-model lmer --table4-timepoint 120 --table4-offset 30
"""
from __future__ import annotations

import argparse
import glob
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Shared primitive + Ohio's figure functions (single source of truth).
import mediation_plotting as mp
from mediation_plotting import (
    COLORS, canonicalize, sort_meal_types, normalize_meal_type,
    format_p, format_est, format_ci,
)
import generate_mediation_outputs as ohio

# Force DejaVu Sans (matplotlib's bundled font). It has a real bold face
# (DejaVuSans-Bold.ttf), so fontweight="bold" actually renders bold; it matches
# the manuscript figures (whose Helvetica->Arial->DejaVu fallback lands on DejaVu
# on a headless render); and it renders identically on the Mac and the cluster.
# NB: macOS Helvetica.ttc has no usable bold face, so leaving the default would
# silently render every "bold" label in regular weight. Set after the imports
# above, which install a Helvetica-first style.
plt.rcParams["font.family"] = "DejaVu Sans"

# Canonical (model, tau) columns for the overview grid; QR taus only shown if present.
# Headers use the unicode tau to match the npj Figure 3 column titles exactly.
MODEL_COLS = [
    ("lmer", None, "LMER (Mean)"),
    ("qr", 0.25, "τ = 0.25"),
    ("qr", 0.50, "τ = 0.50"),
    ("qr", 0.75, "τ = 0.75"),
]
MEAL_ROWS = [
    ("ALL", "Pooled"), ("Breakfast", "Breakfast"), ("Lunch", "Lunch"),
    ("Dinner", "Dinner"), ("Snack", "Snack"),
]
# (canonical est col, lo, hi, p, colour, marker, legend label) -- matches the npj
# manuscript's _MEDIATION_EFFECTS in compose_paper_visualizations.py.
EFFECTS = [
    ("ACME", "ACME_lower", "ACME_upper", "ACME_p", "#2ecc71", "o", "ACME (Indirect)"),
    ("ADE", "ADE_lower", "ADE_upper", "ADE_p", "#e74c3c", "s", "ADE (Direct)"),
    ("total_effect", "total_lower", "total_upper", "total_p", "#3498db", "^", "Total Effect"),
]

# Figures plot every timepoint in the grid (5-min density, matching Ohio); the
# manuscript Table 3 stays at these 30-min summary rows so it doesn't balloon.
SUMMARY_TIMEPOINTS = [60, 90, 120, 150, 180, 210]


def load_grid(paths: list[str]) -> pd.DataFrame:
    """Load + canonicalize every grid CSV; arm -> covariate_mode for Ohio reuse."""
    frames = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            df = pd.read_csv(f)
            if "arm" not in df.columns:
                df["arm"] = Path(f).stem.split("_")[-1]
            frames.append(df)
    if not frames:
        raise SystemExit(f"No grid CSVs matched: {paths}")
    g = canonicalize(pd.concat(frames, ignore_index=True))
    # Ohio's functions key off 'covariate_mode'; the DiaTrend arm is its analog.
    g["covariate_mode"] = g["arm"]
    return g


# --------------------------- overview grid (all meals) ---------------------------
def _cell(df, meal, model, tau, offset):
    sub = df[(df["meal_type"] == meal) & (df["model"] == model)
             & (df["treat_offset"] == offset) & (df["status"] == "success")]
    if model == "qr":
        sub = sub[(sub["quantile_tau"] - tau).abs() < 1e-6]
    return sub


_MEDIATION_XTICKS = [60, 90, 120, 150, 180, 210]


def _plot_npj_panel(ax, sub):
    """One meal×model cell in the npj Figure 3 style (compose_paper_visualizations.py):
    line lw=1.5, faint CI band (alpha 0.15), large star for p<0.05, small shape
    marker otherwise; no per-axes labels/legend (shared at the figure level)."""
    if sub is None or sub.empty:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.5")
        return
    t = sub["minutes"].to_numpy()
    for est, lo, hi, pc, color, marker, _ in EFFECTS:
        vals = sub[est].to_numpy()
        ax.plot(t, vals, "-", color=color, linewidth=1.5)
        ax.fill_between(t, sub[lo].to_numpy(), sub[hi].to_numpy(), color=color, alpha=0.15)
        sig = sub[pc].to_numpy() < 0.05
        if sig.any():
            ax.plot(t[sig], vals[sig], "*", color=color, markersize=14,
                    markeredgecolor="white", markeredgewidth=0.5, linestyle="none")
        if (~sig).any():
            ax.plot(t[~sig], vals[~sig], marker, color=color, markersize=4, linestyle="none")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(55, 215)
    ax.set_xticks(_MEDIATION_XTICKS)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.15)


def _npj_legend_handles():
    h = [Line2D([0], [0], color=color, marker=marker, markersize=7, linewidth=2, label=lbl)
         for _, _, _, _, color, marker, lbl in EFFECTS]
    h.append(Line2D([0], [0], linestyle="none", marker="*", color="gray",
                    markersize=12, label="Significant (p < 0.05)"))
    return h


def make_overview_figure(df_arm: pd.DataFrame, arm: str, offset: int, out: Path):
    """All-meals (rows) x model (cols) overview, styled to match the npj-approved
    manuscript Figure 3 (compose_paper_visualizations.py): bold column headers,
    right-side rotated meal labels with 'N = ', shared 'Effect on Glucose (mg/dL)'
    /'Minutes Post Meal' labels, and a significance entry in the shared legend.
    Per-row shared y so each meal keeps its own scale; no in-figure title (the
    arm is in the filename; journal figures carry a caption instead)."""
    meals = [(k, lbl) for k, lbl in MEAL_ROWS if (df_arm["meal_type"] == k).any()]
    models = [(m, tau, lbl) for m, tau, lbl in MODEL_COLS
              if not _cell(df_arm, meals[0][0], m, tau, offset).empty] if meals else []
    if not meals or not models:
        print(f"  [{arm}] nothing to plot for offset {offset}")
        return
    nrow, ncol = len(meals), len(models)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.0 * nrow),
                             sharex=True, sharey="row", squeeze=False)
    letters = iter(string.ascii_lowercase)
    for i, (mk, mlbl) in enumerate(meals):
        n_meal = int(df_arm[df_arm["meal_type"] == mk]["n_obs"].max())
        for j, (model, tau, modlbl) in enumerate(models):
            ax = axes[i][j]
            _plot_npj_panel(ax, _cell(df_arm, mk, model, tau, offset))
            # Panel letter: small bold, just outside the top-left corner (npj Fig 3).
            ax.text(-0.03, 1.01, next(letters), transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="bottom", ha="right")
            if i == 0:
                ax.set_title(modlbl, fontsize=15, fontweight="bold", pad=10)
            if j == ncol - 1:
                ax.annotate(f"{mlbl} (N = {n_meal})", xy=(1, 0.5),
                            xytext=(10, 0), textcoords="offset points",
                            xycoords="axes fraction", fontsize=14, fontweight="bold",
                            va="center", ha="left", rotation=-90)
    axes[nrow // 2][0].set_ylabel("Effect on Glucose (mg/dL)", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.01, "Minutes Post Meal", ha="center", fontsize=18, fontweight="bold")
    fig.legend(handles=_npj_legend_handles(), loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=14,
               frameon=True, fancybox=True, edgecolor="0.6", framealpha=0.9)
    # Reserve a right margin so the rotated 'N = ...' row labels aren't clipped.
    fig.tight_layout(rect=[0.02, 0.04, 0.93, 0.99])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mediation_{arm}.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [{arm}] wrote fig_mediation_{arm}.png/.pdf (npj Figure 3 style)")


# --------------------------- Ohio-exact per-meal figures ---------------------------
def make_ohio_match_figures(df_arm: pd.DataFrame, arm: str, out: Path):
    """Drive Ohio's own figure functions per meal so the layouts match exactly."""
    offsets = sorted(df_arm["treat_offset"].unique().tolist())
    multi_offset = len(offsets) > 1
    has_qr = (df_arm["model"] == "qr").any()
    meals = sort_meal_types(list(df_arm["meal_type"].unique()))
    for meal in meals:
        d = out / meal.lower()   # `out` is already the arm/inference dir
        d.mkdir(parents=True, exist_ok=True)
        meal_df = df_arm[df_arm["meal_type"] == meal]
        for off in offsets:
            suffix = f"_offset{off}g" if multi_offset else ""
            ohio.plot_individual_effects_figure(
                df_arm, d, cov_mode=arm, meal_type=meal, model_type="lmer",
                offset=off, filename=f"fig_effects_lmer{suffix}.png")
            if has_qr:
                ohio.plot_qr_panel_figure(
                    meal_df, d, offset=off, cov_mode=arm, meal_type=meal,
                    filename=f"fig_qr_panel{suffix}.png")
        if multi_offset:
            ohio.plot_offset_comparison_figure(
                meal_df, d, model_type="lmer", cov_mode=arm, meal_type=meal,
                filename="fig_offset_comparison_lmer.png")
    print(f"  [{arm}] wrote Ohio-style per-meal figures under {arm}/<meal>/")


# --------------------------- LaTeX tables (manuscript Table 3/4) ---------------------------
def table3_pooled(df_arm, arm, model, tau, out):
    """Pooled: timepoint x dose, ACME/ADE/Total (est, CI, p) — manuscript Table 3."""
    d = df_arm[(df_arm["meal_type"] == "ALL") & (df_arm["model"] == model)
               & (df_arm["status"] == "success")]
    if model == "qr":
        d = d[(d["quantile_tau"] - tau).abs() < 1e-6]
    if d.empty:
        print(f"  [{arm}] no pooled {model} rows for Table 3")
        return
    rows = []
    summary_t = [t for t in SUMMARY_TIMEPOINTS if (d["minutes"] == t).any()]
    for t in summary_t:
        for k, off in enumerate(sorted(d["treat_offset"].unique())):
            r = d[(d["minutes"] == t) & (d["treat_offset"] == off)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = [f"{int(t)} min" if k == 0 else "", f"+{int(off)} g"]
            for e, lo, hi, pc in [("ACME", "ACME_lower", "ACME_upper", "ACME_p"),
                                  ("ADE", "ADE_lower", "ADE_upper", "ADE_p"),
                                  ("total_effect", "total_lower", "total_upper", "total_p")]:
                sig = r[pc] < 0.05
                cells += [format_est(r[e], sig), format_ci(r[lo], r[hi], sig), format_p(r[pc], sig)]
            rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    mlabel = "LMER" if model == "lmer" else f"QR ($\\tau={tau:.2f}$)"
    n = int(df_arm[df_arm["meal_type"] == "ALL"]["n_obs"].iloc[0])
    latex = rf"""\begin{{table*}}[ht]
\centering
\caption{{\textbf{{DiaTrend ({arm}) --- Pooled (all meals): {mlabel} causal mediation effects across treatment doses ($N={n}$ meal observations).}} ACME = average causal mediation effect (through insulin); ADE = average direct effect; Total = ACME + ADE. Est.\ in mg/dL; 95\% CI and $p$ from quasi-Bayesian approximation. Significant ($p<0.05$) in bold.}}
\label{{tab:diatrend_pooled_{arm}}}
\small
\begin{{tabular}}{{ll ccc ccc ccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{ACME}} & \multicolumn{{3}}{{c}}{{ADE}} & \multicolumn{{3}}{{c}}{{Total}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-8}}\cmidrule(lr){{9-11}}
Time & Dose & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / f"table3_pooled_{arm}.tex").write_text(latex)
    print(f"  [{arm}] wrote table3_pooled_{arm}.tex")


def table4_summary(df_arm, arm, timepoint, offset, out):
    """meal x model at one (timepoint, offset): ADE/ACME/Total (est, p) — Table 4."""
    d = df_arm[(df_arm["minutes"] == timepoint) & (df_arm["treat_offset"] == offset)
               & (df_arm["status"] == "success")]
    if d.empty:
        print(f"  [{arm}] no rows at t={timepoint}, +{offset}g for Table 4")
        return
    rows = []
    for mk, mlbl in MEAL_ROWS:
        first = True
        for model, tau, modlbl in MODEL_COLS:
            r = d[(d["meal_type"] == mk) & (d["model"] == model)]
            if model == "qr":
                r = r[(r["quantile_tau"] - tau).abs() < 1e-6]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = [mlbl if first else "", modlbl]
            first = False
            for e, pc in [("ADE", "ADE_p"), ("ACME", "ACME_p"), ("total_effect", "total_p")]:
                sig = r[pc] < 0.05
                cells += [format_est(r[e], sig), format_p(r[pc], sig)]
            rows.append(" & ".join(cells) + r" \\")
        if not first:
            rows.append(r"\midrule")
    body = "\n".join(rows)
    latex = rf"""\begin{{table*}}[ht]
\centering
\caption{{\textbf{{DiaTrend ({arm}): mediation effects at {timepoint} min post-meal (+{offset} g contrast).}} ADE = average direct effect; ACME = average causal mediation effect; Total = ADE + ACME (mg/dL). $p$ from quasi-Bayesian approximation; bold = $p<0.05$.}}
\label{{tab:diatrend_summary_{arm}}}
\small
\begin{{tabular}}{{ll cc cc cc}}
\toprule
& & \multicolumn{{2}}{{c}}{{ADE}} & \multicolumn{{2}}{{c}}{{ACME}} & \multicolumn{{2}}{{c}}{{Total}} \\
\cmidrule(lr){{3-4}}\cmidrule(lr){{5-6}}\cmidrule(lr){{7-8}}
Meal & Model & Est. & $p$ & Est. & $p$ & Est. & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    (out / f"table4_t{timepoint}_off{offset}_{arm}.tex").write_text(latex)
    print(f"  [{arm}] wrote table4_t{timepoint}_off{offset}_{arm}.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", action="append", required=True,
                    help="Grid CSV path/glob (repeatable, one or more arms).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Root output dir. Outputs go to <out>/<arm>/<inference>/.")
    ap.add_argument("--inference", default="quasi_bayes",
                    choices=["quasi_bayes", "bootstrap"],
                    help="Inference method these grids came from; names the subdir. "
                         "[quasi_bayes]")
    ap.add_argument("--fig-offset", type=int, default=30,
                    help="Offset for the all-meals overview grid. [30]")
    ap.add_argument("--table3-model", default="lmer")
    ap.add_argument("--table3-tau", type=float, default=0.5)
    ap.add_argument("--table4-timepoint", type=int, default=120)
    ap.add_argument("--table4-offset", type=int, default=30)
    args = ap.parse_args()

    g = load_grid(args.grid)
    for arm in sorted(g["covariate_mode"].unique()):
        da = g[g["covariate_mode"] == arm].copy()
        # Layout: <out>/<arm>/<inference>/ -- arm first, then bootstrap vs not.
        arm_dir = args.out / arm / args.inference
        arm_dir.mkdir(parents=True, exist_ok=True)
        print(f"Arm '{arm}' ({args.inference}): {len(da)} cells -> {arm_dir}")
        make_overview_figure(da, arm, args.fig_offset, arm_dir)
        make_ohio_match_figures(da, arm, arm_dir)
        table3_pooled(da, arm, args.table3_model, args.table3_tau, arm_dir)
        table4_summary(da, arm, args.table4_timepoint, args.table4_offset, arm_dir)
        da.to_csv(arm_dir / f"grid_tidy_{arm}.csv", index=False)
    print(f"\nDone -> {args.out}/<arm>/{args.inference}/")


if __name__ == "__main__":
    main()
