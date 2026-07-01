#!/usr/bin/env python3
"""Per-subject summary across the WHOLE DiaTrend cohort, with demographics.

Generalizes diatrend_subject_profiles.py from a few flagged subjects to ALL
subjects: one point per subject, summarizing their typical meal/insulin/glucose
behavior and relating it to age and HbA1c. Built to see where the
LOSO-influential subjects (default highlight 28/30/38) sit in the cohort and
whether glucose response tracks demographics.

Per subject it computes medians of: carbs (treat_meal_carbs), bolus
(mediator_bolus_for_meal), insulin-to-carb ratio, baseline glucose
(glucose_at_meal = G(0)), and Δglucose @ endpoint (Y_<t>min); plus age, HbA1c,
sex, cohort, and train/test episode counts. Banded cohort-1 age/HbA1c are mapped
to numeric band-midpoints (matches the R parse_num).

Runs on the cluster (real per-episode data stays there); emits a de-identified
per-subject CSV + a 2x3 scatter figure. numpy / pandas / matplotlib only.

Usage (cluster):
  python visualization_code/diatrend_cohort_subject_summary.py \
      --phi-file analysis_data/diatrend/embeddings_full/phi_embeddings_diatrend_full_demo.csv \
      --cohort 1,2 --split all --timepoint 120 --highlight 28,30,38 \
      --out mediation_results/diatrend/figures/_subject_profiles
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "DejaVu Sans"


def parse_num(s):
    """Banded ('40-49', '>9') or numeric string -> mean of the numbers in it."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    nums = re.findall(r"[0-9.]+", str(s))
    return float(np.mean([float(x) for x in nums])) if nums else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi-file", required=True)
    ap.add_argument("--cohort", default="1,2")
    ap.add_argument("--split", default="all", help="test/train/all (for the medians). [all]")
    ap.add_argument("--timepoint", type=int, default=120)
    ap.add_argument("--highlight", default="28,30,38", help="Subject numbers to label.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.phi_file)
    df = df[df["cohort"].isin([int(c) for c in args.cohort.split(",")])]
    df["sid"] = df["subject_id"].astype(str)
    highlight = {f"Subject{n.strip()}" for n in args.highlight.split(",")}

    carb, bolus, g0 = "treat_meal_carbs", "mediator_bolus_for_meal", "glucose_at_meal"
    iob = "iob_at_meal" if "iob_at_meal" in df.columns else None
    ycol = f"Y_{args.timepoint}min"
    for c in ("demo_age", "demo_hba1c"):
        if c in df.columns:
            df[c + "_num"] = df[c].map(parse_num)

    # split-filtered frame for the behavior medians, full frame for counts
    dfm = df if args.split == "all" or "split" not in df else df[df["split"] == args.split]
    dfm = dfm.copy()
    dfm["ratio"] = dfm[bolus] / dfm[carb].replace(0, np.nan)

    rows = []
    for sid, d in dfm.groupby("sid"):
        full = df[df.sid == sid]
        r = {"subject": sid, "cohort": int(d["cohort"].iloc[0]),
             "n_total": len(full),
             "n_train": int((full.get("split") == "train").sum()) if "split" in full else np.nan,
             "n_test": int((full.get("split") == "test").sum()) if "split" in full else np.nan,
             "age": d.get("demo_age_num", pd.Series([np.nan])).median(),
             "hba1c": d.get("demo_hba1c_num", pd.Series([np.nan])).median(),
             "sex": (d["demo_sex"].mode().iloc[0] if "demo_sex" in d and len(d["demo_sex"].mode()) else None),
             "carbs_g": d[carb].median(), "bolus_U": d[bolus].median(),
             "ratio_U_per_g": d["ratio"].median(), "baseline_gluc": d[g0].median(),
             f"dGluc_{args.timepoint}": d[ycol].median()}
        if iob:
            r["iob_at_meal"] = d[iob].median()
        rows.append(r)
    s = pd.DataFrame(rows).sort_values(f"dGluc_{args.timepoint}").reset_index(drop=True)
    s.to_csv(args.out / "cohort_subject_summary.csv", index=False)
    print(f"Per-subject summary ({len(s)} subjects, split={args.split}):")
    print(s.round(2).to_string(index=False))

    # ---- figure: one point per subject, colored by HbA1c, shaped by cohort ----
    dg = f"dGluc_{args.timepoint}"
    hl = s["subject"].isin(highlight)
    cmap = plt.cm.viridis
    a1c = s["hba1c"].to_numpy(dtype=float)
    vmin, vmax = np.nanmin(a1c), np.nanmax(a1c)
    sizes = 40 + 200 * (s["n_total"] / s["n_total"].max())

    def scat(ax, x, y, xlabel, ylabel, title):
        for coh, mk in [(1, "o"), (2, "^")]:
            m = (s["cohort"] == coh)
            sc = ax.scatter(s[x][m], s[y][m], c=s["hba1c"][m], cmap=cmap, vmin=vmin, vmax=vmax,
                            s=sizes[m], marker=mk, edgecolor="0.3", linewidth=0.5, alpha=0.9, zorder=2)
        # highlight ring + label
        ax.scatter(s[x][hl], s[y][hl], s=sizes[hl] + 80, facecolors="none",
                   edgecolors="red", linewidth=2.0, zorder=3)
        for _, r in s[hl].iterrows():
            ax.annotate(r["subject"].replace("Subject", "S"), (r[x], r[y]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9,
                        fontweight="bold", color="red")
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        return sc

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    sc = scat(ax[0][0], "baseline_gluc", dg, "median baseline glucose G(0) (mg/dL)",
              f"median Δglucose @ {args.timepoint} (mg/dL)", "Δglucose vs baseline glucose")
    scat(ax[0][1], "ratio_U_per_g", dg, "median insulin-to-carb ratio (U/g)",
         f"median Δglucose @ {args.timepoint}", "Δglucose vs dosing aggressiveness")
    scat(ax[0][2], "hba1c", dg, "HbA1c", f"median Δglucose @ {args.timepoint}",
         "Δglucose vs HbA1c")
    scat(ax[1][0], "age", dg, "age (yrs)", f"median Δglucose @ {args.timepoint}",
         "Δglucose vs age")
    scat(ax[1][1], "carbs_g", "bolus_U", "median carbs (g)", "median bolus (U)",
         "Dosing: bolus vs carbs")
    scat(ax[1][2], "hba1c", "baseline_gluc", "HbA1c", "median baseline glucose G(0)",
         "Baseline glucose vs HbA1c")
    for a in ax.flat:
        if a.get_title().startswith("Δglucose"):
            a.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)

    cbar = fig.colorbar(sc, ax=ax, location="right", shrink=0.6, pad=0.02)
    cbar.set_label("HbA1c", fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], marker="o", color="w", mfc="0.6", mec="0.3", ms=10, label="cohort 1 (no IOB)"),
               Line2D([0], [0], marker="^", color="w", mfc="0.6", mec="0.3", ms=10, label="cohort 2 (has IOB)"),
               Line2D([0], [0], marker="o", color="w", mec="red", mew=2, ms=12, label="LOSO-influential"),
               Line2D([0], [0], marker="o", color="w", mfc="0.6", mec="0.3", ms=14, label="size ∝ #episodes")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"DiaTrend per-subject summary (n={len(s)} subjects, cohort {args.cohort}, split {args.split})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 0.93, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"cohort_subject_summary.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}/cohort_subject_summary.png + .csv")


if __name__ == "__main__":
    main()
