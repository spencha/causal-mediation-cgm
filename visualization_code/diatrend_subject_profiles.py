#!/usr/bin/env python3
"""Profile specific DiaTrend subjects against the rest of the cohort.

Built to inspect the LOSO-influential subjects (default 28, 30, 38) flagged by
influence_diagnostics.R -- i.e. "do these subjects have unusual glucose reactions
to carbs / insulin?" For each flagged subject it contrasts, against the pooled
"rest of cohort":

  * Delta-glucose trajectory   -- mean Y_t (= G(t) - G(0)) across the postprandial
                                  window; the cohort shown as median + IQR band.
  * Delta-glucose @ endpoint    -- distribution of Y_<timepoint>min.
  * Baseline glucose G(0)       -- glucose_at_meal: were they HIGH pre-meal (a
                                  correction context) vs genuinely hypo-prone?
  * IOB at meal                 -- iob_at_meal: insulin already on board (stacking).
                                  NA for cohort 1 (no IOB); shown only where present.
  * Carb intake                 -- treat_meal_carbs (treatment / meal size).
  * Bolus intake                -- mediator_bolus_for_meal (the mediator).
  * Insulin-to-carb ratio       -- bolus / carbs (units per gram).

Also prints each subject's TRAIN / TEST episode counts (and cohort).

Reads the SAME embeddings CSV the mediation pipeline consumes, so it must run on
the cluster (real per-episode CGM/bolus data never leaves it). Emits only the
de-identified figure + small summary CSVs. numpy / pandas / matplotlib only.

Usage (cluster):
  python visualization_code/diatrend_subject_profiles.py \
      --phi-file analysis_data/diatrend/embeddings_full/phi_embeddings_diatrend_full_demo.csv \
      --subjects 28,30,38 --cohort 1,2 --split all --timepoint 120 \
      --out mediation_results/diatrend/figures/_subject_profiles
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
SUBJ_COLORS = ["#e74c3c", "#9b59b6", "#e67e22", "#16a085"]  # up to 4 highlighted


def resolve_ids(df, subjects):
    """Map requested subject numbers to actual subject_id values in the frame."""
    ids = df["subject_id"].astype(str).unique()
    out = []
    for s in subjects:
        s = s.strip()
        cands = [f"Subject{s}", s, f"subject{s}"]
        hit = next((i for i in ids if i in cands), None)
        if hit is None:  # substring fallback (e.g. cohort-prefixed ids)
            hit = next((i for i in ids if i.endswith(s) or f"_{s}" in i), None)
        print(f"  subject '{s}' -> {hit if hit else 'NOT FOUND'}")
        if hit is not None:
            out.append(hit)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi-file", required=True)
    ap.add_argument("--subjects", default="28,30,38", help="Comma-separated subject numbers.")
    ap.add_argument("--cohort", default="1,2")
    ap.add_argument("--split", default="all", help="test/train/all (for the distributions). [all]")
    ap.add_argument("--timepoint", type=int, default=120, help="Endpoint for Δglucose dist. [120]")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(args.phi_file)
    df_all = df_all[df_all["cohort"].isin([int(c) for c in args.cohort.split(",")])]

    targets = resolve_ids(df_all, args.subjects.split(","))
    if not targets:
        raise SystemExit("No requested subjects found in the embeddings file.")
    sid = df_all["subject_id"].astype(str)

    # ---- TRAIN / TEST counts per subject (computed before any split filter) ----
    print("\nEpisode counts by split (and cohort):")
    cnt_rows = []
    for t in targets:
        d = df_all[sid == t]
        row = {"subject": t, "cohort": int(d["cohort"].iloc[0]) if len(d) else None,
               "train": int((d.get("split") == "train").sum()) if "split" in d else None,
               "test": int((d.get("split") == "test").sum()) if "split" in d else None,
               "total": len(d)}
        cnt_rows.append(row)
    counts = pd.DataFrame(cnt_rows)
    counts.to_csv(args.out / "subject_split_counts.csv", index=False)
    print(counts.to_string(index=False))

    # ---- frame for the distributions (apply --split) ----
    df = df_all
    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split]
    df = df.copy()
    sid = df["subject_id"].astype(str)

    carb, bolus, g0 = "treat_meal_carbs", "mediator_bolus_for_meal", "glucose_at_meal"
    iob = "iob_at_meal" if "iob_at_meal" in df.columns else None
    ycol = f"Y_{args.timepoint}min"
    df["ratio"] = df[bolus] / df[carb].replace(0, np.nan)
    is_t = sid.isin(targets)
    rest = df[~is_t]

    tcols = [f"Y_{t}min" for t in range(60, 211, 5) if f"Y_{t}min" in df.columns]
    tvals = [int(c.split("_")[1].replace("min", "")) for c in tcols]

    # ---- medians summary ----
    def med(d):
        r = {"n_ep": len(d), "carbs_g": d[carb].median(), "bolus_U": d[bolus].median(),
             "baseline_gluc": d[g0].median(), f"dGluc_{args.timepoint}": d[ycol].median(),
             "bolus_per_carb": d["ratio"].median()}
        if iob:
            r["iob_at_meal"] = d[iob].median()
        return r
    rows = [{"group": "Cohort (rest)", **med(rest)}]
    for t in targets:
        rows.append({"group": t, **med(df[sid == t])})
    summ = pd.DataFrame(rows)
    summ.to_csv(args.out / "subject_profiles_summary.csv", index=False)
    print("\nMedians by group:")
    print(summ.round(2).to_string(index=False))

    groups = ["Cohort"] + targets
    gcolors = ["0.6"] + SUBJ_COLORS[:len(targets)]
    glabels = ["Cohort"] + [t.replace("Subject", "S") for t in targets]

    def box_by_group(ax, col, title, xlabel):
        """Boxplot per group, robust to all-NaN groups (e.g. IOB for cohort 1)."""
        positions, data, used = [], [], []
        for i, g in enumerate(groups, start=1):
            d = rest if g == "Cohort" else df[sid == g]
            v = d[col].dropna().to_numpy() if col in d else np.array([])
            if len(v):
                positions.append(i); data.append(v); used.append((i, g, v))
            else:
                ax.text(0.02, i, "n/a", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=9, color="0.5", style="italic")
        if data:
            bp = ax.boxplot(data, positions=positions, vert=False, patch_artist=True,
                            widths=0.6, showfliers=False,
                            medianprops=dict(color="black", lw=1.5))
            for patch, (i, g, _) in zip(bp["boxes"], used):
                patch.set_facecolor(gcolors[i - 1]); patch.set_alpha(0.45)
        for i, g, v in used:                       # strip for highlighted subjects
            if g == "Cohort":
                continue
            jit = (np.random.RandomState(0).rand(len(v)) - 0.5) * 0.3
            ax.plot(v, i + jit, "o", color=gcolors[i - 1], ms=3, alpha=0.5, zorder=3)
        ax.set_yticks(range(1, len(groups) + 1)); ax.set_yticklabels(glabels, fontsize=10)
        ax.set_ylim(0.4, len(groups) + 0.6)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)

    fig, axes = plt.subplots(2, 4, figsize=(21, 9))

    # (a) Δglucose trajectory
    ax = axes[0][0]
    rq1, rmed, rq3 = (rest[tcols].quantile(q) for q in (0.25, 0.5, 0.75))
    ax.fill_between(tvals, rq1.to_numpy(), rq3.to_numpy(), color="0.7", alpha=0.4, label="Cohort IQR")
    ax.plot(tvals, rmed.to_numpy(), color="0.4", lw=2.2, label="Cohort median")
    for t, c in zip(targets, SUBJ_COLORS):
        ax.plot(tvals, df[sid == t][tcols].mean().to_numpy(), color=c, lw=2.0,
                label=t.replace("Subject", "S"))
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("Δ glucose trajectory (mean)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Minutes post meal", fontsize=11, fontweight="bold")
    ax.set_ylabel("Δ glucose (mg/dL)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.2)

    box_by_group(axes[0][1], ycol, f"Δ glucose @ {args.timepoint} min", "Δ glucose (mg/dL)")
    box_by_group(axes[0][2], g0, "Baseline glucose G(0)", "glucose at meal (mg/dL)")
    if iob:
        box_by_group(axes[0][3], iob, "IOB at meal", "insulin on board (U)")
    else:
        axes[0][3].axis("off")
    box_by_group(axes[1][0], carb, "Carb intake", "carbs (g)")
    box_by_group(axes[1][1], bolus, "Bolus intake", "insulin (U)")
    box_by_group(axes[1][2], "ratio", "Insulin-to-carb ratio", "bolus / carb (U/g)")

    # (h) carbs-vs-bolus scatter
    ax = axes[1][3]
    ax.scatter(rest[carb], rest[bolus], s=6, color="0.75", alpha=0.4, label="Cohort")
    for t, c in zip(targets, SUBJ_COLORS):
        d = df[sid == t]
        ax.scatter(d[carb], d[bolus], s=18, color=c, alpha=0.8, label=t.replace("Subject", "S"))
    ax.set_title("Bolus vs carbs", fontsize=13, fontweight="bold")
    ax.set_xlabel("carbs (g)", fontsize=11, fontweight="bold")
    ax.set_ylabel("insulin (U)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, markerscale=1.5); ax.grid(True, alpha=0.2)

    n_rest = rest["subject_id"].nunique()
    fig.suptitle(f"DiaTrend influential-subject profiles vs cohort "
                 f"(cohort {args.cohort}, split {args.split}; rest = {n_rest} subjects)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"subject_profiles.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}/subject_profiles.png + summary CSVs")


if __name__ == "__main__":
    main()
