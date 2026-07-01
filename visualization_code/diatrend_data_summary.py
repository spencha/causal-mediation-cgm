#!/usr/bin/env python3
"""DiaTrend data-distribution summaries, parallel to the OhioT1DM Table 1 /
Figure 2 / Supplementary Figure 1.

Produces, from the DiaTrend embeddings CSV (run on the cluster; only aggregate
de-identified outputs are emitted):
  1. data_characteristics_diatrend.{csv,tex} -- N, carbohydrate intake (treatment),
     insulin bolus (mediator), pre-meal glucose, and % zero-bolus, stratified by
     cohort and meal type (parallel to OhioT1DM Table 1).
  2. fig_diatrend_trajectories.png -- mean +/- 1 SE postprandial Delta-glucose
     trajectories (60-210 min) by meal type, stratified by cohort (a,b) and by
     train/test split (c,d) (parallel to OhioT1DM Figure 2).
  3. fig_diatrend_treatment_mediator.png -- carbohydrate (treatment) vs bolus
     (mediator) scatter by meal type (parallel to Supplementary Figure 1).

numpy / pandas / matplotlib only.

Usage (cluster):
  python visualization_code/diatrend_data_summary.py \
      --phi-file analysis_data/diatrend/embeddings_full/phi_embeddings_diatrend_full_demo.csv \
      --cohort 1,2 --out mediation_results/diatrend/data_summary
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
MEAL_ORDER = ["breakfast", "lunch", "dinner", "snack"]
MEAL_COLOR = {"breakfast": "#8e44ad", "lunch": "#16a085", "dinner": "#d35400", "snack": "#7f8c8d"}
CARB, BOL, G0 = "treat_meal_carbs", "mediator_bolus_for_meal", "glucose_at_meal"


def ms(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return f"{x.mean():.1f} $\\pm$ {x.std():.1f}" if len(x) else "--"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi-file", required=True)
    ap.add_argument("--cohort", default="1,2")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phi_file)
    df = df[df["cohort"].isin([int(c) for c in args.cohort.split(",")])].copy()
    df["meal_type"] = df["meal_type"].astype(str).str.lower()
    df = df[df["meal_type"].isin(MEAL_ORDER)]
    tcols = [f"Y_{t}min" for t in range(60, 211, 5) if f"Y_{t}min" in df.columns]
    tvals = [int(c.split("_")[1].replace("min", "")) for c in tcols]

    # ---- 1. data-characteristics table (by cohort x meal) ----
    def mn(x): x = pd.to_numeric(x, errors="coerce").dropna(); return x.mean() if len(x) else float("nan")
    def sd(x): x = pd.to_numeric(x, errors="coerce").dropna(); return x.std() if len(x) else float("nan")
    rows = []
    for coh in sorted(df["cohort"].unique()):
        for meal in MEAL_ORDER:
            d = df[(df.cohort == coh) & (df.meal_type == meal)]
            if not len(d):
                continue
            rows.append({"Cohort": int(coh), "Meal": meal.capitalize(), "N": len(d),
                         "carb_mean": mn(d[CARB]), "carb_sd": sd(d[CARB]),
                         "bol_mean": mn(d[BOL]), "bol_sd": sd(d[BOL]),
                         "pct_zero": 100 * (pd.to_numeric(d[BOL], errors="coerce") == 0).mean(),
                         "gluc_mean": mn(d[G0]), "gluc_sd": sd(d[G0])})
    tab = pd.DataFrame(rows)
    tab.to_csv(args.out / "data_characteristics_diatrend.csv", index=False)
    # LaTeX, matching the manuscript Table 1 layout (grouped Mean/SD, multirow cohort)
    L = [r"\begin{table*}[ht]", r"\centering", r"\small",
         r"\caption{\textbf{DiaTrend data characteristics by cohort and meal type.} "
         r"Summary statistics for carbohydrate intake (treatment), insulin bolus (mediator), "
         r"and pre-meal glucose stratified by cohort and meal category. "
         r"N = number of meal observations; SD = standard deviation; values are mean $\pm$ SD; "
         r"\% Zero Bolus = percentage of meals with no insulin bolus. "
         r"The causal mediation analysis was performed on the held-out test set "
         r"(N = 1{,}890 demographics-complete meal observations).}",
         r"\label{tab:diatrend_cohort_meal_summary}", r"\begin{tabular}{llccccccc}", r"\toprule",
         r"& & & \multicolumn{2}{c}{Carbs (g)} & \multicolumn{2}{c}{Bolus (U)} & \% Zero & Glucose (mg/dL) \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){9-9}",
         r"Cohort & Meal Type & N & Mean & SD & Mean & SD & Bolus & Mean $\pm$ SD \\", r"\midrule"]
    for ci, coh in enumerate(sorted(tab["Cohort"].unique())):
        sub = tab[tab.Cohort == coh]
        for j, (_, r) in enumerate(sub.iterrows()):
            lead = (r"\multirow{%d}{*}{%d}" % (len(sub), coh)) if j == 0 else ""
            L.append(f"{lead} & {r['Meal']} & {int(r['N'])} & {r['carb_mean']:.1f} & {r['carb_sd']:.1f} & "
                     f"{r['bol_mean']:.2f} & {r['bol_sd']:.2f} & {r['pct_zero']:.1f} & "
                     f"{r['gluc_mean']:.1f} $\\pm$ {r['gluc_sd']:.1f} \\\\")
        if ci < len(tab["Cohort"].unique()) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (args.out / "data_characteristics_diatrend.tex").write_text("\n".join(L))
    print(tab.round(2).to_string(index=False))

    # ---- 2. trajectory figure (parallel to Figure 2) ----
    def panel(ax, sub, title):
        for meal in MEAL_ORDER:
            d = sub[sub.meal_type == meal]
            if not len(d):
                continue
            m = d[tcols].apply(pd.to_numeric, errors="coerce").mean().to_numpy()
            se = d[tcols].apply(pd.to_numeric, errors="coerce").sem().to_numpy()
            ax.plot(tvals, m, "-", color=MEAL_COLOR[meal], lw=1.8, label=meal.capitalize())
            ax.fill_between(tvals, m - se, m + se, color=MEAL_COLOR[meal], alpha=0.15)
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold"); ax.set_xlim(58, 212)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=10)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cohs = sorted(df["cohort"].unique())
    panel(ax[0][0], df[df.cohort == cohs[0]], f"(a) Cohort {cohs[0]}")
    panel(ax[0][1], df[df.cohort == cohs[1]] if len(cohs) > 1 else df, f"(b) Cohort {cohs[1] if len(cohs)>1 else cohs[0]}")
    if "split" in df.columns:
        panel(ax[1][0], df[df.split == "train"], "(c) Training set")
        panel(ax[1][1], df[df.split == "test"], "(d) Test set")
    for a in ax[:, 0]:
        a.set_ylabel("$\\Delta$ Glucose (mg/dL)", fontsize=12, fontweight="bold")
    for a in ax[1, :]:
        a.set_xlabel("Minutes Post Meal", fontsize=12, fontweight="bold")
    ax[0][0].legend(fontsize=9, frameon=False)
    fig.suptitle("DiaTrend: mean postprandial $\\Delta$ glucose trajectories by meal type",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out / "fig_diatrend_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. treatment-mediator scatter (parallel to Supp Fig 1) ----
    fig2, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    for ax, meal in zip(axes, MEAL_ORDER):
        d = df[df.meal_type == meal]
        ax.scatter(pd.to_numeric(d[CARB], errors="coerce"), pd.to_numeric(d[BOL], errors="coerce"),
                   s=8, color=MEAL_COLOR[meal], alpha=0.4)
        ax.set_title(meal.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Carbohydrates (g)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Insulin Bolus (U)", fontsize=11, fontweight="bold")
    fig2.suptitle("DiaTrend: treatment (carbohydrates) vs mediator (insulin bolus) by meal type",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(args.out / "fig_diatrend_treatment_mediator.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"\nWrote data_characteristics_diatrend.{{csv,tex}}, fig_diatrend_trajectories.png, "
          f"fig_diatrend_treatment_mediator.png to {args.out}")


if __name__ == "__main__":
    main()
