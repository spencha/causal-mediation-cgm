#!/usr/bin/env python3
"""Full-cohort robustness/exploratory figures from the co-dated batch outputs.

Generates three figures that complement the npj-style overview produced by
generate_diatrend_mediation_outputs.py:

  1. fig_fullcohort_trajectory.png  -- pooled ACME/ADE/Total time-course (QB and
       subject-bootstrap side by side) + per-meal direct-effect (ADE) trajectories.
       Reads the two timepoint-grid CSVs.
  2. fig_fullcohort_forest_moderation.png -- bootstrap headline (pooled, breakfast)
       + HbA1c and sex moderation (strata + between-stratum difference) forest.
  3. fig_fullcohort_influence_tornado.png -- LOSO ΔACME / ΔADE tornado.

The forest + tornado read the bootstrap/influence STEP LOGS (the
cluster_bootstrap_mediation.R / influence_diagnostics.R scripts print results
rather than writing CSVs), parse them ONCE into tidy CSVs alongside the figures,
and plot from those. numpy / pandas / matplotlib only.

Usage:
  python visualization_code/diatrend_fullcohort_figures.py \
      --qb-grid  mediation_results/diatrend/grid_diatrend_2026-06-18_180932_fullcohort.csv \
      --boot-grid mediation_results/diatrend/grid_diatrend_2026-06-18_222028_fullcohort.csv \
      --logs-dir mediation_results/diatrend/fullcohort_logs \
      --offset 30 --out mediation_results/diatrend/figures/fullcohort
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "DejaVu Sans"
EFF = [("acme", "#2ecc71", "ACME (Indirect)"), ("ade", "#e74c3c", "ADE (Direct)"),
       ("total", "#3498db", "Total")]
MEALS = [("breakfast", "#8e44ad"), ("lunch", "#16a085"), ("dinner", "#d35400"),
         ("snack", "#7f8c8d")]


# ---------------------------------------------------------------- trajectory ---
def trajectory(qb, bs, offset, out):
    def pooled(ax, d, title):
        s = d[(d.meal == "ALL") & (d.model == "lmer") & (d.offset_g == offset)].sort_values("timepoint")
        t = s.timepoint.to_numpy()
        for col, c, _ in EFF:
            v = s[col].to_numpy()
            ax.plot(t, v, "-", color=c, lw=1.8)
            ax.fill_between(t, s[col + "_lo"], s[col + "_hi"], color=c, alpha=0.15)
            sig = s[col + "_p"].to_numpy() < 0.05
            if sig.any():
                ax.plot(t[sig], v[sig], "*", color=c, ms=13, mec="white", mew=0.5, ls="none")
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold"); ax.set_xlim(58, 212)
        ax.grid(True, alpha=0.15); ax.tick_params(labelsize=11)

    def meal_ade(ax, d, title):
        for meal, c in [("ALL", "black")] + MEALS:
            s = d[(d.meal == meal) & (d.model == "lmer") & (d.offset_g == offset)].sort_values("timepoint")
            t, v = s.timepoint.to_numpy(), s.ade.to_numpy()
            ax.plot(t, v, "-", color=c, lw=2.0 if meal == "ALL" else 1.4,
                    label=("Pooled" if meal == "ALL" else meal.capitalize()),
                    alpha=1 if meal == "ALL" else 0.85)
            sig = s.ade_p.to_numpy() < 0.05
            if sig.any():
                ax.plot(t[sig], v[sig], "o", color=c, ms=4, ls="none")
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=14, fontweight="bold"); ax.set_xlim(58, 212)
        ax.grid(True, alpha=0.15); ax.tick_params(labelsize=11)

    n = int(qb[qb.meal == "ALL"].n_episodes.max()); ns = int(qb[qb.meal == "ALL"].n_subjects.max())
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    pooled(ax[0][0], qb, "Pooled — Quasi-Bayes"); pooled(ax[0][1], bs, "Pooled — Subject-cluster Bootstrap")
    meal_ade(ax[1][0], qb, "Direct effect (ADE) by meal — Quasi-Bayes")
    meal_ade(ax[1][1], bs, "Direct effect (ADE) by meal — Bootstrap")
    for a in ax[:, 0]:
        a.set_ylabel("Effect on Glucose (mg/dL)", fontsize=14, fontweight="bold")
    for a in ax[1, :]:
        a.set_xlabel("Minutes Post Meal", fontsize=14, fontweight="bold")
    h = [Line2D([0], [0], color=c, lw=2.5, label=l) for _, c, l in EFF] + \
        [Line2D([0], [0], ls="none", marker="*", color="gray", ms=12, label="p < 0.05")]
    ax[0][1].legend(handles=h, fontsize=10, frameon=True, loc="upper left")
    ax[1][1].legend(fontsize=10, frameon=True, loc="upper left", ncol=2)
    fig.suptitle(f"DiaTrend full cohort ({ns} subj / {n} ep): mediation trajectory, +{offset} g, 60–210 min",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out / "fig_fullcohort_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print("wrote fig_fullcohort_trajectory.png")


# ------------------------------------------------------ parse bootstrap logs ---
_ROW = re.compile(r'(ACME|ADE|TOTAL)\s+(-?\d+\.?\d*)\s+\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]\s+p=(\d+\.?\d*)')


def parse_effects(path, source):
    out, carry = [], None
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        m = _ROW.search(line)
        if not m:
            continue
        pre = line[:m.start()].replace("<-- moderation", "").strip()
        if pre:
            carry = pre
        eff, pt, lo, hi, pv = m.groups()
        out.append(dict(source=source, stratum=carry, effect=eff.title(),
                        point=float(pt), lo=float(lo), hi=float(hi), p=float(pv)))
    return out


def parse_influence(path):
    rex = re.compile(r'(Subject\d+)\s+(\d+)\s*\|\s*(-?\d+\.?\d*)\s+([+-]?\d+\.?\d*)\s+(YES|-)\s*\|\s*(-?\d+\.?\d*)\s+([+-]?\d+\.?\d*)')
    rows, txt = [], path.read_text() if path.exists() else ""
    for line in txt.splitlines():
        m = rex.search(line)
        if m:
            s, n, a_s, da, fl, ad_s, dad = m.groups()
            rows.append(dict(subject=s, n_ep=int(n), acme_minus_s=float(a_s), dACME=float(da),
                             flip=(fl == "YES"), ade_minus_s=float(ad_s), dADE=float(dad)))
    base = re.search(r'baseline ACME\s*:\s*(-?\d+\.?\d*)\s*\(p=(\d+\.?\d*)', txt)
    rng = re.search(r'leave-one-out ACME range\s*:\s*\[(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]', txt)
    meta = {}
    if base:
        meta.update(baseline_acme=float(base.group(1)), baseline_p=float(base.group(2)))
    if rng:
        meta.update(loo_lo=float(rng.group(1)), loo_hi=float(rng.group(2)))
    return pd.DataFrame(rows), meta


# ----------------------------------------------------------------- forest ------
EFFC = {"Acme": "#2ecc71", "Ade": "#e74c3c", "Total": "#3498db"}
EFFO = {"Acme": 0.27, "Ade": 0.0, "Total": -0.27}


def forest(boot, out):
    def panel(ax, df, keycol, groups, glabels, title):
        yt, ytl = [], []
        for gi, g in enumerate(groups):
            yb = len(groups) - 1 - gi; yt.append(yb); ytl.append(glabels[g])
            for eff in ["Acme", "Ade", "Total"]:
                r = df[(df[keycol] == g) & (df.effect == eff)]
                if r.empty:
                    continue
                r = r.iloc[0]; y = yb + EFFO[eff]; sig = r.p < 0.05
                ax.errorbar(r.point, y, xerr=[[r.point - r.lo], [r.hi - r.point]], fmt="o",
                            color=EFFC[eff], ms=8 if sig else 6, capsize=3, lw=1.8,
                            mfc=EFFC[eff] if sig else "white", mec=EFFC[eff], mew=1.6, zorder=3)
                if sig:
                    ax.text(r.hi, y, "*", color=EFFC[eff], fontsize=14, fontweight="bold",
                            va="center", ha="left")
        ax.axvline(0, color="0.4", ls="--", lw=1)
        ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=12, fontweight="bold")
        ax.set_ylim(-0.6, len(groups) - 0.4); ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Effect on Glucose (mg/dL)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    panel(axes[0], boot[boot.source.isin(["Pooled", "Breakfast"])], "source",
          ["Pooled", "Breakfast"], {"Pooled": "Pooled", "Breakfast": "Breakfast"},
          "Headline (bootstrap, B=1000)")
    panel(axes[1], boot[boot.source == "HbA1c"], "stratum",
          ["High", "Mid", "Low", "High_minus_Low"],
          {"High": "High", "Mid": "Mid", "Low": "Low", "High_minus_Low": "High − Low (Δ)"},
          "HbA1c moderation")
    panel(axes[2], boot[boot.source == "Sex"], "stratum",
          ["Female", "Male", "Male_minus_Female"],
          {"Female": "Female", "Male": "Male", "Male_minus_Female": "Male − Female (Δ)"},
          "Sex moderation")
    h = [Line2D([0], [0], marker="o", color=EFFC[e], lw=0, ms=9, label=lbl) for e, lbl in
         [("Acme", "ACME (Indirect)"), ("Ade", "ADE (Direct)"), ("Total", "Total")]]
    h += [Line2D([0], [0], marker="o", color="0.4", lw=0, mfc="0.4", ms=9, label="filled = p<0.05"),
          Line2D([0], [0], marker="o", color="0.4", lw=0, mfc="white", mec="0.4", ms=9, label="open = n.s.")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("DiaTrend full cohort: bootstrap headline + demographic moderation, 120 min, +30 g",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out / "fig_fullcohort_forest_moderation.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("wrote fig_fullcohort_forest_moderation.png")


# ----------------------------------------------------------------- tornado -----
def tornado(inf, meta, out):
    if inf.empty:
        print("  (no influence rows parsed; skipping tornado)"); return
    d = inf.sort_values("dACME").reset_index(drop=True); y = np.arange(len(d))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, col, lab in [(a1, "dACME", "Δ ACME when subject removed"),
                         (a2, "dADE", "Δ ADE when subject removed")]:
        vals = d[col].to_numpy()
        colors = ["#e74c3c" if f else ("#3498db" if v >= 0 else "#9b59b6") for v, f in zip(vals, d.flip)]
        ax.barh(y, vals, color=colors, edgecolor="black", lw=0.5); ax.axvline(0, color="0.3", lw=1)
        ax.set_xlabel(lab, fontsize=12, fontweight="bold"); ax.grid(True, axis="x", alpha=0.25)
    a1.set_yticks(y); a1.set_yticklabels([f"{s} (n={n})" for s, n in zip(d.subject, d.n_ep)], fontsize=10)
    ttl = "LOSO influence"
    if meta:
        ttl = (f"baseline ACME {meta.get('baseline_acme', float('nan')):+.2f} "
               f"(p={meta.get('baseline_p', float('nan')):.2f}); "
               f"LOO range [{meta.get('loo_lo', float('nan')):+.2f}, {meta.get('loo_hi', float('nan')):+.2f}]")
    a1.set_title(ttl, fontsize=12, fontweight="bold")
    a2.set_title("Direct effect sensitivity", fontsize=12, fontweight="bold")
    h = [Line2D([0], [0], marker="s", color="w", mfc="#e74c3c", ms=11, label="removal flips ACME significance"),
         Line2D([0], [0], marker="s", color="w", mfc="#3498db", ms=11, label="ΔACME ≥ 0"),
         Line2D([0], [0], marker="s", color="w", mfc="#9b59b6", ms=11, label="ΔACME < 0")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("DiaTrend full cohort: leave-one-subject-out influence on pooled mediation (top by |ΔACME|)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "fig_fullcohort_influence_tornado.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("wrote fig_fullcohort_influence_tornado.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qb-grid", required=True)
    ap.add_argument("--boot-grid", required=True)
    ap.add_argument("--logs-dir", required=True, type=Path)
    ap.add_argument("--offset", type=int, default=30)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trajectory(pd.read_csv(args.qb_grid), pd.read_csv(args.boot_grid), args.offset, args.out)

    boot = (parse_effects(args.logs_dir / "step6_boot_pooled.log", "Pooled")
            + parse_effects(args.logs_dir / "step6_boot_breakfast.log", "Breakfast")
            + parse_effects(args.logs_dir / "step6_boot_hba1c.log", "HbA1c")
            + parse_effects(args.logs_dir / "step6_boot_sex.log", "Sex"))
    boot = pd.DataFrame(boot)
    if not boot.empty:
        boot.to_csv(args.out / "bootstrap_effects_tidy.csv", index=False)
        forest(boot, args.out)
    else:
        print("  (no bootstrap effect rows parsed; skipping forest)")

    inf, meta = parse_influence(args.logs_dir / "step7_influence.log")
    if not inf.empty:
        inf.to_csv(args.out / "influence_tidy.csv", index=False)
        (args.out / "influence_meta.json").write_text(json.dumps(meta))
    tornado(inf, meta, args.out)
    print(f"\nDone -> {args.out}")


if __name__ == "__main__":
    main()
