#!/usr/bin/env python3
"""DiaTrend causal-assumption diagnostics for the npCBPS-weighted embedding.

Reads an npCBPS weight file (the `npcbps_weights_*.csv` written by
npcbps_weights.R, which carries the covariates + `cbps_weight` + treatment) and
checks the identification assumptions the mediation analysis relies on:

  1. COVARIATE BALANCE (Love plot) -- |corr(covariate, treatment)| before vs
     after weighting; weighted values near 0 => balanced.
  2. CONDITIONAL INDEPENDENCE (permutation test) -- for each covariate, the
     observed *weighted* correlation with treatment vs a permutation null
     (treatment shuffled, same weights). A non-significant observed value =>
     treatment is conditionally independent of that covariate given the weights.
  3. OVERLAP / POSITIVITY -- the weighted conditional distribution of the leading
     embedding dimension (PC_1) across treatment terciles; heavy overlap =>
     common support (every covariate region is represented at every dose).

numpy / pandas / matplotlib only.

Usage (cluster):
  python visualization_code/diatrend_balance_diagnostics.py \
      --weights-file analysis_data/diatrend/weights/npcbps_weights_c2_iob_<id>_demowt.csv \
      --out mediation_results/diatrend/figures/_balance_diagnostics --label demowt
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
TREAT = "treat_meal_carbs"
WCOL = "cbps_weight"


def wmean(a, w):
    return np.sum(w * a) / np.sum(w)


def wcorr(x, t, w):
    """Weighted Pearson correlation."""
    mx, mt = wmean(x, w), wmean(t, w)
    cov = np.sum(w * (x - mx) * (t - mt))
    sx = np.sqrt(np.sum(w * (x - mx) ** 2))
    st = np.sqrt(np.sum(w * (t - mt) ** 2))
    return np.nan if sx == 0 or st == 0 else cov / (sx * st)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-file", required=True)
    ap.add_argument("--covariate-prefix", default="PC")
    ap.add_argument("--balance-n-phi", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--max-na-frac", type=float, default=0.2,
                    help="Drop (not row-filter) covariates with more than this "
                         "fraction missing, e.g. iob_at_meal in the no-IOB full cohort. [0.2]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    df = pd.read_csv(args.weights_file)
    if WCOL not in df.columns:
        raise SystemExit(f"{WCOL} not in {args.weights_file}; pass an npcbps_weights_*.csv")

    # Covariate set actually balanced: PCs + glucose + iob + demographics, when present.
    pcs = [f"{args.covariate_prefix}_{i}" for i in range(1, args.balance_n_phi + 1)]
    extra = ["glucose_at_meal", "iob_at_meal", "demo_age", "demo_sex", "demo_hba1c"]
    cov = [c for c in pcs + extra if c in df.columns]

    # Drop covariates that are mostly missing rather than the rows that lack them.
    # Critical for the full cohort: iob_at_meal is NA for all of cohort 1, so a
    # naive dropna() would silently delete cohort 1 and report cohort-2 balance.
    # Measure TRUE missingness on the raw values -- not via to_numeric, which would
    # coerce present-but-categorical columns (banded demo_age, "M"/"F" demo_sex) to
    # NaN and wrongly flag them as missing. Only genuinely-absent cols (iob) qualify.
    na_frac = df[cov].isna().mean()
    high_na = na_frac[na_frac > args.max_na_frac].index.tolist()
    if high_na:
        print(f"Excluding high-missingness covariate(s) (>{args.max_na_frac:.0%} NA, "
              f"keeps the full sample): {high_na}")
        cov = [c for c in cov if c not in high_na]

    d = df[cov + [TREAT, WCOL]].copy()
    for c in cov:                          # coerce factor covariates (e.g. demo_sex)
        if not pd.api.types.is_numeric_dtype(d[c]):   # robust to pandas StringDtype
            d[c] = pd.factorize(d[c])[0]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    t = d[TREAT].to_numpy(); w = d[WCOL].to_numpy()
    n = len(d)

    # ---- per-covariate balance + conditional-independence permutation p -----
    rows = []
    for c in cov:
        x = d[c].to_numpy()
        unwt = np.corrcoef(x, t)[0, 1]
        wt = wcorr(x, t, w)
        null = np.array([wcorr(x, t[rng.permutation(n)], w) for _ in range(args.n_perm)])
        p = (np.sum(np.abs(null) >= abs(wt)) + 1) / (args.n_perm + 1)
        rows.append(dict(covariate=c, unweighted_r=unwt, weighted_r=wt, perm_p=p))
    bal = pd.DataFrame(rows)
    bal.to_csv(args.out / f"balance_assumptions_{args.label}.csv", index=False)

    print(f"\nCausal-assumption diagnostics ({args.label}, n={n}):")
    print(bal.round(4).to_string(index=False))
    print(f"\nMax |weighted_r| = {bal.weighted_r.abs().max():.4f} | "
          f"covariates with perm_p<0.05 (residual dependence): "
          f"{int((bal.perm_p < 0.05).sum())}/{len(bal)}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (A) Love plot
    ax = axes[0]
    y = np.arange(len(bal))
    ax.scatter(bal.unweighted_r.abs(), y, c="#CC3311", label="unweighted", zorder=3)
    ax.scatter(bal.weighted_r.abs(), y, c="#009988", label="weighted", zorder=3)
    for i in range(len(bal)):
        ax.plot([abs(bal.unweighted_r.iloc[i]), abs(bal.weighted_r.iloc[i])], [i, i],
                color="0.7", zorder=1)
    ax.axvline(0.1, ls="--", color="0.4", label="0.1 threshold")
    ax.set_yticks(y); ax.set_yticklabels(bal.covariate, fontsize=9)
    ax.set_xlabel("|correlation with treatment|", fontweight="bold")
    ax.set_title("Covariate balance (Love plot)", fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, axis="x", alpha=0.3)

    # (B) Conditional independence: permutation null for the most-imbalanced covariate
    ax = axes[1]
    top = bal.iloc[bal.unweighted_r.abs().idxmax()]
    x = d[top.covariate].to_numpy()
    null = np.array([wcorr(x, t[rng.permutation(n)], w) for _ in range(args.n_perm)])
    ax.hist(null, bins=40, color="0.7", edgecolor="white")
    ax.axvline(top.weighted_r, color="#CC3311", lw=2,
               label=f"observed wt r={top.weighted_r:.3f}\nperm p={top.perm_p:.3f}")
    ax.set_xlabel(f"weighted corr({top.covariate}, treat) under H0", fontweight="bold")
    ax.set_ylabel("permutation count", fontweight="bold")
    ax.set_title("Conditional independence\n(permutation test, weighted)", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    # (C) Overlap / positivity: weighted PC_1 distribution by treatment tercile
    ax = axes[2]
    pc1 = d[f"{args.covariate_prefix}_1"].to_numpy()
    terc = np.asarray(pd.qcut(t, 3, labels=["low dose", "mid dose", "high dose"]))
    grid = np.linspace(np.percentile(pc1, 1), np.percentile(pc1, 99), 200)
    for lab, color in zip(["low dose", "mid dose", "high dose"],
                          ["#1b9e77", "#7570b3", "#d95f02"]):
        m = terc == lab
        wi = w[m] / w[m].sum()
        # weighted Gaussian KDE (simple Silverman bandwidth)
        xi = pc1[m]
        bw = 1.06 * np.sqrt(np.cov(xi)) * len(xi) ** (-1 / 5)
        dens = np.array([np.sum(wi * np.exp(-0.5 * ((g - xi) / bw) ** 2)) for g in grid])
        # np.trapz was removed in NumPy 2.x -> np.trapezoid; fall back for 1.x.
        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        dens /= trapz(dens, grid)
        ax.plot(grid, dens, color=color, lw=2, label=lab)
        ax.fill_between(grid, dens, color=color, alpha=0.12)
    ax.set_xlabel(f"{args.covariate_prefix}_1 (leading embedding dim)", fontweight="bold")
    ax.set_ylabel("weighted density", fontweight="bold")
    ax.set_title("Overlap / positivity\n(weighted PC_1 by treatment tercile)", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(f"DiaTrend causal-assumption diagnostics — {args.label} (n={n})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"balance_diagnostics_{args.label}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}/balance_diagnostics_{args.label}.png + .csv")


if __name__ == "__main__":
    main()
