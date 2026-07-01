#!/usr/bin/env python3
"""DiaTrend embedding diagnostics: is the phi/PC embedding actually informative?

Reads a DiaTrend embeddings CSV (the same file the mediation pipeline consumes)
and reports how much signal the embedding carries for the three quantities that
matter in the causal-mediation analysis:

  * TREATMENT  (treat_meal_carbs)        -- how much the embedding predicts meal
                                            size. Should be LOW: it's the
                                            confounding channel, and the npCBPS
                                            balance check already showed treatment
                                            is ~independent of the covariates.
  * MEDIATOR   (mediator_bolus_for_meal) -- the embedding as a confounder/predictor
                                            of the insulin bolus.
  * OUTCOME    (Y_<t>min)                -- post-meal glucose. Should be HIGH (the
                                            pre-meal glucose trajectory strongly
                                            predicts post-meal glucose); this is
                                            where the embedding earns its keep.

For each target it reports in-sample R^2 (OLS, intercept included) under three
predictor sets -- the 6 balance PCs, the 3 model PCs, and the raw phi_* features
-- plus the PCA explained-variance of the phi space, so you can see whether 6 PCs
capture most of the embedding's variance. In-sample R^2 is optimistic (no CV); it
is a descriptive "does the embedding contain this signal" check, not a prediction
benchmark. No sklearn dependency (numpy SVD + lstsq).

Usage (on the cluster, where the embeddings live):
  python visualization_code/diatrend_embedding_diagnostics.py \
      --phi-file analysis_data/diatrend/embeddings/phi_embeddings_diatrend_demo.csv \
      --cohort 2 --split test \
      --out mediation_results/diatrend/figures/_embedding_diagnostics
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


def r2(X: np.ndarray, y: np.ndarray) -> float:
    """In-sample OLS R^2 of y ~ [1, X] (NaN-safe: caller passes complete rows)."""
    if X.ndim == 1:
        X = X[:, None]
    Xi = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xi, y, rcond=None)
    resid = y - Xi @ beta
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float("nan") if ss_tot == 0 else 1.0 - np.sum(resid ** 2) / ss_tot


def explained_variance(phi: np.ndarray) -> np.ndarray:
    """PCA explained-variance ratio of the raw phi matrix via SVD."""
    Xc = phi - phi.mean(axis=0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return (s ** 2) / np.sum(s ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phi-file", required=True)
    ap.add_argument("--cohort", default="2", help="Comma-separated cohort filter. [2]")
    ap.add_argument("--split", default="test", help="test/train/all. [test]")
    ap.add_argument("--covariate-prefix", default="PC")
    ap.add_argument("--balance-n-phi", type=int, default=6)
    ap.add_argument("--model-n-phi", type=int, default=3)
    ap.add_argument("--timepoints", default=",".join(str(t) for t in range(60, 211, 5)))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.phi_file)
    df = df[df["cohort"].isin([int(c) for c in args.cohort.split(",")])]
    if args.split != "all":
        df = df[df["split"] == args.split]
    n_all = len(df)

    pre = args.covariate_prefix
    pc6 = [f"{pre}_{i}" for i in range(1, args.balance_n_phi + 1)]
    pc3 = [f"{pre}_{i}" for i in range(1, args.model_n_phi + 1)]
    phi_cols = sorted([c for c in df.columns if c.startswith("phi_")],
                      key=lambda c: int(c.split("_")[1]))
    timepoints = [int(t) for t in args.timepoints.split(",")]

    predictor_sets = {f"{pre}_1..{args.balance_n_phi}": pc6,
                      f"{pre}_1..{args.model_n_phi}": pc3}
    if phi_cols:
        predictor_sets[f"phi (all {len(phi_cols)})"] = phi_cols

    def cc(cols, target):
        sub = df[cols + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        return sub[cols].to_numpy(), sub[target].to_numpy()

    # ---- R^2 table: target x predictor set --------------------------------
    targets = [("treatment (carbs)", "treat_meal_carbs"),
               ("mediator (bolus)", "mediator_bolus_for_meal"),
               ("outcome (Y_120min)", "Y_120min")]
    rows = []
    for tlabel, tcol in targets:
        if tcol not in df.columns:
            continue
        row = {"target": tlabel}
        for slabel, cols in predictor_sets.items():
            cols = [c for c in cols if c in df.columns]
            if not cols:
                continue
            X, y = cc(cols, tcol)
            row[slabel] = r2(X, y) if len(y) > len(cols) + 2 else float("nan")
        rows.append(row)
    r2_tab = pd.DataFrame(rows).set_index("target")

    # ---- Outcome R^2 across the postprandial window (balance PCs) ----------
    pc6_present = [c for c in pc6 if c in df.columns]
    curve = []
    for t in timepoints:
        col = f"Y_{t}min"
        if col in df.columns and pc6_present:
            X, y = cc(pc6_present, col)
            curve.append((t, r2(X, y) if len(y) > len(pc6_present) + 2 else np.nan))
    curve = pd.DataFrame(curve, columns=["timepoint", "outcome_R2"])

    # ---- PCA explained variance of phi ------------------------------------
    evr = None
    if phi_cols:
        X = df[phi_cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
        if len(X) > len(phi_cols):
            evr = explained_variance(X)

    # ---- report -----------------------------------------------------------
    print(f"\nDiaTrend embedding diagnostics  (cohort={args.cohort}, split={args.split}, "
          f"n={n_all})")
    print("\nIn-sample R^2 (OLS, intercept; optimistic -- descriptive only):")
    print(r2_tab.round(3).to_string())
    if evr is not None:
        cum = np.cumsum(evr)
        print(f"\nphi PCA explained variance: PC1={evr[0]:.1%}, "
              f"cum@{args.model_n_phi}={cum[args.model_n_phi-1]:.1%}, "
              f"cum@{args.balance_n_phi}={cum[args.balance_n_phi-1]:.1%}")
    r2_tab.to_csv(args.out / "embedding_R2.csv")
    curve.to_csv(args.out / "embedding_outcome_R2_by_timepoint.csv", index=False)

    # ---- figure -----------------------------------------------------------
    npan = 3 if evr is not None else 2
    fig, axes = plt.subplots(1, npan, figsize=(5.2 * npan, 4.2))
    # (a) R^2 grouped bars
    ax = axes[0]
    sets = list(r2_tab.columns)
    x = np.arange(len(r2_tab)); w = 0.8 / max(1, len(sets))
    for i, s in enumerate(sets):
        ax.bar(x + i * w, r2_tab[s].values, w, label=s)
    ax.set_xticks(x + w * (len(sets) - 1) / 2)
    ax.set_xticklabels([t.replace(" (", "\n(") for t in r2_tab.index], fontsize=9)
    ax.set_ylabel("In-sample $R^2$", fontweight="bold")
    ax.set_title("Embedding predictiveness", fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, axis="y", alpha=0.3)
    # (b) outcome R^2 across timepoints
    ax = axes[1]
    ax.plot(curve["timepoint"], curve["outcome_R2"], "-o", color="#3498db", ms=4)
    ax.set_xlabel("Minutes post-meal", fontweight="bold")
    ax.set_ylabel("Outcome $R^2$ (balance PCs)", fontweight="bold")
    ax.set_title("Outcome signal over time", fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    # (c) PCA explained variance
    if evr is not None:
        ax = axes[2]
        k = min(len(evr), 12)
        ax.bar(np.arange(1, k + 1), evr[:k], color="#7f7f7f")
        ax.plot(np.arange(1, k + 1), np.cumsum(evr[:k]), "-o", color="#e74c3c", ms=4,
                label="cumulative")
        ax.axvline(args.balance_n_phi + 0.5, ls="--", color="0.4")
        ax.set_xlabel("PC", fontweight="bold")
        ax.set_ylabel("Explained variance", fontweight="bold")
        ax.set_title("phi PCA explained variance", fontweight="bold")
        ax.legend(fontsize=8, frameon=False); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"embedding_diagnostics.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}/embedding_diagnostics.png + .csv files")


if __name__ == "__main__":
    main()
