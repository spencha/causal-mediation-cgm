#!/usr/bin/env python3
"""Reshape the interaction-bootstrap results into the canonical DiaTrend grid
schema so the HOUSE generator (generate_diatrend_mediation_outputs.py) renders
the interaction causal effects in exactly the standard style.

The interaction bootstrap gives, per meal x timepoint (x tau for QR), the
arm-specific decomposition d0/d1 (ACME at control / treated arm), ADE, total.
This writes one canonical grid CSV per arm:
  ACME      <- d1 (treated/+offset arm)  OR  d0 (control arm)
  ADE       <- ade
  total     <- tot
with model='lmer' (from the LMER CSVs) and model='qr' tau in {.25,.5,.75} (from
the QR CSVs), so the canonical meal x {LMER, QR taus} overview just works.

Usage:
  python visualization_code/diatrend_interaction_to_canonical.py \
      --csv-dir mediation_results/diatrend --nref mediation_results/diatrend/grid_diatrend_2026-06-18_222028_fullcohort.csv \
      --arm treated --offset 30 \
      --out mediation_results/diatrend/figures/fullcohort/interaction/grid_interaction_treated.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

MEALS = ["ALL", "breakfast", "lunch", "dinner", "snack"]


def rows_from(df, meal, model, tau, arm, nmap, offset):
    acme = "d1" if arm == "treated" else "d0"
    out = pd.DataFrame({
        "arm": f"{arm}_arm", "meal": meal, "split": "test", "model": model,
        "tau": tau, "offset_g": offset, "timepoint": df["timepoint"],
        "n_episodes": nmap[meal][0], "n_subjects": nmap[meal][1],
        "acme": df[acme], "acme_lo": df[f"{acme}_lo"], "acme_hi": df[f"{acme}_hi"], "acme_p": df[f"{acme}_p"],
        "ade": df["ade"], "ade_lo": df["ade_lo"], "ade_hi": df["ade_hi"], "ade_p": df["ade_p"],
        "total": df["tot"], "total_lo": df["tot_lo"], "total_hi": df["tot_hi"], "total_p": df["tot_p"],
        "prop_mediated": pd.NA, "prop_p": pd.NA})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True, type=Path)
    ap.add_argument("--nref", required=True, help="a main grid CSV, for n_episodes/n_subjects per meal")
    ap.add_argument("--arm", choices=["treated", "control"], default="treated")
    ap.add_argument("--lmer-prefix", default="interaction_bootstrap_fc_int_")
    ap.add_argument("--qr-prefix", default="interaction_bootstrap_fc_int_qr_")
    ap.add_argument("--offset", type=int, default=30)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    nref = pd.read_csv(args.nref)
    nmap = {m: (int(nref[nref.meal == m].n_episodes.max()), int(nref[nref.meal == m].n_subjects.max()))
            for m in MEALS}

    frames = []
    for meal in MEALS:
        lf = args.csv_dir / f"{args.lmer_prefix}{meal}.csv"
        if lf.exists():
            frames.append(rows_from(pd.read_csv(lf), meal, "lmer", 0.5, args.arm, nmap, args.offset))
        qf = args.csv_dir / f"{args.qr_prefix}{meal}.csv"
        if qf.exists():
            q = pd.read_csv(qf)
            for tau in sorted(q.tau.dropna().unique()):
                frames.append(rows_from(q[q.tau == tau], meal, "qr", float(tau), args.arm, nmap, args.offset))
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(grid)} rows, arm={args.arm})")


if __name__ == "__main__":
    main()
