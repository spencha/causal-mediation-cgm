#!/usr/bin/env python3
"""DiaTrend architecture/optimizer/penalty ablation -- pick the best CLAE config.

The DiaTrend analog of run_comprehensive_ae_comparison.py. Because
diatrend_loader.load_diatrend_data() returns the SAME 13-tuple as
resid_ae_utils.load_windows(), the existing trainer (train_causal_linear_ae) and
scorer (evaluate_phi_quality) work unchanged -- we just build the DiaTrend
episode tensors ONCE (parsing the workbooks is expensive) and reuse them across
every config in the sweep.

Sweeps architecture x optimizer x penalty x seed and writes a CSV with the same
columns as the Ohio comparison (architecture, optimizer, penalty_config, seed,
outcome_r2_mean, balance_score, treatment_auc, mediator_r2, ...), so
visualize_architecture_comparison.py renders the same Top-10 figure.

Selection: like Ohio, rank by balance_score then outcome R^2 -- BUT this is a
causal *linear* AE, so prefer configs with linearization on (lin=True); a
"balance_only"/"none" winner has no linear structure for the mediation.

Usage (cluster, in the diatrend env):
  # smoke test: one config, confirm a single run trains + scores
  python ae_python_code/experiments/diatrend_ae_ablation.py --smoke \
      --raw-dir ~/DiaTrend/raw --cohorts 2 --out mediation_results/diatrend/ae_ablation
  # full sweep (use --seeds 42 to cut 3x for CPU):
  python ae_python_code/experiments/diatrend_ae_ablation.py \
      --raw-dir ~/DiaTrend/raw --cohorts 2 --out mediation_results/diatrend/ae_ablation
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))          # ae_python_code/
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))   # repo root (data_processing)

from causal_linear_ae import train_causal_linear_ae          # noqa: E402
from diatrend_loader import load_diatrend_data                # noqa: E402
from experiments.run_comprehensive_ae_comparison import evaluate_phi_quality  # noqa: E402

ARCHITECTURES = ["lstm", "cnn"]
OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "nadam"]
SEEDS = [42, 123, 456]
PENALTY_CONFIGS = [
    {"name": "none", "lin": False, "bal": False, "ci": False, "stab": False},
    {"name": "linear_only", "lin": True, "bal": False, "ci": False, "stab": False},
    {"name": "balance_only", "lin": False, "bal": True, "ci": False, "stab": False},
    {"name": "linear+balance", "lin": True, "bal": True, "ci": False, "stab": False},
    {"name": "linear+balance+ci", "lin": True, "bal": True, "ci": True, "stab": False},
    {"name": "all_penalties", "lin": True, "bal": True, "ci": True, "stab": True},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="~/DiaTrend/raw", help="DiaTrend workbooks dir.")
    ap.add_argument("--cohorts", default="2", help="Comma-separated cohorts to train on. [2]")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--archs", default=",".join(ARCHITECTURES))
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true",
                    help="Train ONE config (cnn/adam/linear+balance, seed 42) and stop.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cohorts = [int(c) for c in args.cohorts.split(",")]

    # --- build DiaTrend tensors ONCE (mirrors load_windows' 13-tuple) -------
    print(f"Loading DiaTrend episodes (cohorts={cohorts}) from {args.raw_dir} ...", flush=True)
    data_tuple, cohort_labels, std_params = load_diatrend_data(
        raw_dir=Path(args.raw_dir).expanduser(), cohorts=cohorts,
        interval_min=5, pre_minutes=120, post_X_minutes=60, post_total_minutes=240,
        standardize=True)
    # data_tuple mirrors load_windows' tuple; grab by position (robust to its
    # exact length): 0 X_ts, 1 X_ts_pre, 2 meal_ohe, 3 subj_ohe, 4 Z, 5 Z_bin,
    # 6 y_seq, 7 mediator_scalar, 8 global_window_id, 9 meal_list, 10 subj_list...
    X_ts_pre, meal_ohe, subj_ohe = data_tuple[1], data_tuple[2], data_tuple[3]
    Z, y_seq, mediator_scalar = data_tuple[4], data_tuple[6], data_tuple[7]
    subj_list = data_tuple[10]
    print(f"  {X_ts_pre.shape[0]} episodes, {len(set(subj_list))} subjects, "
          f"Y horizon {y_seq.shape[1]}", flush=True)

    archs = args.archs.split(",")
    opts = args.optimizers.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    grid = [(a, o, p, s) for a in archs for o in opts for p in PENALTY_CONFIGS for s in seeds]
    if args.smoke:
        grid = [("cnn", "adam", PENALTY_CONFIGS[3], 42)]   # linear+balance
    print(f"Sweeping {len(grid)} configs "
          f"({len(archs)} arch x {len(opts)} opt x {len(PENALTY_CONFIGS)} penalty x {len(seeds)} seed)\n",
          flush=True)

    import tensorflow as tf
    csv = args.out / ("ae_ablation_smoke.csv" if args.smoke else "diatrend_ae_ablation.csv")
    rows = []
    for i, (arch, opt, pen, seed) in enumerate(grid, 1):
        tag = f"{arch}/{opt}/{pen['name']}/seed{seed}"
        print(f"[{i}/{len(grid)}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            _, _, phi, history = train_causal_linear_ae(
                X_ts_pre=X_ts_pre, meal_ohe=meal_ohe, subj_ohe=subj_ohe,
                A_cont=Z, M_scalar=mediator_scalar, Y_seq=y_seq,
                latent_dim=args.latent_dim, encoder_type=arch, optimizer_name=opt,
                use_linearization=pen["lin"], use_balancing=pen["bal"],
                use_ci_penalty=pen["ci"], use_stability=pen["stab"],
                epochs=args.epochs, batch_size=args.batch_size, seed=seed, verbose=0)
            metrics = evaluate_phi_quality(phi, y_seq, Z, mediator_scalar)
            row = {"status": "success", "architecture": arch, "optimizer": opt,
                   "penalty_config": pen["name"], "use_linearization": pen["lin"],
                   "use_balancing": pen["bal"], "use_ci_penalty": pen["ci"],
                   "use_stability": pen["stab"], "seed": seed,
                   "n_samples": int(X_ts_pre.shape[0]), "latent_dim": args.latent_dim,
                   "epochs": args.epochs, "training_time_sec": time.time() - t0,
                   "final_loss": history.history["loss"][-1], **metrics}
            print(f"    balance={metrics['balance_score']:.3f}  "
                  f"outcome_R2={metrics['outcome_r2_mean']:.3f}  "
                  f"({row['training_time_sec']:.0f}s)", flush=True)
        except Exception as e:
            row = {"status": "error", "architecture": arch, "optimizer": opt,
                   "penalty_config": pen["name"], "seed": seed, "error": str(e)}
            print(f"    ERROR: {e}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)   # incremental save
        tf.keras.backend.clear_session(); gc.collect()

    ok = pd.DataFrame(rows)
    ok = ok[ok.status == "success"] if "status" in ok else ok
    if len(ok):
        lin = ok[ok.use_linearization == True] if "use_linearization" in ok else ok
        best = (lin if len(lin) else ok).sort_values(
            ["balance_score", "outcome_r2_mean"], ascending=False).iloc[0]
        print(f"\nBest linearization-on config: {best.architecture}/{best.optimizer}/"
              f"{best.penalty_config} -- balance={best.balance_score:.3f}, "
              f"outcome_R2={best.outcome_r2_mean:.3f}")
    print(f"\nWrote {csv}")


if __name__ == "__main__":
    main()
