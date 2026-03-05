#!/usr/bin/env python3
"""
run_comprehensive_ae_comparison.py
==================================
Comprehensive comparison of autoencoder configurations on combined 2018+2020
training data.

Purpose: Select the best architecture and optimizer. These choices are made
*before* the penalization ablation, so we train on the combined training set.

Sweeps:
1. ARCHITECTURE: LSTM vs CNN
2. OPTIMIZERS: Adam, AdamW, SGD, RMSprop, Nadam
3. PENALIZATION: 6 key penalty configurations

Outputs:
- CSV with all metrics for each configuration
- Summary tables by architecture, optimizer, and penalty
"""

import gc
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import CONFIG
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows

# =============================================================================
# CONFIGURATION
# =============================================================================

ARCHITECTURES = ["lstm", "cnn"]
OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "nadam"]
SEEDS = [42, 123, 456]
EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 16

# Penalization configurations (subset for efficiency)
# Full ablation has 16 configs; here we test key combinations
PENALTY_CONFIGS = [
    {"name": "none", "lin": False, "bal": False, "ci": False, "stab": False},
    {"name": "linear_only", "lin": True, "bal": False, "ci": False, "stab": False},
    {"name": "balance_only", "lin": False, "bal": True, "ci": False, "stab": False},
    {"name": "linear+balance", "lin": True, "bal": True, "ci": False, "stab": False},
    {"name": "linear+balance+ci", "lin": True, "bal": True, "ci": True, "stab": False},
    {"name": "all_penalties", "lin": True, "bal": True, "ci": True, "stab": True},
]

FEATURES = ["glucose", "steps", "basal", "meal", "heart", "bolus"]


def evaluate_phi_quality(phi, Y_seq, A_cont, M_scalar):
    """Evaluate representation quality with standard metrics"""
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    metrics = {}

    # Outcome prediction (main goal)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(phi, Y_seq)
    y_pred = ridge.predict(phi)
    metrics["outcome_r2_mean"] = np.mean([
        r2_score(Y_seq[:, i], y_pred[:, i])
        for i in range(Y_seq.shape[1])
    ])

    # Treatment balance
    A_bin = (A_cont > np.median(A_cont)).astype(int)
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        cv_scores = cross_val_score(lr, phi, A_bin, cv=5, scoring="roc_auc")
        metrics["treatment_auc"] = np.mean(cv_scores)
        metrics["balance_score"] = 1 - abs(0.5 - metrics["treatment_auc"]) * 2
    except Exception:
        metrics["treatment_auc"] = 0.5
        metrics["balance_score"] = 1.0

    # Mediator prediction
    ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge_m.fit(phi, M_scalar)
    metrics["mediator_r2"] = ridge_m.score(phi, M_scalar)

    return metrics


def run_single_configuration(csv_dir, architecture, optimizer,
                              penalty_config, seed):
    """Run a single configuration and return results"""

    # Load data — use absolute path to avoid fallback to wrong dataset
    try:
        (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
         global_window_id, meal_list, subj_list, pre_ints, post_X_ints) = load_windows(
            csv_dir=str(csv_dir),
            features=FEATURES,
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=True,
        )
    except Exception as e:
        return {"status": "data_load_error", "error": str(e)}

    n_samples = X_ts_pre.shape[0]

    # Train model
    t0 = time.time()
    try:
        model, encoder, phi, history = train_causal_linear_ae(
            X_ts_pre=X_ts_pre,
            meal_ohe=meal_ohe,
            subj_ohe=subj_ohe,
            A_cont=Z,
            M_scalar=mediator_scalar,
            Y_seq=y_seq,
            latent_dim=LATENT_DIM,
            encoder_type=architecture,
            optimizer_name=optimizer,
            use_linearization=penalty_config["lin"],
            use_balancing=penalty_config["bal"],
            use_ci_penalty=penalty_config["ci"],
            use_stability=penalty_config["stab"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=seed,
            verbose=0
        )
        training_time = time.time() - t0

        # Evaluate
        metrics = evaluate_phi_quality(phi, y_seq, Z, mediator_scalar)

        result = {
            "status": "success",
            "architecture": architecture,
            "optimizer": optimizer,
            "penalty_config": penalty_config["name"],
            "use_linearization": penalty_config["lin"],
            "use_balancing": penalty_config["bal"],
            "use_ci_penalty": penalty_config["ci"],
            "use_stability": penalty_config["stab"],
            "seed": seed,
            "n_samples": n_samples,
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "training_time_sec": training_time,
            "final_loss": history.history["loss"][-1],
            "final_val_loss": history.history.get("val_loss", [None])[-1],
            "min_val_loss": min(history.history.get("val_loss", [float("inf")])),
            **metrics
        }

    except Exception as e:
        result = {
            "status": "training_error",
            "architecture": architecture,
            "optimizer": optimizer,
            "penalty_config": penalty_config["name"],
            "seed": seed,
            "error": str(e)
        }

    # Prevent TensorFlow memory accumulation across runs
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()

    return result


def run_full_comparison(csv_dir, resume_csv=None):
    """Run the complete comparison across all configurations

    Args:
        csv_dir: Directory containing training CSVs
        resume_csv: Path to partial results CSV to resume from
    """

    results = []
    completed_keys = set()

    # Load partial results if resuming
    if resume_csv and Path(resume_csv).exists():
        prev_df = pd.read_csv(resume_csv)
        results = prev_df.to_dict("records")
        for _, row in prev_df[prev_df["status"] == "success"].iterrows():
            key = (row["architecture"], row["optimizer"],
                   row["penalty_config"], int(row["seed"]))
            completed_keys.add(key)
        print(f"Resuming: loaded {len(completed_keys)} completed runs from {resume_csv}")

    # Incremental save path
    incremental_path = Path(CONFIG.EXPERIMENT_RESULTS_DIR) / "comprehensive_comparison_incremental.csv"
    incremental_path.parent.mkdir(parents=True, exist_ok=True)

    total_configs = len(ARCHITECTURES) * len(OPTIMIZERS) * len(PENALTY_CONFIGS) * len(SEEDS)

    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE AE COMPARISON")
    print(f"Training data: {csv_dir}")
    print(f"Total configurations: {total_configs}")
    if completed_keys:
        print(f"Already completed: {len(completed_keys)} (skipping)")
    print(f"{'='*70}\n")

    config_num = 0
    for architecture in ARCHITECTURES:
        for optimizer in OPTIMIZERS:
            for penalty_config in PENALTY_CONFIGS:
                for seed in SEEDS:
                    config_num += 1
                    key = (architecture, optimizer, penalty_config["name"], seed)

                    if key in completed_keys:
                        print(f"[{config_num}/{total_configs}] "
                              f"{architecture} | {optimizer} | "
                              f"{penalty_config['name']} | seed={seed} -> SKIP (done)")
                        continue

                    print(f"[{config_num}/{total_configs}] "
                          f"{architecture} | {optimizer} | "
                          f"{penalty_config['name']} | seed={seed}")

                    result = run_single_configuration(
                        csv_dir, architecture, optimizer,
                        penalty_config, seed
                    )
                    results.append(result)

                    if result["status"] == "success":
                        print(f"    -> R²={result['outcome_r2_mean']:.4f}, "
                              f"Balance={result['balance_score']:.4f}, "
                              f"Time={result['training_time_sec']:.1f}s")
                    else:
                        print(f"    -> ERROR: {result.get('error', 'unknown')}")

                    # Save incrementally after each run
                    pd.DataFrame(results).to_csv(incremental_path, index=False)

    return pd.DataFrame(results)


def save_results(results_df, output_dir):
    """Save results and generate summary statistics"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results
    results_df.to_csv(output_dir / f"comprehensive_comparison_{timestamp}.csv",
                      index=False)

    # Generate summary tables
    successful = results_df[results_df["status"] == "success"]

    if len(successful) > 0:
        # Summary by architecture
        arch_summary = successful.groupby("architecture").agg({
            "outcome_r2_mean": ["mean", "std"],
            "balance_score": ["mean", "std"],
            "training_time_sec": "mean"
        }).round(4)
        arch_summary.to_csv(output_dir / "summary_by_architecture.csv")

        # Summary by optimizer
        opt_summary = successful.groupby("optimizer").agg({
            "outcome_r2_mean": ["mean", "std"],
            "balance_score": ["mean", "std"],
            "min_val_loss": ["mean", "std"]
        }).round(4)
        opt_summary.to_csv(output_dir / "summary_by_optimizer.csv")

        # Summary by penalty configuration
        penalty_summary = successful.groupby("penalty_config").agg({
            "outcome_r2_mean": ["mean", "std"],
            "balance_score": ["mean", "std"],
            "mediator_r2": ["mean", "std"]
        }).round(4)
        penalty_summary.to_csv(output_dir / "summary_by_penalty.csv")

        # Best configuration: maximize balance, then R² as tiebreaker
        successful = successful.copy()
        successful["_rank"] = successful["balance_score"].rank(ascending=False) * 1000 - successful["outcome_r2_mean"]
        best_idx = successful["_rank"].idxmin()
        best_config = successful.loc[[best_idx], [
            "architecture", "optimizer", "penalty_config",
            "outcome_r2_mean", "balance_score"
        ]]
        best_config.to_csv(output_dir / "best_configurations.csv", index=False)

        print("\n" + "="*70)
        print("BEST CONFIGURATION (max balance, then R²)")
        print("="*70)
        print(best_config.to_string(index=False))

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive AE comparison")
    parser.add_argument("--arch", nargs="+", default=None,
                        choices=["lstm", "cnn"],
                        help="Architectures to test (default: both)")
    parser.add_argument("--optimizers", nargs="+", default=None,
                        choices=["adam", "adamw", "sgd", "rmsprop", "nadam"],
                        help="Optimizers to test (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds (default: [42, 123, 456])")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: lstm only, adamw only, 2 penalties, 1 seed, 10 epochs")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: both archs, adamw only, 3 penalties, 1 seed, 30 epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results CSV to resume from. "
                             "Use --resume auto to resume from incremental save.")
    args = parser.parse_args()

    # Apply presets
    if args.quick:
        ARCHITECTURES = ["lstm"]
        OPTIMIZERS = ["adamw"]
        PENALTY_CONFIGS = [
            {"name": "none", "lin": False, "bal": False, "ci": False, "stab": False},
            {"name": "all_penalties", "lin": True, "bal": True, "ci": True, "stab": True},
        ]
        SEEDS = [42]
        EPOCHS = 10
    elif args.fast:
        ARCHITECTURES = ["lstm", "cnn"]
        OPTIMIZERS = ["adamw"]
        PENALTY_CONFIGS = [
            {"name": "none", "lin": False, "bal": False, "ci": False, "stab": False},
            {"name": "linear+balance", "lin": True, "bal": True, "ci": False, "stab": False},
            {"name": "all_penalties", "lin": True, "bal": True, "ci": True, "stab": True},
        ]
        SEEDS = [42]
        EPOCHS = 30

    # Apply individual overrides
    if args.arch:
        ARCHITECTURES = args.arch
    if args.optimizers:
        OPTIMIZERS = args.optimizers
    if args.seeds:
        SEEDS = args.seeds
    if args.epochs:
        EPOCHS = args.epochs

    # Use combined 2018+2020 training data
    csv_dir = CONFIG.MEAL_WINDOWS_COMBINED_TRAIN_DIR
    if not csv_dir.exists():
        print(f"ERROR: Combined training data not found: {csv_dir}")
        print("Run the train/test split script first.")
        sys.exit(1)

    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Data (combined train): {csv_dir}")
    print(f"  Architectures: {ARCHITECTURES}")
    print(f"  Optimizers: {OPTIMIZERS}")
    print(f"  Penalties: {[p['name'] for p in PENALTY_CONFIGS]}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {EPOCHS}")

    total = len(ARCHITECTURES) * len(OPTIMIZERS) * len(PENALTY_CONFIGS) * len(SEEDS)
    print(f"  Total configs: {total}")

    # Resolve resume path
    resume_csv = None
    if args.resume:
        if args.resume == "auto":
            resume_csv = CONFIG.EXPERIMENT_RESULTS_DIR / "comprehensive_comparison_incremental.csv"
        else:
            resume_csv = Path(args.resume)
        if resume_csv.exists():
            print(f"  Resuming from: {resume_csv}")
        else:
            print(f"  Resume file not found: {resume_csv} (starting fresh)")
            resume_csv = None

    results = run_full_comparison(csv_dir, resume_csv=resume_csv)
    output_dir = save_results(results, CONFIG.EXPERIMENT_RESULTS_DIR)
    print(f"\nResults saved to: {output_dir}")
