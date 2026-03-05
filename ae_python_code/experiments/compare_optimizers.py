#!/usr/bin/env python3
"""
compare_optimizers.py
=====================
Compare different optimizers for the Causal Linear Autoencoder.
Generates comparison metrics and visualizations.

Usage:
    python compare_optimizers.py --data 2020
    python compare_optimizers.py --data combined
    python compare_optimizers.py --data 2018  # default
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from config import CONFIG
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows


def get_meal_windows_dir(args):
    """Get the meal windows directory based on command-line arguments."""
    if args.data_dir:
        return Path(args.data_dir)
    elif args.data == "2020":
        return CONFIG.MEAL_WINDOWS_2020_DIR
    elif args.data == "combined":
        return CONFIG.MEAL_WINDOWS_COMBINED_DIR
    else:  # default to 2018
        return CONFIG.MEAL_WINDOWS_DIR

OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "nadam"]
N_RUNS = 3  # Multiple runs for stability
EPOCHS = 50
SEEDS = [42, 123, 456]


def evaluate_phi_quality(phi, Y_seq, A_cont, M_scalar):
    """Evaluate quality metrics for learned representation"""
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    metrics = {}

    # 1. Linear predictability of outcomes (main goal)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(phi, Y_seq)
    y_pred = ridge.predict(phi)
    metrics["outcome_r2"] = r2_score(Y_seq, y_pred)
    metrics["outcome_r2_mean"] = np.mean([
        r2_score(Y_seq[:, i], y_pred[:, i])
        for i in range(Y_seq.shape[1])
    ])

    # 2. Treatment balance (balancing penalty goal)
    A_bin = (A_cont > np.median(A_cont)).astype(int)
    try:
        lr = LogisticRegression(max_iter=1000, C=1.0)
        cv_scores = cross_val_score(lr, phi, A_bin, cv=5, scoring="roc_auc")
        metrics["treatment_auc"] = np.mean(cv_scores)
        # Lower is better for balance (closer to 0.5)
        metrics["balance_score"] = 1 - abs(0.5 - metrics["treatment_auc"]) * 2
    except:
        metrics["treatment_auc"] = 0.5
        metrics["balance_score"] = 1.0

    # 3. Mediator predictability
    ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge_m.fit(phi, M_scalar)
    metrics["mediator_r2"] = ridge_m.score(phi, M_scalar)

    return metrics


def run_optimizer_comparison(X_ts_pre, meal_ohe, subj_ohe, A_cont, M_scalar, Y_seq):
    """Run comparison across all optimizers"""
    results = []

    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Training with {optimizer}, seed={seed}")
            print('='*60)

            try:
                model, encoder, phi, history = train_causal_linear_ae(
                    X_ts_pre=X_ts_pre,
                    meal_ohe=meal_ohe,
                    subj_ohe=subj_ohe,
                    A_cont=A_cont,
                    M_scalar=M_scalar,
                    Y_seq=Y_seq,
                    optimizer_name=optimizer,
                    epochs=EPOCHS,
                    seed=seed,
                    verbose=0
                )

                # Extract metrics
                metrics = evaluate_phi_quality(phi, Y_seq, A_cont, M_scalar)

                result = {
                    "optimizer": optimizer,
                    "seed": seed,
                    "final_loss": history.history["loss"][-1],
                    "final_val_loss": history.history.get("val_loss", [None])[-1],
                    "min_val_loss": min(history.history.get("val_loss", [float("inf")])),
                    "best_epoch": int(np.argmin(history.history.get("val_loss", history.history["loss"])) + 1),
                    "convergence_epoch": _find_convergence_epoch(history),
                    "final_y_pred_loss": history.history.get("y_pred_loss", [None])[-1],
                    "final_m_pred_loss": history.history.get("m_pred_loss", [None])[-1],
                    **metrics,
                    "status": "success"
                }
            except Exception as e:
                result = {
                    "optimizer": optimizer,
                    "seed": seed,
                    "status": "error",
                    "error": str(e)
                }
            results.append(result)

    return pd.DataFrame(results)


def _find_convergence_epoch(history, patience=5, min_delta=1e-4):
    """Find epoch where loss stabilized"""
    losses = history.history.get("val_loss", history.history["loss"])
    for i in range(patience, len(losses)):
        window = losses[i-patience:i]
        if max(window) - min(window) < min_delta:
            return i - patience
    return len(losses)


def plot_optimizer_comparison(results_df, output_dir):
    """Generate comparison visualizations"""

    # Filter successful runs
    successful = results_df[results_df["status"] == "success"]
    if len(successful) == 0:
        print("No successful runs to plot")
        return None

    # Aggregate by optimizer
    agg = successful.groupby("optimizer").agg({
        "final_loss": ["mean", "std"],
        "final_val_loss": ["mean", "std"],
        "min_val_loss": ["mean", "std"],
        "best_epoch": "mean",
        "convergence_epoch": "mean",
        "outcome_r2_mean": ["mean", "std"],
        "balance_score": ["mean", "std"]
    }).round(4)

    # Bar plot of final validation loss
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Final validation loss
    means = successful.groupby("optimizer")["min_val_loss"].mean()
    stds = successful.groupby("optimizer")["min_val_loss"].std()
    ax = axes[0]
    means.plot(kind="bar", ax=ax, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_title("Minimum Validation Loss by Optimizer")
    ax.set_ylabel("Loss")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    # Plot 2: Convergence speed
    ax = axes[1]
    successful.boxplot(column="convergence_epoch", by="optimizer", ax=ax)
    ax.set_title("Convergence Speed (epochs to stabilize)")
    ax.set_ylabel("Epochs")
    ax.set_xlabel("")
    plt.suptitle("")

    # Plot 3: Outcome prediction R²
    ax = axes[2]
    means = successful.groupby("optimizer")["outcome_r2_mean"].mean()
    stds = successful.groupby("optimizer")["outcome_r2_mean"].std()
    means.plot(kind="bar", ax=ax, yerr=stds, capsize=4, color="coral", alpha=0.8)
    ax.set_title("Outcome Prediction R² (y_pred)")
    ax.set_ylabel("R²")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare optimizers for Causal Linear Autoencoder."
    )
    parser.add_argument(
        "--data",
        choices=["2018", "2020", "combined"],
        default="2018",
        help="Dataset to use: 2018 (default), 2020, or combined"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Custom path to meal windows directory (overrides --data)"
    )
    args = parser.parse_args()

    meal_windows_dir = get_meal_windows_dir(args)

    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON FOR CAUSAL LINEAR AUTOENCODER")
    print("="*60)
    print(f"Data directory: {meal_windows_dir}")

    # Load data
    print("\nLoading data...")
    try:
        (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
         global_window_id, meal_list, subj_list, pre_ints, post_X_ints) = load_windows(
            csv_dir=str(meal_windows_dir),
            features=["glucose", "steps", "basal", "meal", "heart", "bolus"],
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=True,
        )
        print(f"Loaded {X_ts_pre.shape[0]} windows")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure {meal_windows_dir} exists with CSV files")
        exit(1)

    # Run comparison
    results = run_optimizer_comparison(X_ts_pre, meal_ohe, subj_ohe, Z, mediator_scalar, y_seq)

    # Save results
    output_dir = CONFIG.FIGURES_DIR / "optimizer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "optimizer_comparison_results.csv", index=False)
    summary = plot_optimizer_comparison(results, output_dir)

    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("="*60)
    if summary is not None:
        print(summary)
    print(f"\nResults saved to: {output_dir}")
