#!/usr/bin/env python3
"""
incremental_penalization_study.py
=================================
Shows incremental improvement as penalization layers are added.

Tests the following sequence:
1. Baseline (no penalties)
2. + Linearization penalty
3. + Balancing penalty
4. + Conditional independence penalty
5. + Stability regularizer

This demonstrates the marginal contribution of each component.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows

# Incremental configurations (order matters for visualization)
INCREMENTAL_CONFIGS = [
    {"name": "1_baseline", "lin": False, "bal": False, "ci": False, "stab": False},
    {"name": "2_+linearization", "lin": True, "bal": False, "ci": False, "stab": False},
    {"name": "3_+balancing", "lin": True, "bal": True, "ci": False, "stab": False},
    {"name": "4_+CI_penalty", "lin": True, "bal": True, "ci": True, "stab": False},
    {"name": "5_+stability", "lin": True, "bal": True, "ci": True, "stab": True},
]


def evaluate_phi(phi, Y_seq, A_cont, M_scalar):
    """Evaluate phi quality"""
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score

    # Outcome prediction
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(phi, Y_seq)
    y_pred = ridge.predict(phi)
    outcome_r2 = np.mean([
        r2_score(Y_seq[:, i], y_pred[:, i])
        for i in range(Y_seq.shape[1])
    ])

    # Treatment balance
    A_bin = (A_cont > np.median(A_cont)).astype(int)
    try:
        lr = LogisticRegression(max_iter=1000)
        cv_auc = np.mean(cross_val_score(lr, phi, A_bin, cv=5, scoring="roc_auc"))
        balance = 1 - abs(0.5 - cv_auc) * 2
    except:
        balance = 1.0

    return outcome_r2, balance


def run_incremental_study(csv_dir, architecture="lstm", n_runs=3, seeds=None):
    """Run incremental penalization study"""

    if seeds is None:
        seeds = [42, 123, 456][:n_runs]

    # Load data
    print(f"\nLoading data from {csv_dir}...")
    try:
        (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
         global_window_id, meal_list, subj_list, pre_ints, post_X_ints) = load_windows(
            csv_dir=csv_dir,
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
        return pd.DataFrame()

    results = []

    total = len(INCREMENTAL_CONFIGS) * len(seeds)
    count = 0

    for config in INCREMENTAL_CONFIGS:
        for seed in seeds:
            count += 1
            print(f"\n[{count}/{total}] Training: {config['name']} (seed={seed})")

            try:
                model, encoder, phi, history = train_causal_linear_ae(
                    X_ts_pre=X_ts_pre,
                    meal_ohe=meal_ohe,
                    subj_ohe=subj_ohe,
                    A_cont=Z,
                    M_scalar=mediator_scalar,
                    Y_seq=y_seq,
                    encoder_type=architecture,
                    use_linearization=config["lin"],
                    use_balancing=config["bal"],
                    use_ci_penalty=config["ci"],
                    use_stability=config["stab"],
                    seed=seed,
                    verbose=0
                )

                outcome_r2, balance = evaluate_phi(phi, y_seq, Z, mediator_scalar)

                results.append({
                    "config": config["name"],
                    "config_order": int(config["name"][0]),
                    "seed": seed,
                    "outcome_r2": outcome_r2,
                    "balance_score": balance,
                    "final_loss": history.history["loss"][-1],
                    "architecture": architecture
                })
                print(f"    R²={outcome_r2:.4f}, Balance={balance:.4f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "config": config["name"],
                    "config_order": int(config["name"][0]),
                    "seed": seed,
                    "outcome_r2": np.nan,
                    "balance_score": np.nan,
                    "final_loss": np.nan,
                    "architecture": architecture
                })

    return pd.DataFrame(results)


def plot_incremental_improvement(results_df, output_dir, architecture="lstm"):
    """Plot incremental improvement visualization"""

    # Filter out NaN results
    valid_df = results_df.dropna(subset=["outcome_r2", "balance_score"])

    if len(valid_df) == 0:
        print("No valid results to plot")
        return None

    # Aggregate by config
    agg = valid_df.groupby(["config", "config_order"]).agg({
        "outcome_r2": ["mean", "std"],
        "balance_score": ["mean", "std"]
    }).reset_index()
    agg.columns = ["config", "order", "r2_mean", "r2_std", "bal_mean", "bal_std"]
    agg = agg.sort_values("order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Outcome R² improvement
    ax = axes[0]
    x = range(len(agg))
    bars = ax.bar(x, agg["r2_mean"], yerr=agg["r2_std"], capsize=4,
                  color="steelblue", alpha=0.8, edgecolor="navy")

    # Add improvement annotations
    for i in range(1, len(agg)):
        improvement = agg["r2_mean"].iloc[i] - agg["r2_mean"].iloc[i-1]
        color = "green" if improvement > 0 else "red"
        y_pos = agg["r2_mean"].iloc[i] + agg["r2_std"].iloc[i] + 0.01
        ax.annotate(f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}",
                    xy=(i, y_pos),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_", 1)[1] for c in agg["config"]], rotation=45, ha="right")
    ax.set_ylabel("Outcome R² (mean ± std)")
    ax.set_title(f"Incremental Improvement in Outcome Prediction ({architecture.upper()})")
    ax.set_ylim(0, min(1.0, agg["r2_mean"].max() + 0.15))

    # Plot 2: Balance score improvement
    ax = axes[1]
    bars = ax.bar(x, agg["bal_mean"], yerr=agg["bal_std"], capsize=4,
                  color="coral", alpha=0.8, edgecolor="darkred")

    for i in range(1, len(agg)):
        improvement = agg["bal_mean"].iloc[i] - agg["bal_mean"].iloc[i-1]
        color = "green" if improvement > 0 else "red"
        y_pos = agg["bal_mean"].iloc[i] + agg["bal_std"].iloc[i] + 0.01
        ax.annotate(f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}",
                    xy=(i, y_pos),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_", 1)[1] for c in agg["config"]], rotation=45, ha="right")
    ax.set_ylabel("Balance Score (mean ± std)")
    ax.set_title(f"Incremental Improvement in Treatment Balance ({architecture.upper()})")
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / f"incremental_penalization_{architecture}.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    return agg


def get_meal_windows_dir(data_arg, data_dir_arg):
    """Get the meal windows directory based on command-line arguments."""
    if data_dir_arg:
        return str(Path(data_dir_arg))
    elif data_arg == "2020":
        return str(CONFIG.MEAL_WINDOWS_2020_DIR)
    elif data_arg == "combined":
        return str(CONFIG.MEAL_WINDOWS_COMBINED_DIR)
    else:  # default to 2018
        return str(CONFIG.MEAL_WINDOWS_2018_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run incremental penalization study")
    parser.add_argument("--data", choices=["2018", "2020", "combined"], default="2018",
                        help="Dataset to use: 2018 (default), 2020, or combined")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Custom path to meal windows directory (overrides --data)")
    parser.add_argument("--arch", nargs="+", default=["lstm", "cnn"],
                        help="Architectures to test")
    parser.add_argument("--n_runs", type=int, default=3,
                        help="Number of runs per configuration")
    args = parser.parse_args()

    meal_windows_dir = get_meal_windows_dir(args.data, args.data_dir)

    output_dir = CONFIG.FIGURES_DIR / "incremental_penalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {meal_windows_dir}")

    for arch in args.arch:
        print(f"\n{'='*60}")
        print(f"Running incremental study for {arch.upper()}")
        print("="*60)

        results = run_incremental_study(meal_windows_dir, architecture=arch, n_runs=args.n_runs)

        if len(results) > 0:
            results.to_csv(output_dir / f"incremental_results_{arch}.csv", index=False)
            summary = plot_incremental_improvement(results, output_dir, arch)

            if summary is not None:
                print(f"\n{arch.upper()} Summary:")
                print(summary.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")
