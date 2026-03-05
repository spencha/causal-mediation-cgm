#!/usr/bin/env python3
"""
ablation_penalization_layers.py
===============================
Ablation study for CLAE penalization layers.
Tests each penalty in isolation and all combinations.

Usage:
    python ablation_penalization_layers.py              # default: combined train
    python ablation_penalization_layers.py --data 2018
    python ablation_penalization_layers.py --data 2020

Penalties tested:
1. LinearizabilityPenalty - Forces phi to be linearly predictive of outcomes
2. BalancingScorePenalty - Encourages treatment balance across phi values
3. ConditionalIndependencePenalty - Satisfies mediation CI assumptions
4. StabilityRegularizer - Encourages stable bootstrap representations
"""

import argparse
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))

from config import CONFIG
from causal_linear_ae import train_causal_linear_ae
from resid_ae_utils import load_windows


def get_meal_windows_dir(args):
    """Get the meal windows directory based on command-line arguments."""
    if args.data_dir:
        return Path(args.data_dir)
    elif args.data == "2020":
        # Use train subdirectory — top-level 2020 dir has no CSVs
        return CONFIG.MEAL_WINDOWS_2020_TRAIN_DIR
    elif args.data == "combined":
        return CONFIG.MEAL_WINDOWS_COMBINED_TRAIN_DIR
    else:  # 2018
        return CONFIG.MEAL_WINDOWS_DIR


# Ablation configurations
PENALTY_FLAGS = {
    "linearization": True,
    "balancing": True,
    "ci_penalty": True,
    "stability": True
}


def generate_ablation_configs():
    """Generate all 2^4 = 16 configurations"""
    flags = list(PENALTY_FLAGS.keys())
    configs = []

    for combo in product([False, True], repeat=len(flags)):
        config = dict(zip(flags, combo))
        config["name"] = "_".join([f for f, v in config.items() if v]) or "baseline"
        configs.append(config)

    return configs


def evaluate_representation_quality(phi, Y_seq, A_cont, M_scalar, in_range_seq=None):
    """Evaluate quality metrics for learned representation

    Args:
        phi: Learned representation (n, latent_dim)
        Y_seq: Delta glucose sequence (n, T) starting at 60min post-meal
        A_cont: Treatment (meal size)
        M_scalar: Mediator (bolus)
        in_range_seq: Binary in-range sequence (n, T) - glucose 70-140 mg/dL
    """
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.decomposition import PCA

    metrics = {}

    # Timepoints: Y_seq starts at 60min post-meal, 5-min intervals
    # Index mapping: 0=60min, 6=90min, 12=120min
    # For 30min, we need data before Y_seq starts (not available in Y_seq)
    timepoints = {
        "60min": 0,    # Y_seq[0] = 60 min post-meal
        "90min": 6,    # Y_seq[6] = 90 min post-meal
        "120min": 12,  # Y_seq[12] = 120 min post-meal
    }

    # 1. R² at specific timepoints (linear predictability)
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge.fit(phi, Y_seq)
    y_pred = ridge.predict(phi)

    for tp_name, tp_idx in timepoints.items():
        if tp_idx < Y_seq.shape[1]:
            r2_val = r2_score(Y_seq[:, tp_idx], y_pred[:, tp_idx])
            metrics[f"r2_{tp_name}"] = r2_val

    # Overall R²
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
        metrics["balance_score"] = 1 - abs(0.5 - metrics["treatment_auc"]) * 2
    except:
        metrics["treatment_auc"] = 0.5
        metrics["balance_score"] = 1.0

    # 3. Mediator predictability
    ridge_m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    ridge_m.fit(phi, M_scalar)
    metrics["mediator_r2"] = ridge_m.score(phi, M_scalar)

    # 4. Variance explained ratio (stability proxy)
    phi_var = np.var(phi, axis=0)
    metrics["var_ratio"] = np.max(phi_var) / (np.mean(phi_var) + 1e-8)

    # 5. Effective dimensionality
    pca = PCA(n_components=min(phi.shape[1], 10))
    pca.fit(phi)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    metrics["effective_dim_90"] = int(np.searchsorted(cumvar, 0.90) + 1)
    metrics["effective_dim_95"] = int(np.searchsorted(cumvar, 0.95) + 1)

    # 6. In-range AUC at each timepoint (using actual in_range from data)
    for tp_name, tp_idx in timepoints.items():
        if tp_idx < Y_seq.shape[1]:
            # Use actual in_range if available
            if in_range_seq is not None and tp_idx < in_range_seq.shape[1]:
                inrange_label = in_range_seq[:, tp_idx].astype(int)
            else:
                # Fallback: compute from delta glucose (less accurate)
                y_delta = Y_seq[:, tp_idx]
                inrange_label = ((y_delta >= -50) & (y_delta <= 60)).astype(int)

            # Check we have enough samples in each class
            if inrange_label.sum() > 10 and (1 - inrange_label).sum() > 10:
                try:
                    lr_inrange = LogisticRegression(max_iter=1000, class_weight="balanced")
                    inrange_proba = cross_val_predict(lr_inrange, phi, inrange_label, cv=5, method="predict_proba")[:, 1]
                    metrics[f"inrange_auc_{tp_name}"] = roc_auc_score(inrange_label, inrange_proba)
                except:
                    metrics[f"inrange_auc_{tp_name}"] = np.nan
            else:
                metrics[f"inrange_auc_{tp_name}"] = np.nan

    return metrics


def run_ablation_study(X_ts_pre, meal_ohe, subj_ohe, A_cont, M_scalar, Y_seq,
                       in_range_seq=None, epochs=50, seeds=None):
    """Run full ablation study

    Args:
        in_range_seq: Optional in-range sequence (n, T) for AUC calculation
    """
    if seeds is None:
        seeds = [42, 123]

    configs = generate_ablation_configs()
    results = []

    total = len(configs) * len(seeds)
    count = 0

    for config in configs:
        for seed in seeds:
            count += 1
            print(f"\n[{count}/{total}] Config: {config['name']}, seed={seed}")

            try:
                model, encoder, phi, history = train_causal_linear_ae(
                    X_ts_pre=X_ts_pre,
                    meal_ohe=meal_ohe,
                    subj_ohe=subj_ohe,
                    A_cont=A_cont,
                    M_scalar=M_scalar,
                    Y_seq=Y_seq,
                    encoder_type="cnn",       # Stage 1 result
                    optimizer_name="rmsprop", # Stage 1 result
                    use_linearization=config["linearization"],
                    use_balancing=config["balancing"],
                    use_ci_penalty=config["ci_penalty"],
                    use_stability=config["stability"],
                    epochs=epochs,
                    seed=seed,
                    verbose=0
                )

                # Evaluate representation
                metrics = evaluate_representation_quality(phi, Y_seq, A_cont, M_scalar, in_range_seq)

                result = {
                    "config_name": config["name"],
                    "linearization": config["linearization"],
                    "balancing": config["balancing"],
                    "ci_penalty": config["ci_penalty"],
                    "stability": config["stability"],
                    "seed": seed,
                    "final_loss": history.history["loss"][-1],
                    "final_val_loss": history.history.get("val_loss", [None])[-1],
                    **metrics,
                    "status": "success"
                }
                print(f"    R²={metrics['outcome_r2_mean']:.4f}, Balance={metrics['balance_score']:.4f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                result = {
                    "config_name": config["name"],
                    "linearization": config["linearization"],
                    "balancing": config["balancing"],
                    "ci_penalty": config["ci_penalty"],
                    "stability": config["stability"],
                    "seed": seed,
                    "status": "error",
                    "error": str(e)
                }

            # Prevent TensorFlow memory accumulation across runs
            import tensorflow as tf
            tf.keras.backend.clear_session()
            gc.collect()

            results.append(result)

    return pd.DataFrame(results)


def plot_ablation_results(results_df, output_dir):
    """Generate ablation study visualizations

    Args:
        results_df: DataFrame with ablation results
        output_dir: Directory containing results CSV. Figures are saved to
                    a sibling 'figures/' directory to keep data/ clean.
    """
    # Save figures to figures/ directory (sibling of data/)
    figures_dir = Path(output_dir).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    successful = results_df[results_df["status"] == "success"]
    if len(successful) == 0:
        print("No successful runs to plot")
        return None

    # Aggregate results
    agg = successful.groupby("config_name").agg({
        "outcome_r2_mean": ["mean", "std"],
        "balance_score": ["mean", "std"],
        "mediator_r2": ["mean", "std"],
        "effective_dim_95": "mean"
    }).round(4)

    # Heatmap of penalty contributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Prepare data for marginal contributions
    penalty_cols = ["linearization", "balancing", "ci_penalty", "stability"]
    metric_cols = ["outcome_r2_mean", "balance_score", "mediator_r2", "final_val_loss"]

    for idx, metric in enumerate(metric_cols):
        ax = axes.flat[idx]

        # Calculate marginal contribution of each penalty
        contributions = {}
        for penalty in penalty_cols:
            with_penalty = successful[successful[penalty] == True][metric].mean()
            without_penalty = successful[successful[penalty] == False][metric].mean()
            contributions[penalty] = with_penalty - without_penalty

        colors = ["green" if v > 0 else "red" for v in contributions.values()]
        if metric == "final_val_loss":
            # For loss, lower is better, so flip colors
            colors = ["green" if v < 0 else "red" for v in contributions.values()]

        bars = ax.bar(contributions.keys(), contributions.values(), color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title(f"Marginal Contribution to {metric}")
        ax.set_ylabel("Δ metric (with - without)")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_penalty_contributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar chart of top configurations
    fig, ax = plt.subplots(figsize=(12, 6))

    config_means = successful.groupby("config_name")["outcome_r2_mean"].mean().sort_values(ascending=False)
    config_stds = successful.groupby("config_name")["outcome_r2_mean"].std()

    config_means.head(10).plot(kind="bar", ax=ax, yerr=config_stds[config_means.head(10).index],
                               capsize=4, color="steelblue", alpha=0.8)
    ax.set_title("Top 10 Configurations by Outcome R²")
    ax.set_ylabel("Mean R² (outcome prediction)")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_top_configs.png", dpi=300, bbox_inches="tight")
    plt.close()

    return agg


def plot_auc_results(results_df, output_dir):
    """Generate AUC-focused visualizations for ablation study

    Args:
        results_df: DataFrame with ablation results
        output_dir: Directory containing results CSV. Figures are saved to
                    a sibling 'figures/' directory to keep data/ clean.
    """
    # Save figures to figures/ directory (sibling of data/)
    figures_dir = Path(output_dir).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    successful = results_df[results_df["status"] == "success"]
    if len(successful) == 0:
        print("No successful runs to plot AUC results")
        return

    # 1. Treatment Balance AUC Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Treatment AUC by configuration
    ax = axes[0]
    config_means = successful.groupby("config_name")["treatment_auc"].mean().sort_values()
    config_stds = successful.groupby("config_name")["treatment_auc"].std()

    # Color code: green if close to 0.5, red if far from 0.5
    colors = ["green" if abs(v - 0.5) < 0.1 else "orange" if abs(v - 0.5) < 0.2 else "red"
              for v in config_means.values]

    bars = ax.barh(range(len(config_means)), config_means.values,
                   xerr=config_stds[config_means.index].values, capsize=3,
                   color=colors, alpha=0.7)
    ax.set_yticks(range(len(config_means)))
    ax.set_yticklabels(config_means.index, fontsize=8)
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Ideal (0.5)")
    ax.set_xlabel("Treatment Balance AUC")
    ax.set_title("Treatment Balance AUC by Configuration\n(Green = balanced, Red = unbalanced)")
    ax.legend()
    ax.set_xlim(0.4, 1.0)

    # Panel B: Effect of balancing penalty on treatment AUC
    ax = axes[1]
    with_balancing = successful[successful["balancing"] == True]["treatment_auc"]
    without_balancing = successful[successful["balancing"] == False]["treatment_auc"]

    positions = [1, 2]
    bp = ax.boxplot([without_balancing, with_balancing], positions=positions, widths=0.6,
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("salmon")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=2, label="Ideal (0.5)")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Without\nBalancing Penalty", "With\nBalancing Penalty"])
    ax.set_ylabel("Treatment Balance AUC")
    ax.set_title("Effect of Balancing Penalty on Treatment AUC")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_treatment_auc.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. R² at Multiple Timepoints Plot
    timepoints = ["60min", "90min", "120min"]
    r2_cols = [f"r2_{tp}" for tp in timepoints]
    available_r2 = [c for c in r2_cols if c in successful.columns]

    if available_r2:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get top 5 configs by mean R²
        mean_r2 = successful.groupby("config_name")[available_r2].mean().mean(axis=1).sort_values(ascending=False)
        top_configs = mean_r2.head(5).index.tolist()

        x = np.arange(len(available_r2))
        width = 0.15
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_configs)))

        for i, config in enumerate(top_configs):
            config_data = successful[successful["config_name"] == config]
            means = [config_data[col].mean() for col in available_r2]
            stds = [config_data[col].std() for col in available_r2]
            ax.bar(x + i * width, means, width, label=config, color=colors[i],
                   yerr=stds, capsize=3, alpha=0.8)

        ax.set_xlabel("Timepoint (post-meal)")
        ax.set_ylabel("R² (linear predictability)")
        ax.set_title("R² at Different Timepoints by Configuration")
        ax.set_xticks(x + width * (len(top_configs) - 1) / 2)
        ax.set_xticklabels([tp.replace("min", " min") for tp in timepoints[:len(available_r2)]])
        ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(figures_dir / "ablation_r2_timepoints.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. In-Range AUC at Multiple Timepoints
    auc_cols = [f"inrange_auc_{tp}" for tp in timepoints]
    available_auc = [c for c in auc_cols if c in successful.columns]

    if available_auc:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get top 5 configs by mean AUC
        valid_data = successful.dropna(subset=available_auc)
        if len(valid_data) > 0:
            mean_auc = valid_data.groupby("config_name")[available_auc].mean().mean(axis=1).sort_values(ascending=False)
            top_configs = mean_auc.head(5).index.tolist()

            x = np.arange(len(available_auc))
            width = 0.15
            colors = plt.cm.Set2(np.linspace(0, 1, len(top_configs)))

            for i, config in enumerate(top_configs):
                config_data = valid_data[valid_data["config_name"] == config]
                means = [config_data[col].mean() for col in available_auc]
                stds = [config_data[col].std() for col in available_auc]
                ax.bar(x + i * width, means, width, label=config, color=colors[i],
                       yerr=stds, capsize=3, alpha=0.8)

            ax.set_xlabel("Timepoint (post-meal)")
            ax.set_ylabel("AUC (in-range prediction)")
            ax.set_title("In-Range AUC at Different Timepoints by Configuration\n(glucose 70-140 mg/dL)")
            ax.set_xticks(x + width * (len(top_configs) - 1) / 2)
            ax.set_xticklabels([tp.replace("min", " min") for tp in timepoints[:len(available_auc)]])
            ax.legend(title="Configuration", bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
            ax.set_ylim(0.4, 1.0)

            plt.tight_layout()
            plt.savefig(figures_dir / "ablation_inrange_auc_timepoints.png", dpi=300, bbox_inches="tight")
            plt.close()

    # 4. Summary table of AUC metrics
    print("\n" + "="*60)
    print("AUC METRICS SUMMARY")
    print("="*60)
    print("\nTreatment Balance AUC (ideal = 0.5):")
    print("-" * 40)
    treat_auc_summary = successful.groupby("config_name")["treatment_auc"].agg(["mean", "std"]).sort_values("mean")
    for config, row in treat_auc_summary.iterrows():
        status = "+" if abs(row["mean"] - 0.5) < 0.1 else "~" if abs(row["mean"] - 0.5) < 0.2 else "-"
        print(f"  {status} {config:40s}: {row['mean']:.3f} +/- {row['std']:.3f}")

    print("\nR² at timepoints (higher = better):")
    print("-" * 40)
    for col in available_r2 if available_r2 else []:
        best_config = successful.groupby("config_name")[col].mean().idxmax()
        best_value = successful.groupby("config_name")[col].mean().max()
        print(f"  {col:20s}: best={best_value:.3f} ({best_config})")

    print("\nIn-Range AUC at timepoints (higher = better):")
    print("-" * 40)
    for col in available_auc if available_auc else []:
        valid = successful[successful[col].notna()]
        if len(valid) > 0:
            best_config = valid.groupby("config_name")[col].mean().idxmax()
            best_value = valid.groupby("config_name")[col].mean().max()
            print(f"  {col:20s}: best={best_value:.3f} ({best_config})")

    figures_dir = Path(output_dir).parent / "figures"
    print(f"\nPlots saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study for CLAE penalization layers."
    )
    parser.add_argument(
        "--data",
        choices=["2018", "2020", "combined"],
        default="combined",
        help="Dataset to use: combined train (default), 2018, or 2020"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Custom path to meal windows directory (overrides --data)"
    )
    parser.add_argument(
        "--plot_only",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Skip training and only generate plots from existing results CSV"
    )
    args = parser.parse_args()

    # If --plot_only is specified, just generate plots from existing results
    if args.plot_only:
        print("\n" + "="*60)
        print("GENERATING PLOTS FROM EXISTING RESULTS")
        print("="*60)
        print(f"Loading: {args.plot_only}")

        results = pd.read_csv(args.plot_only)
        output_dir = Path(args.plot_only).parent

        summary = plot_ablation_results(results, output_dir)
        plot_auc_results(results, output_dir)

        figures_dir = Path(output_dir).parent / "figures"
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)
        if summary is not None:
            print(summary)
        print(f"\nResults in: {output_dir}")
        print(f"Figures saved to: {figures_dir}")
        exit(0)

    meal_windows_dir = get_meal_windows_dir(args)

    print("\n" + "="*60)
    print("PENALIZATION LAYER ABLATION STUDY")
    print("="*60)
    print(f"Data directory: {meal_windows_dir}")

    # Load data
    print("\nLoading data...")
    try:
        loaded = load_windows(
            csv_dir=str(meal_windows_dir),
            features=["glucose", "steps", "basal", "meal", "heart", "bolus"],
            treat="meal",
            interval_min=5,
            pre_minutes=120,
            post_X_minutes=60,
            post_total_minutes=240,
            standardize=True,
            return_in_range=True,  # Load in_range for AUC calculation
        )
        # Unpack with or without in_range_seq
        if len(loaded) == 14:
            (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
             global_window_id, meal_list, subj_list, pre_ints, post_X_ints, in_range_seq) = loaded
            print(f"Loaded {X_ts_pre.shape[0]} windows with in_range data")
        else:
            (X_ts, X_ts_pre, meal_ohe, subj_ohe, Z, Z_bin, y_seq, mediator_scalar,
             global_window_id, meal_list, subj_list, pre_ints, post_X_ints) = loaded
            in_range_seq = None
            print(f"Loaded {X_ts_pre.shape[0]} windows (no in_range column found)")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure {meal_windows_dir} exists with CSV files")
        exit(1)

    # Run ablation
    results = run_ablation_study(X_ts_pre, meal_ohe, subj_ohe, Z, mediator_scalar, y_seq,
                                  in_range_seq=in_range_seq)

    # Save results
    output_dir = CONFIG.EXPERIMENT_RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / "ablation_results.csv", index=False)
    summary = plot_ablation_results(results, output_dir)

    # Generate AUC-focused plots
    plot_auc_results(results, output_dir)

    figures_dir = output_dir.parent / "figures"
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    if summary is not None:
        print(summary)
    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")
