#!/usr/bin/env python3
"""
visualize_comprehensive_comparison.py
=====================================
Generate publication-quality visualizations comparing:
1. LSTM vs CNN architectures
2. Different optimizers
3. Penalization layer contributions
4. Performance across datasets (2018, 2020, combined)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

COLORS = {
    'lstm': '#2196F3',
    'cnn': '#FF9800',
    '2018': '#4CAF50',
    '2020': '#9C27B0',
    'combined': '#F44336',
    'adam': '#1E88E5',
    'adamw': '#43A047',
    'sgd': '#E53935',
    'rmsprop': '#8E24AA',
    'nadam': '#FB8C00'
}


def load_comparison_results(results_dir):
    """Load the most recent comparison results"""
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob("comprehensive_comparison_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No comparison results found in {results_dir}")

    # Get most recent
    latest = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest}")

    return pd.read_csv(latest)


def plot_architecture_comparison(df, output_dir):
    """
    Figure 1: LSTM vs CNN comparison across datasets
    """
    successful = df[df["status"] == "success"]

    if len(successful) == 0:
        print("No successful runs for architecture comparison")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Outcome R² by architecture and dataset
    ax = axes[0]
    for i, arch in enumerate(["lstm", "cnn"]):
        arch_data = successful[successful["architecture"] == arch]
        if len(arch_data) == 0:
            continue
        means = arch_data.groupby("dataset")["outcome_r2_mean"].mean()
        stds = arch_data.groupby("dataset")["outcome_r2_mean"].std()

        x = np.arange(len(means))
        width = 0.35
        offset = -width/2 if arch == "lstm" else width/2

        ax.bar(x + offset, means, width, yerr=stds,
               label=arch.upper(), color=COLORS.get(arch, "gray"),
               capsize=4, alpha=0.8)

    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Outcome R²")
    ax.set_title("Outcome Prediction by Architecture")
    ax.legend()
    ax.set_ylim(0, 1)

    # Plot 2: Training time comparison
    ax = axes[1]
    if "training_time_sec" in successful.columns:
        arch_time = successful.groupby(["dataset", "architecture"])["training_time_sec"].mean().unstack()
        if len(arch_time.columns) > 0:
            arch_time.plot(kind="bar", ax=ax,
                          color=[COLORS.get(c, "gray") for c in arch_time.columns], alpha=0.8)
            ax.set_ylabel("Training Time (seconds)")
            ax.set_title("Training Efficiency")
            ax.legend(title="Architecture")
            ax.tick_params(axis='x', rotation=45)

    # Plot 3: Balance score comparison
    ax = axes[2]
    for i, arch in enumerate(["lstm", "cnn"]):
        arch_data = successful[successful["architecture"] == arch]
        if len(arch_data) == 0:
            continue
        means = arch_data.groupby("dataset")["balance_score"].mean()
        stds = arch_data.groupby("dataset")["balance_score"].std()

        x = np.arange(len(means))
        width = 0.35
        offset = -width/2 if arch == "lstm" else width/2

        ax.bar(x + offset, means, width, yerr=stds,
               label=arch.upper(), color=COLORS.get(arch, "gray"), capsize=4, alpha=0.8)

    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index)
    ax.set_ylabel("Balance Score")
    ax.set_title("Treatment Balance by Architecture")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_architecture_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig1_architecture_comparison.pdf", bbox_inches="tight")
    plt.close()

    print("Saved: fig1_architecture_comparison.png/pdf")


def plot_optimizer_comparison(df, output_dir):
    """
    Figure 2: Optimizer comparison across architectures
    """
    successful = df[df["status"] == "success"]

    if len(successful) == 0:
        print("No successful runs for optimizer comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    optimizers = ["adam", "adamw", "sgd", "rmsprop", "nadam"]
    available_opts = [o for o in optimizers if o in successful["optimizer"].values]

    # Plot 1: Outcome R² by optimizer (aggregated)
    ax = axes[0, 0]
    opt_summary = successful.groupby("optimizer")["outcome_r2_mean"].agg(["mean", "std"])
    opt_summary = opt_summary.reindex([o for o in optimizers if o in opt_summary.index])

    if len(opt_summary) > 0:
        colors = [COLORS.get(opt, "gray") for opt in opt_summary.index]
        ax.bar(opt_summary.index, opt_summary["mean"], yerr=opt_summary["std"],
               color=colors, capsize=4, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Outcome R²")
        ax.set_title("Outcome Prediction by Optimizer")
        ax.tick_params(axis='x', rotation=45)

    # Plot 2: Min validation loss by optimizer
    ax = axes[0, 1]
    if "min_val_loss" in successful.columns:
        loss_summary = successful.groupby("optimizer")["min_val_loss"].agg(["mean", "std"])
        loss_summary = loss_summary.reindex([o for o in optimizers if o in loss_summary.index])

        if len(loss_summary) > 0:
            colors = [COLORS.get(opt, "gray") for opt in loss_summary.index]
            ax.bar(loss_summary.index, loss_summary["mean"], yerr=loss_summary["std"],
                   color=colors, capsize=4, alpha=0.8, edgecolor="black")
            ax.set_ylabel("Minimum Validation Loss")
            ax.set_title("Convergence Quality by Optimizer")
            ax.tick_params(axis='x', rotation=45)

    # Plot 3: Optimizer performance by architecture
    ax = axes[1, 0]
    for i, arch in enumerate(["lstm", "cnn"]):
        arch_data = successful[successful["architecture"] == arch]
        if len(arch_data) == 0:
            continue
        means = arch_data.groupby("optimizer")["outcome_r2_mean"].mean()
        means = means.reindex([o for o in optimizers if o in means.index])

        if len(means) > 0:
            x = np.arange(len(means))
            width = 0.35
            offset = -width/2 if arch == "lstm" else width/2

            ax.bar(x + offset, means, width, label=arch.upper(),
                   color=COLORS.get(arch, "gray"), alpha=0.8)

    ax.set_xticks(np.arange(len(available_opts)))
    ax.set_xticklabels(available_opts, rotation=45)
    ax.set_ylabel("Outcome R²")
    ax.set_title("Optimizer × Architecture Interaction")
    ax.legend()

    # Plot 4: Heatmap of optimizer × dataset
    ax = axes[1, 1]
    pivot = successful.pivot_table(
        values="outcome_r2_mean",
        index="optimizer",
        columns="dataset",
        aggfunc="mean"
    )
    if len(pivot) > 0:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
                    vmin=0, vmax=max(0.1, pivot.max().max() * 1.1),
                    cbar_kws={"label": "Outcome R²"})
        ax.set_title("Optimizer × Dataset Performance")

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_optimizer_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig2_optimizer_comparison.pdf", bbox_inches="tight")
    plt.close()

    print("Saved: fig2_optimizer_comparison.png/pdf")


def plot_penalization_comparison(df, output_dir):
    """
    Figure 3: Penalization layer contributions
    """
    successful = df[df["status"] == "success"]

    if len(successful) == 0:
        print("No successful runs for penalization comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Performance by penalty configuration
    ax = axes[0, 0]
    penalty_summary = successful.groupby("penalty_config")["outcome_r2_mean"].agg(["mean", "std"]).reset_index()
    penalty_summary.columns = ["config", "mean", "std"]
    penalty_summary = penalty_summary.sort_values("mean", ascending=True)

    if len(penalty_summary) > 0:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(penalty_summary)))
        ax.barh(penalty_summary["config"], penalty_summary["mean"],
                xerr=penalty_summary["std"], color=colors, capsize=3, alpha=0.8)
        ax.set_xlabel("Outcome R²")
        ax.set_title("Performance by Penalty Configuration")

    # Plot 2: Marginal contribution of each penalty
    ax = axes[0, 1]
    penalties = ["use_linearization", "use_balancing", "use_ci_penalty", "use_stability"]
    penalty_labels = ["Linearization", "Balancing", "CI Penalty", "Stability"]

    available_penalties = [p for p in penalties if p in successful.columns]

    contributions = []
    for penalty in available_penalties:
        with_penalty = successful[successful[penalty] == True]["outcome_r2_mean"].mean()
        without_penalty = successful[successful[penalty] == False]["outcome_r2_mean"].mean()
        contributions.append(with_penalty - without_penalty)

    if len(contributions) > 0:
        colors = ["green" if c > 0 else "red" for c in contributions]
        labels = [penalty_labels[penalties.index(p)] for p in available_penalties]
        bars = ax.bar(labels, contributions, color=colors, alpha=0.7, edgecolor="black")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Δ Outcome R²")
        ax.set_title("Marginal Contribution of Each Penalty")
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, val in zip(bars, contributions):
            height = bar.get_height()
            ax.annotate(f'{val:+.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -12),
                        textcoords="offset points",
                        ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Penalty effects by architecture
    ax = axes[1, 0]
    for i, arch in enumerate(["lstm", "cnn"]):
        arch_data = successful[successful["architecture"] == arch]
        if len(arch_data) == 0:
            continue

        contributions_arch = []
        for penalty in available_penalties:
            with_p = arch_data[arch_data[penalty] == True]["outcome_r2_mean"].mean()
            without_p = arch_data[arch_data[penalty] == False]["outcome_r2_mean"].mean()
            contributions_arch.append(with_p - without_p)

        if len(contributions_arch) > 0:
            x = np.arange(len(available_penalties))
            width = 0.35
            offset = -width/2 if arch == "lstm" else width/2

            ax.bar(x + offset, contributions_arch, width, label=arch.upper(),
                   color=COLORS.get(arch, "gray"), alpha=0.8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(available_penalties)))
    labels = [penalty_labels[penalties.index(p)] for p in available_penalties]
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Δ Outcome R²")
    ax.set_title("Penalty Effects by Architecture")
    ax.legend()

    # Plot 4: Penalty effects on balance score
    ax = axes[1, 1]
    balance_contributions = []
    for penalty in available_penalties:
        with_penalty = successful[successful[penalty] == True]["balance_score"].mean()
        without_penalty = successful[successful[penalty] == False]["balance_score"].mean()
        balance_contributions.append(with_penalty - without_penalty)

    if len(balance_contributions) > 0:
        colors = ["green" if c > 0 else "red" for c in balance_contributions]
        labels = [penalty_labels[penalties.index(p)] for p in available_penalties]
        bars = ax.bar(labels, balance_contributions, color=colors, alpha=0.7, edgecolor="black")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Δ Balance Score")
        ax.set_title("Penalty Effects on Treatment Balance")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_penalization_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig3_penalization_comparison.pdf", bbox_inches="tight")
    plt.close()

    print("Saved: fig3_penalization_comparison.png/pdf")


def plot_dataset_comparison(df, output_dir):
    """
    Figure 4: Performance comparison across datasets
    """
    successful = df[df["status"] == "success"]

    if len(successful) == 0:
        print("No successful runs for dataset comparison")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = ["2018", "2020", "combined"]
    available_datasets = [d for d in datasets if d in successful["dataset"].values]

    # Plot 1: Overall performance by dataset
    ax = axes[0]
    dataset_summary = successful.groupby("dataset").agg({
        "outcome_r2_mean": ["mean", "std"],
        "n_samples": "first"
    })
    dataset_summary.columns = ["r2_mean", "r2_std", "n_samples"]
    dataset_summary = dataset_summary.reindex([d for d in datasets if d in dataset_summary.index])

    if len(dataset_summary) > 0:
        colors = [COLORS.get(d, "gray") for d in dataset_summary.index]
        bars = ax.bar(dataset_summary.index, dataset_summary["r2_mean"],
                      yerr=dataset_summary["r2_std"],
                      color=colors, capsize=4, alpha=0.8, edgecolor="black")

        # Add sample size annotations
        for bar, (idx, row) in zip(bars, dataset_summary.iterrows()):
            if not np.isnan(row["n_samples"]):
                ax.annotate(f'n={int(row["n_samples"])}',
                            xy=(bar.get_x() + bar.get_width() / 2, 0.02),
                            ha='center', fontsize=9, color='white', fontweight='bold')

        ax.set_ylabel("Outcome R²")
        ax.set_title("Performance by Dataset")
        ax.set_ylim(0, 1)

    # Plot 2: Best architecture per dataset
    ax = axes[1]
    best_arch = successful.groupby(["dataset", "architecture"])["outcome_r2_mean"].mean().unstack()
    best_arch = best_arch.reindex([d for d in datasets if d in best_arch.index])

    if len(best_arch) > 0:
        best_arch.plot(kind="bar", ax=ax,
                       color=[COLORS.get(c, "gray") for c in best_arch.columns], alpha=0.8)
        ax.set_ylabel("Outcome R²")
        ax.set_title("Architecture Performance by Dataset")
        ax.legend(title="Architecture")
        ax.tick_params(axis='x', rotation=0)

    # Plot 3: Configuration sensitivity by dataset
    ax = axes[2]
    var_by_dataset = successful.groupby("dataset")["outcome_r2_mean"].std()
    var_by_dataset = var_by_dataset.reindex([d for d in datasets if d in var_by_dataset.index])

    if len(var_by_dataset) > 0:
        colors = [COLORS.get(d, "gray") for d in var_by_dataset.index]
        ax.bar(var_by_dataset.index, var_by_dataset, color=colors, alpha=0.8, edgecolor="black")
        ax.set_ylabel("Std Dev of Outcome R²")
        ax.set_title("Configuration Sensitivity by Dataset")

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_dataset_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig4_dataset_comparison.pdf", bbox_inches="tight")
    plt.close()

    print("Saved: fig4_dataset_comparison.png/pdf")


def generate_summary_table(df, output_dir):
    """
    Generate LaTeX-formatted summary tables
    """
    successful = df[df["status"] == "success"]

    if len(successful) == 0:
        print("No successful runs for summary table")
        return None

    # Table 1: Best configuration per dataset
    best_per_dataset = []
    for dataset in successful["dataset"].unique():
        dataset_df = successful[successful["dataset"] == dataset]
        best_idx = dataset_df["outcome_r2_mean"].idxmax()
        best = dataset_df.loc[best_idx]
        best_per_dataset.append({
            "Dataset": dataset,
            "Architecture": best["architecture"].upper(),
            "Optimizer": best["optimizer"],
            "Penalties": best["penalty_config"],
            "Outcome R²": f"{best['outcome_r2_mean']:.4f}",
            "Balance": f"{best['balance_score']:.4f}"
        })

    best_df = pd.DataFrame(best_per_dataset)

    # Save as CSV and LaTeX
    best_df.to_csv(output_dir / "table_best_configurations.csv", index=False)

    latex = best_df.to_latex(index=False, caption="Best Configuration per Dataset",
                              label="tab:best_configs")
    with open(output_dir / "table_best_configurations.tex", "w") as f:
        f.write(latex)

    print("Saved: table_best_configurations.csv/tex")

    return best_df


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE COMPARISON FIGURES")
    print("="*60)

    results_dir = CONFIG.FIGURES_DIR / "comprehensive_comparison"
    output_dir = CONFIG.FIGURES_DIR / "comparison_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    try:
        df = load_comparison_results(results_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run run_comprehensive_ae_comparison.py first to generate results.")
        return

    print(f"\nLoaded {len(df)} configurations")
    print(f"Successful: {(df['status'] == 'success').sum()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Architectures: {df['architecture'].unique()}")

    # Generate all figures
    print("\nGenerating figures...")
    plot_architecture_comparison(df, output_dir)
    plot_optimizer_comparison(df, output_dir)
    plot_penalization_comparison(df, output_dir)
    plot_dataset_comparison(df, output_dir)

    # Generate summary table
    print("\nGenerating summary tables...")
    best_df = generate_summary_table(df, output_dir)

    if best_df is not None:
        print("\n" + "="*60)
        print("BEST CONFIGURATIONS")
        print("="*60)
        print(best_df.to_string(index=False))

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
