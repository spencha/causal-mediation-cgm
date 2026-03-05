#!/usr/bin/env python3
"""
generate_architecture_comparisons.py
=====================================
Generate publication-quality figures for autoencoder experiments
using the combined 2018+2020 dataset.

Visualizes results from:
- Architecture comparison (CNN vs LSTM)
- Penalization ablation studies

All experiments train on the full combined dataset (100% of both 2018
and 2020 data).  The incremental data augmentation sweep (varying
pct_2018) has been removed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import argparse

# Add parent paths for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ae_python_code"))

from config import CONFIG

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'font.family': 'sans-serif',
})

COLORS = {
    'lstm': '#2196F3',  # Blue
    'cnn': '#FF9800',   # Orange
    'lin_bal': '#4CAF50',  # Green
    'none': '#9E9E9E',  # Gray
}

# Output directories live under visualizations/ (separate from code)
FIGURES_DIR = PROJECT_ROOT / "visualizations" / "incremental_data_experiment" / "figures"
TABLES_DIR = PROJECT_ROOT / "visualizations" / "incremental_data_experiment" / "tables"


def _resolve_col(df, candidates, label="column"):
    """Return the first column name from candidates that exists in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_experiment_results():
    """Find all experiment result files."""
    results = {}

    # Look in experiment results directory
    results_dir = CONFIG.EXPERIMENT_RESULTS_DIR
    if results_dir.exists():
        # Architecture comparison
        arch_files = list(results_dir.glob("comprehensive_comparison_*.csv"))
        if arch_files:
            arch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            results['architecture'] = pd.read_csv(arch_files[0])
            print(f"Found architecture comparison: {arch_files[0]}")

        # Penalization ablation
        ablation_files = list(results_dir.glob("ablation_results*.csv"))
        if ablation_files:
            ablation_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            results['ablation'] = pd.read_csv(ablation_files[0])
            print(f"Found ablation results: {ablation_files[0]}")

    return results


def plot_comprehensive_comparison(df):
    """
    Plot comprehensive comparison results: architecture × optimizer × penalty.

    Generates:
    - Fig 1: R² and Balance heatmaps for Architecture × Optimizer
    - Fig 2: R² and Balance heatmaps for Architecture × Penalty
    - Fig 3: R² vs Balance scatter colored by architecture, shaped by optimizer
    - Fig 4: Top 10 configs ranked by balance then R²
    """
    if df is None or len(df) == 0:
        print("No architecture comparison results")
        return

    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()

    if len(df) == 0:
        print("No successful architecture runs")
        return

    r2_col = _resolve_col(df, ['test_outcome_r2', 'outcome_r2_mean', 'outcome_R2'])
    balance_col = _resolve_col(df, ['test_balance_score', 'balance_score'])
    opt_col = _resolve_col(df, ['optimizer'])
    pen_col = _resolve_col(df, ['penalty_config', 'penalty', 'penalization'])

    if not r2_col or 'architecture' not in df.columns:
        print("Missing required columns for comprehensive comparison")
        return

    # --- Fig 1: Architecture × Optimizer heatmaps ---
    if opt_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # R² heatmap
        pivot_r2 = df.groupby(['architecture', opt_col])[r2_col].mean().unstack()
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
                    linewidths=0.5, cbar_kws={'label': 'Outcome R²'})
        axes[0].set_title('A. Outcome R² by Architecture × Optimizer', fontweight='bold')
        axes[0].set_ylabel('Architecture')
        axes[0].set_xlabel('Optimizer')

        # Balance heatmap
        if balance_col:
            pivot_bal = df.groupby(['architecture', opt_col])[balance_col].mean().unstack()
            sns.heatmap(pivot_bal, annot=True, fmt='.3f', cmap='YlGn', ax=axes[1],
                        linewidths=0.5, cbar_kws={'label': 'Balance Score'})
            axes[1].set_title('B. Balance Score by Architecture × Optimizer', fontweight='bold')
            axes[1].set_ylabel('Architecture')
            axes[1].set_xlabel('Optimizer')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fig_arch_optimizer_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'fig_arch_optimizer_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'fig_arch_optimizer_heatmap.png'}")

    # --- Fig 2: Architecture × Penalty heatmaps ---
    if pen_col:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pivot_r2 = df.groupby(['architecture', pen_col])[r2_col].mean().unstack()
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
                    linewidths=0.5, cbar_kws={'label': 'Outcome R²'})
        axes[0].set_title('A. Outcome R² by Architecture × Penalty', fontweight='bold')
        axes[0].set_ylabel('Architecture')
        axes[0].set_xlabel('Penalty Config')
        axes[0].tick_params(axis='x', rotation=45)

        if balance_col:
            pivot_bal = df.groupby(['architecture', pen_col])[balance_col].mean().unstack()
            sns.heatmap(pivot_bal, annot=True, fmt='.3f', cmap='YlGn', ax=axes[1],
                        linewidths=0.5, cbar_kws={'label': 'Balance Score'})
            axes[1].set_title('B. Balance Score by Architecture × Penalty', fontweight='bold')
            axes[1].set_ylabel('Architecture')
            axes[1].set_xlabel('Penalty Config')
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fig_arch_penalty_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'fig_arch_penalty_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'fig_arch_penalty_heatmap.png'}")

    # --- Fig 3: R² vs Balance scatter ---
    if balance_col and opt_col and pen_col:
        # Average over seeds
        group_cols = ['architecture', opt_col, pen_col]
        avg = df.groupby(group_cols).agg({r2_col: 'mean', balance_col: 'mean'}).reset_index()

        fig, ax = plt.subplots(figsize=(10, 7))
        markers = {'adam': 'o', 'adamw': 's', 'sgd': '^', 'rmsprop': 'D', 'nadam': 'v'}
        arch_colors = {'cnn': '#FF9800', 'lstm': '#2196F3'}

        for _, row in avg.iterrows():
            ax.scatter(row[balance_col], row[r2_col],
                      c=arch_colors.get(row['architecture'], 'gray'),
                      marker=markers.get(row[opt_col], 'o'),
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Legend for architectures
        for arch, color in arch_colors.items():
            ax.scatter([], [], c=color, s=80, label=arch.upper(), edgecolors='black', linewidth=0.5)
        # Legend for optimizers
        for opt, marker in markers.items():
            ax.scatter([], [], c='gray', marker=marker, s=80, label=opt, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Balance Score (higher = better)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Outcome R²', fontsize=12, fontweight='bold')
        ax.set_title('R² vs Balance: All Configurations', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fig_r2_vs_balance_scatter.png', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'fig_r2_vs_balance_scatter.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'fig_r2_vs_balance_scatter.png'}")

    # --- Fig 4: Top configs bar chart ---
    if balance_col and opt_col and pen_col:
        group_cols = ['architecture', opt_col, pen_col]
        avg = df.groupby(group_cols).agg({r2_col: 'mean', balance_col: 'mean'}).reset_index()
        avg = avg.sort_values([balance_col, r2_col], ascending=[False, False]).head(10)
        avg['label'] = avg.apply(
            lambda r: f"{r['architecture'].upper()} / {r[opt_col]} / {r[pen_col]}", axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Balance bars
        ax = axes[0]
        colors = ['#4CAF50' if b > 0.9 else '#FF9800' if b > 0.7 else '#F44336'
                  for b in avg[balance_col]]
        ax.barh(range(len(avg)), avg[balance_col].values, color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(avg)))
        ax.set_yticklabels(avg['label'].values, fontsize=9)
        ax.set_xlabel('Balance Score')
        ax.set_title('A. Top 10 by Balance Score', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        # R² bars for same configs
        ax = axes[1]
        ax.barh(range(len(avg)), avg[r2_col].values, color='steelblue',
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(avg)))
        ax.set_yticklabels(avg['label'].values, fontsize=9)
        ax.set_xlabel('Outcome R²')
        ax.set_title('B. Corresponding R² for Top 10', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fig_top_configs.png', dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / 'fig_top_configs.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {FIGURES_DIR / 'fig_top_configs.png'}")


def plot_penalization_ablation(df):
    """
    Plot penalization layer ablation results.
    """
    if df is None or len(df) == 0:
        print("No penalization ablation results")
        return

    # Filter for successful runs
    if 'status' in df.columns:
        df = df[df['status'] == 'success'].copy()

    if len(df) == 0:
        print("No successful ablation runs")
        return

    pen_col = _resolve_col(df, ['config_name', 'penalty_config', 'penalty', 'penalization', 'penalty_type'])
    r2_col = _resolve_col(df, ['test_outcome_r2', 'outcome_r2_mean', 'outcome_R2'])
    balance_col = _resolve_col(df, ['test_balance_score', 'balance_score'])

    if pen_col is None:
        print("No penalization column found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Outcome R² by penalization
    ax = axes[0]
    if r2_col:
        grouped = df.groupby(pen_col)[r2_col].agg(['mean', 'std'])
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'],
               color=[COLORS.get(p, 'gray') for p in grouped.index],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in grouped.index], rotation=45, ha='right')
        ax.set_ylabel('Outcome R²')
        ax.set_title('A. Outcome by Penalization Type')
        ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Balance Score by penalization
    ax = axes[1]
    if balance_col:
        grouped = df.groupby(pen_col)[balance_col].agg(['mean', 'std'])
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'],
               color=[COLORS.get(p, 'gray') for p in grouped.index],
               capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in grouped.index], rotation=45, ha='right')
        ax.set_ylabel('Balance Score')
        ax.set_title('B. Balance by Penalization Type')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig_penalization_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'fig_penalization_ablation.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'fig_penalization_ablation.png'}")


def generate_experiment_summary_table(results):
    """Generate detailed summary tables of all experiment results."""

    # --- Comprehensive comparison detailed table ---
    if 'architecture' in results:
        df = results['architecture']
        if 'status' in df.columns:
            df = df[df['status'] == 'success']

        r2_col = _resolve_col(df, ['test_outcome_r2', 'outcome_r2_mean', 'outcome_R2'])
        balance_col = _resolve_col(df, ['test_balance_score', 'balance_score'])
        opt_col = _resolve_col(df, ['optimizer'])
        pen_col = _resolve_col(df, ['penalty_config', 'penalty', 'penalization'])

        if r2_col and len(df) > 0:
            # Full breakdown: arch × optimizer × penalty, averaged over seeds
            group_cols = [c for c in ['architecture', opt_col, pen_col] if c]
            agg_dict = {r2_col: ['mean', 'std']}
            if balance_col:
                agg_dict[balance_col] = ['mean', 'std']

            full_table = df.groupby(group_cols).agg(agg_dict).round(4)
            full_table.columns = ['_'.join(c).strip('_') for c in full_table.columns]
            full_table = full_table.reset_index()

            # Sort by balance (desc), then R² (desc)
            bal_mean_col = f'{balance_col}_mean' if balance_col else None
            r2_mean_col = f'{r2_col}_mean'
            if bal_mean_col and bal_mean_col in full_table.columns:
                full_table = full_table.sort_values(
                    [bal_mean_col, r2_mean_col], ascending=[False, False])

            full_table.to_csv(TABLES_DIR / 'comprehensive_full_breakdown.csv', index=False)
            print(f"Saved: {TABLES_DIR / 'comprehensive_full_breakdown.csv'}")

            # Top 10 configs
            top10 = full_table.head(10)
            top10.to_csv(TABLES_DIR / 'comprehensive_top10.csv', index=False)
            print(f"Saved: {TABLES_DIR / 'comprehensive_top10.csv'}")


def _fmt_est(val, decimals=3):
    """Format a point estimate."""
    return f"{val:.{decimals}f}"


def _fmt_pm(mean, std, decimals=3):
    r"""Format mean \pm std for LaTeX."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


# Mapping from internal config/penalty names to publication-quality labels.
_DISPLAY_NAMES = {
    # Penalty configurations (comprehensive comparison)
    "none":                "None",
    "linear_only":         "Linear only",
    "balance_only":        "Balance only",
    "linear+balance":      "Linear + Balance",
    "linear+balance+ci":   "Linear + Balance + CI",
    "all_penalties":       "All penalties",
    # Ablation config names (run_incremental_data_experiment)
    "lin_bal_ci":          "Linear + Balance + CI",
    "lin_bal_stab":        "Linear + Balance + Stability",
    "lin_bal":             "Linear + Balance",
    "bal_stab":            "Balance + Stability",
    # Marginal penalty-layer flags
    "linearization":       "Linearization",
    "balancing":           "Balancing",
    "ci_penalty":          "CI penalty",
    "stability":           "Stability",
}


def _display_name(raw: str) -> str:
    """Return a publication-quality display name for a config/penalty string."""
    return _DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def generate_comprehensive_comparison_latex(df):
    """Generate LaTeX table for architecture × optimizer × penalty comparison.

    Produces table_comprehensive_top10.tex in TABLES_DIR.
    """
    if df is None or len(df) == 0:
        return

    if 'status' in df.columns:
        df = df[df['status'] == 'success']

    r2_col = _resolve_col(df, ['test_outcome_r2', 'outcome_r2_mean', 'outcome_R2'])
    balance_col = _resolve_col(df, ['test_balance_score', 'balance_score'])
    opt_col = _resolve_col(df, ['optimizer'])
    pen_col = _resolve_col(df, ['penalty_config', 'penalty', 'penalization'])

    if not r2_col or 'architecture' not in df.columns:
        print("Missing columns for comprehensive LaTeX table")
        return

    group_cols = [c for c in ['architecture', opt_col, pen_col] if c]
    agg_dict = {r2_col: ['mean', 'std']}
    if balance_col:
        agg_dict[balance_col] = ['mean', 'std']

    summary = df.groupby(group_cols).agg(agg_dict).round(4)
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary = summary.reset_index()

    bal_mean = f'{balance_col}_mean' if balance_col else None
    r2_mean = f'{r2_col}_mean'
    r2_std = f'{r2_col}_std'
    bal_std = f'{balance_col}_std' if balance_col else None

    if bal_mean and bal_mean in summary.columns:
        summary = summary.sort_values([bal_mean, r2_mean], ascending=[False, False])

    n_seeds = int(df.groupby(group_cols).size().median())
    n_total = len(summary)
    top = summary.head(10)

    # Build rows
    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        arch = row['architecture'].upper()
        opt = row[opt_col] if opt_col else '--'
        pen = _display_name(row[pen_col]) if pen_col else '--'
        r2_str = _fmt_pm(row[r2_mean], row[r2_std])
        if bal_mean:
            bal_str = _fmt_pm(row[bal_mean], row[bal_std])
        else:
            bal_str = '--'

        rows.append(f"{rank} & {arch} & {opt} & {pen} & {r2_str} & {bal_str} \\\\")

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Top 10 autoencoder configurations ranked by balance score then outcome $R^2$ (""" + str(n_total) + r""" total configurations, """ + str(n_seeds) + r""" seeds each).}
Each configuration is a combination of encoder architecture, optimizer, and penalty regime.
Balance score $= 1 - |0.5 - \text{AUC}_{\text{treatment}}| \times 2$ (1.0 = ideal).
Training on combined 2018+2020 data.}
\label{tab:comprehensive_top10}
\resizebox{\textwidth}{!}{%
\begin{tabular}{clllcc}
\toprule
Rank & Architecture & Optimizer & Penalty & Outcome $R^2$ & Balance Score \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    save_path = TABLES_DIR / 'table_comprehensive_top10.tex'
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {save_path}")

    # --- Full breakdown by architecture × optimizer (marginal) ---
    if opt_col and balance_col:
        agg2 = df.groupby(['architecture', opt_col]).agg(
            {r2_col: ['mean', 'std'], balance_col: ['mean', 'std']}
        ).round(4)
        agg2.columns = ['r2_mean', 'r2_std', 'bal_mean', 'bal_std']
        agg2 = agg2.reset_index().sort_values(['bal_mean', 'r2_mean'], ascending=[False, False])

        rows2 = []
        for _, row in agg2.iterrows():
            rows2.append(
                f"{row['architecture'].upper()} & {row[opt_col]} & "
                f"{_fmt_pm(row['r2_mean'], row['r2_std'])} & "
                f"{_fmt_pm(row['bal_mean'], row['bal_std'])} \\\\"
            )

        latex2 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Architecture $\times$ optimizer comparison (averaged over penalties and seeds).}
Sorted by balance score, then $R^2$.}
\label{tab:arch_optimizer_summary}
\begin{tabular}{llcc}
\toprule
Architecture & Optimizer & Outcome $R^2$ & Balance Score \\
\midrule
""" + "\n".join(rows2) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        save_path2 = TABLES_DIR / 'table_arch_optimizer_summary.tex'
        with open(save_path2, 'w') as f:
            f.write(latex2)
        print(f"Saved: {save_path2}")


def generate_ablation_latex(df):
    """Generate LaTeX tables for the penalization layer ablation study.

    Produces:
    - table_ablation_top_configs.tex   (top 10 penalty combinations)
    - table_ablation_marginal.tex      (marginal contribution of each penalty)
    """
    if df is None or len(df) == 0:
        return

    if 'status' in df.columns:
        df = df[df['status'] == 'success']

    if len(df) == 0:
        return

    pen_col = _resolve_col(df, ['config_name', 'penalty_config', 'penalty'])
    r2_col = _resolve_col(df, ['test_outcome_r2', 'outcome_r2_mean'])
    balance_col = _resolve_col(df, ['test_balance_score', 'balance_score'])
    mediator_col = _resolve_col(df, ['mediator_r2'])
    treat_auc_col = _resolve_col(df, ['treatment_auc'])

    if not pen_col or not r2_col:
        print("Missing columns for ablation LaTeX table")
        return

    # --- Table 1: Top configurations ---
    agg_dict = {r2_col: ['mean', 'std']}
    if balance_col:
        agg_dict[balance_col] = ['mean', 'std']
    if mediator_col:
        agg_dict[mediator_col] = ['mean', 'std']
    if treat_auc_col:
        agg_dict[treat_auc_col] = ['mean', 'std']

    summary = df.groupby(pen_col).agg(agg_dict).round(4)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()

    # Sort by balance then R²
    bal_m = f'{balance_col}_mean' if balance_col else None
    r2_m = f'{r2_col}_mean'
    if bal_m and bal_m in summary.columns:
        summary = summary.sort_values([bal_m, r2_m], ascending=[False, False])
    else:
        summary = summary.sort_values(r2_m, ascending=False)

    n_seeds = int(df.groupby(pen_col).size().median())
    top = summary.head(10)

    # Determine which columns to show
    has_bal = bal_m and bal_m in top.columns
    has_med = mediator_col and f'{mediator_col}_mean' in top.columns
    has_auc = treat_auc_col and f'{treat_auc_col}_mean' in top.columns

    # Build column spec
    col_headers = ['Rank', 'Configuration', 'Outcome $R^2$']
    col_spec = 'cllc'  # extra l for config name (can be long)
    n_cols = 3
    if has_bal:
        col_headers.append('Balance Score')
        n_cols += 1
    if has_med:
        col_headers.append('Mediator $R^2$')
        n_cols += 1
    if has_auc:
        col_headers.append('Treatment AUC')
        n_cols += 1

    # Adjust col_spec: rank=c, config=l, rest=c
    col_spec = 'cl' + 'c' * (n_cols - 2)

    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        config = _display_name(row[pen_col]) if isinstance(row[pen_col], str) else str(row[pen_col])
        cells = [str(rank), config,
                 _fmt_pm(row[f'{r2_col}_mean'], row[f'{r2_col}_std'])]
        if has_bal:
            cells.append(_fmt_pm(row[bal_m], row[f'{balance_col}_std']))
        if has_med:
            cells.append(_fmt_pm(row[f'{mediator_col}_mean'], row[f'{mediator_col}_std']))
        if has_auc:
            cells.append(_fmt_pm(row[f'{treat_auc_col}_mean'], row[f'{treat_auc_col}_std']))

        rows.append(' & '.join(cells) + ' \\\\')

    header_line = ' & '.join(col_headers) + ' \\\\'

    latex_top = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Top 10 penalization configurations from the ablation study (""" + str(n_seeds) + r""" seeds each).}
All $2^4 = 16$ combinations of the four penalty layers are evaluated.
Architecture: CNN; optimizer: rmsprop.
Training on combined 2018+2020 data.
Treatment AUC ideal is 0.5 (no predictive power $\Rightarrow$ balanced embedding).}
\label{tab:ablation_top_configs}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
""" + header_line + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    save_path = TABLES_DIR / 'table_ablation_top_configs.tex'
    with open(save_path, 'w') as f:
        f.write(latex_top)
    print(f"Saved: {save_path}")

    # --- Table 2: Marginal contributions of each penalty layer ---
    penalty_flags = ['linearization', 'balancing', 'ci_penalty', 'stability']
    available_flags = [f for f in penalty_flags if f in df.columns]

    if not available_flags:
        return

    metric_cols = []
    metric_labels = []
    if r2_col:
        metric_cols.append(r2_col)
        metric_labels.append('$\\Delta R^2$')
    if balance_col:
        metric_cols.append(balance_col)
        metric_labels.append('$\\Delta$ Balance')
    if mediator_col:
        metric_cols.append(mediator_col)
        metric_labels.append('$\\Delta$ Mediator $R^2$')

    marg_rows = []
    for flag in available_flags:
        with_flag = df[df[flag] == True]
        without_flag = df[df[flag] == False]

        cells = [_display_name(flag)]
        for mc in metric_cols:
            delta = with_flag[mc].mean() - without_flag[mc].mean()
            sign = '+' if delta >= 0 else ''
            cells.append(f"{sign}{delta:.4f}")

        marg_rows.append(' & '.join(cells) + ' \\\\')

    marg_header = 'Penalty Layer & ' + ' & '.join(metric_labels) + ' \\\\'
    marg_col_spec = 'l' + 'c' * len(metric_cols)

    latex_marg = r"""\begin{table}[ht]
\centering
\caption{\textbf{Marginal contribution of each penalty layer.}
$\Delta$ = mean metric with penalty enabled $-$ mean metric with penalty disabled,
averaged across all other penalty combinations and seeds.
Positive $\Delta R^2$ and $\Delta$ Balance are desirable.}
\label{tab:ablation_marginal}
\begin{tabular}{""" + marg_col_spec + r"""}
\toprule
""" + marg_header + r"""
\midrule
""" + "\n".join(marg_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    save_path2 = TABLES_DIR / 'table_ablation_marginal.tex'
    with open(save_path2, 'w') as f:
        f.write(latex_marg)
    print(f"Saved: {save_path2}")


def print_summary_to_console(results):
    """Print experiment summary to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY (combined 2018+2020 data)")
    print("=" * 60)

    if 'architecture' in results:
        df = results['architecture']
        print(f"\nArchitecture Comparison: {len(df)} runs")

    if 'ablation' in results:
        df = results['ablation']
        print(f"Penalization Ablation: {len(df)} runs")

    print("\nAll experiments use 100% of combined 2018+2020 dataset.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory containing experiment results')
    args = parser.parse_args()

    # Ensure output directories exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("EXPERIMENT VISUALIZATION")
    print("=" * 60)

    # Find and load results
    results = find_experiment_results()

    if not results:
        print("\nNo experiment results found.")
        print("Expected location: ", CONFIG.EXPERIMENT_RESULTS_DIR)
        return

    # Print summary
    print_summary_to_console(results)

    # Generate visualizations
    print("\nGenerating figures...")
    if 'architecture' in results:
        plot_comprehensive_comparison(results['architecture'])
    if 'ablation' in results:
        plot_penalization_ablation(results['ablation'])

    # Generate tables (CSV + LaTeX)
    print("\nGenerating tables...")
    generate_experiment_summary_table(results)

    print("\nGenerating LaTeX tables...")
    if 'architecture' in results:
        generate_comprehensive_comparison_latex(results['architecture'])
    if 'ablation' in results:
        generate_ablation_latex(results['ablation'])

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
