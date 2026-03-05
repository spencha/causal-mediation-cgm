#!/usr/bin/env python3
"""
generate_embedding_diagnostics.py
==================================
Comprehensive visualization and diagnostics for autoencoder results.
Creates figures and diagnostics saved to causal_ae/results_visualizations/images/ae_evals
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, LassoCV
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AE_CODE_DIR = PROJECT_ROOT / "ae_python_code"

# Add ae_python_code to path for imports
if str(AE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_CODE_DIR))

from config import CONFIG
CONFIG.ensure_dirs()

OUTPUT_DIR = CONFIG.ANALYSIS_DATA_DIR
# Output directories live under visualizations/ (separate from code)
FIGURES_DIR = PROJECT_ROOT / "visualizations" / "ae_embeddings" / "figures"
TABLES_DIR = PROJECT_ROOT / "visualizations" / "ae_embeddings" / "tables"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Legacy alias for backward compatibility
VIZ_DIR = FIGURES_DIR

# Import modules
import resid_ae_utils as RAE
import causal_linear_ae as CLAE

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})


def format_phi_label(col_name):
    """Format phi column names as proper LaTeX math subscripts using varphi.

    Always uses LaTeX formatting for consistent rendering across matplotlib.
    """
    if col_name.startswith('phi_'):
        subscript = col_name.replace('phi_', '')
        return r'$\varphi_{' + subscript + r'}$'
    elif col_name == 'glucose_at_meal':
        return 'Glucose'
    return col_name


def load_results():
    """Load all available results from the analysis"""
    results = {}

    # Load comparison metrics (optional)
    metrics_path = OUTPUT_DIR / "ae_comparison_metrics.csv"
    if metrics_path.exists():
        results['metrics'] = pd.read_csv(metrics_path)
        print(f"Loaded metrics: {metrics_path.name}")

    # Embeddings are in: cma_cluster/analysis_data/embeddings/
    # File patterns:
    #   phi_embeddings_combined_{arch}_{pct}pct_{penalty}_seed{seed}.csv
    #   phi_embeddings_train_{arch}_{pct}pct_{penalty}_seed{seed}.csv
    #   phi_embeddings_test_{arch}_{pct}pct_{penalty}_seed{seed}.csv
    embeddings_dir = OUTPUT_DIR / 'embeddings'

    if not embeddings_dir.exists():
        print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
        return results

    # Load combined embeddings
    combined_files = sorted(embeddings_dir.glob('phi_embeddings_combined_*.csv'), reverse=True)
    if combined_files:
        combined_path = combined_files[0]
        results['phi_combined'] = pd.read_csv(combined_path)
        results['phi_main'] = results['phi_combined']
        print(f"Loaded combined phi: {combined_path.name}")

    # Load train embeddings
    train_files = sorted(embeddings_dir.glob('phi_embeddings_train_*.csv'), reverse=True)
    if train_files:
        results['phi_train'] = pd.read_csv(train_files[0])
        print(f"Loaded train phi: {train_files[0].name}")

    # Load test embeddings
    test_files = sorted(embeddings_dir.glob('phi_embeddings_test_*.csv'), reverse=True)
    if test_files:
        results['phi_test'] = pd.read_csv(test_files[0])
        print(f"Loaded test phi: {test_files[0].name}")

    return results

def plot_model_comparison(metrics_df):
    """Create comprehensive model comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Linearity ratio Comparison
    ax = axes[0, 0]
    if 'model' in metrics_df.columns:
        metrics_pivot = metrics_df.pivot_table(
            index='seed', 
            columns='model', 
            values='linearity_ratio'
        )
        metrics_pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title('Linearity ratio by model and seed', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Linearity ratio')
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target')
        ax.legend(title='Model', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # 2. Outcome RÂ² Comparison
    ax = axes[0, 1]
    if 'outcome_R2' in metrics_df.columns:
        metrics_pivot = metrics_df.pivot_table(
            index='seed', 
            columns='model', 
            values='outcome_R2'
        )
        metrics_pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title('Outcome R\u00b2 (mediation validation)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('RÂ²')
        ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Good')
        ax.legend(title='Model', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # 3. Mediator RÂ² Comparison
    ax = axes[0, 2]
    if 'mediator_R2' in metrics_df.columns:
        metrics_pivot = metrics_df.pivot_table(
            index='seed', 
            columns='model', 
            values='mediator_R2'
        )
        metrics_pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title('Mediator prediction RÂ²', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('RÂ²')
        ax.legend(title='Model', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # 4. Trade-off Scatter Plot
    ax = axes[1, 0]
    if 'linearity_ratio' in metrics_df.columns and 'outcome_R2' in metrics_df.columns:
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            ax.scatter(
                model_data['linearity_ratio'], 
                model_data['outcome_R2'],
                label=model, s=100, alpha=0.7, edgecolors='black', linewidth=1
            )
        ax.set_xlabel('Linearity ratio')
        ax.set_ylabel('Outcome RÂ²')
        ax.set_title('Linearity vs predictive power trade-off', fontsize=12, fontweight='bold')
        ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.3)
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)
    
    # 5. Sparsity (Proportion non-zero)
    ax = axes[1, 1]
    if 'prop_nonzero' in metrics_df.columns:
        metrics_pivot = metrics_df.pivot_table(
            index='seed', 
            columns='model', 
            values='prop_nonzero'
        )
        metrics_pivot.plot(kind='bar', ax=ax, width=0.7)
        ax.set_title('Feature sparsity (prop. non-zero)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Proportion non-zero')
        ax.legend(title='Model', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # 6. Performance summary Table
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary statistics
    summary = metrics_df.groupby('model').agg({
        'linearity_ratio': ['mean', 'std'],
        'outcome_R2': ['mean', 'std'],
        'mediator_R2': ['mean', 'std']
    }).round(3)
    
    # Format for display
    summary_display = []
    for model in summary.index:
        row = [model]
        for metric in ['linearity_ratio', 'outcome_R2', 'mediator_R2']:
            if metric in summary.columns.levels[0]:
                mean = summary.loc[model, (metric, 'mean')]
                std = summary.loc[model, (metric, 'std')]
                row.append(f"{mean:.3f} Â± {std:.3f}")
        summary_display.append(row)
    
    table = ax.table(
        cellText=summary_display,
        colLabels=['Model', 'Linearity', 'Outcome RÂ²', 'Mediator RÂ²'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Performance summary (mean Â± std)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Autoencoder model comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved model comparison plot to {VIZ_DIR / 'model_comparison.png'}")
    plt.close()

def plot_phi_distributions(phi_df, model_name=''):
    """Plot distributions of phi features"""
    # Get phi columns
    phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]
    n_phi = len(phi_cols)
    
    if n_phi == 0:
        print(f"No phi columns found in {model_name} data")
        return
    
    # Create figure with subplots
    n_rows = int(np.ceil(n_phi / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3))
    axes = np.atleast_1d(axes).flatten()
    
    for i, col in enumerate(phi_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Plot histogram with KDE
            phi_values = phi_df[col].dropna()
            ax.hist(phi_values, bins=30, alpha=0.6, color='blue', edgecolor='black')
            ax.axvline(phi_values.mean(), color='red', linestyle='--', label=f'Mean: {phi_values.mean():.2f}')
            ax.axvline(phi_values.median(), color='green', linestyle='--', label=f'Median: {phi_values.median():.2f}')
            
            ax.set_title(format_phi_label(col), fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_phi, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} φ Feature distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'phi_distributions_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved phi distributions to {VIZ_DIR / f'phi_distributions_{model_name.lower()}.png'}")
    plt.close()

def plot_phi_correlations(phi_df, model_name=''):
    """Plot correlation matrix of phi features"""
    phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]

    if len(phi_cols) == 0:
        print(f"No phi columns found in {model_name} data")
        return

    # Compute correlation matrix
    corr_matrix = phi_df[phi_cols].corr()

    # Format labels using LaTeX phi symbols
    formatted_labels = [format_phi_label(col) for col in phi_cols]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Heatmap with formatted labels
    ax = axes[0]
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        xticklabels=formatted_labels,
        yticklabels=formatted_labels,
        cbar_kws={'label': 'Correlation'}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_title(f'{model_name} φ Correlation matrix', fontsize=12, fontweight='bold')
    
    # 2. Distribution of correlations
    ax = axes[1]
    # Get upper triangle of correlation matrix (excluding diagonal)
    upper_tri = np.triu(corr_matrix.values, k=1)
    correlations = upper_tri[upper_tri != 0].flatten()
    
    ax.hist(correlations, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero correlation')
    ax.set_xlabel('Correlation coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of pairwise correlations', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    textstr = f'Mean: {np.mean(correlations):.3f}\n'
    textstr += f'Std: {np.std(correlations):.3f}\n'
    textstr += f'Max: {np.max(np.abs(correlations)):.3f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'phi_correlations_{model_name.lower()}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved phi correlations to {VIZ_DIR / f'phi_correlations_{model_name.lower()}.png'}")
    plt.close()

def plot_phi_by_meal_type(phi_df, model_name=''):
    """Plot phi features grouped by meal type.

    Uses the first 3 PCs and shows two rows:
      Row 1: PC1 vs PC2 coloured by meal type, treatment, mediator
      Row 2: PC1 vs PC3 coloured by the same variables (highlights PC3)
    """
    if 'meal_type' not in phi_df.columns:
        print("No meal_type column found")
        return

    phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]

    n_pcs = min(3, len(phi_cols))
    if n_pcs < 2:
        print("Need at least 2 phi features for visualization")
        return

    phi_values = phi_df[phi_cols].values
    pca = PCA(n_components=n_pcs)
    phi_pca = pca.fit_transform(phi_values)

    var_pct = [f'{pca.explained_variance_ratio_[i]:.1%}' for i in range(n_pcs)]

    # Two rows: top = PC1 vs PC2, bottom = PC1 vs PC3 (if available)
    n_rows = 2 if n_pcs >= 3 else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 7 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    for row, (yi, ylabel) in enumerate(
        [(1, f'PC2 ({var_pct[1]} var)'),
         (2, f'PC3 ({var_pct[2]} var)')][:n_rows]
    ):
        xlabel = f'PC1 ({var_pct[0]} var)'

        # 1. Meal type
        ax = axes[row, 0]
        meal_types = sorted(phi_df['meal_type'].unique())
        colors = sns.color_palette("Set2", len(meal_types))
        for i, meal in enumerate(meal_types):
            mask = phi_df['meal_type'] == meal
            ax.scatter(phi_pca[mask, 0], phi_pca[mask, yi],
                       label=meal, alpha=0.6, s=25, color=colors[i],
                       edgecolors='none', rasterized=True)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{model_name} — Meal type', fontsize=14, fontweight='bold')
        ax.legend(title='Meal type', fontsize=10, title_fontsize=11,
                  markerscale=1.5, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        # 2. Treatment
        ax = axes[row, 1]
        if 'treat_meal_carbs' in phi_df.columns:
            scatter = ax.scatter(phi_pca[:, 0], phi_pca[:, yi],
                                 c=phi_df['treat_meal_carbs'],
                                 cmap='YlOrBr', alpha=0.65, s=25,
                                 edgecolors='none', rasterized=True)
            plt.colorbar(scatter, ax=ax, label='Meal Carbs (g)', shrink=0.8)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title('Coloured by treatment', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)

        # 3. Mediator
        ax = axes[row, 2]
        if 'mediator_bolus_for_meal' in phi_df.columns:
            scatter = ax.scatter(phi_pca[:, 0], phi_pca[:, yi],
                                 c=phi_df['mediator_bolus_for_meal'],
                                 cmap='cividis', alpha=0.65, s=25,
                                 edgecolors='none', rasterized=True)
            plt.colorbar(scatter, ax=ax, label='Bolus (units)', shrink=0.8)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title('Coloured by mediator', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'{model_name} — Embeddings by meal type, treatment & mediator',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f'phi_by_meal_{model_name.lower()}.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved phi by meal type to "
          f"{VIZ_DIR / f'phi_by_meal_{model_name.lower()}.png'}")
    plt.close()

def _add_density_contours(ax, x, y, color='black', levels=5):
    """Add KDE density contours to a scatter plot axis."""
    try:
        from scipy.stats import gaussian_kde
        # Remove NaN/inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]
        if len(x_clean) < 10:
            return
        xy = np.vstack([x_clean, y_clean])
        kde = gaussian_kde(xy)
        xmin, xmax = x_clean.min(), x_clean.max()
        ymin, ymax = y_clean.min(), y_clean.max()
        pad_x = (xmax - xmin) * 0.05
        pad_y = (ymax - ymin) * 0.05
        xx, yy = np.mgrid[
            xmin - pad_x : xmax + pad_x : 100j,
            ymin - pad_y : ymax + pad_y : 100j,
        ]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = np.reshape(kde(positions), xx.shape)
        ax.contour(xx, yy, zz, levels=levels, colors=color,
                   linewidths=0.7, alpha=0.5)
    except Exception:
        pass  # gracefully skip if KDE fails (e.g. singular covariance)


def plot_phi_pca_3d(phi_df, model_name=''):
    """Enhanced PCA visualization emphasizing all three leading PCs.

    Produces three separate figures saved as individual files:

    Figure 1 – Pairwise scatter plots (PC1-PC2, PC1-PC3, PC2-PC3) coloured by
               the *third* PC value so that all three dimensions are visible in
               every panel.  Density contours are overlaid to reveal structure.

    Figure 2 – The same three pairwise views coloured by an external variable
               (outcome delta glucose, treatment carbs, or mediator bolus) to
               show how the PC space relates to the causal quantities.

    Figure 3 – The same three pairwise views coloured by meal type (categorical)
               to reveal meal-dependent clustering in the PC space.
    """
    phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]

    if len(phi_cols) < 3:
        print(f"Need at least 3 phi features for 3D visualization (found {len(phi_cols)})")
        return

    # ---- PCA ----
    phi_values = phi_df[phi_cols].values
    n_components = min(10, len(phi_cols))
    pca = PCA(n_components=n_components)
    phi_pca = pca.fit_transform(phi_values)
    cumvar_3pc = np.sum(pca.explained_variance_ratio_[:3])

    var_labels = [
        f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})'
        for i in range(3)
    ]

    # (variance info is included in the axis labels already)

    pair_specs = [
        # (x_idx, y_idx, colour_idx)
        (0, 1, 2),   # PC1 vs PC2, colour = PC3
        (0, 2, 1),   # PC1 vs PC3, colour = PC2
        (1, 2, 0),   # PC2 vs PC3, colour = PC1
    ]

    panel_labels = ['a', 'b', 'c']

    # ================================================================
    # Figure 1: Pairwise scatter plots coloured by the *third* PC
    # ================================================================
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 6))

    for col_idx, (xi, yi, ci) in enumerate(pair_specs):
        ax = axes1[col_idx]
        scatter = ax.scatter(
            phi_pca[:, xi], phi_pca[:, yi],
            c=phi_pca[:, ci],
            cmap='viridis', s=18, alpha=0.7,
            edgecolors='none', rasterized=True,
        )
        _add_density_contours(ax, phi_pca[:, xi], phi_pca[:, yi],
                              color='white', levels=6)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(var_labels[ci], fontsize=10)
        ax.set_xlabel(var_labels[xi], fontsize=11)
        ax.set_ylabel(var_labels[yi], fontsize=11)
        ax.set_title(f'{var_labels[xi].split(" ")[0]} vs {var_labels[yi].split(" ")[0]}',
                     fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.15)
        ax.text(-0.12, 1.08, panel_labels[col_idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                fontfamily='sans-serif')

    fig1.suptitle(
        f'{model_name} PCA — Pairwise views coloured by third PC',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig1.tight_layout()
    fname1 = VIZ_DIR / f'phi_pca_pc_color_{model_name.lower()}.png'
    fig1.savefig(fname1, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname1}")
    plt.close(fig1)

    # ================================================================
    # Figure 2: Pairwise views coloured by external causal variables
    # ================================================================
    ext_vars = []
    if 'outcome_delta_glucose' in phi_df.columns:
        ext_vars.append(('outcome_delta_glucose', 'plasma', r'$\Delta$ Glucose (mg/dL)'))
    if 'treat_meal_carbs' in phi_df.columns:
        ext_vars.append(('treat_meal_carbs', 'YlOrBr', 'Meal Carbs (g)'))
    if 'mediator_bolus_for_meal' in phi_df.columns:
        ext_vars.append(('mediator_bolus_for_meal', 'cividis', 'Bolus (units)'))

    # Pad to exactly 3 panels (re-use first if fewer available)
    while len(ext_vars) < 3:
        ext_vars.append(ext_vars[0] if ext_vars else (None, None, None))

    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))

    for col_idx in range(3):
        ax = axes2[col_idx]
        xi, yi = pair_specs[col_idx][:2]
        var_name, cmap_name, label = ext_vars[col_idx]

        if var_name is not None and var_name in phi_df.columns:
            c_vals = phi_df[var_name].values
            scatter = ax.scatter(
                phi_pca[:, xi], phi_pca[:, yi],
                c=c_vals, cmap=cmap_name, s=18, alpha=0.7,
                edgecolors='none', rasterized=True,
            )
            _add_density_contours(ax, phi_pca[:, xi], phi_pca[:, yi],
                                  color='grey', levels=5)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(label, fontsize=10)
            ax.set_title(label, fontweight='bold', fontsize=12)
        else:
            ax.scatter(phi_pca[:, xi], phi_pca[:, yi],
                       s=18, alpha=0.5, color='steelblue',
                       edgecolors='none', rasterized=True)
            ax.set_title(f'{var_labels[xi].split(" ")[0]} vs {var_labels[yi].split(" ")[0]}',
                         fontweight='bold', fontsize=12)

        ax.set_xlabel(var_labels[xi], fontsize=11)
        ax.set_ylabel(var_labels[yi], fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.text(-0.12, 1.08, panel_labels[col_idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                fontfamily='sans-serif')

    fig2.suptitle(
        f'{model_name} PCA — Pairwise views coloured by causal variables',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig2.tight_layout()
    fname2 = VIZ_DIR / f'phi_pca_causal_color_{model_name.lower()}.png'
    fig2.savefig(fname2, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname2}")
    plt.close(fig2)

    # ================================================================
    # Figure 3: Pairwise views coloured by meal type (categorical)
    # ================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6))

    if 'meal_type' in phi_df.columns:
        meal_types = sorted(phi_df['meal_type'].unique())
        palette = sns.color_palette("Set2", len(meal_types))
        color_map = {meal: palette[i] for i, meal in enumerate(meal_types)}

        for col_idx, (xi, yi, _ci) in enumerate(pair_specs):
            ax = axes3[col_idx]
            for meal in meal_types:
                mask = phi_df['meal_type'] == meal
                ax.scatter(
                    phi_pca[mask, xi], phi_pca[mask, yi],
                    color=color_map[meal], label=meal,
                    s=18, alpha=0.7, edgecolors='none', rasterized=True,
                )
            _add_density_contours(ax, phi_pca[:, xi], phi_pca[:, yi],
                                  color='grey', levels=5)
            ax.set_xlabel(var_labels[xi], fontsize=11)
            ax.set_ylabel(var_labels[yi], fontsize=11)
            ax.set_title(f'{var_labels[xi].split(" ")[0]} vs {var_labels[yi].split(" ")[0]}',
                         fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.15)
            ax.legend(title='Meal type', fontsize=9, title_fontsize=10,
                      loc='upper right', frameon=True, framealpha=0.9,
                      edgecolor='0.6', fancybox=True)
            ax.text(-0.12, 1.08, panel_labels[col_idx], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left',
                    fontfamily='sans-serif')
    else:
        # Fallback: no meal_type column — plain scatter
        for col_idx, (xi, yi, _ci) in enumerate(pair_specs):
            ax = axes3[col_idx]
            ax.scatter(phi_pca[:, xi], phi_pca[:, yi],
                       s=18, alpha=0.5, color='steelblue',
                       edgecolors='none', rasterized=True)
            ax.set_xlabel(var_labels[xi], fontsize=11)
            ax.set_ylabel(var_labels[yi], fontsize=11)
            ax.set_title(f'{var_labels[xi].split(" ")[0]} vs {var_labels[yi].split(" ")[0]}',
                         fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.15)
            ax.text(-0.12, 1.08, panel_labels[col_idx], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left',
                    fontfamily='sans-serif')

    fig3.suptitle(
        f'{model_name} PCA — Pairwise views by meal type',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig3.tight_layout()
    fname3 = VIZ_DIR / f'phi_pca_meal_type_{model_name.lower()}.png'
    fig3.savefig(fname3, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname3}")
    plt.close(fig3)

    return pca, phi_pca


def plot_phi_comparison(results):
    """Create direct comparison plots between residual and causal phi features"""
    if 'phi_residual' not in results or 'phi_causal' not in results:
        print("Need both residual and causal phi for comparison")
        return
    
    phi_resid = results['phi_residual']
    phi_causal = results['phi_causal']
    
    # Get phi columns for each
    phi_cols_resid = [col for col in phi_resid.columns if col.startswith('phi_')]
    phi_cols_causal = [col for col in phi_causal.columns if col.startswith('phi_')]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Compare number of features and their distributions
    ax = axes[0, 0]
    ax.bar(['Residual', 'Causal'], [len(phi_cols_resid), len(phi_cols_causal)], 
           color=['coral', 'skyblue'], edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of φ Features')
    ax.set_title('Feature dimensionality comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, (name, count) in enumerate(zip(['Residual', 'Causal'], 
                                          [len(phi_cols_resid), len(phi_cols_causal)])):
        ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    # 2. Compare correlation structure
    ax = axes[0, 1]
    
    # Calculate correlation statistics for each model
    corr_resid = phi_resid[phi_cols_resid].corr()
    corr_causal = phi_causal[phi_cols_causal].corr()
    
    # Get off-diagonal correlations
    mask_resid = np.triu(np.ones_like(corr_resid), k=1).astype(bool)
    mask_causal = np.triu(np.ones_like(corr_causal), k=1).astype(bool)
    
    corr_vals_resid = corr_resid.values[mask_resid]
    corr_vals_causal = corr_causal.values[mask_causal]
    
    # Plot distributions
    ax.hist(np.abs(corr_vals_resid), bins=30, alpha=0.5, label='Residual', 
            color='coral', density=True, edgecolor='black')
    ax.hist(np.abs(corr_vals_causal), bins=30, alpha=0.5, label='Causal', 
            color='skyblue', density=True, edgecolor='black')
    
    ax.set_xlabel('Absolute correlation')
    ax.set_ylabel('Density')
    ax.set_title('Feature correlation distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    textstr = f'Residual mean: {np.mean(np.abs(corr_vals_resid)):.3f}\n'
    textstr += f'Causal mean: {np.mean(np.abs(corr_vals_causal)):.3f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Compare variance explained (PCA)
    ax = axes[1, 0]
    
    # Fit PCA for each
    from sklearn.decomposition import PCA
    
    pca_resid = PCA().fit(phi_resid[phi_cols_resid].values)
    pca_causal = PCA().fit(phi_causal[phi_cols_causal].values)
    
    # Plot cumulative variance explained
    n_comps = min(10, len(phi_cols_resid), len(phi_cols_causal))
    cum_var_resid = np.cumsum(pca_resid.explained_variance_ratio_[:n_comps])
    cum_var_causal = np.cumsum(pca_causal.explained_variance_ratio_[:n_comps])
    
    ax.plot(range(1, n_comps+1), cum_var_resid, 'o-', label='Residual', 
            color='coral', linewidth=2, markersize=8)
    ax.plot(range(1, n_comps+1), cum_var_causal, 's-', label='Causal', 
            color='skyblue', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative variance explained')
    ax.set_title('PCA variance explained', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_comps+1))
    
    # 4. Compare feature ranges/scales
    ax = axes[1, 1]
    
    # Calculate feature statistics
    ranges_resid = []
    ranges_causal = []
    
    for col in phi_cols_resid[:min(len(phi_cols_resid), 20)]:  # Limit to first 20
        vals = phi_resid[col].values
        ranges_resid.append(np.ptp(vals))  # peak-to-peak (max - min)
    
    for col in phi_cols_causal[:min(len(phi_cols_causal), 20)]:
        vals = phi_causal[col].values
        ranges_causal.append(np.ptp(vals))
    
    # Box plot of ranges
    bp = ax.boxplot([ranges_resid, ranges_causal], 
                     labels=['Residual', 'Causal'],
                     patch_artist=True)
    
    # Color the boxes
    colors = ['coral', 'skyblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Feature range (max - min)')
    ax.set_title('Feature scale comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values
    ax.scatter([1, 2], [np.mean(ranges_resid), np.mean(ranges_causal)], 
               color='red', s=100, zorder=3, marker='D', label='Mean')
    ax.legend()
    
    plt.suptitle('Residual vs causal φ feature comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'phi_model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved phi comparison plot to {VIZ_DIR / 'phi_model_comparison.png'}")
    plt.close()


def plot_performance_over_seeds(metrics_df):
    """Plot performance metrics across different random seeds"""
    if 'seed' not in metrics_df.columns:
        print("No seed column found in metrics")
        return
    
    metrics_to_plot = ['linearity_ratio', 'outcome_R2', 'mediator_R2', 'prop_nonzero']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    n_metrics = len(available_metrics)
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Plot lines for each model
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            ax.plot(
                model_data['seed'], 
                model_data[metric],
                marker='o', 
                label=model,
                linewidth=2,
                markersize=8
            )
        
        ax.set_xlabel('Random seed')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} across seeds', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model stability across random seeds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'performance_stability.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved stability plot to {VIZ_DIR / 'performance_stability.png'}")
    plt.close()

def plot_mediation_diagnostics(results):
    """Create detailed diagnostics for mediation analysis readiness"""
    if 'phi_causal' not in results and 'phi_main' not in results:
        print("No causal phi found for mediation diagnostics")
        return
    
    # Use best available phi
    phi_df = results.get('phi_causal', results.get('phi_main'))
    phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]
    
    if 'treat_meal_carbs' not in phi_df.columns or 'mediator_bolus_for_meal' not in phi_df.columns:
        print("Missing treatment or mediator columns for diagnostics")
        return
    
    # Create figure with mediation-specific diagnostics
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Extract data
    A = phi_df['treat_meal_carbs'].values
    M = phi_df['mediator_bolus_for_meal'].values
    phi_values = phi_df[phi_cols].values
    
    # 1. Treatment-mediator relationship
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(A, M, alpha=0.5, s=20)
    ax.set_xlabel('Treatment (meal carbs)')
    ax.set_ylabel('Mediator (bolus)')
    ax.set_title('Treatment-mediator relationship', fontweight='bold')
    
    # Add regression line
    z = np.polyfit(A[np.isfinite(A) & np.isfinite(M)], 
                   M[np.isfinite(A) & np.isfinite(M)], 1)
    p = np.poly1d(z)
    ax.plot(np.sort(A), p(np.sort(A)), "r-", alpha=0.8, label=f'slope={z[0]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mediator Distribution by Treatment group
    ax = fig.add_subplot(gs[0, 1])
    A_binary = A > np.median(A)
    
    data_to_plot = [M[~A_binary], M[A_binary]]
    bp = ax.boxplot(data_to_plot, labels=['Low Carbs', 'High Carbs'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_ylabel('Mediator (bolus)')
    ax.set_title('Mediator by treatment group', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add means
    ax.scatter([1, 2], [np.mean(M[~A_binary]), np.mean(M[A_binary])], 
               color='red', s=100, zorder=3, marker='D', label='Mean')
    ax.legend()
    
    # 3. Partial Correlations
    ax = fig.add_subplot(gs[0, 2:4])
    
    # Calculate partial correlations between A and M given each phi
    partial_corrs = []
    for i in range(phi_values.shape[1]):
        # Residualize A and M with respect to phi_i
        from sklearn.linear_model import LinearRegression
        lr_a = LinearRegression()
        lr_m = LinearRegression()
        
        phi_i = phi_values[:, i].reshape(-1, 1)
        lr_a.fit(phi_i, A)
        lr_m.fit(phi_i, M)
        
        A_resid = A - lr_a.predict(phi_i)
        M_resid = M - lr_m.predict(phi_i)
        
        corr = np.corrcoef(A_resid, M_resid)[0, 1]
        partial_corrs.append(corr)
    
    ax.bar(range(len(partial_corrs)), partial_corrs, color='steelblue', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Â±0.1 threshold')
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('φ Feature Index')
    ax.set_ylabel('Partial Correlation(A, M | φᵢ)')
    ax.set_title('Partial correlations: testing AâŠ¥M|Î¦', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Feature Importance for Mediator
    ax = fig.add_subplot(gs[1, 0:2])
    
    from sklearn.ensemble import RandomForestRegressor
    rf_m = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_m.fit(phi_values, M)
    importance_m = rf_m.feature_importances_
    
    indices = np.argsort(importance_m)[-10:]  # Top 10
    ax.barh(range(len(indices)), importance_m[indices], color='green', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([format_phi_label(phi_cols[i]) for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 features for mediator prediction', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Residual Analysis
    ax = fig.add_subplot(gs[1, 2:4])
    
    # Predict M from phi
    from sklearn.linear_model import RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13))
    ridge.fit(phi_values, M)
    M_pred = ridge.predict(phi_values)
    residuals = M - M_pred
    
    ax.scatter(M_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Predicted mediator')
    ax.set_ylabel('Residuals')
    ax.set_title(f'Mediator prediction residuals (RÂ²={ridge.score(phi_values, M):.3f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add Â±2 std bands
    std_resid = np.std(residuals)
    ax.axhline(y=2*std_resid, color='orange', linestyle=':', alpha=0.5, label='Â±2Ïƒ')
    ax.axhline(y=-2*std_resid, color='orange', linestyle=':', alpha=0.5)
    ax.legend()
    
    # 6. Confounding Assessment: Compare treated vs control in phi space
    ax = fig.add_subplot(gs[2, :])
    
    # PCA of phi
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    phi_pca = pca.fit_transform(phi_values)
    
    ax.scatter(phi_pca[~A_binary, 0], phi_pca[~A_binary, 1], 
              alpha=0.5, s=30, color='blue', label='Low Carbs')
    ax.scatter(phi_pca[A_binary, 0], phi_pca[A_binary, 1], 
              alpha=0.5, s=30, color='red', label='High Carbs')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Treatment groups in phi space (good overlap = good balance)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Correlation Heatmap: A, M, and top phi features
    ax = fig.add_subplot(gs[3, 0:2])
    
    # Select top 5 phi features by importance
    top_5_idx = np.argsort(importance_m)[-5:]
    data_for_corr = np.column_stack([A, M, phi_values[:, top_5_idx]])
    labels = ['A', 'M'] + [format_phi_label(phi_cols[i]) for i in top_5_idx]
    
    corr_matrix = np.corrcoef(data_for_corr.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_title('Correlation: treatment, mediator, top phi', fontweight='bold')
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # 8. Propensity score Distribution
    ax = fig.add_subplot(gs[3, 2:4])
    
    # Estimate propensity scores
    from sklearn.linear_model import LogisticRegression
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(phi_values, A_binary)
    propensity = ps_model.predict_proba(phi_values)[:, 1]
    
    ax.hist(propensity[~A_binary], bins=30, alpha=0.5, label='Low Carbs', color='blue', density=True)
    ax.hist(propensity[A_binary], bins=30, alpha=0.5, label='High Carbs', color='red', density=True)
    ax.set_xlabel('Propensity score')
    ax.set_ylabel('Density')
    ax.set_title('Propensity score overlap (good overlap = identifiability)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Mediation analysis diagnostics', fontsize=16, fontweight='bold')
    plt.savefig(VIZ_DIR / 'mediation_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved mediation diagnostics to {VIZ_DIR / 'mediation_diagnostics.png'}")
    plt.close()


def plot_treatment_mediator_outcome_pathways(results):
    """Visualize the treatment â†’ mediator â†’ outcome pathways"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if 'phi_causal' in results or 'phi_main' in results:
        phi_df = results.get('phi_causal', results.get('phi_main'))
        
        if all(col in phi_df.columns for col in ['treat_meal_carbs', 'mediator_bolus_for_meal']):
            A = phi_df['treat_meal_carbs'].values
            M = phi_df['mediator_bolus_for_meal'].values
            
            # 1. A â†’ M relationship strength
            ax = axes[0]
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(A, M)
            
            ax.text(0.5, 0.7, 'Treatment â†’ Mediator', ha='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.5, f'Correlation: {np.corrcoef(A, M)[0,1]:.3f}', ha='center', fontsize=12)
            ax.text(0.5, 0.4, f'Slope: {slope:.3f} Â± {std_err:.3f}', ha='center', fontsize=12)
            ax.text(0.5, 0.3, f'RÂ²: {r_value**2:.3f}', ha='center', fontsize=12)
            ax.text(0.5, 0.2, f'p-value: {p_value:.2e}', ha='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Direct effect statistics', fontweight='bold')
            
            # 2. Distribution shifts
            ax = axes[1]
            A_groups = pd.qcut(A, q=3, labels=['Low', 'Medium', 'High'])
            
            for i, group in enumerate(['Low', 'Medium', 'High']):
                mask = A_groups == group
                if mask.sum() > 0:
                    kernel = stats.gaussian_kde(M[mask])
                    x_range = np.linspace(M.min(), M.max(), 100)
                    ax.plot(x_range, kernel(x_range), label=f'{group} Carbs', linewidth=2)
            
            ax.set_xlabel('Mediator (bolus)')
            ax.set_ylabel('Density')
            ax.set_title('Mediator distribution by treatment level', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Variance analysis
            ax = axes[2]
            variances = []
            means = []
            groups = ['Low', 'Medium', 'High']
            
            for group in groups:
                mask = A_groups == group
                if mask.sum() > 0:
                    variances.append(np.var(M[mask]))
                    means.append(np.mean(M[mask]))
            
            x = np.arange(len(groups))
            width = 0.35
            
            ax2 = ax.twinx()
            bars1 = ax.bar(x - width/2, means, width, label='Mean', color='skyblue', edgecolor='black')
            bars2 = ax2.bar(x + width/2, variances, width, label='Variance', color='coral', edgecolor='black')
            
            ax.set_xlabel('Treatment group')
            ax.set_ylabel('Mean mediator', color='skyblue')
            ax2.set_ylabel('Variance', color='coral')
            ax.set_title('Mean and variance by treatment', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.tick_params(axis='y', labelcolor='skyblue')
            ax2.tick_params(axis='y', labelcolor='coral')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Treatment-mediator-outcome pathway analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'pathway_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved pathway analysis to {VIZ_DIR / 'pathway_analysis.png'}")
    plt.close()


def create_detailed_mediation_report(results):
    """Create a detailed text report focused on mediation analysis readiness"""
    report = []
    report.append("="*70)
    report.append("DETAILED MEDIATION ANALYSIS READINESS REPORT")
    report.append("="*70)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append("")
    
    # Check for required data
    phi_df = results.get('phi_causal', results.get('phi_main'))
    
    if phi_df is not None and all(col in phi_df.columns for col in ['treat_meal_carbs', 'mediator_bolus_for_meal']):
        phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]
        A = phi_df['treat_meal_carbs'].values
        M = phi_df['mediator_bolus_for_meal'].values
        phi_values = phi_df[phi_cols].values
        
        report.append("MEDIATION PATHWAY STATISTICS")
        report.append("-"*40)
        
        # Treatment-Mediator relationship
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(A, M)
        
        report.append("\n1. TREATMENT â†’ MEDIATOR")
        report.append(f"   Correlation: {np.corrcoef(A, M)[0,1]:.4f}")
        report.append(f"   Linear slope: {slope:.4f} Â± {std_err:.4f}")
        report.append(f"   RÂ²: {r_value**2:.4f}")
        report.append(f"   p-value: {p_value:.2e}")
        
        if p_value < 0.001:
            report.append("   âœ“ Strong treatment-mediator relationship")
        else:
            report.append("   âš ï¸ Weak treatment-mediator relationship")
        
        # Conditional independence check
        report.append("\n2. CONDITIONAL INDEPENDENCE AâŠ¥M|Î¦")
        
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(phi_values, M)
        M_resid = M - lr.predict(phi_values)
        
        lr_a = LinearRegression()
        lr_a.fit(phi_values, A)
        A_resid = A - lr_a.predict(phi_values)
        
        partial_corr = np.corrcoef(A_resid, M_resid)[0, 1]
        
        report.append(f"   Partial correlation(A,M|Î¦): {partial_corr:.4f}")
        report.append(f"   Mediator RÂ² from Î¦: {lr.score(phi_values, M):.4f}")
        
        if abs(partial_corr) < 0.1:
            report.append("   âœ“ Good conditional independence")
        elif abs(partial_corr) < 0.2:
            report.append("   âš ï¸ Moderate conditional independence")
        else:
            report.append("   âŒ Poor conditional independence - may affect ACME")
        
        # Feature diagnostics
        report.append("\n3. PHI FEATURE DIAGNOSTICS")
        report.append(f"   Number of features: {len(phi_cols)}")
        report.append(f"   Samples: {len(phi_df)}")
        
        # Sparsity
        from sklearn.linear_model import LassoCV
        lasso = LassoCV(cv=5)
        lasso.fit(phi_values, M)
        n_nonzero = np.sum(lasso.coef_ != 0)
        
        report.append(f"   Active features for M (Lasso): {n_nonzero}/{len(phi_cols)} ({100*n_nonzero/len(phi_cols):.1f}%)")
        
        # Collinearity check
        corr_matrix = np.corrcoef(phi_values.T)
        np.fill_diagonal(corr_matrix, 0)
        max_corr = np.max(np.abs(corr_matrix))
        high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.8) // 2
        
        report.append(f"   Max correlation between features: {max_corr:.3f}")
        report.append(f"   Highly correlated pairs (|r|>0.8): {high_corr_pairs}")
        
        if high_corr_pairs > 3:
            report.append("   âš ï¸ High collinearity detected - may inflate standard errors")
        
        # Balance assessment
        report.append("\n4. TREATMENT GROUP BALANCE")
        A_binary = A > np.median(A)
        
        phi_treated = phi_values[A_binary]
        phi_control = phi_values[~A_binary]
        
        # Standardized mean differences
        smd = np.abs(phi_treated.mean(axis=0) - phi_control.mean(axis=0)) / np.sqrt(
            (phi_treated.var(axis=0) + phi_control.var(axis=0)) / 2 + 1e-8
        )
        
        report.append(f"   Max standardized mean difference: {np.max(smd):.3f}")
        report.append(f"   Mean SMD: {np.mean(smd):.3f}")
        report.append(f"   Features with SMD > 0.2: {np.sum(smd > 0.2)}/{len(smd)}")
        
        if np.max(smd) < 0.2:
            report.append("   âœ“ Good balance between treatment groups")
        else:
            report.append("   âš ï¸ Some imbalance between treatment groups")
        
        # Sample size considerations
        report.append("\n5. SAMPLE SIZE AND POWER")
        report.append(f"   Total samples: {len(phi_df)}")
        report.append(f"   Treated: {np.sum(A_binary)} ({100*np.mean(A_binary):.1f}%)")
        report.append(f"   Control: {np.sum(~A_binary)} ({100*np.mean(~A_binary):.1f}%)")
        
        if len(phi_df) < 500:
            report.append("   âš ï¸ Sample size may be low for detecting small mediation effects")
        
        # Overall assessment
        report.append("\n" + "="*40)
        report.append("OVERALL MEDIATION READINESS ASSESSMENT")
        report.append("-"*40)
        
        issues = []
        if abs(partial_corr) > 0.15:
            issues.append("- High residual correlation between A and M given Î¦")
        if high_corr_pairs > 3:
            issues.append("- High collinearity among phi features")
        if np.max(smd) > 0.25:
            issues.append("- Imbalance between treatment groups")
        if lr.score(phi_values, M) < 0.2:
            issues.append("- Low mediator predictability from phi")
        
        if len(issues) == 0:
            report.append("âœ… Features appear well-suited for mediation analysis")
        else:
            report.append("âš ï¸ Potential issues that may affect ACME significance:")
            for issue in issues:
                report.append(f"  {issue}")
        
        # Recommendations
        report.append("\nRECOMMENDATIONS:")
        if abs(partial_corr) > 0.15:
            report.append("1. Consider adding more confounders to achieve AâŠ¥M|Î¦")
        if high_corr_pairs > 3:
            report.append("2. Use feature selection or regularization to reduce collinearity")
        if np.max(smd) > 0.25:
            report.append("3. Consider propensity score weighting for better balance")
        if lr.score(phi_values, M) < 0.2:
            report.append("4. Features may not capture mediator determinants well")
    
    # Save report (to tables directory, not figures)
    report_path = TABLES_DIR / 'mediation_readiness_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved detailed mediation report to {report_path}")

    # Also print to console
    print('\n'.join(report))

    return report


def create_summary_report(results):
    """Create a text summary report of the results"""
    report = []
    report.append("="*70)
    report.append("AUTOENCODER EVALUATION SUMMARY REPORT")
    report.append("="*70)
    report.append("")
    
    # Best model identification
    if 'best_model' in results:
        report.append(f"BEST MODEL: {results['best_model'].upper()}")
        report.append("(This is the model saved as the main phi file)")
        report.append("")
    
    # Model comparison
    if 'metrics' in results:
        metrics_df = results['metrics']
        report.append("MODEL COMPARISON SUMMARY")
        report.append("-"*40)
        
        # Find best performing model based on metrics
        if 'outcome_R2' in metrics_df.columns:
            best_by_outcome = metrics_df.groupby('model')['outcome_R2'].mean().idxmax()
            report.append(f"Best by Outcome RÂ²: {best_by_outcome}")
        
        if 'linearity_ratio' in metrics_df.columns:
            best_by_linearity = metrics_df.groupby('model')['linearity_ratio'].mean().idxmax()
            report.append(f"Best by Linearity: {best_by_linearity}")
        
        report.append("")
        
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            report.append(f"\n{model}:")
            
            if 'linearity_ratio' in model_data.columns:
                mean_lin = model_data['linearity_ratio'].mean()
                std_lin = model_data['linearity_ratio'].std()
                report.append(f"  Linearity ratio: {mean_lin:.3f} Â± {std_lin:.3f}")
            
            if 'outcome_R2' in model_data.columns:
                mean_r2 = model_data['outcome_R2'].mean()
                std_r2 = model_data['outcome_R2'].std()
                report.append(f"  Outcome RÂ²: {mean_r2:.3f} Â± {std_r2:.3f}")
            
            if 'mediator_R2' in model_data.columns:
                mean_med = model_data['mediator_R2'].mean()
                std_med = model_data['mediator_R2'].std()
                report.append(f"  Mediator RÂ²: {mean_med:.3f} Â± {std_med:.3f}")
            
            if 'prop_nonzero' in model_data.columns:
                mean_sparse = model_data['prop_nonzero'].mean()
                report.append(f"  Feature Sparsity: {mean_sparse:.3f}")
    
    # Phi feature statistics
    report.append("\n" + "="*40)
    report.append("PHI FEATURE STATISTICS")
    report.append("-"*40)
    
    for key in results:
        if key.startswith('phi_'):
            model_name = key.replace('phi_', '')
            
            # Skip main if it's identical to another model
            if model_name == 'main' and 'best_model' in results:
                report.append(f"\n{key} (identical to {results['best_model']}):")
                report.append("  [Skipped - same as best model]")
                continue
            
            phi_df = results[key]
            phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]
            
            report.append(f"\n{key}:")
            report.append(f"  Number of phi features: {len(phi_cols)}")
            report.append(f"  Number of samples: {len(phi_df)}")
            
            if len(phi_cols) > 0:
                phi_values = phi_df[phi_cols].values
                corr_matrix = np.corrcoef(phi_values.T)
                # Off-diagonal correlations only
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                off_diag_corr = np.abs(corr_matrix[mask])
                
                report.append(f"  Mean absolute correlation: {np.mean(off_diag_corr):.3f}")
                report.append(f"  Max absolute correlation: {np.max(off_diag_corr):.3f}")
                report.append(f"  Correlation > 0.5: {np.sum(off_diag_corr > 0.5)} pairs")
                report.append(f"  Correlation > 0.8: {np.sum(off_diag_corr > 0.8)} pairs")
    
    # Save report (to tables directory, not figures)
    report_path = TABLES_DIR / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved summary report to {report_path}")
    
    # Also print to console
    print('\n'.join(report))

    return report


def create_comprehensive_summary(results):
    """Create a single comprehensive summary figure with all key diagnostics"""
    
    # Check what data we have
    has_metrics = 'metrics' in results
    has_phi = any(key.startswith('phi_') for key in results.keys())
    
    if not has_metrics and not has_phi:
        print("Insufficient data for comprehensive summary")
        return
    
    # Create a large figure with multiple panels
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 6, hspace=0.3, wspace=0.3)
    
    # Panel 1: Model comparison summary
    if has_metrics:
        metrics_df = results['metrics']
        
        ax = fig.add_subplot(gs[0, 0:2])
        if 'model' in metrics_df.columns and 'linearity_ratio' in metrics_df.columns:
            model_means = metrics_df.groupby('model')['linearity_ratio'].mean()
            ax.bar(range(len(model_means)), model_means.values, 
                  color=['coral', 'skyblue'][:len(model_means)])
            ax.set_xticks(range(len(model_means)))
            ax.set_xticklabels(model_means.index, rotation=45)
            ax.set_ylabel('Linearity ratio')
            ax.set_title('Model linearity comparison', fontweight='bold')
            ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
    
    # Panel 2: Outcome prediction performance
    if has_metrics and 'outcome_R2' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 2:4])
        model_means = metrics_df.groupby('model')['outcome_R2'].mean()
        model_stds = metrics_df.groupby('model')['outcome_R2'].std()
        
        x_pos = range(len(model_means))
        ax.bar(x_pos, model_means.values, yerr=model_stds.values,
               color=['coral', 'skyblue'][:len(model_means)], 
               capsize=5, edgecolor='black', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_means.index, rotation=45)
        ax.set_ylabel('RÂ²')
        ax.set_title('Outcome prediction performance', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Panel 3: Mediator prediction performance  
    if has_metrics and 'mediator_R2' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 4:6])
        model_means = metrics_df.groupby('model')['mediator_R2'].mean()
        model_stds = metrics_df.groupby('model')['mediator_R2'].std()
        
        x_pos = range(len(model_means))
        ax.bar(x_pos, model_means.values, yerr=model_stds.values,
               color=['coral', 'skyblue'][:len(model_means)],
               capsize=5, edgecolor='black', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_means.index, rotation=45)
        ax.set_ylabel('RÂ²')
        ax.set_title('Mediator prediction performance', fontweight='bold')
        ax.axhline(y=0.25, color='g', linestyle='--', alpha=0.5, label='Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Get best phi for remaining analyses
    phi_df = None
    if 'phi_causal' in results:
        phi_df = results['phi_causal']
        model_name = 'Causal'
    elif 'phi_main' in results:
        phi_df = results['phi_main']
        model_name = 'Main'
    
    if phi_df is not None:
        phi_cols = [col for col in phi_df.columns if col.startswith('phi_')]
        
        # Panel 4: Feature correlation structure
        ax = fig.add_subplot(gs[1, 0:3])
        if len(phi_cols) > 0:
            corr_matrix = phi_df[phi_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
            corr_values = np.abs(corr_matrix.values[mask])
            
            ax.hist(corr_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(x=0.5, color='orange', linestyle='--', label='Moderate correlation')
            ax.axvline(x=0.8, color='red', linestyle='--', label='High correlation')
            ax.set_xlabel('Absolute correlation')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{model_name} model: feature correlation distribution', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            textstr = f'Mean: {np.mean(corr_values):.3f}\n'
            textstr += f'Max: {np.max(corr_values):.3f}\n'
            textstr += f'> 0.8: {np.sum(corr_values > 0.8)} pairs'
            ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 5: Treatment-Mediator relationship
        if all(col in phi_df.columns for col in ['treat_meal_carbs', 'mediator_bolus_for_meal']):
            ax = fig.add_subplot(gs[1, 3:6])
            
            A = phi_df['treat_meal_carbs'].values
            M = phi_df['mediator_bolus_for_meal'].values
            
            # Create hexbin plot for dense data
            hb = ax.hexbin(A, M, gridsize=30, cmap='YlOrRd', mincnt=1)
            ax.set_xlabel('Treatment (meal carbs)')
            ax.set_ylabel('Mediator (bolus)')
            ax.set_title('Treatment-mediator relationship', fontweight='bold')
            plt.colorbar(hb, ax=ax, label='Count')
            
            # Add regression line
            z = np.polyfit(A[np.isfinite(A) & np.isfinite(M)], 
                          M[np.isfinite(A) & np.isfinite(M)], 1)
            p = np.poly1d(z)
            ax.plot(np.sort(A), p(np.sort(A)), "b-", linewidth=2, alpha=0.8)
            
            # Add correlation
            corr = np.corrcoef(A, M)[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel 6-7: PCA visualization
        if len(phi_cols) >= 2:
            from sklearn.decomposition import PCA
            phi_values = phi_df[phi_cols].values
            pca = PCA(n_components=min(10, len(phi_cols)))
            pca.fit(phi_values)
            
            # Scree plot
            ax = fig.add_subplot(gs[2, 0:2])
            ax.plot(range(1, len(pca.explained_variance_ratio_)+1), 
                   pca.explained_variance_ratio_, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Principal component')
            ax.set_ylabel('Variance explained')
            ax.set_title('PCA scree plot', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Cumulative variance
            ax2 = ax.twinx()
            ax2.plot(range(1, len(pca.explained_variance_ratio_)+1),
                    np.cumsum(pca.explained_variance_ratio_), 's-', 
                    color='red', linewidth=2, markersize=6, alpha=0.7)
            ax2.set_ylabel('Cumulative variance', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.axhline(y=0.9, color='red', linestyle=':', alpha=0.5)
        
        # Panel 8: Feature sparsity analysis
        if 'treat_meal_carbs' in phi_df.columns and 'mediator_bolus_for_meal' in phi_df.columns:
            ax = fig.add_subplot(gs[2, 2:4])
            
            from sklearn.linear_model import LassoCV
            # For mediator
            lasso_m = LassoCV(cv=5, max_iter=2000)
            lasso_m.fit(phi_values, M)
            
            # For treatment
            lasso_a = LassoCV(cv=5, max_iter=2000)
            lasso_a.fit(phi_values, A)
            
            nonzero_m = np.sum(lasso_m.coef_ != 0)
            nonzero_a = np.sum(lasso_a.coef_ != 0)
            
            categories = ['Mediator\nPrediction', 'Treatment\nPrediction']
            values = [nonzero_m, nonzero_a]
            colors = ['green', 'orange']
            
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
            ax.set_ylabel('Non-zero features')
            ax.set_title('Feature selection (Lasso)', fontweight='bold')
            ax.set_ylim(0, len(phi_cols))
            ax.axhline(y=len(phi_cols), color='red', linestyle='--', alpha=0.5, 
                      label=f'Total features: {len(phi_cols)}')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add percentage labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val}/{len(phi_cols)}\n({100*val/len(phi_cols):.1f}%)',
                       ha='center', va='bottom')
        
        # Panel 9: Sample size and balance
        ax = fig.add_subplot(gs[2, 4:6])
        
        if 'meal_type' in phi_df.columns:
            meal_counts = phi_df['meal_type'].value_counts()
            colors_meal = plt.cm.Set3(np.linspace(0, 1, len(meal_counts)))
            
            wedges, texts, autotexts = ax.pie(meal_counts.values, 
                                              labels=meal_counts.index,
                                              colors=colors_meal,
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax.set_title(f'Meal type distribution (N={len(phi_df)})', fontweight='bold')
            
            # Make percentage text more visible
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')
        
        # Panel 10-12: Key statistics summary
        ax = fig.add_subplot(gs[3, :])
        ax.axis('off')
        
        # Create summary text
        summary_text = "KEY FINDINGS\n" + "="*50 + "\n\n"
        
        if has_metrics:
            best_model = metrics_df.groupby('model')['outcome_R2'].mean().idxmax()
            best_r2 = metrics_df.groupby('model')['outcome_R2'].mean().max()
            summary_text += f"Best Model: {best_model} (Outcome RÂ² = {best_r2:.3f})\n"
            
            if 'mediator_R2' in metrics_df.columns:
                best_med_r2 = metrics_df[metrics_df['model'] == best_model]['mediator_R2'].mean()
                summary_text += f"Mediator RÂ²: {best_med_r2:.3f}\n"
        
        if phi_df is not None:
            summary_text += f"\nFeatures: {len(phi_cols)} phi dimensions\n"
            summary_text += f"Samples: {len(phi_df)} windows\n"
            
            if 'treat_meal_carbs' in phi_df.columns and 'mediator_bolus_for_meal' in phi_df.columns:
                corr_am = np.corrcoef(phi_df['treat_meal_carbs'], 
                                     phi_df['mediator_bolus_for_meal'])[0, 1]
                summary_text += f"Treatment-Mediator Correlation: {corr_am:.3f}\n"
        
        summary_text += "\n" + "="*50 + "\n"
        summary_text += "For significant ACME in mediation analysis:\n"
        summary_text += "â€¢ Need strong mediator prediction (RÂ² > 0.25)\n"
        summary_text += "â€¢ Need conditional independence AâŠ¥M|Î¦\n"
        summary_text += "â€¢ Avoid high collinearity among features\n"
        summary_text += "â€¢ Ensure good treatment group balance"
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
               fontsize=11, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Comprehensive autoencoder evaluation summary', fontsize=18, fontweight='bold')
    plt.savefig(VIZ_DIR / 'comprehensive_summary.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive summary to {VIZ_DIR / 'comprehensive_summary.png'}")
    plt.close()


def main():
    """Main function to run all visualizations"""
    print("\n" + "="*70)
    print("AUTOENCODER RESULTS VISUALIZATION")
    print("="*70)
    
    # Load results
    print("\nðŸ“Š Loading results...")
    results = load_results()
    
    if not results:
        print("âŒ No results found. Please run the autoencoder analysis first.")
        return
    
    # Create visualizations
    print("\nðŸ“ˆ Creating visualizations...")
    
    # 1. Model comparison
    if 'metrics' in results:
        print("\n1. Model Comparison...")
        plot_model_comparison(results['metrics'])
        plot_performance_over_seeds(results['metrics'])
    
    # 2. Phi feature analysis for each model
    # Determine which models to analyze (skip main if it's a duplicate)
    models_to_analyze = []
    best_model = results.get('best_model', None)
    
    for model in ['residual', 'causal']:
        phi_key = f'phi_{model}'
        if phi_key in results:
            models_to_analyze.append((model, phi_key))
    
    # Only add main if it's different from the individual models
    if 'phi_main' in results:
        if best_model:
            print(f"\nâ„¹ï¸  Main phi is identical to {best_model.upper()} model (skipping duplicate plots)")
        else:
            # Main is different or we couldn't determine which model it is
            models_to_analyze.append(('main', 'phi_main'))
    
    # Create plots for each unique model
    for model_name, phi_key in models_to_analyze:
        print(f"\n2. Analyzing {model_name} phi features...")
        phi_df = results[phi_key]
        plot_phi_distributions(phi_df, model_name=model_name.title())
        plot_phi_correlations(phi_df, model_name=model_name.title())
        plot_phi_by_meal_type(phi_df, model_name=model_name.title())
        
        # Add enhanced 3D PCA visualization
        print(f"   Creating enhanced 3D PCA visualization for {model_name}...")
        plot_phi_pca_3d(phi_df, model_name=model_name.title())
    
    # 3. Create comparison plots if we have multiple models
    if 'phi_residual' in results and 'phi_causal' in results:
        print("\n3. Creating model comparison plots...")
        plot_phi_comparison(results)
    
    # 4. Mediation-specific diagnostics
    print("\n4. Creating mediation diagnostics...")
    plot_mediation_diagnostics(results)
    plot_treatment_mediator_outcome_pathways(results)
    
    # 5. Create detailed reports
    print("\nðŸ“ Creating summary reports...")
    create_summary_report(results)
    detailed_report = create_detailed_mediation_report(results)
    
    # 6. Create a comprehensive summary figure
    print("\n5. Creating comprehensive summary figure...")
    create_comprehensive_summary(results)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Tables/Reports saved to: {TABLES_DIR}")
    print("\nKey outputs:")
    print("  Figures:")
    print(f"    - model_comparison.png - Model performance comparison")
    print(f"    - mediation_diagnostics.png - Detailed mediation readiness")
    print(f"    - pathway_analysis.png - Treatment-mediator-outcome pathways")
    print(f"    - phi_distributions_*.png - Feature distributions")
    print(f"    - phi_correlations_*.png - Feature correlations")
    print(f"    - phi_pca_pc_color_*.png - Pairwise PCs coloured by third PC")
    print(f"    - phi_pca_causal_color_*.png - Pairwise PCs coloured by causal variables")
    print(f"    - phi_pca_meal_type_*.png - Pairwise PCs coloured by meal type")
    print("  Tables/Reports:")
    print(f"    - summary_report.txt - Basic summary")
    print(f"    - mediation_readiness_report.txt - Detailed mediation assessment")
    
    return results

if __name__ == "__main__":
    results = main()