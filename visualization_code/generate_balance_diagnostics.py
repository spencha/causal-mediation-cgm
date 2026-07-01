#!/usr/bin/env python3
"""
generate_balance_diagnostics.py
================================
Comprehensive diagnostics to verify that treatment is conditionally independent
of covariates after applying npCBPS balancing weights.

This script generates publication-quality figures and tables demonstrating that
the causal mediation analysis assumptions are satisfied.

Key Assumptions Verified:
1. Covariate Balance: Treatment is conditionally independent of confounders
   after weighting (T ⊥ X | weights)
2. Overlap/Positivity: Sufficient overlap in propensity scores between
   treatment levels
3. Weight Quality: Effective sample size and weight distribution

Usage:
  python generate_balance_diagnostics.py                  # Use phi features (default)
  python generate_balance_diagnostics.py --use-pca        # Use PC features
  python generate_balance_diagnostics.py --use-pca --n-phi 3  # Use first 3 PCs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import WLS
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False
})


def format_phi_label(col_name):
    """Format phi/PC column names as proper LaTeX math subscripts."""
    if col_name.startswith('phi_'):
        subscript = col_name.replace('phi_', '')
        return r'$\varphi_{' + subscript + r'}$'
    elif col_name.startswith('PC_'):
        subscript = col_name.replace('PC_', '')
        return f'PC{subscript}'
    elif col_name == 'glucose_at_meal':
        return 'Glucose at Meal (std.)'  # Standardized glucose at meal start
    elif col_name == 'glucose':
        return 'Glucose'
    return col_name

# =============================================================================
# COLORBLIND-FRIENDLY PALETTE (Nature Style)
# =============================================================================
# Following Paul Tol's colorblind-safe palette
# https://personal.sron.nl/~pault/#sec:qualitative
COLORS = {
    'unweighted': '#CC3311',    # Vermillion (before weighting)
    'weighted': '#009988',      # Teal (after weighting)
    'threshold': '#EE7733',     # Orange (threshold line)
    'balanced': '#009988',      # Teal (good balance)
    'imbalanced': '#CC3311',    # Vermillion (needs attention)
    'treatment': '#0077BB',     # Blue (treatment variable)
    'covariate': '#33BBEE',     # Cyan (covariates)
    'primary': '#332288',       # Indigo (emphasis)
    'secondary': '#BBBBBB'      # Gray (secondary elements)
}

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AE_CODE_DIR = PROJECT_ROOT / "ae_python_code"

# Default number of phi/PC features used for balancing (must match npcbps_weights.R)
DEFAULT_N_PHI_FEATURES = 6

# Default split to use (must match npcbps_weights.R default)
DEFAULT_SPLIT = "test"

# Global config (will be set by command line args)
USE_PCA = False
N_PHI_FEATURES = DEFAULT_N_PHI_FEATURES

# Add ae_python_code to path for imports
if str(AE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_CODE_DIR))

from config import CONFIG
CONFIG.ensure_dirs()

WEIGHTS_DIR = CONFIG.WEIGHTS_DIR
PHI_DIR = CONFIG.ANALYSIS_DATA_DIR
# Output directories live under visualizations/ (separate from code)
FIGURES_DIR = PROJECT_ROOT / "visualizations" / "npcbps_balance" / "figures"
TABLES_DIR = PROJECT_ROOT / "visualizations" / "npcbps_balance" / "tables"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(use_pca=False, n_features=None):
    """Load phi/PC embeddings and weights data.

    Parameters
    ----------
    use_pca : bool
        If True, use PC_ columns instead of phi_ columns
    n_features : int, optional
        Number of features to use (defaults to N_PHI_FEATURES)
    """
    global N_PHI_FEATURES, USE_PCA

    if n_features is not None:
        N_PHI_FEATURES = n_features
    USE_PCA = use_pca

    feature_type = "PC" if use_pca else "phi"

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    print(f"Feature type: {feature_type} (using first {N_PHI_FEATURES} features)")

    # Embeddings are in: analysis_data/embeddings/
    # File pattern: phi_embeddings_combined_{arch}_{pct}pct_{penalty}_seed{seed}.csv
    embeddings_dir = PHI_DIR / 'embeddings'

    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    combined_files = list(embeddings_dir.glob('phi_embeddings_combined_*.csv'))

    if not combined_files:
        raise FileNotFoundError(
            f"No combined embeddings found in: {embeddings_dir}\n"
            f"Expected: phi_embeddings_combined_*.csv\n"
            f"Run train_and_export_embeddings.py first."
        )

    # Use most recently modified file (matches npcbps_weights.R logic)
    phi_path = max(combined_files, key=lambda f: f.stat().st_mtime)
    phi_df = pd.read_csv(phi_path)
    print(f"\nLoaded {len(phi_df):,} total observations from: {phi_path.name}")

    # Filter to test split only (must match npcbps_weights.R default)
    if 'split' in phi_df.columns:
        n_before = len(phi_df)
        phi_df = phi_df[phi_df['split'] == DEFAULT_SPLIT].copy()
        print(f"Filtered to split='{DEFAULT_SPLIT}': {n_before} -> {len(phi_df)} observations")
    else:
        print("WARNING: No 'split' column found - using all observations")

    # Load weights
    weights_path = WEIGHTS_DIR / 'npCBPS_weights.csv'
    if weights_path.exists():
        weights_df = pd.read_csv(weights_path)
        print(f"Loaded weights for {len(weights_df):,} observations")

        # Merge weights with phi data
        phi_df = phi_df.merge(weights_df, on='global_window_id', how='left')

        # Fill missing weights with 1
        if 'treatment_weight' in phi_df.columns:
            phi_df['treatment_weight'] = phi_df['treatment_weight'].fillna(1.0)
            print(f"Weight range: [{phi_df['treatment_weight'].min():.3f}, {phi_df['treatment_weight'].max():.3f}]")
        else:
            print("WARNING: 'treatment_weight' column not found")
            phi_df['treatment_weight'] = 1.0
    else:
        print("WARNING: Weights file not found - using unit weights")
        phi_df['treatment_weight'] = 1.0

    # Identify phi or PC columns based on use_pca flag
    if use_pca:
        # Use PC columns
        all_cols = [col for col in phi_df.columns if col.startswith('PC_')]
        if not all_cols:
            raise ValueError("No PC_ columns found in embeddings file. "
                           "Make sure PCA was computed when exporting embeddings.")
        all_cols = sorted(all_cols, key=lambda x: int(x.replace('PC_', '')))
        feature_cols = all_cols[:N_PHI_FEATURES]
        print(f"Found {len(all_cols)} PC dimensions, using first {len(feature_cols)} for balance assessment")
    else:
        # Use phi columns
        all_cols = [col for col in phi_df.columns if col.startswith('phi_')]
        all_cols = sorted(all_cols, key=lambda x: int(x.replace('phi_', '')))
        feature_cols = all_cols[:N_PHI_FEATURES]
        print(f"Found {len(all_cols)} phi dimensions, using first {len(feature_cols)} for balance assessment")

    print(f"  Using: {feature_cols}")
    print(f"  (This matches the n_phi_features={N_PHI_FEATURES} setting in npcbps_weights.R)")

    # Add glucose_at_meal if present
    if 'glucose_at_meal' in phi_df.columns:
        print("Including glucose_at_meal as additional covariate")

    return phi_df, feature_cols


# =============================================================================
# BALANCE METRICS
# =============================================================================

def compute_weighted_correlation(x, y, weights=None):
    """
    Compute weighted Pearson correlation between x and y.
    
    Parameters
    ----------
    x, y : array-like
        Variables to correlate
    weights : array-like, optional
        Observation weights
        
    Returns
    -------
    correlation : float
    p_value : float (approximate)
    """
    valid_idx = ~(np.isnan(x) | np.isnan(y))
    if weights is not None:
        valid_idx &= ~np.isnan(weights)
    
    x_clean = np.array(x)[valid_idx]
    y_clean = np.array(y)[valid_idx]
    
    if len(x_clean) < 3:
        return np.nan, np.nan
    
    if weights is None:
        corr, p_val = pearsonr(x_clean, y_clean)
    else:
        w_clean = np.array(weights)[valid_idx]
        
        # Weighted means
        w_sum = w_clean.sum()
        x_mean = np.average(x_clean, weights=w_clean)
        y_mean = np.average(y_clean, weights=w_clean)
        
        # Weighted covariance and variances
        cov_xy = np.sum(w_clean * (x_clean - x_mean) * (y_clean - y_mean)) / w_sum
        var_x = np.sum(w_clean * (x_clean - x_mean)**2) / w_sum
        var_y = np.sum(w_clean * (y_clean - y_mean)**2) / w_sum
        
        if var_x > 0 and var_y > 0:
            corr = cov_xy / np.sqrt(var_x * var_y)
        else:
            corr = 0.0
        
        # Approximate p-value using effective sample size
        n_eff = w_sum**2 / np.sum(w_clean**2)
        if n_eff > 2 and abs(corr) < 1:
            t_stat = corr * np.sqrt(n_eff - 2) / np.sqrt(1 - corr**2)
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff - 2))
        else:
            p_val = np.nan
    
    return corr, p_val


def compute_standardized_mean_difference(x, treatment, weights=None):
    """
    Compute standardized mean difference (SMD) for continuous treatment.
    
    For continuous treatment, we compute the correlation-based SMD:
    SMD = r * sqrt(1 + r²) where r is the correlation between x and treatment.
    
    This is equivalent to the standardized regression coefficient.
    
    Parameters
    ----------
    x : array-like
        Covariate values
    treatment : array-like
        Treatment values (continuous)
    weights : array-like, optional
        Observation weights
        
    Returns
    -------
    smd : float
        Standardized mean difference
    """
    corr, _ = compute_weighted_correlation(x, treatment, weights)
    
    if np.isnan(corr):
        return np.nan
    
    # For continuous treatment, SMD ≈ correlation
    # This is the standardized regression coefficient
    return corr


def compute_balance_statistics(phi_df, phi_cols, treatment_col='treat_meal_carbs'):
    """
    Compute comprehensive balance statistics.
    
    Parameters
    ----------
    phi_df : DataFrame
        Data with covariates and weights
    phi_cols : list
        List of covariate column names
    treatment_col : str
        Name of treatment column
        
    Returns
    -------
    balance_df : DataFrame
        Balance statistics for each covariate
    """
    treatment = phi_df[treatment_col].values
    weights = phi_df['treatment_weight'].values
    
    results = []
    
    # Add glucose_at_meal if present
    covariates = list(phi_cols)
    if 'glucose_at_meal' in phi_df.columns:
        covariates.append('glucose_at_meal')
    
    for cov in covariates:
        x = phi_df[cov].values
        
        # Unweighted statistics
        corr_unw, p_unw = compute_weighted_correlation(x, treatment, weights=None)
        smd_unw = compute_standardized_mean_difference(x, treatment, weights=None)
        
        # Weighted statistics
        corr_w, p_w = compute_weighted_correlation(x, treatment, weights=weights)
        smd_w = compute_standardized_mean_difference(x, treatment, weights=weights)
        
        # Improvement
        if not np.isnan(corr_unw) and not np.isnan(corr_w):
            reduction_pct = (1 - abs(corr_w) / abs(corr_unw)) * 100 if abs(corr_unw) > 0.001 else 0
        else:
            reduction_pct = np.nan
        
        results.append({
            'Covariate': cov,  # Keep original name, format for display later
            'Corr_Unweighted': corr_unw,
            'Corr_Weighted': corr_w,
            'P_Unweighted': p_unw,
            'P_Weighted': p_w,
            'SMD_Unweighted': smd_unw,
            'SMD_Weighted': smd_w,
            'Reduction_Pct': reduction_pct,
            'Balanced_Before': abs(corr_unw) < 0.1 if not np.isnan(corr_unw) else False,
            'Balanced_After': abs(corr_w) < 0.1 if not np.isnan(corr_w) else False
        })
    
    return pd.DataFrame(results)


def compute_effective_sample_size(weights):
    """
    Compute effective sample size (ESS) given weights.
    
    ESS = (Σw)² / Σw²
    
    This measures how much the weighting reduces the effective information
    in the sample.
    """
    valid_weights = weights[~np.isnan(weights)]
    if len(valid_weights) == 0:
        return 0
    
    ess = (valid_weights.sum()**2) / (valid_weights**2).sum()
    return ess


# =============================================================================
# FIGURE 1: LOVE PLOT (COVARIATE BALANCE)
# =============================================================================

def plot_love_plot(balance_df, save_dir):
    """
    Create Love plot showing balance improvement.
    
    Love plots display the absolute correlation (or SMD) for each covariate
    before and after weighting, with lines connecting them.
    """
    fig, ax = plt.subplots(figsize=(10, max(6, len(balance_df) * 0.4)))
    
    y_pos = np.arange(len(balance_df))
    
    # Plot points
    ax.scatter(np.abs(balance_df['Corr_Unweighted']), y_pos, 
              color=COLORS['unweighted'], s=100, label='Unweighted', zorder=3,
              marker='o', edgecolor='white', linewidth=1)
    ax.scatter(np.abs(balance_df['Corr_Weighted']), y_pos,
              color=COLORS['weighted'], s=100, label='Weighted', zorder=3,
              marker='s', edgecolor='white', linewidth=1)
    
    # Connect with lines
    for i in range(len(balance_df)):
        ax.plot([np.abs(balance_df['Corr_Unweighted'].iloc[i]), 
                np.abs(balance_df['Corr_Weighted'].iloc[i])],
               [y_pos[i], y_pos[i]], 
               color='gray', linewidth=1, alpha=0.5, zorder=1)
    
    # Balance threshold
    ax.axvline(x=0.1, color=COLORS['threshold'], linestyle='--', 
              linewidth=2, label='Balance threshold (|r| = 0.1)')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_phi_label(c) for c in balance_df['Covariate']])
    ax.set_xlabel('Absolute correlation with treatment', fontsize=11)
    ax.set_title('Covariate balance: love plot\nLines connect unweighted to weighted values',
                fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(-0.02, max(0.35, np.abs(balance_df['Corr_Unweighted']).max() * 1.1))
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add summary annotation
    n_balanced_before = balance_df['Balanced_Before'].sum()
    n_balanced_after = balance_df['Balanced_After'].sum()
    n_total = len(balance_df)
    
    summary_text = (f"Balanced (|r| < 0.1):\n"
                   f"  Before: {n_balanced_before}/{n_total}\n"
                   f"  After:  {n_balanced_after}/{n_total}")
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    save_path = save_dir / 'fig1_love_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 2: CORRELATION COMPARISON
# =============================================================================

def plot_correlation_comparison(balance_df, save_dir):
    """
    Bar chart comparing correlations before and after weighting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_pos = np.arange(len(balance_df))
    width = 0.35
    
    # Panel A: Signed correlations
    ax = axes[0]
    
    ax.bar(x_pos - width/2, balance_df['Corr_Unweighted'], width,
          label='Unweighted', color=COLORS['unweighted'], alpha=0.8)
    ax.bar(x_pos + width/2, balance_df['Corr_Weighted'], width,
          label='Weighted', color=COLORS['weighted'], alpha=0.8)
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.1, color=COLORS['threshold'], linestyle='--', 
              linewidth=1.5, alpha=0.7)
    ax.axhline(y=-0.1, color=COLORS['threshold'], linestyle='--', 
              linewidth=1.5, alpha=0.7)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_phi_label(c) for c in balance_df['Covariate']], rotation=45, ha='right')
    ax.set_ylabel('Correlation with treatment')
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Treatment-covariate correlations', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Absolute correlations
    ax = axes[1]

    ax.bar(x_pos - width/2, np.abs(balance_df['Corr_Unweighted']), width,
          label='Unweighted', color=COLORS['unweighted'], alpha=0.8)
    ax.bar(x_pos + width/2, np.abs(balance_df['Corr_Weighted']), width,
          label='Weighted', color=COLORS['weighted'], alpha=0.8)

    ax.axhline(y=0.1, color=COLORS['threshold'], linestyle='--',
              linewidth=2, label='Balance threshold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_phi_label(c) for c in balance_df['Covariate']], rotation=45, ha='right')
    ax.set_ylabel('|Correlation| with treatment')
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Absolute correlations (balance assessment)', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Covariate balance before and after weighting',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig2_correlation_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 3: WEIGHT DISTRIBUTION AND EFFECTIVE SAMPLE SIZE
# =============================================================================

def plot_weight_diagnostics(phi_df, save_dir):
    """
    Visualize weight distribution and effective sample size.
    """
    weights = phi_df['treatment_weight'].values
    treatment = phi_df['treat_meal_carbs'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Weight distribution
    ax = axes[0, 0]
    
    ax.hist(weights, bins=50, color=COLORS['weighted'], alpha=0.7, 
           edgecolor='black', linewidth=0.5)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, label='Unit weight')
    ax.axvline(x=np.median(weights), color=COLORS['threshold'], linestyle='-', 
              linewidth=2, label=f'Median = {np.median(weights):.2f}')
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Distribution of npCBPS weights', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Weights vs Treatment
    ax = axes[0, 1]
    
    ax.scatter(treatment, weights, alpha=0.3, s=10, color=COLORS['weighted'])
    
    # Add trend line
    valid_idx = ~(np.isnan(treatment) | np.isnan(weights))
    z = np.polyfit(treatment[valid_idx], weights[valid_idx], 1)
    p = np.poly1d(z)
    x_range = np.linspace(treatment[valid_idx].min(), treatment[valid_idx].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
    
    ax.set_xlabel('Treatment (carbohydrates)')
    ax.set_ylabel('Weight')
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Weight vs treatment level', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel C: Cumulative weight distribution
    ax = axes[1, 0]
    
    sorted_weights = np.sort(weights)
    cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
    
    ax.plot(sorted_weights, cumulative, color=COLORS['weighted'], linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=np.median(weights), color=COLORS['threshold'], linestyle='--', 
              alpha=0.7, label=f'Median = {np.median(weights):.2f}')
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Cumulative proportion')
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Cumulative weight distribution', fontweight='bold', loc='left')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Panel D: Effective Sample size summary
    ax = axes[1, 1]
    ax.axis('off')
    
    n_total = len(weights)
    ess = compute_effective_sample_size(weights)
    ess_pct = ess / n_total * 100
    
    # Weight statistics
    weight_stats = {
        'N (Total)': f'{n_total:,}',
        'ESS': f'{ess:,.1f}',
        'ESS %': f'{ess_pct:.1f}%',
        'Weight Mean': f'{np.mean(weights):.3f}',
        'Weight SD': f'{np.std(weights):.3f}',
        'Weight Min': f'{np.min(weights):.3f}',
        'Weight Max': f'{np.max(weights):.3f}',
        'Weight Median': f'{np.median(weights):.3f}',
        'Weight IQR': f'[{np.percentile(weights, 25):.2f}, {np.percentile(weights, 75):.2f}]',
        '% Weights > 2': f'{100 * np.mean(weights > 2):.1f}%',
        '% Weights < 0.5': f'{100 * np.mean(weights < 0.5):.1f}%'
    }
    
    # Create table
    cell_text = [[k, v] for k, v in weight_stats.items()]
    table = ax.table(cellText=cell_text, 
                    colLabels=['Metric', 'Value'],
                    loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Weight summary statistics', fontweight='bold', loc='left', pad=20)
    
    fig.suptitle('npCBPS weight diagnostics',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig3_weight_diagnostics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 4: CORRELATION HEATMAPS
# =============================================================================

def plot_correlation_heatmaps(phi_df, phi_cols, save_dir):
    """
    Compare correlation structure before and after weighting.
    """
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    # Add glucose_at_meal if present
    covariates = list(phi_cols)
    if 'glucose_at_meal' in phi_df.columns:
        covariates.append('glucose_at_meal')
    
    n_covs = len(covariates)
    
    # Compute correlation matrices
    # Unweighted: treatment with each covariate
    corr_unweighted = np.zeros(n_covs)
    corr_weighted = np.zeros(n_covs)
    
    for i, cov in enumerate(covariates):
        x = phi_df[cov].values
        
        corr_unweighted[i], _ = compute_weighted_correlation(x, treatment, weights=None)
        corr_weighted[i], _ = compute_weighted_correlation(x, treatment, weights=weights)
    
    # Full covariate correlation matrix (unweighted)
    cov_data = phi_df[covariates].values
    valid_idx = ~np.any(np.isnan(cov_data), axis=1)
    full_corr_unw = np.corrcoef(cov_data[valid_idx].T)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Treatment-covariate correlations comparison
    ax = axes[0]
    
    data = np.array([corr_unweighted, corr_weighted])
    max_abs = max(np.abs(data).max(), 0.3)
    
    im = ax.imshow(data, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')
    
    ax.set_xticks(range(n_covs))
    ax.set_xticklabels([format_phi_label(c) for c in covariates], rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Unweighted', 'Weighted'])
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Treatment-covariate correlations', fontweight='bold', loc='left')

    # Add text annotations
    for i in range(2):
        for j in range(n_covs):
            val = data[i, j]
            color = 'white' if abs(val) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
    
    # Panel B: Covariate correlation matrix
    ax = axes[1]
    
    im = ax.imshow(full_corr_unw, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(n_covs))
    ax.set_xticklabels([format_phi_label(c) for c in covariates], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_covs))
    ax.set_yticklabels([format_phi_label(c) for c in covariates], fontsize=8)
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Covariate correlation structure', fontweight='bold', loc='left')
    
    plt.colorbar(im, ax=ax, label='Correlation', shrink=0.8)
    
    # Panel C: Balance improvement
    ax = axes[2]
    
    improvement = np.abs(corr_unweighted) - np.abs(corr_weighted)
    
    colors = [COLORS['balanced'] if imp > 0 else COLORS['imbalanced'] for imp in improvement]
    bars = ax.barh(range(n_covs), improvement, color=colors, alpha=0.8)
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(range(n_covs))
    ax.set_yticklabels([format_phi_label(c) for c in covariates])
    ax.set_xlabel('Reduction in |correlation|')
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Balance improvement\n(positive = improved)', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle('Correlation analysis: unweighted vs weighted',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig4_correlation_heatmaps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 5: SCATTER PLOTS FOR TOP COVARIATES
# =============================================================================

def plot_scatter_diagnostics(phi_df, balance_df, phi_cols, save_dir):
    """
    Create scatter plots showing treatment vs covariates before/after weighting.
    """
    # Find top 4 most imbalanced covariates (before weighting)
    top_covs = balance_df.nlargest(4, 'Corr_Unweighted', keep='first')
    
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, (_, row) in enumerate(top_covs.iterrows()):
        cov_name = row['Covariate']  # Already in original format
        
        if cov_name not in phi_df.columns:
            continue
        
        x = phi_df[cov_name].values
        valid_idx = ~(np.isnan(x) | np.isnan(treatment) | np.isnan(weights))
        
        # Unweighted scatter
        ax = axes[0, i]
        ax.scatter(x[valid_idx], treatment[valid_idx], alpha=0.3, s=10, 
                  color=COLORS['unweighted'])
        
        # Add trend line
        z = np.polyfit(x[valid_idx], treatment[valid_idx], 1)
        p = np.poly1d(z)
        x_range = np.linspace(x[valid_idx].min(), x[valid_idx].max(), 100)
        ax.plot(x_range, p(x_range), 'k-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(format_phi_label(row['Covariate']))
        if i == 0:
            ax.set_ylabel('Treatment')
        ax.set_title(f'Unweighted\nr = {row["Corr_Unweighted"]:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Weighted scatter (size proportional to weight)
        ax = axes[1, i]
        
        # Normalize weights for visualization
        w_norm = weights[valid_idx] / weights[valid_idx].max() * 50
        
        ax.scatter(x[valid_idx], treatment[valid_idx], alpha=0.4, s=w_norm,
                  color=COLORS['weighted'])
        
        # Weighted trend line
        z_w = np.polyfit(x[valid_idx], treatment[valid_idx], 1, w=weights[valid_idx])
        p_w = np.poly1d(z_w)
        ax.plot(x_range, p_w(x_range), 'k-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(format_phi_label(row['Covariate']))
        if i == 0:
            ax.set_ylabel('Treatment')
        ax.set_title(f'Weighted\nr = {row["Corr_Weighted"]:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Treatment vs top imbalanced covariates\n'
                 '(point size in bottom row proportional to weight)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig5_scatter_diagnostics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 6: RESIDUAL ANALYSIS
# =============================================================================

def plot_residual_analysis(phi_df, phi_cols, save_dir):
    """
    Analyze residuals from treatment ~ covariates regression.
    """
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    # Prepare covariates
    X = phi_df[phi_cols].values
    if 'glucose_at_meal' in phi_df.columns:
        X = np.column_stack([X, phi_df['glucose_at_meal'].values])
    
    valid_idx = ~(np.any(np.isnan(X), axis=1) | np.isnan(treatment) | np.isnan(weights))
    X_clean = X[valid_idx]
    treatment_clean = treatment[valid_idx]
    weights_clean = weights[valid_idx]
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Fit unweighted regression
    model_unw = LinearRegression()
    model_unw.fit(X_scaled, treatment_clean)
    resid_unw = treatment_clean - model_unw.predict(X_scaled)
    
    # Fit weighted regression
    model_w = LinearRegression()
    model_w.fit(X_scaled, treatment_clean, sample_weight=weights_clean)
    resid_w = treatment_clean - model_w.predict(X_scaled)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Unweighted
    ax = axes[0, 0]
    ax.scatter(model_unw.predict(X_scaled), resid_unw, alpha=0.3, s=10, 
              color=COLORS['unweighted'])
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Unweighted: residuals vs fitted', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(resid_unw, bins=50, density=True, color=COLORS['unweighted'], 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    x_norm = np.linspace(resid_unw.min(), resid_unw.max(), 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, np.mean(resid_unw), np.std(resid_unw)),
           'k-', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Unweighted: residual distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    stats.probplot(resid_unw, dist="norm", plot=ax)
    ax.set_title('Unweighted: Q-Q plot', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Weighted
    ax = axes[1, 0]
    scatter = ax.scatter(model_w.predict(X_scaled), resid_w, alpha=0.3, s=10,
                        c=weights_clean, cmap='Blues')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Weighted: residuals vs fitted', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Weight')
    
    ax = axes[1, 1]
    ax.hist(resid_w, bins=50, density=True, color=COLORS['weighted'],
           alpha=0.7, edgecolor='black', linewidth=0.5, 
           weights=weights_clean/weights_clean.sum()*len(weights_clean))
    ax.plot(x_norm, stats.norm.pdf(x_norm, np.average(resid_w, weights=weights_clean), 
                                    np.sqrt(np.average((resid_w - np.average(resid_w, weights=weights_clean))**2, 
                                                       weights=weights_clean))),
           'k-', linewidth=2, label='Normal fit')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Density')
    ax.set_title('Weighted: residual distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    # Weighted Q-Q plot approximation
    resid_standardized = (resid_w - np.average(resid_w, weights=weights_clean)) / \
                         np.sqrt(np.average((resid_w - np.average(resid_w, weights=weights_clean))**2, 
                                           weights=weights_clean))
    stats.probplot(resid_standardized, dist="norm", plot=ax)
    ax.set_title('Weighted: Q-Q plot', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add R² annotations
    r2_unw = model_unw.score(X_scaled, treatment_clean)
    r2_w = 1 - np.sum(weights_clean * resid_w**2) / np.sum(weights_clean * (treatment_clean - np.average(treatment_clean, weights=weights_clean))**2)
    
    fig.text(0.02, 0.52, f'Unweighted R² = {r2_unw:.3f}', fontsize=11, fontweight='bold',
            transform=fig.transFigure)
    fig.text(0.02, 0.02, f'Weighted R² = {r2_w:.3f}', fontsize=11, fontweight='bold',
            transform=fig.transFigure)
    
    fig.suptitle('Residual analysis: treatment ~ covariates',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig6_residual_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 7: PERMUTATION TEST VISUALIZATION
# =============================================================================

def plot_permutation_tests(phi_df, balance_df, phi_cols, save_dir, n_permutations=500):
    """
    Conduct and visualize permutation tests for conditional independence.
    """
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    # Select top 4 covariates by absolute unweighted correlation
    top_covs = balance_df.nlargest(4, 'Corr_Unweighted', keep='first')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    np.random.seed(42)
    
    for i, (_, row) in enumerate(top_covs.iterrows()):
        ax = axes[i // 2, i % 2]
        
        cov_name = row['Covariate']  # Already in original format
        
        if cov_name not in phi_df.columns:
            continue
        
        x = phi_df[cov_name].values
        valid_idx = ~(np.isnan(x) | np.isnan(treatment) | np.isnan(weights))
        
        x_clean = x[valid_idx]
        t_clean = treatment[valid_idx]
        w_clean = weights[valid_idx]
        
        # Observed weighted correlation
        obs_corr, _ = compute_weighted_correlation(x_clean, t_clean, w_clean)
        
        # Permutation distribution
        perm_corrs = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(len(t_clean))
            perm_corr, _ = compute_weighted_correlation(x_clean, t_clean[perm_idx], w_clean)
            perm_corrs.append(perm_corr)
        
        perm_corrs = np.array(perm_corrs)
        
        # P-value
        p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_corr))
        
        # Plot
        ax.hist(perm_corrs, bins=40, density=True, alpha=0.7, 
               color=COLORS['weighted'], edgecolor='black', linewidth=0.5)
        ax.axvline(x=obs_corr, color='red', linestyle='-', linewidth=2.5,
                  label=f'Observed r = {obs_corr:.3f}')
        ax.axvline(x=-obs_corr, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Correlation (permuted)')
        ax.set_ylabel('Density')
        ax.set_title(f'{format_phi_label(row["Covariate"])}\np = {p_value:.3f}', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Color code significance
        if p_value > 0.05:
            ax.set_facecolor('#e8f5e9')  # Light green for independent
        else:
            ax.set_facecolor('#ffebee')  # Light red for dependent
    
    fig.suptitle('Permutation tests for conditional independence (weighted)\n'
                 'Green background = p > 0.05 (independent)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig7_permutation_tests.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 8: PROPENSITY SCORE / GENERALIZED PROPENSITY SCORE ANALYSIS
# =============================================================================

def plot_propensity_analysis(phi_df, phi_cols, save_dir):
    """
    Analyze the generalized propensity score (GPS) for continuous treatment.
    
    For continuous treatments, we use the predicted treatment values from
    the regression model as a proxy for the GPS density.
    """
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    # Prepare covariates
    X = phi_df[phi_cols].values
    if 'glucose_at_meal' in phi_df.columns:
        X = np.column_stack([X, phi_df['glucose_at_meal'].values])
    
    valid_idx = ~(np.any(np.isnan(X), axis=1) | np.isnan(treatment) | np.isnan(weights))
    X_clean = X[valid_idx]
    treatment_clean = treatment[valid_idx]
    weights_clean = weights[valid_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Fit model to get predicted treatment (GPS proxy)
    model = LinearRegression()
    model.fit(X_scaled, treatment_clean)
    predicted_treatment = model.predict(X_scaled)
    residuals = treatment_clean - predicted_treatment
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel A: Treatment distribution
    ax = axes[0, 0]
    ax.hist(treatment_clean, bins=50, density=True, alpha=0.7,
           color=COLORS['unweighted'], edgecolor='black', linewidth=0.5,
           label='Unweighted')
    ax.hist(treatment_clean, bins=50, density=True, alpha=0.5,
           weights=weights_clean/weights_clean.sum()*len(weights_clean),
           color=COLORS['weighted'], edgecolor='black', linewidth=0.5,
           label='Weighted')
    ax.set_xlabel('Treatment (carbohydrates)')
    ax.set_ylabel('Density')
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Treatment distribution', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Predicted vs Actual treatment
    ax = axes[0, 1]
    ax.scatter(predicted_treatment, treatment_clean, alpha=0.2, s=5,
              color=COLORS['unweighted'], label='Unweighted')
    ax.plot([treatment_clean.min(), treatment_clean.max()],
           [treatment_clean.min(), treatment_clean.max()],
           'k--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Predicted treatment')
    ax.set_ylabel('Actual treatment')
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Predicted vs actual treatment', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel C: Residual vs Weight
    ax = axes[0, 2]
    scatter = ax.scatter(residuals, weights_clean, alpha=0.3, s=10,
                        c=treatment_clean, cmap='viridis')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Residual (actual - predicted)')
    ax.set_ylabel('Weight')
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Residual vs weight', fontweight='bold', loc='left')
    plt.colorbar(scatter, ax=ax, label='Treatment')
    ax.grid(True, alpha=0.3)
    
    # Panel D: Weight by treatment tertile
    ax = axes[1, 0]
    
    tertiles = np.percentile(treatment_clean, [33, 67])
    tertile_labels = ['Low\n(<33%)', 'Medium\n(33-67%)', 'High\n(>67%)']
    tertile_weights = [
        weights_clean[treatment_clean <= tertiles[0]],
        weights_clean[(treatment_clean > tertiles[0]) & (treatment_clean <= tertiles[1])],
        weights_clean[treatment_clean > tertiles[1]]
    ]
    
    bp = ax.boxplot(tertile_weights, labels=tertile_labels, patch_artist=True)
    colors_box = ['#90CAF9', '#42A5F5', '#1565C0']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Treatment tertile')
    ax.set_ylabel('Weight')
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Weight distribution by treatment level', fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Effective sample size by tertile
    ax = axes[1, 1]
    
    ess_tertiles = []
    n_tertiles = []
    for tw in tertile_weights:
        ess = compute_effective_sample_size(tw)
        ess_tertiles.append(ess)
        n_tertiles.append(len(tw))
    
    x_pos = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, n_tertiles, width, label='N', 
                  color='lightgray', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, ess_tertiles, width, label='ESS',
                  color=COLORS['weighted'], edgecolor='black')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tertile_labels)
    ax.set_xlabel('Treatment tertile')
    ax.set_ylabel('Sample size')
    ax.text(-0.12, 1.08, 'e', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Sample size vs ESS by tertile', fontweight='bold', loc='left')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel F: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate overlap statistics
    overall_ess = compute_effective_sample_size(weights_clean)
    
    summary_data = [
        ['Metric', 'Value'],
        ['N (Total)', f'{len(treatment_clean):,}'],
        ['ESS (Overall)', f'{overall_ess:,.0f}'],
        ['ESS Ratio', f'{overall_ess/len(treatment_clean)*100:.1f}%'],
        ['Treatment Mean', f'{np.mean(treatment_clean):.1f}g'],
        ['Treatment SD', f'{np.std(treatment_clean):.1f}g'],
        ['Treatment Range', f'[{np.min(treatment_clean):.0f}, {np.max(treatment_clean):.0f}]g'],
        ['Model R²', f'{model.score(X_scaled, treatment_clean):.3f}'],
        ['Residual SD', f'{np.std(residuals):.1f}g']
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.text(-0.12, 1.08, 'f', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Summary statistics', fontweight='bold', loc='left', pad=15)

    fig.suptitle('Generalized propensity score analysis\n'
                 'Assessing overlap and positivity for continuous treatment',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig8_propensity_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 9: COVARIATE BALANCE BY TREATMENT LEVEL
# =============================================================================

def plot_balance_by_treatment_level(phi_df, phi_cols, save_dir):
    """
    Examine covariate balance across different treatment levels.
    
    This checks whether the weighting achieves balance uniformly across
    the treatment distribution.
    """
    treatment = phi_df['treat_meal_carbs'].values
    weights = phi_df['treatment_weight'].values
    
    # Create treatment bins
    n_bins = 5
    treatment_bins = pd.qcut(treatment, n_bins, labels=False, duplicates='drop')
    bin_edges = pd.qcut(treatment, n_bins, duplicates='drop').categories
    
    # Select covariates (phi_cols is already limited to N_PHI_FEATURES)
    covariates = list(phi_cols)
    if 'glucose_at_meal' in phi_df.columns and 'glucose_at_meal' not in covariates:
        covariates = covariates + ['glucose_at_meal']
    
    fig, axes = plt.subplots(2, len(covariates)//2 + len(covariates)%2, 
                             figsize=(4*len(covariates)//2, 8), squeeze=False)
    axes = axes.flatten()
    
    for i, cov in enumerate(covariates):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        x = phi_df[cov].values
        
        # Calculate correlation within each bin
        bin_corrs_unw = []
        bin_corrs_w = []
        bin_labels = []
        
        for b in range(n_bins):
            mask = treatment_bins == b
            if mask.sum() > 10:
                x_bin = x[mask]
                t_bin = treatment[mask]
                w_bin = weights[mask]
                
                # Unweighted correlation
                valid = ~(np.isnan(x_bin) | np.isnan(t_bin))
                if valid.sum() > 2:
                    corr_unw, _ = pearsonr(x_bin[valid], t_bin[valid])
                else:
                    corr_unw = np.nan
                
                # Weighted correlation
                valid = ~(np.isnan(x_bin) | np.isnan(t_bin) | np.isnan(w_bin))
                if valid.sum() > 2:
                    corr_w, _ = compute_weighted_correlation(x_bin[valid], t_bin[valid], w_bin[valid])
                else:
                    corr_w = np.nan
                
                bin_corrs_unw.append(corr_unw)
                bin_corrs_w.append(corr_w)
                bin_labels.append(f'Q{b+1}')
        
        x_pos = np.arange(len(bin_labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, bin_corrs_unw, width, label='Unweighted',
              color=COLORS['unweighted'], alpha=0.7)
        ax.bar(x_pos + width/2, bin_corrs_w, width, label='Weighted',
              color=COLORS['weighted'], alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.1, color=COLORS['threshold'], linestyle='--', alpha=0.5)
        ax.axhline(y=-0.1, color=COLORS['threshold'], linestyle='--', alpha=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel('Treatment quintile')
        ax.set_ylabel('Correlation')
        ax.set_title(format_phi_label(cov), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused axes
    for j in range(len(covariates), len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Covariate balance by treatment quintile\n'
                 'Checking uniform balance across treatment distribution',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'fig9_balance_by_treatment.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# FIGURE 10: SUMMARY FIGURE
# =============================================================================

def plot_summary_figure(phi_df, balance_df, save_dir):
    """
    Create publication-ready summary figure.
    """
    weights = phi_df['treatment_weight'].values
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: Love plot (condensed)
    ax = fig.add_subplot(gs[0, 0])
    
    y_pos = np.arange(len(balance_df))
    
    ax.scatter(np.abs(balance_df['Corr_Unweighted']), y_pos,
              color=COLORS['unweighted'], s=60, label='Unweighted', 
              marker='o', zorder=3)
    ax.scatter(np.abs(balance_df['Corr_Weighted']), y_pos,
              color=COLORS['weighted'], s=60, label='Weighted',
              marker='s', zorder=3)
    
    for i in range(len(balance_df)):
        ax.plot([np.abs(balance_df['Corr_Unweighted'].iloc[i]),
                np.abs(balance_df['Corr_Weighted'].iloc[i])],
               [y_pos[i], y_pos[i]], color='gray', linewidth=0.8, alpha=0.5)
    
    ax.axvline(x=0.1, color=COLORS['threshold'], linestyle='--', linewidth=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_phi_label(c) for c in balance_df['Covariate']], fontsize=8)
    ax.set_xlabel('|Correlation|')
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Covariate balance', fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-0.02, None)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Panel B: Weight distribution
    ax = fig.add_subplot(gs[0, 1])
    
    ax.hist(weights, bins=40, color=COLORS['weighted'], alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=np.median(weights), color=COLORS['threshold'], linestyle='-',
              linewidth=2, label=f'Median = {np.median(weights):.2f}')
    
    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Weight distribution', fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Summary statistics table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    
    n_total = len(weights)
    ess = compute_effective_sample_size(weights)
    n_balanced_before = balance_df['Balanced_Before'].sum()
    n_balanced_after = balance_df['Balanced_After'].sum()
    n_covs = len(balance_df)
    
    mean_abs_corr_before = np.abs(balance_df['Corr_Unweighted']).mean()
    mean_abs_corr_after = np.abs(balance_df['Corr_Weighted']).mean()
    
    summary_data = [
        ['Metric', 'Value'],
        ['N (Total)', f'{n_total:,}'],
        ['ESS', f'{ess:,.0f} ({100*ess/n_total:.1f}%)'],
        ['Balanced Before', f'{n_balanced_before}/{n_covs}'],
        ['Balanced After', f'{n_balanced_after}/{n_covs}'],
        ['Mean |r| Before', f'{mean_abs_corr_before:.3f}'],
        ['Mean |r| After', f'{mean_abs_corr_after:.3f}'],
        ['Reduction', f'{100*(1 - mean_abs_corr_after/mean_abs_corr_before):.1f}%']
    ]
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Summary statistics', fontweight='bold', loc='left', pad=15)
    
    # Panel D: Correlation comparison bar chart
    ax = fig.add_subplot(gs[1, :2])
    
    x_pos = np.arange(len(balance_df))
    width = 0.35
    
    ax.bar(x_pos - width/2, np.abs(balance_df['Corr_Unweighted']), width,
          label='Unweighted', color=COLORS['unweighted'], alpha=0.8)
    ax.bar(x_pos + width/2, np.abs(balance_df['Corr_Weighted']), width,
          label='Weighted', color=COLORS['weighted'], alpha=0.8)
    
    ax.axhline(y=0.1, color=COLORS['threshold'], linestyle='--', linewidth=2,
              label='Balance threshold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_phi_label(c) for c in balance_df['Covariate']], rotation=45, ha='right')
    ax.set_ylabel('|Correlation| with Treatment')
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Balance improvement by covariate', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel E: Interpretation
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    if n_balanced_after >= n_covs * 0.8:
        status = "✓ BALANCE ACHIEVED"
        color = '#43A047'
        interpretation = ("All major confounders are balanced\n"
                         "after applying npCBPS weights.\n\n"
                         "The conditional independence\n"
                         "assumption is supported.")
    elif n_balanced_after >= n_covs * 0.5:
        status = "⚠ PARTIAL BALANCE"
        color = '#FFA726'
        interpretation = ("Most confounders are balanced,\n"
                         "but some residual imbalance remains.\n\n"
                         "Results should be interpreted\n"
                         "with caution.")
    else:
        status = "✗ IMBALANCE DETECTED"
        color = '#E53935'
        interpretation = ("Significant imbalance remains\n"
                         "after weighting.\n\n"
                         "Consider alternative methods\n"
                         "or additional covariates.")
    
    ax.text(0.5, 0.7, status, ha='center', va='center',
           fontsize=14, fontweight='bold', color=color,
           transform=ax.transAxes)
    ax.text(0.5, 0.35, interpretation, ha='center', va='center',
           fontsize=10, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.text(-0.12, 1.08, 'e', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')
    ax.set_title('Assessment', fontweight='bold', loc='left', pad=15)
    
    fig.suptitle('Covariate balance assessment for causal mediation analysis',
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = save_dir / 'fig8_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.show()
    
    return fig


# =============================================================================
# TABLE GENERATION
# =============================================================================

def generate_tables(phi_df, balance_df, tables_dir):
    """
    Generate a single consolidated LaTeX diagnostics table and supporting CSVs.

    Produces ``npcbps_diagnostics.tex`` with per-covariate balance and key
    weight statistics in a table note.

    Parameters
    ----------
    phi_df : DataFrame
        Data with covariates and weights.
    balance_df : DataFrame
        Balance statistics (from ``compute_balance_statistics``).
    tables_dir : Path
        Directory to save tables.
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    weights = phi_df['treatment_weight'].values
    n_total = len(weights)
    ess = compute_effective_sample_size(weights)
    ess_pct = ess / n_total * 100

    n_covs = len(balance_df)
    n_balanced_before = int(balance_df['Balanced_Before'].sum())
    n_balanced_after = int(balance_df['Balanced_After'].sum())

    mean_abs_before = np.abs(balance_df['Corr_Unweighted']).mean()
    mean_abs_after = np.abs(balance_df['Corr_Weighted']).mean()
    max_abs_before = np.abs(balance_df['Corr_Unweighted']).max()
    max_abs_after = np.abs(balance_df['Corr_Weighted']).max()

    w_mean = np.mean(weights)
    w_sd = np.std(weights)
    w_median = np.median(weights)
    w_q25 = np.percentile(weights, 25)
    w_q75 = np.percentile(weights, 75)
    w_min = np.min(weights)
    w_max = np.max(weights)
    pct_gt2 = 100 * np.mean(weights > 2)
    pct_lt05 = 100 * np.mean(weights < 0.5)

    # -----------------------------------------------------------------
    # Per-covariate rows
    # -----------------------------------------------------------------
    def _fmt_r(val):
        """Format correlation with \\phantom{-} alignment for LaTeX."""
        s = f'{val:.3f}'
        if val >= 0:
            return r'$\phantom{-}' + s + '$'
        return f'${s}$'

    cov_rows = []
    for _, row in balance_df.iterrows():
        cov_label = format_phi_label(row['Covariate'])
        r_unw = row['Corr_Unweighted']
        r_w = row['Corr_Weighted']
        reduction = (1 - abs(r_w) / abs(r_unw)) * 100 if abs(r_unw) > 1e-6 else 0.0
        status = 'Balanced' if abs(r_w) < 0.1 else 'Imbalanced'

        cov_rows.append(
            f'{cov_label:<23} & {_fmt_r(r_unw)} & {_fmt_r(r_w)} & {reduction:.1f} & {status} \\\\'
        )

    cov_body = '\n'.join(cov_rows)

    # -----------------------------------------------------------------
    # Build the consolidated LaTeX table
    # -----------------------------------------------------------------
    latex_content = (
        r'\begin{table}[H]' '\n'
        r'\centering' '\n'
        r'\caption{npCBPS covariate balance before and after weighting.}' '\n'
        r'\label{tab:npcbps_diagnostics}' '\n'
        r'\small' '\n'
        r'\begin{tabular}{lcccc}' '\n'
        r'\toprule' '\n'
        r'Covariate & $r$ (Unwtd) & $r$ (Wtd) & Red.\ (\%) & Status \\' '\n'
        r'\midrule' '\n'
        f'{cov_body}\n'
        r'\bottomrule' '\n'
        r'\end{tabular}' '\n'
        r'\vspace{4pt}' '\n'
        r'\begin{minipage}{0.85\textwidth}' '\n'
        r'\small' '\n'
        f'\\textit{{Note:}} $N = {n_total}$, '
        f'ESS $= {ess:.1f}$ ({ess_pct:.1f}\\%), '
        f'weight mean (SD) $= {w_mean:.3f}$ ({w_sd:.3f}), '
        f'weight range $= [{w_min:.3f},\\; {w_max:.3f}]$. '
        r'Balance achieved when $|r| < 0.1$. '
        r'ESS $= (\sum w_i)^2 / \sum w_i^2$.' '\n'
        r'\end{minipage}' '\n'
        r'\end{table}' '\n'
    )

    latex_path = tables_dir / 'npcbps_diagnostics.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"Saved: {latex_path}")

    # -----------------------------------------------------------------
    # CSV exports (unchanged – handy for downstream programmatic use)
    # -----------------------------------------------------------------
    table1 = balance_df.copy()
    table1['Status_After'] = table1['Balanced_After'].apply(
        lambda x: 'Balanced' if x else 'Imbalanced'
    )
    table1_display = table1[['Covariate', 'Corr_Unweighted', 'Corr_Weighted',
                             'Reduction_Pct', 'Status_After']].copy()
    table1_display.columns = ['Covariate', 'r (Unweighted)', 'r (Weighted)',
                              'Reduction (%)', 'Status']
    csv_path = tables_dir / 'balance_statistics.csv'
    table1_display.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved: {csv_path}")

    weight_summary = {
        'Statistic': ['N', 'ESS', 'ESS (%)', 'Mean', 'SD', 'Median',
                     'Min', 'Max', 'IQR', '% > 2', '% < 0.5'],
        'Value': [
            f'{n_total:,}', f'{ess:,.1f}', f'{ess_pct:.1f}',
            f'{w_mean:.3f}', f'{w_sd:.3f}', f'{w_median:.3f}',
            f'{w_min:.3f}', f'{w_max:.3f}',
            f'[{w_q25:.2f}, {w_q75:.2f}]',
            f'{pct_gt2:.1f}', f'{pct_lt05:.1f}'
        ]
    }
    table2 = pd.DataFrame(weight_summary)
    csv_path = tables_dir / 'weight_summary.csv'
    table2.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    overall_summary = {
        'Metric': ['Covariates Balanced (|r| < 0.1)', 'Mean |Correlation|',
                  'Max |Correlation|'],
        'Before Weighting': [f'{n_balanced_before}/{n_covs}',
                            f'{mean_abs_before:.3f}', f'{max_abs_before:.3f}'],
        'After Weighting': [f'{n_balanced_after}/{n_covs}',
                           f'{mean_abs_after:.3f}', f'{max_abs_after:.3f}']
    }
    table3 = pd.DataFrame(overall_summary)
    csv_path = tables_dir / 'overall_balance_summary.csv'
    table3.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # P-values for independence tests
    pvalue_table = balance_df[['Covariate', 'P_Unweighted', 'P_Weighted']].copy()
    pvalue_table['Independent_Before'] = pvalue_table['P_Unweighted'].apply(
        lambda p: 'Y' if p > 0.05 else 'N' if not np.isnan(p) else '--'
    )
    pvalue_table['Independent_After'] = pvalue_table['P_Weighted'].apply(
        lambda p: 'Y' if p > 0.05 else 'N' if not np.isnan(p) else '--'
    )
    csv_path = tables_dir / 'independence_tests.csv'
    pvalue_table.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved: {csv_path}")

    # Detailed per-covariate balance
    detailed_balance = balance_df.copy()
    detailed_balance['Abs_Corr_Before'] = np.abs(detailed_balance['Corr_Unweighted'])
    detailed_balance['Abs_Corr_After'] = np.abs(detailed_balance['Corr_Weighted'])
    detailed_balance['Improvement'] = detailed_balance['Abs_Corr_Before'] - detailed_balance['Abs_Corr_After']
    detailed_balance['Pct_Improvement'] = 100 * detailed_balance['Improvement'] / detailed_balance['Abs_Corr_Before']
    csv_path = tables_dir / 'detailed_covariate_balance.csv'
    detailed_balance.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Saved: {csv_path}")

    return table1, table2, table3, pvalue_table, detailed_balance


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Verify covariate balance after npCBPS weighting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Use phi features (default 6)
  %(prog)s --use-pca              # Use PC features (default 6)
  %(prog)s --use-pca --n-phi 3    # Use first 3 PCs
  %(prog)s --n-phi 10             # Use first 10 phi features
        """
    )
    parser.add_argument('--use-pca', action='store_true', default=False,
                        help='Use PC_ columns instead of phi_ columns (default: False)')
    parser.add_argument('--n-phi', '-n', type=int, default=None,
                        help=f'Number of phi/PC features to use (default: {DEFAULT_N_PHI_FEATURES})')
    return parser.parse_args()


def main(use_pca=False, n_phi=None):
    """Run complete balance verification analysis.

    Parameters
    ----------
    use_pca : bool
        If True, use PC_ columns instead of phi_ columns
    n_phi : int, optional
        Number of features to use
    """
    feature_type = "PC" if use_pca else "phi"
    n_features = n_phi if n_phi else DEFAULT_N_PHI_FEATURES

    print("\n" + "="*70)
    print("COVARIATE BALANCE VERIFICATION FOR CAUSAL MEDIATION ANALYSIS")
    print("="*70)
    print(f"Feature type: {feature_type}")
    print(f"Number of features: {n_features}")

    # Create output directories with feature type suffix
    suffix = f"_{feature_type.lower()}{n_features}" if use_pca or n_phi else ""
    # Output directories live under visualizations/ (separate from code)
    figures_dir = PROJECT_ROOT / "visualizations" / "npcbps_balance" / f"figures{suffix}"
    tables_dir = PROJECT_ROOT / "visualizations" / "npcbps_balance" / f"tables{suffix}"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    phi_df, phi_cols = load_data(use_pca=use_pca, n_features=n_phi)

    # Compute balance statistics
    print("\n" + "="*70)
    print("COMPUTING BALANCE STATISTICS")
    print("="*70)

    balance_df = compute_balance_statistics(phi_df, phi_cols)

    print("\nBalance summary:")
    print(f"  Covariates: {len(balance_df)}")
    print(f"  Balanced before (|r| < 0.1): {balance_df['Balanced_Before'].sum()}/{len(balance_df)}")
    print(f"  Balanced after (|r| < 0.1): {balance_df['Balanced_After'].sum()}/{len(balance_df)}")
    print(f"  Mean |r| before: {np.abs(balance_df['Corr_Unweighted']).mean():.4f}")
    print(f"  Mean |r| after: {np.abs(balance_df['Corr_Weighted']).mean():.4f}")

    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    print("\n1. Love plot...")
    plot_love_plot(balance_df, figures_dir)

    print("\n2. Correlation comparison...")
    plot_correlation_comparison(balance_df, figures_dir)

    print("\n3. Weight diagnostics...")
    plot_weight_diagnostics(phi_df, figures_dir)

    print("\n4. Correlation heatmaps...")
    plot_correlation_heatmaps(phi_df, phi_cols, figures_dir)

    print("\n5. Scatter diagnostics...")
    plot_scatter_diagnostics(phi_df, balance_df, phi_cols, figures_dir)

    print("\n6. Residual analysis...")
    plot_residual_analysis(phi_df, phi_cols, figures_dir)

    print("\n7. Permutation tests...")
    plot_permutation_tests(phi_df, balance_df, phi_cols, figures_dir)

    print("\n8. Propensity score analysis...")
    plot_propensity_analysis(phi_df, phi_cols, figures_dir)

    print("\n9. Balance by treatment level...")
    plot_balance_by_treatment_level(phi_df, phi_cols, figures_dir)

    print("\n10. Summary figure...")
    plot_summary_figure(phi_df, balance_df, figures_dir)

    # Generate tables
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)

    tables = generate_tables(phi_df, balance_df, tables_dir)

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    n_balanced_after = balance_df['Balanced_After'].sum()
    n_covs = len(balance_df)

    print(f"\nKey findings:")
    print(f"  Covariates balanced: {n_balanced_after}/{n_covs} ({100*n_balanced_after/n_covs:.1f}%)")
    print(f"  Mean correlation reduction: {balance_df['Reduction_Pct'].mean():.1f}%")
    print(f"  ESS: {compute_effective_sample_size(phi_df['treatment_weight'].values):,.0f}")

    if n_balanced_after >= n_covs * 0.8:
        print("\n  BALANCE ACHIEVED: Conditional independence assumption supported")
    elif n_balanced_after >= n_covs * 0.5:
        print("\n  PARTIAL BALANCE: Some residual imbalance remains")
    else:
        print("\n  IMBALANCE DETECTED: Consider alternative methods")

    print(f"\nFigures saved to: {figures_dir.absolute()}")
    print(f"Tables saved to: {tables_dir.absolute()}")

    # List generated files
    print("\nGenerated figures:")
    for f in sorted(figures_dir.glob('*.png')):
        print(f"  - {f.name}")

    print("\nGenerated tables:")
    for f in sorted(tables_dir.glob('*.csv')):
        print(f"  - {f.name}")
    for f in sorted(tables_dir.glob('*.tex')):
        print(f"  - {f.name}")

    return phi_df, balance_df


if __name__ == "__main__":
    args = parse_args()
    phi_df, balance_df = main(use_pca=args.use_pca, n_phi=args.n_phi)