#!/usr/bin/env python3
"""
Generate publication-quality figures and tables for causal mediation analysis.

Supports:
- Separate outputs per covariate mode (phi vs PCA)
- Separate outputs per meal type (ALL, Breakfast, Lunch, Dinner, Snack)
- Multi-model results (lmer + QR at different quantiles)
- Multiple treatment offsets (15g, 30g, 45g)
- Meal-type comparison tables with double col/row headers
- Covariate-mode comparison tables (phi vs PCA)
- Individual figures for journal abc panel labeling

Outputs:
1. Per-covariate, per-meal figures: ACME, ADE, ATE over time
2. Meal comparison tables with multirow timepoints
3. Covariate comparison tables (phi vs PCA side-by-side)
4. Individual panel figures for journal submission
5. CSV exports with full labeling

Usage:
  python generate_mediation_outputs.py
  python generate_mediation_outputs.py --results path/to/results.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add ae_python_code to path for config
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AE_CODE_DIR = PROJECT_ROOT / "ae_python_code"

if str(AE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_CODE_DIR))

try:
    from config import CONFIG
except ImportError:
    CONFIG = None

# Style settings - increased sizes for presentation quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

COLORS = {
    'ACME': '#2ecc71',  # Green - indirect effect
    'ADE': '#e74c3c',   # Red - direct effect
    'Total': '#3498db',  # Blue - total effect
}

# Tau values to exclude from main paper outputs
# 0.95 excluded: extreme tail quantile adds little interpretive value
# and can be unstable with limited sample sizes.
EXCLUDED_TAUS = [0.95]

# Canonical meal type ordering for consistent output
MEAL_TYPE_ORDER = ['ALL', 'Breakfast', 'Lunch', 'Dinner', 'Snack']

# Display labels for covariate modes
COV_MODE_LABELS = {
    'phi': r'$\varphi$',
    'pca': 'PCA',
}
COV_MODE_LABELS_PLAIN = {
    'phi': 'Phi',
    'pca': 'PCA',
}


def load_results(filepath):
    """Load mediation results CSV."""
    df = pd.read_csv(filepath)
    return df


def _normalize_meal_type(raw_name):
    """Normalize a meal type name by stripping quotes and fixing capitalization."""
    cleaned = raw_name.strip('"\'')
    canonical = {'all': 'ALL', 'breakfast': 'Breakfast', 'lunch': 'Lunch',
                 'dinner': 'Dinner', 'snack': 'Snack'}
    return canonical.get(cleaned.lower(), cleaned)


def _sort_meal_types(meal_types):
    """Sort meal types in canonical order."""
    order = {m: i for i, m in enumerate(MEAL_TYPE_ORDER)}
    return sorted(meal_types, key=lambda x: order.get(x, 99))


def aggregate_results_from_directory(base_dir, covariate_mode='phi'):
    """
    Aggregate mediation results from the nested directory structure.

    New structure: mediation_results/{phi,pca}/{ALL,Breakfast,Lunch,Dinner,Snack}/*.csv
    Also handles quoted directory names (e.g., "Breakfast") from SLURM quoting issues.

    Returns a combined DataFrame with all results, adding meal_type column if not present.
    """
    base_path = Path(base_dir)
    cov_dir = base_path / covariate_mode

    if not cov_dir.exists():
        return None

    all_dfs = []

    # Scan all subdirectories in cov_dir, not just expected names,
    # to handle quoted variants like "Breakfast" or 'ALL'
    for meal_dir in sorted(cov_dir.iterdir()):
        if not meal_dir.is_dir():
            continue

        # Normalize the directory name to a canonical meal type
        meal_type = _normalize_meal_type(meal_dir.name)

        # Find all CSV files in this meal directory
        csv_files = list(meal_dir.glob('mediation_*.csv'))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add meal_type column if not present
                if 'meal_type' not in df.columns:
                    df['meal_type'] = meal_type
                # Add covariate_mode column
                df['covariate_mode'] = covariate_mode
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")

    if not all_dfs:
        return None

    # Combine all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates (same timepoint, model, offset, meal_type)
    key_cols = ['minutes', 'model', 'treat_offset', 'meal_type']
    key_cols = [c for c in key_cols if c in combined.columns]
    if 'quantile_tau' in combined.columns:
        key_cols.append('quantile_tau')

    combined = combined.drop_duplicates(subset=key_cols, keep='last')

    return combined


def _detect_covariate_mode(df, filename=''):
    """Detect covariate mode (phi or pca) from dataframe columns or filename."""
    # Check the use_pca column if present
    if 'use_pca' in df.columns:
        if df['use_pca'].any():
            return 'pca'
        return 'phi'
    # Check for phi/pca columns in the data
    phi_cols = [c for c in df.columns if c.startswith('phi_')]
    pca_cols = [c for c in df.columns if c.startswith('PC_')]
    if pca_cols and not phi_cols:
        return 'pca'
    if phi_cols and not pca_cols:
        return 'phi'
    # Fall back to filename
    fname_lower = str(filename).lower()
    if '_pca_' in fname_lower:
        return 'pca'
    if '_phi_' in fname_lower:
        return 'phi'
    return 'phi'  # default


def find_and_load_all_results(mediation_dir):
    """
    Find and load all mediation results.

    Handles multiple directory layouts:
    1. Nested:  mediation_results/{phi,pca}/{ALL,Breakfast,...}/mediation_*.csv
    2. Cov-dir: mediation_results/{phi,pca}/mediation_*.csv
    3. Flat:    mediation_results/mediation_*.csv  (covariate mode inferred from filename/data)
    """
    mediation_path = Path(mediation_dir)
    all_results = []

    # --- Strategy 1 & 2: Look inside phi/ and pca/ subdirectories ---
    for cov_mode in ['phi', 'pca']:
        cov_dir = mediation_path / cov_mode
        if not cov_dir.exists():
            continue

        # Check if using nested structure (has meal subdirectories)
        # Also check for quoted variants like "Breakfast" from SLURM quoting issues
        meal_names = ['ALL', 'Breakfast', 'Lunch', 'Dinner', 'Snack']
        has_nested = any((cov_dir / m).exists() for m in meal_names)
        if not has_nested:
            # Check for any subdirectory whose name (stripped of quotes) matches a meal type
            has_nested = any(
                _normalize_meal_type(d.name) in meal_names
                for d in cov_dir.iterdir() if d.is_dir()
            ) if cov_dir.exists() else False

        if has_nested:
            # Nested structure: phi/{ALL,Breakfast,...}/mediation_*.csv
            df = aggregate_results_from_directory(mediation_path, cov_mode)
            if df is not None:
                all_results.append(df)
                print(f"  Loaded {len(df)} results from {cov_mode}/ (nested structure)")
        else:
            # Flat cov-dir structure: phi/mediation_*.csv
            combined_file = cov_dir / 'mediation_all_timepoints_all.csv'
            if combined_file.exists():
                df = pd.read_csv(combined_file)
                df['covariate_mode'] = cov_mode
                all_results.append(df)
                print(f"  Loaded {len(df)} results from {combined_file.name}")
            else:
                csv_files = list(cov_dir.glob('mediation_*.csv'))
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        df['covariate_mode'] = cov_mode
                        all_results.append(df)
                    except Exception as e:
                        print(f"Warning: Could not load {csv_file}: {e}")
                if csv_files:
                    print(f"  Loaded results from {len(csv_files)} files in {cov_mode}/")

    # --- Strategy 3: Flat structure - files directly in mediation_results/ ---
    if not all_results and mediation_path.exists():
        # Look for combined all-timepoints files first (preferred)
        combined_files = sorted(mediation_path.glob('mediation_all_timepoints_*.csv'))
        if combined_files:
            for csv_file in combined_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'covariate_mode' not in df.columns:
                        df['covariate_mode'] = _detect_covariate_mode(df, csv_file.name)
                    # Infer meal_type from filename if not in data
                    if 'meal_type' not in df.columns:
                        # Pattern: mediation_all_timepoints_{meal}.csv or
                        #          mediation_all_timepoints_{meal}_offset{N}g.csv
                        stem = csv_file.stem  # e.g. mediation_all_timepoints_breakfast_offset30g
                        parts = stem.replace('mediation_all_timepoints_', '')
                        meal_part = parts.split('_offset')[0]  # e.g. "breakfast" or "all"
                        df['meal_type'] = meal_part.upper() if meal_part.lower() == 'all' else meal_part.capitalize()
                    all_results.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")
            if all_results:
                print(f"  Loaded results from {len(combined_files)} combined files in {mediation_path.name}/")
        else:
            # Fall back to individual per-timepoint files
            csv_files = sorted(mediation_path.glob('mediation_*.csv'))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'covariate_mode' not in df.columns:
                        df['covariate_mode'] = _detect_covariate_mode(df, csv_file.name)
                    all_results.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")
            if csv_files:
                print(f"  Loaded results from {len(csv_files)} files in {mediation_path.name}/")

    if not all_results:
        return None

    combined = pd.concat(all_results, ignore_index=True)

    # Normalize meal_type values (strip quotes from SLURM quoting issues)
    if 'meal_type' in combined.columns:
        combined['meal_type'] = combined['meal_type'].apply(
            lambda x: _normalize_meal_type(str(x)) if pd.notna(x) else x
        )

    # Remove duplicates (same timepoint, model, offset, meal_type, covariate_mode)
    key_cols = [c for c in ['minutes', 'model', 'treat_offset', 'meal_type', 'covariate_mode']
                if c in combined.columns]
    if 'quantile_tau' in combined.columns:
        key_cols.append('quantile_tau')
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols, keep='last')

    return combined


# =============================================================================
# Formatting helpers
# =============================================================================

def _format_ci(lower, upper):
    """Format confidence interval as (lower, upper)."""
    return f"({lower:.2f}, {upper:.2f})"


def _format_p(p, bold=False):
    """Format p-value for LaTeX, optionally bold."""
    if p < 0.001:
        txt = "$<$0.001"
    elif p < 0.01:
        txt = f"{p:.3f}"
    else:
        txt = f"{p:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def _format_est(est, bold=False):
    """Format point estimate, optionally bold."""
    txt = f"{est:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def _format_ci_bold(lower, upper, bold=False):
    """Format confidence interval, optionally bold."""
    txt = f"({lower:.2f}, {upper:.2f})"
    return rf"\textbf{{{txt}}}" if bold else txt


def _format_est_ci_star(est, lower, upper, p):
    """Format estimate with CI and significance star, bold if significant."""
    sig = p < 0.05
    star = '*' if sig else ''
    txt = f"{est:.2f}{star} ({lower:.2f}, {upper:.2f})"
    return rf"\textbf{{{txt}}}" if sig else txt


# =============================================================================
# Plotting helpers
# =============================================================================

def _plot_effects_on_ax(ax, df_success, show_legend=True):
    """Plot ACME, ADE, Total Effect with significance stars on a single axes."""
    df_success = df_success.sort_values('minutes')
    minutes = df_success['minutes'].values

    # Separate significant and non-significant points
    acme_sig = df_success['ACME_p'] < 0.05
    ade_sig = df_success['ADE_p'] < 0.05
    total_sig = df_success['total_p'] < 0.05

    # Plot ACME line and confidence band
    ax.plot(minutes, df_success['ACME'], '-', color=COLORS['ACME'],
            label='ACME (Indirect)', linewidth=2)
    ax.fill_between(minutes, df_success['ACME_lower'], df_success['ACME_upper'],
                    color=COLORS['ACME'], alpha=0.2)
    if acme_sig.any():
        ax.plot(minutes[acme_sig], df_success.loc[acme_sig, 'ACME'], '*',
                color=COLORS['ACME'], markersize=18, markeredgecolor='white', markeredgewidth=0.5)
    if (~acme_sig).any():
        ax.plot(minutes[~acme_sig], df_success.loc[~acme_sig, 'ACME'], 'o',
                color=COLORS['ACME'], markersize=8)

    # Plot ADE line and confidence band
    ax.plot(minutes, df_success['ADE'], '-', color=COLORS['ADE'],
            label='ADE (Direct)', linewidth=2)
    ax.fill_between(minutes, df_success['ADE_lower'], df_success['ADE_upper'],
                    color=COLORS['ADE'], alpha=0.2)
    if ade_sig.any():
        ax.plot(minutes[ade_sig], df_success.loc[ade_sig, 'ADE'], '*',
                color=COLORS['ADE'], markersize=18, markeredgecolor='white', markeredgewidth=0.5)
    if (~ade_sig).any():
        ax.plot(minutes[~ade_sig], df_success.loc[~ade_sig, 'ADE'], 's',
                color=COLORS['ADE'], markersize=8)

    # Plot Total Effect line and confidence band
    ax.plot(minutes, df_success['total_effect'], '-', color=COLORS['Total'],
            label='Total Effect', linewidth=2)
    ax.fill_between(minutes, df_success['total_lower'], df_success['total_upper'],
                    color=COLORS['Total'], alpha=0.2)
    if total_sig.any():
        ax.plot(minutes[total_sig], df_success.loc[total_sig, 'total_effect'], '*',
                color=COLORS['Total'], markersize=18, markeredgecolor='white', markeredgewidth=0.5)
    if (~total_sig).any():
        ax.plot(minutes[~total_sig], df_success.loc[~total_sig, 'total_effect'], '^',
                color=COLORS['Total'], markersize=8)

    # Add reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Minutes post-meal', fontsize=14)
    ax.set_ylabel('Effect on glucose (mg/dL)', fontsize=14)

    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Thin out x-tick labels for 5-min intervals
    if len(minutes) > 10:
        ax.set_xticks(minutes[::6])  # Every 30 min
    else:
        ax.set_xticks(minutes)


# =============================================================================
# Individual figure generation (single panel per file)
# =============================================================================

def plot_individual_effects_figure(df, save_dir, cov_mode, meal_type, model_type='lmer',
                                   offset=30, quantile_tau=None, filename=None):
    """
    Create a single-panel figure for one covariate mode, meal type, model, and offset.

    For QR models, quantile_tau must be specified to select a single quantile.
    """
    df_success = df[df['status'] == 'success'].copy()

    # Filter by covariate mode
    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]

    # Filter by meal type
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]

    # Filter by model
    if 'model' in df_success.columns:
        df_success = df_success[df_success['model'] == model_type]

    # Filter by quantile tau (critical for QR models to avoid overlapping taus)
    if quantile_tau is not None and 'quantile_tau' in df_success.columns:
        df_success = df_success[abs(df_success['quantile_tau'] - quantile_tau) < 0.01]

    # Filter by offset
    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]

    if len(df_success) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_effects_on_ax(ax, df_success)

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
    n_obs = int(df_success['n_obs'].iloc[0]) if 'n_obs' in df_success.columns else 0
    title_parts = [meal_type, cov_label, f'n={n_obs}']
    if model_type == 'qr' and quantile_tau is not None:
        title_parts.append(f'QR tau={quantile_tau:.2f}')
    ax.set_title(f'{" — ".join(title_parts[:2])} ({", ".join(title_parts[2:])})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if filename is None:
        filename = f'fig_effects_{cov_mode}_{meal_type.lower()}_{model_type}_offset{offset}g.png'
    save_path = Path(save_dir) / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_causal_effects_figure(df, save_dir, cov_mode=None, meal_type=None,
                               filename='fig_causal_effects.png'):
    """
    Create publication figure showing ACME, ADE, and Total Effect over time.
    Single panel for a single model configuration.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    df_success = df[df['status'] == 'success'].copy()
    if len(df_success) == 0:
        print("No successful results to plot")
        plt.close()
        return

    # Determine offset for title
    offset = df_success['treat_offset'].iloc[0] if 'treat_offset' in df_success.columns else 30
    model = df_success['model'].iloc[0] if 'model' in df_success.columns else 'lmer'

    _plot_effects_on_ax(ax, df_success)

    title = 'Causal mediation effects across postprandial period'
    if model == 'qr' and 'quantile_tau' in df_success.columns:
        tau = df_success['quantile_tau'].iloc[0]
        title += f' (QR, tau={tau:.2f})'
    # Add context labels
    context_parts = []
    if cov_mode:
        context_parts.append(COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode))
    if meal_type:
        context_parts.append(meal_type)
    if context_parts:
        title += f' — {", ".join(context_parts)}'
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_qr_panel_figure(df, save_dir, offset=30, cov_mode=None, meal_type=None,
                         filename='fig_qr_panel.png'):
    """
    Create panel figure comparing LMER and QR results at different quantiles.
    Each panel is a different model (lmer, qr tau=0.25, 0.50, 0.75).
    """
    df_success = df[(df['status'] == 'success') & (df['treat_offset'] == offset)].copy()
    if len(df_success) == 0:
        print(f"No successful results for offset={offset}g")
        return

    # Identify model configurations present in the data
    panels = []

    # LMER panel
    lmer_df = df_success[df_success['model'] == 'lmer']
    if len(lmer_df) > 0:
        panels.append(('LMER (Mean)', lmer_df))

    # QR panels (sorted by tau, excluding unstable extreme quantiles)
    qr_df = df_success[df_success['model'] == 'qr']
    if len(qr_df) > 0 and 'quantile_tau' in qr_df.columns:
        for tau in sorted(qr_df['quantile_tau'].dropna().unique()):
            # Skip excluded tau values (e.g., 0.95 which is unstable)
            if any(abs(tau - excl) < 0.01 for excl in EXCLUDED_TAUS):
                continue
            tau_df = qr_df[abs(qr_df['quantile_tau'] - tau) < 0.01]
            if len(tau_df) > 0:
                panels.append((f'QR (tau={tau:.2f})', tau_df))

    if len(panels) == 0:
        print("No model panels to plot")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for i, (title, panel_df) in enumerate(panels):
        _plot_effects_on_ax(axes[i], panel_df, show_legend=(i == 0))
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        if i > 0:
            axes[i].set_ylabel('')

    context_parts = []
    if cov_mode:
        context_parts.append(COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode))
    if meal_type:
        context_parts.append(meal_type)
    context_str = f' — {", ".join(context_parts)}' if context_parts else ''
    fig.suptitle(f'Causal mediation effects (+{offset}g carbs){context_str}',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_offset_comparison_figure(df, save_dir, model_type='lmer', quantile_tau=None,
                                   cov_mode=None, meal_type=None,
                                   filename='fig_offset_comparison.png'):
    """
    Create panel figure comparing results across treatment offsets.
    Each panel is a different offset (e.g., +15g, +30g, +45g).
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == model_type)].copy()

    if model_type == 'qr' and quantile_tau is not None:
        df_success = df_success[abs(df_success['quantile_tau'] - quantile_tau) < 0.01]

    if len(df_success) == 0:
        print(f"No successful results for model={model_type}")
        return

    offsets = sorted(df_success['treat_offset'].unique())
    if len(offsets) <= 1:
        print("Only one offset found, skipping offset comparison")
        return

    n_panels = len(offsets)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for i, offset in enumerate(offsets):
        offset_df = df_success[df_success['treat_offset'] == offset]
        _plot_effects_on_ax(axes[i], offset_df, show_legend=(i == 0))
        axes[i].set_title(f'+{offset}g carbs', fontsize=14, fontweight='bold')
        if i > 0:
            axes[i].set_ylabel('')

    model_label = 'LMER' if model_type == 'lmer' else f'QR (tau={quantile_tau:.2f})'
    context_parts = [model_label]
    if cov_mode:
        context_parts.append(COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode))
    if meal_type:
        context_parts.append(meal_type)
    fig.suptitle(f'Causal mediation effects by treatment offset ({", ".join(context_parts)})',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / filename
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_diagnostics_figure(df, save_dir, cov_mode=None, meal_type=None,
                            filename='fig_model_diagnostics.png'):
    """
    Plot model diagnostics: residual statistics across timepoints.
    Shows skewness, kurtosis, and Shapiro-Wilk p-values for both models.
    """
    df_success = df[df['status'] == 'success'].copy()

    if cov_mode and 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if meal_type and 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]

    # Check if diagnostic columns exist
    diag_cols = ['m_resid_skew', 'm_resid_kurt', 'm_shapiro_p',
                 'y_resid_skew', 'y_resid_kurt', 'y_shapiro_p']
    if not all(col in df_success.columns for col in diag_cols):
        print("Diagnostic columns not found in results, skipping diagnostics figure")
        return

    if len(df_success) == 0:
        return

    # If multiple models/offsets, just use first config
    if 'model' in df_success.columns:
        model = df_success['model'].iloc[0]
        df_success = df_success[df_success['model'] == model]
    if 'treat_offset' in df_success.columns:
        offset = df_success['treat_offset'].iloc[0]
        df_success = df_success[df_success['treat_offset'] == offset]

    df_success = df_success.sort_values('minutes')
    minutes = df_success['minutes'].values

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Mediator model diagnostics
    axes[0, 0].plot(minutes, df_success['m_resid_skew'], 'o-', color='#2c3e50')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Mediator: residual skewness')
    axes[0, 0].set_ylabel('Skewness')

    axes[0, 1].plot(minutes, df_success['m_resid_kurt'], 'o-', color='#2c3e50')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Mediator: residual kurtosis')
    axes[0, 1].set_ylabel('Excess kurtosis')

    axes[0, 2].plot(minutes, df_success['m_shapiro_p'], 'o-', color='#2c3e50')
    axes[0, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[0, 2].set_title('Mediator: Shapiro-Wilk p-value')
    axes[0, 2].set_ylabel('p-value')
    axes[0, 2].legend()

    # Outcome model diagnostics
    axes[1, 0].plot(minutes, df_success['y_resid_skew'], 's-', color='#8e44ad')
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Outcome: residual skewness')
    axes[1, 0].set_ylabel('Skewness')
    axes[1, 0].set_xlabel('Minutes post-meal')

    axes[1, 1].plot(minutes, df_success['y_resid_kurt'], 's-', color='#8e44ad')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Outcome: residual kurtosis')
    axes[1, 1].set_ylabel('Excess kurtosis')
    axes[1, 1].set_xlabel('Minutes post-meal')

    axes[1, 2].plot(minutes, df_success['y_shapiro_p'], 's-', color='#8e44ad')
    axes[1, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    axes[1, 2].set_title('Outcome: Shapiro-Wilk p-value')
    axes[1, 2].set_ylabel('p-value')
    axes[1, 2].set_xlabel('Minutes post-meal')
    axes[1, 2].legend()

    for ax_row in axes:
        for ax in ax_row:
            ax.grid(True, alpha=0.3)
            if len(minutes) > 10:
                ax.set_xticks(minutes[::6])

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, '') if cov_mode else ''
    meal_label = meal_type if meal_type else ''
    subtitle_parts = [p for p in [cov_label, meal_label] if p]
    subtitle = f' ({", ".join(subtitle_parts)})' if subtitle_parts else ''
    fig.suptitle(f'Model diagnostics across timepoints{subtitle}', fontsize=18, fontweight='bold')
    plt.tight_layout()

    save_path = Path(save_dir) / filename
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path.name}")


# =============================================================================
# Table generation: per covariate mode, per meal type
# =============================================================================

def generate_per_meal_effects_table_latex(df, save_dir, cov_mode, meal_type, model_type='lmer',
                                          offset=30, filename=None):
    """
    Generate a LaTeX table for one covariate mode, meal type, model, and offset.
    Shows effects at summary timepoints (30-min intervals).
    """
    df_success = df[df['status'] == 'success'].copy()

    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]
    if 'model' in df_success.columns:
        df_success = df_success[df_success['model'] == model_type]
    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]

    df_success = df_success.sort_values('minutes')
    summary_timepoints = [60, 90, 120, 150, 180, 210]
    summary_df = df_success[df_success['minutes'].isin(summary_timepoints)]

    if len(summary_df) == 0:
        return

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
    n_obs = int(summary_df['n_obs'].iloc[0])

    rows = []
    for _, row in summary_df.iterrows():
        acme_sig = row['ACME_p'] < 0.05
        ade_sig = row['ADE_p'] < 0.05
        total_sig = row['total_p'] < 0.05
        row_str = (
            f"{int(row['minutes'])} min & "
            f"{_format_est(row['ACME'], acme_sig)} & {_format_ci_bold(row['ACME_lower'], row['ACME_upper'], acme_sig)} & {_format_p(row['ACME_p'], acme_sig)} & "
            f"{_format_est(row['ADE'], ade_sig)} & {_format_ci_bold(row['ADE_lower'], row['ADE_upper'], ade_sig)} & {_format_p(row['ADE_p'], ade_sig)} & "
            f"{_format_est(row['total_effect'], total_sig)} & {_format_ci_bold(row['total_lower'], row['total_upper'], total_sig)} & {_format_p(row['total_p'], total_sig)} \\\\"
        )
        rows.append(row_str)

    inference_method = 'quasi-Bayesian approximation with 1000 Monte Carlo simulations' if model_type == 'lmer' else 'bootstrap'

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{""" + cov_label + r""" covariate --- """ + meal_type + r""" meals: causal mediation effects ($N = """ + str(n_obs) + r"""$, +""" + str(offset) + r"""g treatment).}
Time = postprandial measurement time point in minutes after meal start.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin);
Total = ACME + ADE.
Est.\ = point estimate (mg/dL);
95\% CI from """ + inference_method + r""";
$p$ = two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:effects_""" + cov_mode + r"""_""" + meal_type.lower() + r"""_""" + model_type + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{ACME} & \multicolumn{3}{c}{ADE} & \multicolumn{3}{c}{Total} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Time & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_effects_{cov_mode}_{meal_type.lower()}_{model_type}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


def generate_full_timepoints_table_latex(df, save_dir, cov_mode, meal_type, model_type='lmer',
                                         offset=30, filename=None):
    """
    Generate a full (5-min intervals) LaTeX table for appendix.
    One covariate mode, meal type, model, and offset.
    """
    df_success = df[df['status'] == 'success'].copy()

    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]
    if 'model' in df_success.columns:
        df_success = df_success[df_success['model'] == model_type]
    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]

    df_success = df_success.sort_values('minutes')

    if len(df_success) == 0:
        return

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
    n_obs = int(df_success['n_obs'].iloc[0])

    rows = []
    for _, row in df_success.iterrows():
        acme_sig = row['ACME_p'] < 0.05
        ade_sig = row['ADE_p'] < 0.05
        total_sig = row['total_p'] < 0.05
        row_str = (
            f"{int(row['minutes'])} min & "
            f"{_format_est(row['ACME'], acme_sig)} & {_format_ci_bold(row['ACME_lower'], row['ACME_upper'], acme_sig)} & {_format_p(row['ACME_p'], acme_sig)} & "
            f"{_format_est(row['ADE'], ade_sig)} & {_format_ci_bold(row['ADE_lower'], row['ADE_upper'], ade_sig)} & {_format_p(row['ADE_p'], ade_sig)} & "
            f"{_format_est(row['total_effect'], total_sig)} & {_format_ci_bold(row['total_lower'], row['total_upper'], total_sig)} & {_format_p(row['total_p'], total_sig)} \\\\"
        )
        rows.append(row_str)

    inference_method = 'quasi-Bayesian approximation with 1000 Monte Carlo simulations' if model_type == 'lmer' else 'bootstrap'

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{""" + cov_label + r""" covariate --- """ + meal_type + r""" meals: causal mediation effects at 5-minute intervals ($N = """ + str(n_obs) + r"""$, +""" + str(offset) + r"""g treatment).}
Time = postprandial measurement time point in minutes after meal start (reported at 5-minute intervals).
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin);
Total = ACME + ADE.
Est.\ = point estimate (mg/dL);
95\% CI from """ + inference_method + r""";
$p$ = two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:effects_full_""" + cov_mode + r"""_""" + meal_type.lower() + r"""_""" + model_type + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
& \multicolumn{3}{c}{ACME} & \multicolumn{3}{c}{ADE} & \multicolumn{3}{c}{Total} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
Time & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_effects_full_{cov_mode}_{meal_type.lower()}_{model_type}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# Meal comparison table: double col/row headers, one per covariate mode
# =============================================================================

def generate_meal_comparison_table_latex(df, save_dir, cov_mode, model_type='lmer',
                                          offset=30, filename=None):
    """
    Generate a publication-quality LaTeX table comparing causal effects across meal types.

    Structure uses double column headers:
      - Top level: Meal types (Breakfast, Lunch, Dinner, Snack)
      - Second level: ACME, ADE for each meal type
      - Rows: multirow timepoints

    Filters to a single covariate mode so phi and pca are never mixed.
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == model_type)].copy()

    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]
    if 'meal_type' not in df_success.columns or len(df_success) == 0:
        return

    # Get meal types excluding ALL, in canonical order
    meal_types = [m for m in df_success['meal_type'].unique() if m.upper() != 'ALL']
    meal_types = _sort_meal_types(meal_types)

    if len(meal_types) == 0:
        return

    summary_timepoints = [60, 90, 120, 150, 180, 210]
    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)

    rows = []
    for minutes in summary_timepoints:
        row_parts = [f"{int(minutes)} min"]
        for meal in meal_types:
            meal_time_df = df_success[(df_success['meal_type'] == meal) &
                                       (df_success['minutes'] == minutes)]
            if len(meal_time_df) > 0:
                r = meal_time_df.iloc[0]
                acme_str = _format_est_ci_star(r['ACME'], r['ACME_lower'], r['ACME_upper'], r['ACME_p'])
                ade_str = _format_est_ci_star(r['ADE'], r['ADE_lower'], r['ADE_upper'], r['ADE_p'])
                row_parts.extend([acme_str, ade_str])
            else:
                row_parts.extend(['--', '--'])
        rows.append(" & ".join(row_parts) + r" \\")

    # Build header with meal types
    n_meals = len(meal_types)
    header_meals = " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in meal_types])
    cmidrules = " ".join([f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(n_meals)])
    subheader = "Time & " + " & ".join(["ACME & ADE"] * n_meals) + r" \\"

    # Get sample sizes per meal
    n_obs_parts = []
    for m in meal_types:
        m_df = df_success[df_success['meal_type'] == m]
        if len(m_df) > 0:
            n_obs_parts.append(f"{m}: $N={int(m_df['n_obs'].iloc[0])}$")
    n_obs_str = ", ".join(n_obs_parts)

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Meal-type comparison of causal mediation effects (""" + cov_label + r""" covariates, """ + model_type.upper() + r""", +""" + str(offset) + r"""g treatment).}
Sample sizes: """ + n_obs_str + r""".
Time = postprandial measurement time point in minutes after meal start.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin).
Point estimates (mg/dL) with 95\% CI in parentheses.
* indicates $p < 0.05$, where $p$ is the two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:meal_comparison_""" + cov_mode + r"""_""" + model_type + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l""" + "cc" * n_meals + r"""}
\toprule
& """ + header_meals + r""" \\
""" + cmidrules + r"""
""" + subheader + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_meal_comparison_{cov_mode}_{model_type}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# Covariate comparison table: phi vs PCA side-by-side for each meal type
# =============================================================================

def generate_covariate_comparison_table_latex(df, save_dir, meal_type, model_type='lmer',
                                               offset=30, filename=None):
    """
    Generate a publication-quality LaTeX table comparing phi vs PCA for a single meal type.

    Structure uses double column/row headers:
      - Top level: Phi, PCA
      - Second level: ACME (Est, 95% CI, p), ADE (Est, 95% CI, p)
      - Rows: timepoints
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == model_type)].copy()

    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]
    if 'covariate_mode' not in df_success.columns or len(df_success) == 0:
        return

    cov_modes = sorted(df_success['covariate_mode'].unique())
    if len(cov_modes) < 2:
        return

    summary_timepoints = [60, 90, 120, 150, 180, 210]

    rows = []
    for minutes in summary_timepoints:
        row_parts = [f"{int(minutes)} min"]
        for cov_mode in cov_modes:
            cov_time_df = df_success[(df_success['covariate_mode'] == cov_mode) &
                                      (df_success['minutes'] == minutes)]
            if len(cov_time_df) > 0:
                r = cov_time_df.iloc[0]
                acme_sig = r['ACME_p'] < 0.05
                ade_sig = r['ADE_p'] < 0.05
                row_parts.extend([
                    _format_est(r['ACME'], acme_sig),
                    _format_ci_bold(r['ACME_lower'], r['ACME_upper'], acme_sig),
                    _format_p(r['ACME_p'], acme_sig),
                    _format_est(r['ADE'], ade_sig),
                    _format_ci_bold(r['ADE_lower'], r['ADE_upper'], ade_sig),
                    _format_p(r['ADE_p'], ade_sig),
                ])
            else:
                row_parts.extend(['--'] * 6)
        rows.append(" & ".join(row_parts) + r" \\")

    # Build header
    n_cov = len(cov_modes)
    cov_labels = [COV_MODE_LABELS_PLAIN.get(c, c) for c in cov_modes]

    # Top-level: covariate modes spanning 6 columns each (ACME: Est, CI, p; ADE: Est, CI, p)
    header_top = " & ".join([f"\\multicolumn{{6}}{{c}}{{{label}}}" for label in cov_labels])
    cmidrules_top = " ".join([f"\\cmidrule(lr){{{2+i*6}-{7+i*6}}}" for i in range(n_cov)])

    # Second level: ACME and ADE within each covariate mode
    header_mid_parts = []
    cmidrules_mid_parts = []
    for i in range(n_cov):
        base_col = 2 + i * 6
        header_mid_parts.append(f"\\multicolumn{{3}}{{c}}{{ACME}} & \\multicolumn{{3}}{{c}}{{ADE}}")
        cmidrules_mid_parts.append(f"\\cmidrule(lr){{{base_col}-{base_col+2}}} \\cmidrule(lr){{{base_col+3}-{base_col+5}}}")
    header_mid = " & ".join(header_mid_parts)
    cmidrules_mid = " ".join(cmidrules_mid_parts)

    # Third level: Est, CI, p repeated
    header_bottom = "Time & " + " & ".join(["Est. & 95\\% CI & $p$"] * (2 * n_cov)) + r" \\"

    # Sample sizes
    n_obs_parts = []
    for cov_mode in cov_modes:
        cov_df = df_success[df_success['covariate_mode'] == cov_mode]
        if len(cov_df) > 0:
            label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
            n_obs_parts.append(f"{label}: $N={int(cov_df['n_obs'].iloc[0])}$")
    n_obs_str = ", ".join(n_obs_parts)

    col_spec = "l" + "cccccc" * n_cov

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Covariate comparison (Phi vs PCA) for """ + meal_type + r""" meals (""" + model_type.upper() + r""", +""" + str(offset) + r"""g treatment).}
Sample sizes: """ + n_obs_str + r""".
Time = postprandial measurement time point in minutes after meal start.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin).
Est.\ = point estimate (mg/dL); 95\% CI in parentheses;
$p$ = two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:cov_comparison_""" + meal_type.lower() + r"""_""" + model_type + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
& """ + header_top + r""" \\
""" + cmidrules_top + r"""
& """ + header_mid + r""" \\
""" + cmidrules_mid + r"""
""" + header_bottom + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_cov_comparison_{meal_type.lower()}_{model_type}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# Combined meal x covariate comparison table
# =============================================================================

def generate_combined_comparison_table_latex(df, save_dir, model_type='lmer',
                                              offset=30, filename=None):
    """
    Generate a large comparison table: rows = timepoint x covariate mode,
    columns = meal types. Uses multirow for timepoints and a sub-row for each
    covariate mode.

    Structure:
      Time | Cov  | Breakfast ACME | Breakfast ADE | Lunch ACME | Lunch ADE | ...
      60   | Phi  | ...            | ...           | ...        | ...
           | PCA  | ...            | ...           | ...        | ...
      90   | Phi  | ...
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == model_type)].copy()

    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]

    if 'meal_type' not in df_success.columns or 'covariate_mode' not in df_success.columns:
        return
    if len(df_success) == 0:
        return

    meal_types = [m for m in df_success['meal_type'].unique() if m.upper() != 'ALL']
    meal_types = _sort_meal_types(meal_types)
    cov_modes = sorted(df_success['covariate_mode'].unique())

    if len(meal_types) == 0 or len(cov_modes) < 2:
        return

    summary_timepoints = [60, 90, 120, 150, 180, 210]

    rows = []
    for t_idx, minutes in enumerate(summary_timepoints):
        for c_idx, cov_mode in enumerate(cov_modes):
            cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)

            if c_idx == 0:
                time_cell = f"\\multirow{{{len(cov_modes)}}}{{*}}{{{int(minutes)} min}}"
            else:
                time_cell = ""

            row_parts = [time_cell, cov_label]
            for meal in meal_types:
                cell_df = df_success[
                    (df_success['meal_type'] == meal) &
                    (df_success['minutes'] == minutes) &
                    (df_success['covariate_mode'] == cov_mode)
                ]
                if len(cell_df) > 0:
                    r = cell_df.iloc[0]
                    acme_str = _format_est_ci_star(r['ACME'], r['ACME_lower'], r['ACME_upper'], r['ACME_p'])
                    ade_str = _format_est_ci_star(r['ADE'], r['ADE_lower'], r['ADE_upper'], r['ADE_p'])
                    row_parts.extend([acme_str, ade_str])
                else:
                    row_parts.extend(['--', '--'])
            rows.append(" & ".join(row_parts) + r" \\")

        # Add midrule between timepoint groups (not after last)
        if t_idx < len(summary_timepoints) - 1:
            rows.append("\\midrule")

    # Build header
    n_meals = len(meal_types)
    header_meals = " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in meal_types])
    cmidrules = " ".join([f"\\cmidrule(lr){{{3+i*2}-{4+i*2}}}" for i in range(n_meals)])
    subheader = "Time & Cov. & " + " & ".join(["ACME & ADE"] * n_meals) + r" \\"

    col_spec = "ll" + "cc" * n_meals

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Causal mediation effects: meal type $\times$ covariate comparison (""" + model_type.upper() + r""", +""" + str(offset) + r"""g treatment).}
Time = postprandial measurement time point in minutes after meal start.
Rows group timepoints with sub-rows for each covariate type (Phi = learned autoencoder embeddings; PCA = principal components).
Cov.\ = covariate mode used to represent glucose dynamics.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin).
Point estimates (mg/dL) with 95\% CI in parentheses.
* indicates $p < 0.05$, where $p$ is the two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:combined_comparison_""" + model_type + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + col_spec + r"""}
\toprule
& & """ + header_meals + r""" \\
""" + cmidrules + r"""
""" + subheader + r"""
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_combined_comparison_{model_type}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# LMER multi-offset table (per covariate mode + meal type)
# =============================================================================

def generate_lmer_offset_table_latex(df, save_dir, cov_mode, meal_type, filename=None):
    """
    Generate LMER table with multirow timepoints and sub-rows per offset.
    Filters by covariate mode and meal type.
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == 'lmer')].copy()

    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]

    if len(df_success) == 0:
        return

    offsets = sorted(df_success['treat_offset'].unique()) if 'treat_offset' in df_success.columns else []
    if len(offsets) <= 1:
        return

    summary_timepoints = [60, 90, 120, 150, 180, 210]
    summary_df = df_success[df_success['minutes'].isin(summary_timepoints)]
    if len(summary_df) == 0:
        return

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
    n_obs = int(summary_df['n_obs'].iloc[0])

    rows = []
    for t_idx, minutes in enumerate(summary_timepoints):
        time_data = summary_df[summary_df['minutes'] == minutes]
        if len(time_data) == 0:
            continue

        valid_offsets = [o for o in offsets if len(time_data[time_data['treat_offset'] == o]) > 0]

        for off_idx, offset in enumerate(offsets):
            offset_row = time_data[time_data['treat_offset'] == offset]
            if len(offset_row) == 0:
                continue
            row = offset_row.iloc[0]

            if off_idx == 0:
                time_label = f"\\multirow{{{len(valid_offsets)}}}{{*}}{{{int(minutes)} min}}"
            else:
                time_label = ""

            acme_sig = row['ACME_p'] < 0.05
            ade_sig = row['ADE_p'] < 0.05
            total_sig = row['total_p'] < 0.05
            row_str = (
                f"{time_label} & +{offset}g & "
                f"{_format_est(row['ACME'], acme_sig)} & {_format_ci_bold(row['ACME_lower'], row['ACME_upper'], acme_sig)} & {_format_p(row['ACME_p'], acme_sig)} & "
                f"{_format_est(row['ADE'], ade_sig)} & {_format_ci_bold(row['ADE_lower'], row['ADE_upper'], ade_sig)} & {_format_p(row['ADE_p'], ade_sig)} & "
                f"{_format_est(row['total_effect'], total_sig)} & {_format_ci_bold(row['total_lower'], row['total_upper'], total_sig)} & {_format_p(row['total_p'], total_sig)} \\\\"
            )
            rows.append(row_str)

        if t_idx < len(summary_timepoints) - 1:
            rows.append("\\midrule")

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{""" + cov_label + r""" covariate --- """ + meal_type + r""" meals: LMER causal mediation effects across treatment doses ($N = """ + str(n_obs) + r"""$).}
Time = postprandial measurement time point in minutes after meal start.
Dose = hypothetical increase in carbohydrate intake (grams) above the meal-type-specific median.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin);
Total = ACME + ADE.
Est.\ = point estimate (mg/dL); 95\% CI from quasi-Bayesian approximation with 1000 Monte Carlo simulations;
$p$ = two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:lmer_offsets_""" + cov_mode + r"""_""" + meal_type.lower() + r"""}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
& & \multicolumn{3}{c}{ACME} & \multicolumn{3}{c}{ADE} & \multicolumn{3}{c}{Total} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
Time & Dose & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_lmer_offsets_{cov_mode}_{meal_type.lower()}.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# QR multi-tau table (per covariate mode + meal type)
# =============================================================================

def generate_qr_tau_table_latex(df, save_dir, cov_mode, meal_type, offset=30, filename=None):
    """
    Generate QR table with multirow timepoints and sub-rows per tau.
    Filters by covariate mode and meal type.
    """
    df_success = df[(df['status'] == 'success') & (df['model'] == 'qr')].copy()

    if 'covariate_mode' in df_success.columns:
        df_success = df_success[df_success['covariate_mode'] == cov_mode]
    if 'meal_type' in df_success.columns:
        df_success = df_success[df_success['meal_type'] == meal_type]
    if 'treat_offset' in df_success.columns:
        df_success = df_success[df_success['treat_offset'] == offset]

    if len(df_success) == 0 or 'quantile_tau' not in df_success.columns:
        return

    all_taus = sorted(df_success['quantile_tau'].dropna().unique())
    taus = [t for t in all_taus if not any(abs(t - excl) < 0.01 for excl in EXCLUDED_TAUS)]

    if len(taus) == 0:
        return

    summary_timepoints = [60, 90, 120, 150, 180, 210]
    summary_df = df_success[df_success['minutes'].isin(summary_timepoints)]
    if len(summary_df) == 0:
        return

    cov_label = COV_MODE_LABELS_PLAIN.get(cov_mode, cov_mode)
    n_obs = int(summary_df['n_obs'].iloc[0])

    rows = []
    for t_idx, minutes in enumerate(summary_timepoints):
        time_data = summary_df[summary_df['minutes'] == minutes]
        if len(time_data) == 0:
            continue

        valid_taus = [tau for tau in taus if len(time_data[abs(time_data['quantile_tau'] - tau) < 0.01]) > 0]

        for tau_idx, tau in enumerate(taus):
            tau_row = time_data[abs(time_data['quantile_tau'] - tau) < 0.01]
            if len(tau_row) == 0:
                continue
            row = tau_row.iloc[0]

            if tau_idx == 0:
                time_label = f"\\multirow{{{len(valid_taus)}}}{{*}}{{{int(minutes)} min}}"
            else:
                time_label = ""

            acme_sig = row['ACME_p'] < 0.05
            ade_sig = row['ADE_p'] < 0.05
            total_sig = row['total_p'] < 0.05
            row_str = (
                f"{time_label} & $\\tau={tau:.2f}$ & "
                f"{_format_est(row['ACME'], acme_sig)} & {_format_ci_bold(row['ACME_lower'], row['ACME_upper'], acme_sig)} & {_format_p(row['ACME_p'], acme_sig)} & "
                f"{_format_est(row['ADE'], ade_sig)} & {_format_ci_bold(row['ADE_lower'], row['ADE_upper'], ade_sig)} & {_format_p(row['ADE_p'], ade_sig)} & "
                f"{_format_est(row['total_effect'], total_sig)} & {_format_ci_bold(row['total_lower'], row['total_upper'], total_sig)} & {_format_p(row['total_p'], total_sig)} \\\\"
            )
            rows.append(row_str)

        if t_idx < len(summary_timepoints) - 1:
            rows.append("\\midrule")

    latex = r"""\begin{table*}[ht]
\centering
\caption{\textbf{""" + cov_label + r""" covariate --- """ + meal_type + r""" meals: QR causal mediation effects ($N = """ + str(n_obs) + r"""$, +""" + str(offset) + r"""g treatment).}
Time = postprandial measurement time point in minutes after meal start.
Quantile = conditional quantile ($\tau$) of the glucose response distribution being modeled.
Results shown for quantiles $\tau \in \{""" + ", ".join([f"{t:.2f}" for t in taus]) + r"""\}$.
ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);
ADE = Average Direct Effect (effect not mediated through insulin);
Total = ACME + ADE.
Est.\ = point estimate (mg/dL); 95\% CI from bootstrap;
$p$ = two-sided p-value testing the null hypothesis that the effect equals zero.
Significant results ($p < 0.05$) are shown in \textbf{bold}.}
\label{tab:qr_taus_""" + cov_mode + r"""_""" + meal_type.lower() + r"""_""" + str(offset) + r"""g}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccccc}
\toprule
& & \multicolumn{3}{c}{ACME} & \multicolumn{3}{c}{ADE} & \multicolumn{3}{c}{Total} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
Time & Quantile & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    if filename is None:
        filename = f'table_qr_taus_{cov_mode}_{meal_type.lower()}_offset{offset}g.tex'
    save_path = Path(save_dir) / filename
    with open(save_path, 'w') as f:
        f.write(latex)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# CSV export (with full labeling)
# =============================================================================

def generate_effects_table_csv(df, save_dir, filename='table_causal_effects.csv'):
    """
    Generate CSV table of causal effects with full labeling (covariate_mode, meal_type).
    """
    df_success = df[df['status'] == 'success'].sort_values(
        [c for c in ['covariate_mode', 'meal_type', 'model', 'treat_offset', 'minutes']
         if c in df.columns]
    )

    if len(df_success) == 0:
        print("No successful results for CSV table")
        return

    cols = []
    for c in ['covariate_mode', 'meal_type', 'model', 'quantile_tau', 'treat_offset', 'minutes', 'n_obs']:
        if c in df_success.columns:
            cols.append(c)

    cols += [
        'ACME', 'ACME_lower', 'ACME_upper', 'ACME_p',
        'ADE', 'ADE_lower', 'ADE_upper', 'ADE_p',
        'total_effect', 'total_lower', 'total_upper', 'total_p',
        'prop_mediated',
    ]

    # Add diagnostics columns if present
    diag_cols = ['m_resid_skew', 'm_resid_kurt', 'm_shapiro_p',
                 'y_resid_skew', 'y_resid_kurt', 'y_shapiro_p']
    for col in diag_cols:
        if col in df_success.columns:
            cols.append(col)

    available_cols = [c for c in cols if c in df_success.columns]
    table_df = df_success[available_cols].copy()

    save_path = Path(save_dir) / filename
    table_df.to_csv(save_path, index=False)
    print(f"  Saved: {save_path.name}")


# =============================================================================
# Console summary table (properly grouped)
# =============================================================================

def print_summary_table(df):
    """
    Print a nicely formatted summary table to console.
    Groups by covariate_mode, meal_type, model, quantile_tau, and treat_offset
    so that each section shows results for exactly one combination.
    """
    df_success = df[df['status'] == 'success'].sort_values('minutes')

    if len(df_success) == 0:
        print("No successful results to display")
        return

    def fmt_effect(est, lower, upper):
        return f"{est:>7.2f} [{lower:>7.2f}, {upper:>7.2f}]"

    def fmt_p(p):
        if p < 0.001:
            return "<0.001"
        else:
            return f"{p:.4f}"

    # Build groupby columns: always include covariate_mode and meal_type if present
    group_cols = []
    for c in ['covariate_mode', 'meal_type', 'model', 'quantile_tau', 'treat_offset']:
        if c in df_success.columns:
            group_cols.append(c)

    if not group_cols:
        # No grouping columns at all - treat as single group
        groups = [(None, df_success)]
    else:
        groups = df_success.groupby(group_cols, dropna=False)

    for group_key, group_df in groups:
        group_df = group_df.sort_values('minutes')

        # Build label from group key
        if group_key is None:
            label = "LMER"
        elif isinstance(group_key, tuple):
            parts = []
            for col, val in zip(group_cols, group_key):
                if col == 'covariate_mode':
                    parts.append(f"Cov: {COV_MODE_LABELS_PLAIN.get(val, val)}")
                elif col == 'meal_type':
                    parts.append(f"Meal: {val}")
                elif col == 'model':
                    parts.append(f"Model: {val.upper()}")
                elif col == 'quantile_tau':
                    if pd.notna(val):
                        parts.append(f"tau={val:.2f}")
                elif col == 'treat_offset':
                    parts.append(f"+{int(val)}g")
            label = ", ".join(parts)
        else:
            label = str(group_key)

        print("\n" + "=" * 100)
        print(f"CAUSAL MEDIATION ANALYSIS: {label}")
        print("=" * 100)

        header = f"{'Min':>4} {'N':>4} {'ACME [95% CI]':>28} {'p':>8} {'ADE [95% CI]':>28} {'p':>8} {'%Med':>6}"
        print(header)
        print("-" * 100)

        for _, row in group_df.iterrows():
            acme_str = fmt_effect(row['ACME'], row['ACME_lower'], row['ACME_upper'])
            ade_str = fmt_effect(row['ADE'], row['ADE_lower'], row['ADE_upper'])
            prop_med = f"{row['prop_mediated'] * 100:>5.1f}%" if pd.notna(row.get('prop_mediated')) else "   NA"

            line = f"{int(row['minutes']):>4} {int(row['n_obs']):>4} {acme_str:>28} {fmt_p(row['ACME_p']):>8} {ade_str:>28} {fmt_p(row['ADE_p']):>8} {prop_med}"
            print(line)

        print("-" * 100)

        # Total effect summary
        print(f"\n  Total Effect Summary:")
        print(f"  {'Min':>4} {'Total Effect [95% CI]':>32} {'p':>8}")
        print("  " + "-" * 50)
        for _, row in group_df.iterrows():
            total_str = f"{row['total_effect']:>7.2f} [{row['total_lower']:>7.2f}, {row['total_upper']:>7.2f}]"
            p_str = f"{row['total_p']:.4f}" if row['total_p'] >= 0.001 else "<0.001"
            print(f"  {int(row['minutes']):>4} {total_str:>32} {p_str:>8}")
        print()


# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Generate mediation analysis visualizations.

    This script visualizes causal mediation results (ACME, ADE, Total Effect).
    All outputs are separated by covariate mode (phi vs PCA) and meal type.
    Figures are saved as individual panels for journal abc labeling.
    """
    parser = argparse.ArgumentParser(description='Generate mediation analysis outputs')
    parser.add_argument('--results', '-r', type=str, default=None,
                       help='Path to mediation results CSV')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                       help='Output directory for figures and tables')
    args = parser.parse_args()

    # Set default paths
    project_root = PROJECT_ROOT
    mediation_dir = project_root / 'cma_cluster' / 'mediation_results'

    # Output base directories
    # Output directories live under visualizations/ (separate from code)
    figures_base = PROJECT_ROOT / 'visualizations' / 'mediation_visualizations' / 'figures'
    tables_base = PROJECT_ROOT / 'visualizations' / 'mediation_visualizations' / 'tables'
    figures_base.mkdir(parents=True, exist_ok=True)
    tables_base.mkdir(parents=True, exist_ok=True)

    print(f"Figures base: {figures_base}")
    print(f"Tables base: {tables_base}")
    print()

    # Load results - either from specified file or by aggregating from directory structure
    if args.results is not None:
        print(f"Loading from specified file: {args.results}")
        df = load_results(args.results)
        print(f"Loaded {len(df)} results")
    else:
        # Try to aggregate from directory structure (handles both old and new layouts)
        print(f"Scanning mediation results directory: {mediation_dir}")
        df = find_and_load_all_results(mediation_dir)

        if df is None or len(df) == 0:
            # Fall back to legacy file search
            candidates = [
                mediation_dir / 'phi' / 'mediation_all_timepoints_all.csv',
                mediation_dir / 'pca' / 'mediation_all_timepoints_all.csv',
                mediation_dir / 'mediation_all_timepoints_all.csv',
                mediation_dir / 'mediation_lmer_all_timepoints.csv',
            ]
            for path in candidates:
                if path.exists():
                    df = load_results(str(path))
                    print(f"Loaded {len(df)} results from {path}")
                    break

            # If still nothing, try loading all mediation_all_timepoints_*.csv files
            if (df is None or len(df) == 0) and mediation_dir.exists():
                combined_files = sorted(mediation_dir.glob('mediation_all_timepoints_*.csv'))
                if combined_files:
                    dfs = []
                    for path in combined_files:
                        try:
                            part = pd.read_csv(path)
                            dfs.append(part)
                        except Exception as e:
                            print(f"Warning: Could not load {path}: {e}")
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        print(f"Loaded {len(df)} results from {len(combined_files)} combined files")

        if df is None or len(df) == 0:
            print("ERROR: Could not find any results files.")
            print(f"Searched in: {mediation_dir}")
            print("Specify a file with --results or ensure results exist in the expected locations.")
            sys.exit(1)

        print(f"Total: {len(df)} results loaded")
    print(f"Successful: {(df['status'] == 'success').sum()}")

    # Detect data dimensions
    has_model = 'model' in df.columns
    has_qr = has_model and 'qr' in df['model'].values
    has_multi_offset = 'treat_offset' in df.columns and df['treat_offset'].nunique() > 1
    has_cov_modes = 'covariate_mode' in df.columns and df['covariate_mode'].nunique() > 0
    has_meal_types = 'meal_type' in df.columns and df['meal_type'].nunique() > 0

    cov_modes = sorted(df['covariate_mode'].unique()) if has_cov_modes else ['phi']
    all_meal_types = _sort_meal_types(list(df['meal_type'].unique())) if has_meal_types else ['ALL']
    meal_types_no_all = [m for m in all_meal_types if m.upper() != 'ALL']
    offsets = sorted(df['treat_offset'].unique().tolist()) if 'treat_offset' in df.columns else [30]
    default_offset = offsets[0] if len(offsets) == 1 else (30 if 30 in offsets else offsets[0])

    print(f"Covariate modes: {cov_modes}")
    print(f"Meal types: {all_meal_types}")
    print(f"Models: {df['model'].unique().tolist() if has_model else ['lmer']}")
    if has_multi_offset:
        print(f"Offsets: {offsets}")
    else:
        print(f"Offset: {default_offset}g")
    print()

    # =========================================================================
    # Helper to get output directories for a given covariate mode + meal type
    # Structure: {base}/{cov_mode}/{meal_type}/
    # Cross-cutting comparisons go in {base}/{cov_mode}/ or {base}/comparison/
    # =========================================================================
    def _fig_dir(cov_mode, meal_type):
        d = figures_base / cov_mode / meal_type.lower()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tab_dir(cov_mode, meal_type):
        d = tables_base / cov_mode / meal_type.lower()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fig_cov_dir(cov_mode):
        d = figures_base / cov_mode
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _tab_cov_dir(cov_mode):
        d = tables_base / cov_mode
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _comparison_fig_dir():
        d = figures_base / 'comparison'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _comparison_tab_dir():
        d = tables_base / 'comparison'
        d.mkdir(parents=True, exist_ok=True)
        return d

    # =========================================================================
    # FIGURES
    # =========================================================================
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # 1. Individual effects figures: one per (covariate_mode, meal_type, offset)
    #    Saved to: figures/{cov_mode}/{meal_type}/
    print("\n--- Individual effects figures (for journal panels) ---")
    for cov_mode in cov_modes:
        for meal_type in all_meal_types:
            fig_dir = _fig_dir(cov_mode, meal_type)
            for offset in offsets:
                offset_suffix = f'_offset{offset}g' if has_multi_offset else ''

                # LMER individual figure
                plot_individual_effects_figure(
                    df, fig_dir, cov_mode=cov_mode, meal_type=meal_type,
                    model_type='lmer', offset=offset,
                    filename=f'fig_effects_lmer{offset_suffix}.png'
                )

                # QR individual figures per tau
                if has_qr:
                    cov_meal_df = df[
                        (df['status'] == 'success') &
                        (df['model'] == 'qr')
                    ].copy()
                    if 'covariate_mode' in cov_meal_df.columns:
                        cov_meal_df = cov_meal_df[cov_meal_df['covariate_mode'] == cov_mode]
                    if 'meal_type' in cov_meal_df.columns:
                        cov_meal_df = cov_meal_df[cov_meal_df['meal_type'] == meal_type]
                    if 'quantile_tau' in cov_meal_df.columns:
                        for tau in sorted(cov_meal_df['quantile_tau'].dropna().unique()):
                            if any(abs(tau - excl) < 0.01 for excl in EXCLUDED_TAUS):
                                continue
                            tau_label = f"tau{tau:.2f}".replace('.', '')
                            plot_individual_effects_figure(
                                df, fig_dir, cov_mode=cov_mode, meal_type=meal_type,
                                model_type='qr', offset=offset, quantile_tau=tau,
                                filename=f'fig_effects_qr_{tau_label}{offset_suffix}.png'
                            )

    # 2. QR panel figures (LMER + QR taus in panels) per covariate mode + meal type
    #    Saved to: figures/{cov_mode}/{meal_type}/
    if has_qr:
        print("\n--- QR panel figures (per covariate mode + meal type) ---")
        for cov_mode in cov_modes:
            for meal_type in all_meal_types:
                fig_dir = _fig_dir(cov_mode, meal_type)
                cov_meal_df = df.copy()
                if 'covariate_mode' in cov_meal_df.columns:
                    cov_meal_df = cov_meal_df[cov_meal_df['covariate_mode'] == cov_mode]
                if 'meal_type' in cov_meal_df.columns:
                    cov_meal_df = cov_meal_df[cov_meal_df['meal_type'] == meal_type]

                for offset in offsets:
                    offset_suffix = f'_offset{offset}g' if has_multi_offset else ''
                    plot_qr_panel_figure(
                        cov_meal_df, fig_dir, offset=offset,
                        cov_mode=cov_mode, meal_type=meal_type,
                        filename=f'fig_qr_panel{offset_suffix}.png'
                    )

    # 3. Offset comparison figures per covariate mode + meal type
    #    Saved to: figures/{cov_mode}/{meal_type}/
    if has_multi_offset:
        print("\n--- Offset comparison figures (per covariate mode + meal type) ---")
        for cov_mode in cov_modes:
            for meal_type in all_meal_types:
                fig_dir = _fig_dir(cov_mode, meal_type)
                cov_meal_df = df.copy()
                if 'covariate_mode' in cov_meal_df.columns:
                    cov_meal_df = cov_meal_df[cov_meal_df['covariate_mode'] == cov_mode]
                if 'meal_type' in cov_meal_df.columns:
                    cov_meal_df = cov_meal_df[cov_meal_df['meal_type'] == meal_type]

                plot_offset_comparison_figure(
                    cov_meal_df, fig_dir, model_type='lmer',
                    cov_mode=cov_mode, meal_type=meal_type,
                    filename=f'fig_offset_comparison_lmer.png'
                )

    # 4. Diagnostics figures per covariate mode + meal type
    #    Saved to: figures/{cov_mode}/{meal_type}/
    print("\n--- Diagnostics figures ---")
    for cov_mode in cov_modes:
        for meal_type in all_meal_types:
            fig_dir = _fig_dir(cov_mode, meal_type)
            plot_diagnostics_figure(
                df, fig_dir, cov_mode=cov_mode, meal_type=meal_type,
                filename=f'fig_diagnostics.png'
            )

    # =========================================================================
    # TABLES
    # =========================================================================
    print()
    print("=" * 60)
    print("GENERATING TABLES")
    print("=" * 60)

    # 1. Per covariate mode, per meal type summary tables
    #    Saved to: tables/{cov_mode}/{meal_type}/
    print("\n--- Per-meal effects tables ---")
    for cov_mode in cov_modes:
        for meal_type in all_meal_types:
            tab_dir = _tab_dir(cov_mode, meal_type)
            for offset in offsets:
                offset_suffix = f'_offset{offset}g' if has_multi_offset else ''

                # Summary table (30-min intervals)
                generate_per_meal_effects_table_latex(
                    df, tab_dir, cov_mode=cov_mode, meal_type=meal_type,
                    model_type='lmer', offset=offset,
                    filename=f'table_effects_lmer{offset_suffix}.tex'
                )

                # Full table (5-min intervals, appendix)
                generate_full_timepoints_table_latex(
                    df, tab_dir, cov_mode=cov_mode, meal_type=meal_type,
                    model_type='lmer', offset=offset,
                    filename=f'table_effects_full_lmer{offset_suffix}.tex'
                )

    # 2. Meal comparison tables (per covariate mode)
    #    Saved to: tables/{cov_mode}/ (cross-cutting across meal types)
    if len(meal_types_no_all) > 1:
        print("\n--- Meal comparison tables (per covariate mode) ---")
        for cov_mode in cov_modes:
            tab_dir = _tab_cov_dir(cov_mode)
            for offset in offsets:
                offset_suffix = f'_offset{offset}g' if has_multi_offset else ''
                generate_meal_comparison_table_latex(
                    df, tab_dir, cov_mode=cov_mode, model_type='lmer', offset=offset,
                    filename=f'table_meal_comparison_lmer{offset_suffix}.tex'
                )

    # 3. Covariate comparison tables (phi vs PCA, per meal type)
    #    Saved to: tables/comparison/ (cross-cutting across covariate modes)
    if len(cov_modes) > 1:
        print("\n--- Covariate comparison tables (phi vs PCA, per meal type) ---")
        comp_tab_dir = _comparison_tab_dir()
        for meal_type in all_meal_types:
            for offset in offsets:
                offset_suffix = f'_offset{offset}g' if has_multi_offset else ''
                generate_covariate_comparison_table_latex(
                    df, comp_tab_dir, meal_type=meal_type, model_type='lmer', offset=offset,
                    filename=f'table_cov_comparison_{meal_type.lower()}_lmer{offset_suffix}.tex'
                )

    # 4. Combined meal x covariate comparison table
    #    Saved to: tables/comparison/
    if len(cov_modes) > 1 and len(meal_types_no_all) > 1:
        print("\n--- Combined meal x covariate comparison tables ---")
        comp_tab_dir = _comparison_tab_dir()
        for offset in offsets:
            offset_suffix = f'_offset{offset}g' if has_multi_offset else ''
            generate_combined_comparison_table_latex(
                df, comp_tab_dir, model_type='lmer', offset=offset,
                filename=f'table_combined_comparison_lmer{offset_suffix}.tex'
            )

    # 5. LMER multi-offset tables (per covariate mode + meal type)
    #    Saved to: tables/{cov_mode}/{meal_type}/
    if has_multi_offset:
        print("\n--- LMER multi-offset tables ---")
        for cov_mode in cov_modes:
            for meal_type in all_meal_types:
                tab_dir = _tab_dir(cov_mode, meal_type)
                generate_lmer_offset_table_latex(
                    df, tab_dir, cov_mode=cov_mode, meal_type=meal_type,
                    filename=f'table_lmer_offsets.tex'
                )

    # 6. QR multi-tau tables (per covariate mode + meal type + offset)
    #    Saved to: tables/{cov_mode}/{meal_type}/
    if has_qr:
        print("\n--- QR multi-tau tables ---")
        for cov_mode in cov_modes:
            for meal_type in all_meal_types:
                tab_dir = _tab_dir(cov_mode, meal_type)
                for offset in offsets:
                    offset_suffix = f'_offset{offset}g' if has_multi_offset else ''
                    generate_qr_tau_table_latex(
                        df, tab_dir, cov_mode=cov_mode, meal_type=meal_type, offset=offset,
                        filename=f'table_qr_taus{offset_suffix}.tex'
                    )

    # 7. Full CSV export with all labels
    #    Saved to: tables/ (top-level, contains all data)
    print("\n--- CSV export ---")
    generate_effects_table_csv(df, tables_base)

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================
    print()
    print("=" * 60)
    print("CONSOLE SUMMARY")
    print("=" * 60)
    print_summary_table(df)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {figures_base}")
    print(f"Tables saved to: {tables_base}")
    print(f"\nDirectory structure:")
    print(f"  {{figures,tables}}/{{cov_mode}}/{{meal_type}}/  -- per-meal outputs")
    print(f"  {{tables}}/{{cov_mode}}/                       -- meal comparison tables")
    print(f"  {{tables}}/comparison/                         -- cross-covariate tables")


if __name__ == "__main__":
    main()
