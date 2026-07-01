#!/usr/bin/env python3
"""Shared mediation-figure primitives so OhioT1DM and DiaTrend figures match.

Both ``generate_mediation_outputs.py`` (Ohio) and
``generate_diatrend_mediation_outputs.py`` (DiaTrend) import the SAME plotting
primitive, style block, and formatters from here, so the two datasets' figures
are identical by construction rather than by two scripts independently trying to
look alike.

The primitive expects the OhioT1DM column convention (``minutes``, ``ACME``,
``ACME_lower/upper/p``, ``ADE...``, ``total_effect...``). ``canonicalize()``
maps the DiaTrend grid convention (``timepoint``, ``acme``, ``acme_lo/hi/p`` ...)
onto it, so a single code path serves both. Figures are always built from result
data frames (grid CSVs / per-cell rds), never from parsed printed tables.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Style: applied once on import so every figure inherits it (Ohio's block) -
STYLE = {
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
}
plt.rcParams.update(STYLE)

COLORS = {
    'ACME': '#2ecc71',   # green  - indirect effect
    'ADE': '#e74c3c',    # red    - direct effect
    'Total': '#3498db',  # blue   - total effect
}

MEAL_TYPE_ORDER = ['ALL', 'Breakfast', 'Lunch', 'Dinner', 'Snack']


# =============================================================================
# Schema normalization
# =============================================================================

# DiaTrend grid column -> canonical (Ohio) column.
_DIATREND_RENAME = {
    'timepoint': 'minutes',
    'acme': 'ACME', 'acme_lo': 'ACME_lower', 'acme_hi': 'ACME_upper', 'acme_p': 'ACME_p',
    'ade': 'ADE', 'ade_lo': 'ADE_lower', 'ade_hi': 'ADE_upper', 'ade_p': 'ADE_p',
    'total': 'total_effect', 'total_lo': 'total_lower', 'total_hi': 'total_upper',
    'total_p': 'total_p',
    'offset_g': 'treat_offset', 'tau': 'quantile_tau', 'n_episodes': 'n_obs',
}

_CANONICAL_EFFECT_COLS = [
    'minutes', 'ACME', 'ACME_lower', 'ACME_upper', 'ACME_p',
    'ADE', 'ADE_lower', 'ADE_upper', 'ADE_p',
    'total_effect', 'total_lower', 'total_upper', 'total_p',
]


def normalize_meal_type(raw_name) -> str:
    """Strip quotes/case and map to canonical capitalization (ALL, Breakfast...)."""
    cleaned = str(raw_name).strip().strip('"\'')
    canonical = {'all': 'ALL', 'breakfast': 'Breakfast', 'lunch': 'Lunch',
                 'dinner': 'Dinner', 'snack': 'Snack'}
    return canonical.get(cleaned.lower(), cleaned)


def sort_meal_types(meal_types):
    order = {m: i for i, m in enumerate(MEAL_TYPE_ORDER)}
    return sorted(meal_types, key=lambda x: order.get(x, 99))


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with OhioT1DM-canonical columns regardless of input schema.

    Accepts either the Ohio convention (already canonical) or the DiaTrend grid
    convention. Adds a 'status' column (default 'success') and a normalized
    'meal_type' if only 'meal' is present, so downstream code is schema-blind.
    """
    out = df.copy()
    rename = {k: v for k, v in _DIATREND_RENAME.items()
              if k in out.columns and v not in out.columns}
    out = out.rename(columns=rename)
    if 'meal_type' not in out.columns and 'meal' in out.columns:
        out['meal_type'] = out['meal']
    if 'meal_type' in out.columns:
        out['meal_type'] = out['meal_type'].map(normalize_meal_type)
    if 'status' not in out.columns:
        out['status'] = 'success'
    # The bootstrap/grid LMER rows can carry an empty tau; treat blanks as NaN.
    if 'quantile_tau' in out.columns:
        out['quantile_tau'] = pd.to_numeric(out['quantile_tau'], errors='coerce')
    return out


# =============================================================================
# Formatting helpers (shared by both table generators)
# =============================================================================

def format_p(p, bold=False):
    if p < 0.001:
        txt = "$<$0.001"
    elif p < 0.01:
        txt = f"{p:.3f}"
    else:
        txt = f"{p:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def format_est(est, bold=False):
    txt = f"{est:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def format_ci(lower, upper, bold=False):
    txt = f"({lower:.2f}, {upper:.2f})"
    return rf"\textbf{{{txt}}}" if bold else txt


def format_est_ci_star(est, lower, upper, p):
    sig = p < 0.05
    star = '*' if sig else ''
    txt = f"{est:.2f}{star} ({lower:.2f}, {upper:.2f})"
    return rf"\textbf{{{txt}}}" if sig else txt


# =============================================================================
# THE primitive: ACME / ADE / Total over time on one axes (Ohio's, verbatim)
# =============================================================================

def plot_effects_on_ax(ax, df_success, show_legend=True, set_labels=True):
    """Plot ACME, ADE, Total Effect with significance stars on a single axes.

    df_success must use canonical columns (call canonicalize() first) and be
    pre-filtered to a single model configuration. set_labels=False suppresses the
    per-axes x/y labels (for grids that label at the figure level); it defaults
    to True so existing single/row-panel callers are unchanged.
    """
    df_success = df_success.sort_values('minutes')
    minutes = df_success['minutes'].values

    acme_sig = df_success['ACME_p'] < 0.05
    ade_sig = df_success['ADE_p'] < 0.05
    total_sig = df_success['total_p'] < 0.05

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

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    if set_labels:
        ax.set_xlabel('Minutes post-meal', fontsize=14)
        ax.set_ylabel('Effect on glucose (mg/dL)', fontsize=14)
    if show_legend:
        ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    if len(minutes) > 10:
        ax.set_xticks(minutes[::6])   # every 30 min for 5-min-interval data
    else:
        ax.set_xticks(minutes)


# Back-compat alias for the private name the Ohio script used internally.
_plot_effects_on_ax = plot_effects_on_ax
