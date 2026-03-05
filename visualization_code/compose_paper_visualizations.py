#!/usr/bin/env python3
"""Compose publication-ready figures and tables for the paper.

Single source for ALL publication figures and tables.  Generates composite
figures, condensed QR tables, and copies pre-existing analysis outputs into
a unified paper_visualizations directory.

Intermediate files are staged internally (in a hidden ``.staging/`` directory)
and cleaned up automatically.  The final ``paper_visualizations/`` directory
contains **only** journal-named figures and tables (e.g. "Figure 1.pdf",
"Supplementary Table 3.tex").

Usage examples
--------------
    python visualization_code/compose_paper_visualizations.py --all               # produce everything
    python visualization_code/compose_paper_visualizations.py figure3              # single composite figure
    python visualization_code/compose_paper_visualizations.py --tables             # condensed QR tables only
    python visualization_code/compose_paper_visualizations.py --copy               # copy raw outputs only
    python visualization_code/compose_paper_visualizations.py figure3 --dpi 600    # high-res composite
    python visualization_code/compose_paper_visualizations.py --submission         # export submission figures & tables
"""

import argparse
import re
import shutil
import string
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================================================================
# Configuration
# =========================================================================

VISUALIZATION_CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_CODE_DIR.parent

# Output directories (all generated outputs live under visualizations/)
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
PAPER_VIS_DIR = VISUALIZATIONS_DIR / "paper_visualizations"
PAPER_VIS_FIGURES_DIR = PAPER_VIS_DIR / "figures"
PAPER_VIS_TABLES_DIR = PAPER_VIS_DIR / "tables"

# Internal staging area for intermediate files (composites, panel copies, etc.)
# Cleaned up after the final export so paper_visualizations/ only has journal-named files.
_STAGING_DIR = PAPER_VIS_DIR / ".staging"
_STAGING_FIGURES_DIR = _STAGING_DIR / "figures"
_STAGING_TABLES_DIR = _STAGING_DIR / "tables"

# Raw analysis directories (inputs — generated outputs from individual scripts)
RAW_DIRS = {
    "ae_embeddings": VISUALIZATIONS_DIR / "ae_embeddings",
    "data_distribution": VISUALIZATIONS_DIR / "data_distribution",
    "mediation_vis": VISUALIZATIONS_DIR / "mediation_visualizations",
    "npcbps_balance": VISUALIZATIONS_DIR / "npcbps_balance",
    "incremental": VISUALIZATIONS_DIR / "incremental_data_experiment",
}

# Meal type configuration
MEAL_CONFIG = {
    "all":       {"display_name": "Pooled (all meals)",  "N": 190},
    "breakfast": {"display_name": "Breakfast",   "N": 51},
    "lunch":     {"display_name": "Lunch",       "N": 58},
    "dinner":    {"display_name": "Dinner",      "N": 42},
    "snack":     {"display_name": "Snack",       "N": 39},
}

# Canonical ordering for table generation
MEAL_TYPE_ORDER = ["all", "breakfast", "lunch", "dinner", "snack"]

# Taus to include in condensed tables
CONDENSED_TAUS = [0.25, 0.50, 0.75]

# Timepoints shown in summary tables
SUMMARY_TIMEPOINTS = [60, 90, 120, 150, 180, 210]

# Treatment doses and quantile tau strings for file naming
DOSES = ["15.0g", "30.0g", "45.0g"]
TAU_STRINGS = ["025", "050", "075"]

# =========================================================================
# Submission export configuration
# =========================================================================

SUBMISSION_FIGURES_DIR = PAPER_VIS_FIGURES_DIR
SUBMISSION_TABLES_DIR = PAPER_VIS_TABLES_DIR

# Keep backward-compatible alias used throughout this file.
SUBMISSION_DIR = SUBMISSION_FIGURES_DIR

# Mapping from journal-required filename -> source path relative to
# _STAGING_FIGURES_DIR.  A value of None means the figure is compiled
# separately (e.g. TikZ standalone) and handled by special-case logic in
# export_submission_figures().
SUBMISSION_FIGURES = {
    "Figure 1.pdf": None,  # Compiled separately from figure1_dag.tex
    "Figure 2.png": "trajectory_composite.png",
    "Figure 3.png": "figure3_composite.png",
    "Supplementary Figure 1.png": "fig08_treatment_mediator_by_meal.png",
    "Supplementary Figure 2.png": "fig1_love_plot.png",
    "Supplementary Figure 3.png": "phi_pca_meal_type_main.png",  # was main Figure 2
    "Supplementary Figure 4.png": "all/fig_lmer_composite_all.png",
    "Supplementary Figure 5.png": "all/fig_qr_composite_all.png",
    "Supplementary Figure 6.png": "dinner/fig_lmer_composite_dinner.png",
    "Supplementary Figure 7.png": "dinner/fig_qr_composite_dinner.png",
    "Supplementary Figure 8.png": "breakfast/fig_lmer_composite_breakfast.png",
    "Supplementary Figure 9.png": "breakfast/fig_qr_composite_breakfast.png",
    "Supplementary Figure 10.png": "lunch/fig_lmer_composite_lunch.png",
    "Supplementary Figure 11.png": "lunch/fig_qr_composite_lunch.png",
    "Supplementary Figure 12.png": "snack/fig_lmer_composite_snack.png",
    "Supplementary Figure 13.png": "snack/fig_qr_composite_snack.png",
}

# Mapping from journal-required filename -> source path relative to
# _STAGING_TABLES_DIR.  Table 2 is generated by generate_clae_config_table().
# Table 4 is generated by generate_qr_summary_table().
SUBMISSION_TABLES = {
    # Main text
    "Table 1.tex": "cohort_meal_summary.tex",
    "Table 2.tex": "clae_config.tex",  # generated (static)
    "Table 3.tex": "all/table_lmer_offsets.tex",
    "Table 4.tex": "qr_summary_main.tex",  # generated
    # Supplementary
    "Supplementary Table 1.tex": "split_meal_summary.tex",
    "Supplementary Table 2.tex": "npcbps_diagnostics.tex",
    "Supplementary Table 3.tex": "lmer_condensed_all.tex",
    "Supplementary Table 4.tex": "qr_condensed_all.tex",
    "Supplementary Table 5.tex": "lmer_condensed_dinner.tex",
    "Supplementary Table 6.tex": "qr_condensed_dinner.tex",
    "Supplementary Table 7.tex": "lmer_condensed_breakfast.tex",
    "Supplementary Table 8.tex": "qr_condensed_breakfast.tex",
    "Supplementary Table 9.tex": "lmer_condensed_lunch.tex",
    "Supplementary Table 10.tex": "qr_condensed_lunch.tex",
    "Supplementary Table 11.tex": "lmer_condensed_snack.tex",
    "Supplementary Table 12.tex": "qr_condensed_snack.tex",
    "Supplementary Table 13.tex": "table_arch_optimizer_summary.tex",
    "Supplementary Table 14.tex": "table_ablation_marginal.tex",
    "Supplementary Table 15.tex": "table_ablation_top_configs.tex",
    "Supplementary Table 16.tex": "table_comprehensive_top10.tex",
}


# =========================================================================
# Copy manifest
# =========================================================================

def _build_copy_manifest():
    """Build the manifest of pre-existing files to copy into staging.

    Only includes files that the submission export needs but that are NOT
    generated by the composition/table functions — i.e. raw outputs that
    already exist in their original ``visualizations/`` subdirectories.

    Each entry is ``(source_relative_to_VISUALIZATIONS_DIR,
    dest_relative_to_STAGING_DIR)``.
    """
    manifest = []

    # --- Figures that exist in original locations ---
    manifest.append((
        "ae_embeddings/figures/phi_pca_meal_type_main.png",
        "figures/phi_pca_meal_type_main.png",
    ))
    manifest.append((
        "data_distribution/figures/fig08_treatment_mediator_by_meal.png",
        "figures/fig08_treatment_mediator_by_meal.png",
    ))
    manifest.append((
        "npcbps_balance/figures_pc3/fig1_love_plot.png",
        "figures/fig1_love_plot.png",
    ))

    # --- Tables that exist in original locations ---
    manifest.append((
        "data_distribution/tables/cohort_meal_summary.tex",
        "tables/cohort_meal_summary.tex",
    ))
    manifest.append((
        "data_distribution/tables/split_meal_summary.tex",
        "tables/split_meal_summary.tex",
    ))
    manifest.append((
        "npcbps_balance/tables_pc3/npcbps_diagnostics.tex",
        "tables/npcbps_diagnostics.tex",
    ))

    return manifest


COPY_MANIFEST = _build_copy_manifest()


# =========================================================================
# Helper functions
# =========================================================================

def copy_to_paper_vis(src_relative, dst_relative):
    """Copy a single file from a raw analysis directory to paper_visualizations/.

    *src_relative* is relative to ``VISUALIZATIONS_DIR``.
    *dst_relative* is relative to ``_STAGING_DIR``.

    Returns ``True`` on success, ``False`` if the source file is missing.
    """
    src = VISUALIZATIONS_DIR / src_relative
    dst = _STAGING_DIR / dst_relative
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _panel_letters():
    """Return an infinite generator of panel letters: a, b, c, ..."""
    yield from string.ascii_lowercase


# =========================================================================
# LaTeX formatting helpers
# =========================================================================

def _fmt_est(value, bold=False):
    """Format a point estimate."""
    txt = f"{value:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def _fmt_ci(lower, upper, bold=False):
    """Format a 95% confidence interval."""
    txt = f"({lower:.2f}, {upper:.2f})"
    return rf"\textbf{{{txt}}}" if bold else txt


def _fmt_p(p, bold=False):
    """Format a p-value."""
    if p < 0.001:
        txt = "$<$0.001"
    elif p < 0.01:
        txt = f"{p:.3f}"
    else:
        txt = f"{p:.2f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def _write_latex_table(path, content):
    """Write LaTeX content to *path*, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _sample_size_symbol(meal_key):
    """Return 'N' (number of meal observations used in the analysis)."""
    return "N"


# =========================================================================
# Composition engines
# =========================================================================

def _crop_title_region(img, crop_fraction=0.08):
    """Crop the top portion of an image array to remove baked-in titles.

    *crop_fraction* is the fraction of image height to remove from the top.
    """
    h = img.shape[0]
    crop_px = int(h * crop_fraction)
    return img[crop_px:, :, :]


# Mediation effect colours / markers (consistent across all panels)
_MEDIATION_EFFECTS = [
    ("ACME",         "ACME_lower", "ACME_upper",  "#2ecc71", "o",  "ACME (Indirect)"),
    ("ADE",          "ADE_lower",  "ADE_upper",   "#e74c3c", "s",  "ADE (Direct)"),
    ("total_effect", "total_lower", "total_upper", "#3498db", "^",  "Total Effect"),
]


def _get_panel_data(df, model, meal_key, dose, tau=None):
    """Extract mediation results for a single panel from *df*.

    Returns a DataFrame sorted by minutes, or *None*.
    """
    meal_label = "ALL" if meal_key == "all" else meal_key.capitalize()
    sub = df[(df["status"] == "success") & (df["model"] == model)].copy()

    if "covariate_mode" in sub.columns:
        sub = sub[sub["covariate_mode"] == "pca"]
    if "meal_type" in sub.columns:
        sub = sub[sub["meal_type"] == meal_label]
    if "treat_offset" in sub.columns:
        sub = sub[sub["treat_offset"] == dose]
    if tau is not None and "quantile_tau" in sub.columns:
        sub = sub[abs(sub["quantile_tau"] - tau) < 0.01]

    return sub.sort_values("minutes") if not sub.empty else None


def _compute_common_ylim(panel_data_list, padding=0.10):
    """Compute a shared y-axis range across all panels.

    Inspects the CI bounds (lower/upper) of every non-None panel DataFrame.
    Returns ``(ymin, ymax)`` with *padding* fraction of the range added.
    """
    ymin, ymax = float("inf"), float("-inf")
    ci_lower_cols = ["ACME_lower", "ADE_lower", "total_lower"]
    ci_upper_cols = ["ACME_upper", "ADE_upper", "total_upper"]

    for pdata in panel_data_list:
        if pdata is None or pdata.empty:
            continue
        for col in ci_lower_cols:
            if col in pdata.columns:
                ymin = min(ymin, pdata[col].min())
        for col in ci_upper_cols:
            if col in pdata.columns:
                ymax = max(ymax, pdata[col].max())

    if ymin == float("inf"):
        return (-30, 30)  # sensible fallback
    span = ymax - ymin
    return (ymin - padding * span, ymax + padding * span)


_MEDIATION_XTICKS = [60, 90, 120, 150, 180, 210]

# p-value columns for significance testing (parallel to _MEDIATION_EFFECTS)
_MEDIATION_P_COLS = ["ACME_p", "ADE_p", "total_p"]


def _plot_mediation_panel(ax, pdata, ylim):
    """Plot ACME, ADE, Total with CIs and significance stars on *ax*.

    Significant points (p < 0.05) are shown as large stars; non-significant
    points keep their regular shape marker.  No title, no individual legend,
    no axis labels — tick label visibility is handled by sharey/sharex.
    """
    if pdata is None or pdata.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="0.5")
        ax.set_ylim(ylim)
        return

    minutes = pdata["minutes"].values

    for (est_col, lo_col, hi_col, color, marker, _), p_col in zip(
        _MEDIATION_EFFECTS, _MEDIATION_P_COLS
    ):
        if est_col not in pdata.columns:
            continue
        vals = pdata[est_col].values
        lo = pdata[lo_col].values if lo_col in pdata.columns else None
        hi = pdata[hi_col].values if hi_col in pdata.columns else None

        # Line + confidence band
        ax.plot(minutes, vals, "-", color=color, linewidth=1.5)
        if lo is not None and hi is not None:
            ax.fill_between(minutes, lo, hi, color=color, alpha=0.15)

        # Significant vs non-significant markers
        if p_col in pdata.columns:
            sig = pdata[p_col].values < 0.05
            if sig.any():
                ax.plot(minutes[sig], vals[sig], "*", color=color,
                        markersize=14, markeredgecolor="white",
                        markeredgewidth=0.5)
            if (~sig).any():
                ax.plot(minutes[~sig], vals[~sig], marker, color=color,
                        markersize=4)
        else:
            ax.plot(minutes, vals, marker, color=color, markersize=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(ylim)
    ax.set_xlim(55, 215)
    ax.set_xticks(_MEDIATION_XTICKS)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.15)


def compose_grid(cfg, dpi, output):
    """Compose a grid of individual panel PNGs with row/column headers.

    Falls back to the image-based approach (with title cropping) only when
    CSV-based regeneration has already failed.  The ``compose_grid_from_csv``
    function is the preferred path.
    """
    base_dir = cfg["base_dir"]
    rows = cfg["rows"]
    cols = cfg["cols"]
    n_rows = len(rows)
    n_cols = len(cols)
    crop_frac = cfg.get("crop_fraction", 0.08)

    # Load and crop images
    images = {}
    for r, (_, subdir) in enumerate(rows):
        for c, (_, fname) in enumerate(cols):
            path = base_dir / subdir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing panel image: {path}")
            img = mpimg.imread(str(path))
            images[(r, c)] = _crop_title_region(img, crop_frac)

    sample = images[(0, 0)]
    img_h, img_w = sample.shape[:2]
    aspect = img_w / img_h

    panel_width = cfg.get("panel_width", 4.0)
    panel_height = panel_width / aspect
    left_margin = cfg.get("left_margin", 0.80)
    top_margin = cfg.get("top_margin", 0.45)
    hgap = cfg.get("hgap", 0.05)
    vgap = cfg.get("vgap", 0.05)
    letter_fs = cfg.get("letter_fontsize", 12)
    header_fs = cfg.get("header_fontsize", 15)

    fig_width = left_margin + n_cols * panel_width + (n_cols - 1) * hgap
    fig_height = top_margin + n_rows * panel_height + (n_rows - 1) * vgap

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    letters = _panel_letters()

    for r in range(n_rows):
        for c in range(n_cols):
            x = (left_margin + c * (panel_width + hgap)) / fig_width
            y = 1.0 - (top_margin + (r + 1) * panel_height + r * vgap) / fig_height
            w = panel_width / fig_width
            h = panel_height / fig_height

            ax = fig.add_axes([x, y, w, h])
            ax.imshow(images[(r, c)])
            ax.set_axis_off()

            ax.text(
                0.02, 0.97, next(letters),
                transform=ax.transAxes,
                fontsize=letter_fs, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.8, pad=2),
            )

    # Column headers (larger, bold)
    for c, (header, _) in enumerate(cols):
        cx = (left_margin + c * (panel_width + hgap) + panel_width / 2) / fig_width
        cy = 1.0 - (top_margin * 0.40) / fig_height
        fig.text(cx, cy, header,
                 ha="center", va="center",
                 fontsize=header_fs, fontweight="bold")

    # Row labels (larger, bold, rotated 90 degrees)
    for r, (label, _) in enumerate(rows):
        rx = (left_margin * 0.30) / fig_width
        ry = 1.0 - (top_margin + r * (panel_height + vgap) + panel_height / 2) / fig_height
        fig.text(rx, ry, label,
                 ha="center", va="center",
                 fontsize=header_fs, fontweight="bold", rotation=90)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def compose_vstack(cfg, dpi, output):
    """Vertically stack wide source images and add panel letters."""
    sources = cfg["sources"]
    panels_per_row = cfg.get("panels_per_row", 2)
    fig_width = cfg.get("fig_width", 14.0)
    vgap = cfg.get("vgap", 0.15)
    letter_fs = cfg.get("letter_fontsize", 18)
    x_positions = cfg.get("letter_x_positions", [0.02, 0.50])

    # Load images
    imgs = []
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        imgs.append(mpimg.imread(str(src)))

    # Compute row heights proportional to each image's aspect ratio
    row_heights = [fig_width * (img.shape[0] / img.shape[1]) for img in imgs]
    total_height = sum(row_heights) + vgap * (len(imgs) - 1)

    fig = plt.figure(figsize=(fig_width, total_height), facecolor="white")
    letters = _panel_letters()

    # Place rows top-to-bottom
    y_cursor = total_height  # start from top
    for img, row_h in zip(imgs, row_heights):
        y_cursor -= row_h
        y_frac = y_cursor / total_height
        h_frac = row_h / total_height

        ax = fig.add_axes([0.0, y_frac, 1.0, h_frac])
        ax.imshow(img)
        ax.set_axis_off()

        for x_pos in x_positions[:panels_per_row]:
            ax.text(
                x_pos, 0.97, next(letters),
                transform=ax.transAxes,
                fontsize=letter_fs, fontweight="bold",
                va="top", ha="left",
                fontfamily="sans-serif",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.8, pad=2),
            )

        y_cursor -= vgap

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


ENGINES = {
    "grid": compose_grid,
    "vstack": compose_vstack,
}


# =========================================================================
# Figure registry -- add new figures here
# =========================================================================

FIGURES = {}


def register(name, cfg):
    """Register a figure configuration."""
    FIGURES[name] = cfg


# -- Figure 3 (main text): mediation effects 3x4 grid --------------------

_fig3_base = RAW_DIRS["mediation_vis"] / "figures" / "pca"

register("figure3", dict(
    description="Mediation effects 3x4 grid (Pooled / Breakfast / Dinner x LMER / QR taus)",
    mode="grid",
    base_dir=_fig3_base,
    output=_STAGING_FIGURES_DIR / "figure3_composite.png",
    # Each row: (row_label, subdirectory)
    rows=[
        ("Pooled (N = 190)", "all"),
        ("Breakfast (N = 51)", "breakfast"),
        ("Dinner (N = 42)", "dinner"),
    ],
    # Each col: (column_header, filename)
    cols=[
        ("LMER (Mean)",  "fig_effects_lmer_offset30.0g.png"),
        ("\u03c4 = 0.25", "fig_effects_qr_tau025_offset30.0g.png"),
        ("\u03c4 = 0.50", "fig_effects_qr_tau050_offset30.0g.png"),
        ("\u03c4 = 0.75", "fig_effects_qr_tau075_offset30.0g.png"),
    ],
    panel_width=4.0,
    left_margin=0.80,
    top_margin=0.45,
    bottom_margin=0.60,
    hgap=0.05,
    vgap=0.05,
    letter_fontsize=12,
    header_fontsize=15,
    crop_fraction=0.08,
))


def generate_figure3_from_csv(df, dpi=300):
    """Generate main-text Figure 3 (3x4 mediation grid) from CSV data.

    Rows: Pooled / Breakfast / Dinner.
    Cols: LMER(Mean) / tau=0.25 / tau=0.50 / tau=0.75.
    Dose: +30 g.

    All 12 panels share a common y-axis range for direct comparison.
    Only the left column has y-tick labels; only the bottom row has x-ticks.
    A single shared x-label, y-label, and legend are drawn once.

    Returns 1 on success, 0 on failure.
    """
    if df is None or df.empty:
        return 0

    output = _STAGING_FIGURES_DIR / "figure3_composite.png"

    meal_keys = ["all", "breakfast", "dinner"]
    row_labels = ["Pooled (N = 190)", "Breakfast (N = 51)", "Dinner (N = 42)"]
    col_specs = [
        ("lmer", None, "LMER (Mean)"),
        ("qr", 0.25,   "\u03c4 = 0.25"),
        ("qr", 0.50,   "\u03c4 = 0.50"),
        ("qr", 0.75,   "\u03c4 = 0.75"),
    ]
    dose = 30
    n_rows = len(meal_keys)
    n_cols = len(col_specs)

    # Collect all panel data
    all_panels = []   # [row][col]
    flat_panels = []
    for meal_key in meal_keys:
        row = []
        for model, tau, _ in col_specs:
            pdata = _get_panel_data(df, model, meal_key, dose, tau)
            row.append(pdata)
            flat_panels.append(pdata)
        all_panels.append(row)

    if not any(p is not None for p in flat_panels):
        print("  No panel data available for figure3")
        return 0

    ylim = _compute_common_ylim(flat_panels)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows),
        sharey=True, sharex=True,
    )
    letters_iter = _panel_letters()

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            _plot_mediation_panel(ax, all_panels[r][c], ylim)
            # Panel letter just outside top-left corner
            ax.text(
                -0.03, 1.01, next(letters_iter),
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right",
            )

    # Column headers
    for c, (_, _, header) in enumerate(col_specs):
        axes[0, c].set_title(header, fontsize=16, fontweight="bold", pad=10)

    # Row labels on the RIGHT side, close to the panel edge
    for r, rl in enumerate(row_labels):
        axes[r, n_cols - 1].annotate(
            rl, xy=(1, 0.5),
            xytext=(10, 0), textcoords="offset points",
            xycoords="axes fraction",
            fontsize=14, fontweight="bold",
            va="center", ha="left", rotation=-90,
        )

    # Shared Y-axis label — bold, larger than row labels
    axes[1, 0].set_ylabel("Effect on Glucose (mg/dL)", fontsize=18,
                          fontweight="bold")
    # Shared X-axis label — bold
    fig.text(0.5, 0.01, "Minutes Post Meal", ha="center",
             fontsize=18, fontweight="bold")

    # Common legend
    fig.legend(
        handles=_make_legend_handles(), loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=4, fontsize=14, frameon=True, fancybox=True,
        edgecolor="0.6", framealpha=0.9,
    )
    fig.subplots_adjust(
        left=0.07, right=0.92, bottom=0.06, top=0.94,
        hspace=0.15, wspace=0.10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved figure3 (from CSV): {output}")
    return 1


# -- Trajectory composite: stack two 2-panel images ----------------------

_traj_base = RAW_DIRS["data_distribution"] / "figures"

register("trajectory", dict(
    description="Trajectory 2x2 stack (cohort split / train-test split)",
    mode="vstack",
    output=_STAGING_FIGURES_DIR / "trajectory_composite.png",
    # Ordered list of source images (top to bottom)
    sources=[
        _traj_base / "fig16_trajectory_by_meal_type.png",
        _traj_base / "fig19_trajectory_train_vs_test.png",
    ],
    # Number of subplots per source image (used for letter placement)
    panels_per_row=2,
    fig_width=14.0,
    vgap=0.15,
    letter_fontsize=18,
    # x-positions (in axes fraction) for each sub-panel within a source image
    letter_x_positions=[0.02, 0.50],
))


# =========================================================================
# Data loading (reuses the same CSV files as generate_mediation_outputs.py)
# =========================================================================

def _normalize_meal_type(raw_name):
    """Normalize a meal type name by stripping quotes and fixing case."""
    cleaned = raw_name.strip("\"'")
    canonical = {
        "all": "ALL", "breakfast": "Breakfast", "lunch": "Lunch",
        "dinner": "Dinner", "snack": "Snack",
    }
    return canonical.get(cleaned.lower(), cleaned)


def load_mediation_results(mediation_dir):
    """Load and combine mediation result CSVs from the nested directory structure.

    Handles: ``mediation_results/{phi,pca}/{ALL,Breakfast,...}/mediation_*.csv``
    and flat layouts.  Returns a single combined DataFrame or *None*.
    """
    base = Path(mediation_dir)
    all_dfs = []

    for cov_mode in ("phi", "pca"):
        cov_dir = base / cov_mode
        if not cov_dir.exists():
            continue

        for meal_dir in sorted(cov_dir.iterdir()):
            if not meal_dir.is_dir():
                continue
            meal_type = _normalize_meal_type(meal_dir.name)
            for csv_file in sorted(meal_dir.glob("mediation_*.csv")):
                try:
                    df = pd.read_csv(csv_file)
                    if "meal_type" not in df.columns:
                        df["meal_type"] = meal_type
                    df["covariate_mode"] = cov_mode
                    all_dfs.append(df)
                except Exception as exc:
                    print(f"  Warning: skipping {csv_file}: {exc}")

    # Flat fallback: CSVs directly in base or base/{phi,pca}/
    if not all_dfs:
        for pattern in ("mediation_all_timepoints_*.csv", "mediation_*.csv"):
            for csv_file in sorted(base.glob(pattern)):
                try:
                    df = pd.read_csv(csv_file)
                    all_dfs.append(df)
                except Exception as exc:
                    print(f"  Warning: skipping {csv_file}: {exc}")
            if all_dfs:
                break

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    if "meal_type" in combined.columns:
        combined["meal_type"] = combined["meal_type"].apply(
            lambda x: _normalize_meal_type(str(x)) if pd.notna(x) else x
        )

    # De-duplicate
    key_cols = [c for c in ("minutes", "model", "treat_offset",
                            "meal_type", "covariate_mode", "quantile_tau")
                if c in combined.columns]
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols, keep="last")

    return combined


# =========================================================================
# Table generation -- condensed QR tables
# =========================================================================

def _build_condensed_qr_table(df, meal_key):
    """Build a single condensed QR LaTeX table for *meal_key*.

    Selects the +30 g contrast, PCA covariate mode, and combines tau = 0.25,
    0.50, 0.75 into one table with rows grouped by timepoint.

    Returns the LaTeX string, or *None* if insufficient data.
    """
    cfg = MEAL_CONFIG[meal_key]
    display_name = cfg["display_name"]
    n_obs = cfg["N"]

    # Filter to QR, +30 g, PCA, target meal
    meal_label = "ALL" if meal_key == "all" else meal_key.capitalize()
    sub = df[
        (df["status"] == "success")
        & (df["model"] == "qr")
    ].copy()

    if "covariate_mode" in sub.columns:
        sub = sub[sub["covariate_mode"] == "pca"]
    if "meal_type" in sub.columns:
        sub = sub[sub["meal_type"] == meal_label]
    if "treat_offset" in sub.columns:
        sub = sub[sub["treat_offset"] == 30]

    if sub.empty or "quantile_tau" not in sub.columns:
        return None

    sub = sub[sub["minutes"].isin(SUMMARY_TIMEPOINTS)]
    if sub.empty:
        return None

    # Use observed n_obs if available
    if "n_obs" in sub.columns and not sub["n_obs"].isna().all():
        n_obs = int(sub["n_obs"].iloc[0])

    rows = []
    for tp_idx, minutes in enumerate(SUMMARY_TIMEPOINTS):
        tp_data = sub[sub["minutes"] == minutes]
        if tp_data.empty:
            continue

        available_taus = [
            tau for tau in CONDENSED_TAUS
            if (tp_data["quantile_tau"] - tau).abs().min() < 0.01
        ]

        for tau_idx, tau in enumerate(CONDENSED_TAUS):
            tau_row = tp_data[abs(tp_data["quantile_tau"] - tau) < 0.01]
            if tau_row.empty:
                continue
            r = tau_row.iloc[0]

            # Multirow time label on first tau of each timepoint
            if tau_idx == 0:
                time_cell = (
                    rf"\multirow{{{len(available_taus)}}}{{*}}{{{int(minutes)} min}}"
                )
            else:
                time_cell = ""

            acme_sig = r["ACME_p"] < 0.05
            ade_sig = r["ADE_p"] < 0.05
            total_sig = r["total_p"] < 0.05

            row_str = (
                f"{time_cell} & $\\tau={tau:.2f}$ & "
                f"{_fmt_est(r['ACME'], acme_sig)} & "
                f"{_fmt_ci(r['ACME_lower'], r['ACME_upper'], acme_sig)} & "
                f"{_fmt_p(r['ACME_p'], acme_sig)} & "
                f"{_fmt_est(r['ADE'], ade_sig)} & "
                f"{_fmt_ci(r['ADE_lower'], r['ADE_upper'], ade_sig)} & "
                f"{_fmt_p(r['ADE_p'], ade_sig)} & "
                f"{_fmt_est(r['total_effect'], total_sig)} & "
                f"{_fmt_ci(r['total_lower'], r['total_upper'], total_sig)} & "
                f"{_fmt_p(r['total_p'], total_sig)} \\\\"
            )
            rows.append(row_str)

        # \midrule between timepoint groups (not after the last one)
        if tp_idx < len(SUMMARY_TIMEPOINTS) - 1 and rows and not rows[-1].endswith("\\midrule"):
            rows.append("\\midrule")

    if not rows:
        return None

    body = "\n".join(rows)

    sym = _sample_size_symbol(meal_key)
    latex = rf"""\begin{{table}}[H]
\centering
\small
\caption{{\textbf{{\boldmath QR causal mediation effects --- {display_name} (${sym} = {n_obs}$ meal observations, $+30$\,g contrast).}} ACME = Average Causal Mediation Effect; ADE = Average Direct Effect; Total = ACME + ADE. Est.\ = point estimate (mg/dL); CI = confidence interval. Significant results ($p < 0.05$) are shown in bold.}}
\label{{tab:qr_condensed_{meal_key}}}
\begin{{tabular}}{{llccccccccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{ACME}} & \multicolumn{{3}}{{c}}{{ADE}} & \multicolumn{{3}}{{c}}{{Total}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}}
Time & Quantile & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    return latex


def generate_condensed_qr_tables(df):
    """Generate all condensed QR tables to the paper_visualizations tables directory.

    Produces one .tex file per meal type plus an ``all`` pooled table.
    Returns the number of tables written.
    """
    _STAGING_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for meal_key in MEAL_TYPE_ORDER:
        latex = _build_condensed_qr_table(df, meal_key)
        if latex is None:
            print(f"  Skipping {meal_key}: no QR data for +30 g contrast")
            continue
        out_path = _STAGING_TABLES_DIR / f"qr_condensed_{meal_key}.tex"
        _write_latex_table(out_path, latex)
        print(f"  Saved: {out_path}")
        count += 1

    return count


# =========================================================================
# Table generation -- condensed LMER tables
# =========================================================================

# Dose labels for the condensed LMER tables
_DOSE_MAP = {15: "+15\\,g", 30: "+30\\,g", 45: "+45\\,g"}
_DOSE_VALUES = [15, 30, 45]


def _condensed_lmer_caption_and_label(meal_key, n_obs):
    """Return the condensed caption and label strings for an LMER table."""
    display_name = MEAL_CONFIG[meal_key]["display_name"]
    sym = _sample_size_symbol(meal_key)
    caption = (
        rf"\caption{{\textbf{{\boldmath LMER causal mediation effects "
        rf"--- {display_name} (${sym} = {n_obs}$ meal observations, all contrast sizes).}} "
        rf"ACME = Average Causal Mediation Effect; ADE = Average Direct Effect; "
        rf"Total = ACME + ADE. Est.\ = point estimate (mg/dL); CI = confidence interval. "
        rf"Significant results ($p < 0.05$) are shown in bold.}}"
    )
    label = rf"\label{{tab:lmer_condensed_{meal_key}}}"
    return caption, label


def _find_balanced_brace(text, start):
    """Return the index of the ``}`` that balances the ``{`` at *start*.

    *start* must point to an opening ``{``.  Returns ``-1`` if unbalanced.
    """
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _build_condensed_lmer_from_tex(tex_content, meal_key):
    """Build a condensed LMER table by replacing caption/label in existing LaTeX.

    Returns the modified LaTeX string.
    """
    cfg = MEAL_CONFIG[meal_key]
    n_obs = cfg["N"]
    new_caption, new_label = _condensed_lmer_caption_and_label(meal_key, n_obs)

    # Replace \caption{...} handling nested braces (e.g. \textbf{...})
    result = tex_content
    cap_match = re.search(r"\\caption\{", result)
    if cap_match:
        brace_open = cap_match.end() - 1          # index of the '{'
        brace_close = _find_balanced_brace(result, brace_open)
        if brace_close != -1:
            result = result[:cap_match.start()] + new_caption + result[brace_close + 1:]

    # Replace \label{...}
    result = re.sub(
        r"\\label\{[^}]*\}",
        lambda _: new_label,
        result,
        count=1,
    )
    return result


def _lmer_rows_from_csv(df, meal_key):
    """Filter LMER CSV data for *meal_key* and build table body rows.

    Returns ``(rows_list, n_obs)`` or ``(None, None)`` when data is
    insufficient.  Shared by the condensed and per-meal-subfolder generators.
    """
    cfg = MEAL_CONFIG[meal_key]
    n_obs = cfg["N"]

    meal_label = "ALL" if meal_key == "all" else meal_key.capitalize()
    sub = df[
        (df["status"] == "success")
        & (df["model"] == "lmer")
    ].copy()

    if "covariate_mode" in sub.columns:
        sub = sub[sub["covariate_mode"] == "pca"]
    if "meal_type" in sub.columns:
        sub = sub[sub["meal_type"] == meal_label]

    if sub.empty:
        return None, None

    sub = sub[sub["minutes"].isin(SUMMARY_TIMEPOINTS)]
    if sub.empty:
        return None, None

    if "n_obs" in sub.columns and not sub["n_obs"].isna().all():
        n_obs = int(sub["n_obs"].iloc[0])

    rows = []
    for tp_idx, minutes in enumerate(SUMMARY_TIMEPOINTS):
        tp_data = sub[sub["minutes"] == minutes]
        if tp_data.empty:
            continue

        available_doses = [
            d for d in _DOSE_VALUES
            if "treat_offset" not in tp_data.columns
            or not tp_data[tp_data["treat_offset"] == d].empty
        ]

        for dose_idx, dose in enumerate(_DOSE_VALUES):
            if "treat_offset" in tp_data.columns:
                dose_row = tp_data[tp_data["treat_offset"] == dose]
            else:
                dose_row = tp_data
            if dose_row.empty:
                continue
            r = dose_row.iloc[0]

            if dose_idx == 0:
                time_cell = (
                    rf"\multirow{{{len(available_doses)}}}{{*}}{{{int(minutes)} min}}"
                )
            else:
                time_cell = ""

            acme_sig = r["ACME_p"] < 0.05
            ade_sig = r["ADE_p"] < 0.05
            total_sig = r["total_p"] < 0.05

            row_str = (
                f"{time_cell} & ${_DOSE_MAP[dose]}$ & "
                f"{_fmt_est(r['ACME'], acme_sig)} & "
                f"{_fmt_ci(r['ACME_lower'], r['ACME_upper'], acme_sig)} & "
                f"{_fmt_p(r['ACME_p'], acme_sig)} & "
                f"{_fmt_est(r['ADE'], ade_sig)} & "
                f"{_fmt_ci(r['ADE_lower'], r['ADE_upper'], ade_sig)} & "
                f"{_fmt_p(r['ADE_p'], ade_sig)} & "
                f"{_fmt_est(r['total_effect'], total_sig)} & "
                f"{_fmt_ci(r['total_lower'], r['total_upper'], total_sig)} & "
                f"{_fmt_p(r['total_p'], total_sig)} \\\\"
            )
            rows.append(row_str)

        if tp_idx < len(SUMMARY_TIMEPOINTS) - 1 and rows and not rows[-1].endswith("\\midrule"):
            rows.append("\\midrule")

    if not rows:
        return None, None

    return rows, n_obs


def _build_condensed_lmer_from_csv(df, meal_key):
    """Build a single condensed LMER LaTeX table for *meal_key* from CSV data.

    Selects PCA covariate mode, all three dose contrasts (+15g, +30g, +45g),
    and builds rows grouped by timepoint.

    Returns the LaTeX string, or *None* if insufficient data.
    """
    rows, n_obs = _lmer_rows_from_csv(df, meal_key)
    if rows is None:
        return None

    display_name = MEAL_CONFIG[meal_key]["display_name"]
    sym = _sample_size_symbol(meal_key)
    body = "\n".join(rows)

    latex = rf"""\begin{{table}}[H]
\centering
\small
\caption{{\textbf{{\boldmath LMER causal mediation effects --- {display_name} (${sym} = {n_obs}$ meal observations, all contrast sizes).}} ACME = Average Causal Mediation Effect; ADE = Average Direct Effect; Total = ACME + ADE. Est.\ = point estimate (mg/dL); CI = confidence interval. Significant results ($p < 0.05$) are shown in bold.}}
\label{{tab:lmer_condensed_{meal_key}}}
\begin{{tabular}}{{llccccccccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{ACME}} & \multicolumn{{3}}{{c}}{{ADE}} & \multicolumn{{3}}{{c}}{{Total}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}} \cmidrule(lr){{9-11}}
Time & Dose & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    return latex


def generate_condensed_lmer_tables(df):
    """Generate all condensed LMER tables to the paper_visualizations tables directory.

    Strategy: prefer generating from CSV data (honours SUMMARY_TIMEPOINTS).
    Falls back to reading existing ``.tex`` files and replacing their
    caption/label when CSV data is unavailable.
    Returns the number of tables written.
    """
    _STAGING_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for meal_key in MEAL_TYPE_ORDER:
        out_path = _STAGING_TABLES_DIR / f"lmer_condensed_{meal_key}.tex"

        # Strategy 1: generate from CSV data (respects SUMMARY_TIMEPOINTS)
        if df is not None and not df.empty:
            latex = _build_condensed_lmer_from_csv(df, meal_key)
            if latex is not None:
                _write_latex_table(out_path, latex)
                print(f"  Saved (from CSV): {out_path}")
                count += 1
                continue

        # Strategy 2: read existing .tex and replace caption/label
        src_tex = (
            VISUALIZATIONS_DIR
            / "mediation_visualizations" / "tables" / "pca"
            / meal_key / "table_lmer_offsets.tex"
        )
        if src_tex.exists():
            tex_content = src_tex.read_text()
            latex = _build_condensed_lmer_from_tex(tex_content, meal_key)
            _write_latex_table(out_path, latex)
            print(f"  Saved (from .tex): {out_path}")
            count += 1
            continue

        print(f"  Skipping {meal_key}: no LMER data or existing .tex found")

    return count


def _build_lmer_offsets_from_csv(df, meal_key):
    """Build a ``table_lmer_offsets.tex`` table for *meal_key* from CSV data.

    Uses the same row body as the condensed LMER tables but wraps with the
    caption / label expected by the main-text and appendix ``\\input`` calls.

    Returns the LaTeX string, or *None* if insufficient data.
    """
    rows, n_obs = _lmer_rows_from_csv(df, meal_key)
    if rows is None:
        return None

    display_name = MEAL_CONFIG[meal_key]["display_name"]
    sym = _sample_size_symbol(meal_key)
    body = "\n".join(rows)

    latex = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{\boldmath PCA covariate --- "
        + display_name
        + (r": LMER causal mediation effects across treatment doses ($"
           if "meals" in display_name.lower() else
           r" meals: LMER causal mediation effects across treatment doses ($")
        + sym + r" = "
        + str(n_obs)
        + r"$ meal observations, all contrast sizes).}" "\n"
        r"Time = postprandial measurement time point in minutes after meal start." "\n"
        r"Dose = hypothetical increase in carbohydrate intake (grams) above the meal-type-specific median." "\n"
        r"ACME = Average Causal Mediation Effect (indirect effect mediated through insulin);" "\n"
        r"ADE = Average Direct Effect (effect not mediated through insulin);" "\n"
        r"Total = ACME + ADE." "\n"
        r"Est.\ = point estimate (mg/dL); 95\% CI from quasi-Bayesian approximation with 1000 Monte Carlo simulations;" "\n"
        r"$p$ = two-sided p-value testing the null hypothesis that the effect equals zero." "\n"
        r"Significant results ($p < 0.05$) are shown in bold.}" "\n"
        r"\label{tab:lmer_offsets_pca_"
        + meal_key
        + r"}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{llccccccccc}" "\n"
        r"\toprule" "\n"
        r"& & \multicolumn{3}{c}{ACME} & \multicolumn{3}{c}{ADE} & \multicolumn{3}{c}{Total} \\" "\n"
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}" "\n"
        r"Time & Dose & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ & Est. & 95\% CI & $p$ \\" "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"}" "\n"
        r"\end{table*}" "\n"
    )
    return latex


def generate_per_meal_lmer_tables(df):
    """Generate ``table_lmer_offsets.tex`` in each meal-type subdirectory.

    These are the tables that the main text and appendix ``\\input`` directly.
    Built from CSV data so they respect ``SUMMARY_TIMEPOINTS`` (including
    210 min).  Falls back to copying any pre-existing source files.
    Returns the number of tables written.
    """
    count = 0
    for meal_key in MEAL_TYPE_ORDER:
        out_dir = _STAGING_TABLES_DIR / meal_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "table_lmer_offsets.tex"

        # Strategy 1: generate from CSV data
        if df is not None and not df.empty:
            latex = _build_lmer_offsets_from_csv(df, meal_key)
            if latex is not None:
                _write_latex_table(out_path, latex)
                print(f"  Saved (from CSV): {out_path}")
                count += 1
                continue

        # Strategy 2: copy pre-existing source file
        src_tex = (
            VISUALIZATIONS_DIR
            / "mediation_visualizations" / "tables" / "pca"
            / meal_key / "table_lmer_offsets.tex"
        )
        if src_tex.exists():
            shutil.copy2(src_tex, out_path)
            print(f"  Copied (from source): {out_path}")
            count += 1
            continue

        print(f"  Skipping {meal_key}/table_lmer_offsets.tex: no data or source file")

    return count


# =========================================================================
# Table generation -- incremental data experiment tables (from CSV)
# =========================================================================

# Data directory containing pre-computed experiment CSVs
_EXPERIMENT_DATA_DIR = VISUALIZATIONS_DIR / "incremental_data_experiment" / "data"

# Mapping from internal config/penalty names to publication-quality labels.
_PENALTY_DISPLAY_NAMES = {
    "none":                "None",
    "linear_only":         "Linear only",
    "balance_only":        "Balance only",
    "linear+balance":      "Linear + Balance",
    "linear+balance+ci":   "Linear + Balance + CI",
    "all_penalties":       "All penalties",
    "baseline":            "Baseline (none)",
    "lin_bal_ci":          "Linear + Balance + CI",
    "lin_bal_stab":        "Linear + Balance + Stability",
    "lin_bal":             "Linear + Balance",
    "bal_stab":            "Balance + Stability",
    "linearization":       "Linearization",
    "balancing":           "Balancing",
    "ci_penalty":          "CI penalty",
    "stability":           "Stability",
    "balancing_ci_penalty": "Balancing + CI",
    "balancing_stability":  "Balancing + Stability",
    "ci_penalty_stability": "CI + Stability",
    "linearization_balancing": "Linearization + Balancing",
    "linearization_ci_penalty": "Linearization + CI",
    "linearization_stability":  "Linearization + Stability",
    "balancing_ci_penalty_stability": "Balancing + CI + Stability",
    "linearization_balancing_ci_penalty": "Lin. + Bal. + CI",
    "linearization_balancing_stability":  "Lin. + Bal. + Stability",
    "linearization_balancing_ci_penalty_stability": "All penalties",
}


def _penalty_display(raw: str) -> str:
    """Return a publication-quality display name for a penalty config string."""
    return _PENALTY_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def _fmt_pm(mean, std, decimals=3):
    r"""Format mean \pm std for LaTeX."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def _load_experiment_csvs():
    """Load experiment CSV files from the data directory.

    Returns a dict with keys 'architecture' and/or 'ablation', each
    mapping to a DataFrame.  Returns an empty dict if no data is found.
    """
    results = {}
    if not _EXPERIMENT_DATA_DIR.exists():
        return results

    # Architecture comparison: comprehensive_comparison_*.csv
    arch_files = sorted(
        _EXPERIMENT_DATA_DIR.glob("comprehensive_comparison_*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if arch_files:
        try:
            results["architecture"] = pd.read_csv(arch_files[0])
            print(f"  Loaded architecture comparison: {arch_files[0].name}")
        except Exception as exc:
            print(f"  Warning: failed to read {arch_files[0]}: {exc}")

    # Penalization ablation: ablation_results*.csv
    ablation_files = sorted(
        _EXPERIMENT_DATA_DIR.glob("ablation_results*.csv"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if ablation_files:
        try:
            results["ablation"] = pd.read_csv(ablation_files[0])
            print(f"  Loaded ablation results: {ablation_files[0].name}")
        except Exception as exc:
            print(f"  Warning: failed to read {ablation_files[0]}: {exc}")

    return results


def _resolve_col(df, candidates):
    """Return the first column name from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _generate_comprehensive_top10_tex(df):
    r"""Generate ``table_comprehensive_top10.tex`` from architecture comparison data.

    Returns the LaTeX string, or *None* if data is insufficient.
    """
    if df is None or df.empty:
        return None

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if df.empty or "architecture" not in df.columns:
        return None

    r2_col = _resolve_col(df, ["test_outcome_r2", "outcome_r2_mean", "outcome_R2"])
    balance_col = _resolve_col(df, ["test_balance_score", "balance_score"])
    opt_col = _resolve_col(df, ["optimizer"])
    pen_col = _resolve_col(df, ["penalty_config", "penalty", "penalization"])

    if not r2_col:
        return None

    group_cols = [c for c in ["architecture", opt_col, pen_col] if c]
    agg_dict = {r2_col: ["mean", "std"]}
    if balance_col:
        agg_dict[balance_col] = ["mean", "std"]

    summary = df.groupby(group_cols).agg(agg_dict).round(4)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    bal_mean = f"{balance_col}_mean" if balance_col else None
    r2_mean = f"{r2_col}_mean"
    r2_std = f"{r2_col}_std"
    bal_std = f"{balance_col}_std" if balance_col else None

    if bal_mean and bal_mean in summary.columns:
        summary = summary.sort_values([bal_mean, r2_mean], ascending=[False, False])

    n_seeds = int(df.groupby(group_cols).size().median())
    n_total = len(summary)
    top = summary.head(10)

    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        arch = row["architecture"].upper()
        opt = row[opt_col] if opt_col else "--"
        pen = _penalty_display(row[pen_col]) if pen_col else "--"
        r2_str = _fmt_pm(row[r2_mean], row[r2_std])
        bal_str = _fmt_pm(row[bal_mean], row[bal_std]) if bal_mean else "--"
        rows.append(f"{rank} & {arch} & {opt} & {pen} & {r2_str} & {bal_str} \\\\")

    latex = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{\boldmath Top 10 autoencoder configurations ranked by balance score "
        r"then outcome $R^2$ (" + str(n_total) + r" total configurations, "
        + str(n_seeds) + r" seeds each).}" "\n"
        r"Each configuration is a combination of encoder architecture, optimizer, "
        r"and penalty regime." "\n"
        r"Balance score $= 1 - |0.5 - \text{AUC}_{\text{treatment}}| \times 2$ "
        r"(1.0 = ideal)." "\n"
        r"Values shown as mean $\pm$ SD across seeds. "
        r"Training on combined 2018+2020 data.}" "\n"
        r"\label{tab:comprehensive_top10}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{clllcc}" "\n"
        r"\toprule" "\n"
        r"Rank & Architecture & Optimizer & Penalty & Outcome $R^2$ & Balance Score \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"}" "\n"
        r"\end{table*}" "\n"
    )
    return latex


def _generate_arch_optimizer_summary_tex(df):
    r"""Generate ``table_arch_optimizer_summary.tex`` from architecture comparison data.

    Returns the LaTeX string, or *None* if data is insufficient.
    """
    if df is None or df.empty:
        return None

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if df.empty or "architecture" not in df.columns:
        return None

    r2_col = _resolve_col(df, ["test_outcome_r2", "outcome_r2_mean", "outcome_R2"])
    balance_col = _resolve_col(df, ["test_balance_score", "balance_score"])
    opt_col = _resolve_col(df, ["optimizer"])

    if not r2_col or not opt_col or not balance_col:
        return None

    agg = df.groupby(["architecture", opt_col]).agg(
        {r2_col: ["mean", "std"], balance_col: ["mean", "std"]}
    ).round(4)
    agg.columns = ["r2_mean", "r2_std", "bal_mean", "bal_std"]
    agg = agg.reset_index().sort_values(
        ["bal_mean", "r2_mean"], ascending=[False, False]
    )

    rows = []
    for _, row in agg.iterrows():
        rows.append(
            f"{row['architecture'].upper()} & {row[opt_col]} & "
            f"{_fmt_pm(row['r2_mean'], row['r2_std'])} & "
            f"{_fmt_pm(row['bal_mean'], row['bal_std'])} \\\\"
        )

    latex = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{\boldmath Architecture $\times$ optimizer comparison "
        r"(averaged over penalties and seeds).}" "\n"
        r"Values shown as mean $\pm$ SD. "
        r"Sorted by balance score, then $R^2$.}" "\n"
        r"\label{tab:arch_optimizer_summary}" "\n"
        r"\begin{tabular}{llcc}" "\n"
        r"\toprule" "\n"
        r"Architecture & Optimizer & Outcome $R^2$ & Balance Score \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    return latex


def _generate_ablation_top_configs_tex(df):
    r"""Generate ``table_ablation_top_configs.tex`` from ablation data.

    Returns the LaTeX string, or *None* if data is insufficient.
    """
    if df is None or df.empty:
        return None

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if df.empty:
        return None

    pen_col = _resolve_col(df, ["config_name", "penalty_config", "penalty"])
    r2_col = _resolve_col(df, ["test_outcome_r2", "outcome_r2_mean"])
    balance_col = _resolve_col(df, ["test_balance_score", "balance_score"])
    mediator_col = _resolve_col(df, ["mediator_r2"])
    treat_auc_col = _resolve_col(df, ["treatment_auc"])

    if not pen_col or not r2_col:
        return None

    agg_dict = {r2_col: ["mean", "std"]}
    if balance_col:
        agg_dict[balance_col] = ["mean", "std"]
    if mediator_col:
        agg_dict[mediator_col] = ["mean", "std"]
    if treat_auc_col:
        agg_dict[treat_auc_col] = ["mean", "std"]

    summary = df.groupby(pen_col).agg(agg_dict).round(4)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    bal_m = f"{balance_col}_mean" if balance_col else None
    r2_m = f"{r2_col}_mean"
    if bal_m and bal_m in summary.columns:
        summary = summary.sort_values([bal_m, r2_m], ascending=[False, False])
    else:
        summary = summary.sort_values(r2_m, ascending=False)

    n_seeds = int(df.groupby(pen_col).size().median())
    top = summary.head(10)

    has_bal = bal_m and bal_m in top.columns
    has_med = mediator_col and f"{mediator_col}_mean" in top.columns
    has_auc = treat_auc_col and f"{treat_auc_col}_mean" in top.columns

    col_headers = ["Rank", "Configuration", r"Outcome $R^2$"]
    if has_bal:
        col_headers.append("Balance Score")
    if has_med:
        col_headers.append(r"Mediator $R^2$")
    if has_auc:
        col_headers.append("Treatment AUC")

    col_spec = "cl" + "c" * (len(col_headers) - 2)

    rows = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        config = _penalty_display(row[pen_col]) if isinstance(row[pen_col], str) else str(row[pen_col])
        cells = [str(rank), config,
                 _fmt_pm(row[f"{r2_col}_mean"], row[f"{r2_col}_std"])]
        if has_bal:
            cells.append(_fmt_pm(row[bal_m], row[f"{balance_col}_std"]))
        if has_med:
            cells.append(_fmt_pm(row[f"{mediator_col}_mean"], row[f"{mediator_col}_std"]))
        if has_auc:
            cells.append(_fmt_pm(row[f"{treat_auc_col}_mean"], row[f"{treat_auc_col}_std"]))
        rows.append(" & ".join(cells) + " \\\\")

    header_line = " & ".join(col_headers) + " \\\\"

    latex = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{Top 10 penalization configurations from the ablation study ("
        + str(n_seeds) + r" seeds each).}" "\n"
        r"All $2^4 = 16$ combinations of the four penalty layers are evaluated." "\n"
        r"Architecture: CNN; optimizer: rmsprop." "\n"
        r"Values shown as mean $\pm$ SD across seeds. "
        r"Training on combined 2018+2020 data." "\n"
        r"Treatment AUC ideal is 0.5 (no predictive power "
        r"$\Rightarrow$ balanced embedding).}" "\n"
        r"\label{tab:ablation_top_configs}" "\n"
        r"\resizebox{\textwidth}{!}{%" "\n"
        r"\begin{tabular}{" + col_spec + r"}" "\n"
        r"\toprule" "\n"
        + header_line + "\n"
        r"\midrule" "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}%" "\n"
        r"}" "\n"
        r"\end{table*}" "\n"
    )
    return latex


def _generate_ablation_marginal_tex(df):
    r"""Generate ``table_ablation_marginal.tex`` from ablation data.

    Returns the LaTeX string, or *None* if data is insufficient.
    """
    if df is None or df.empty:
        return None

    if "status" in df.columns:
        df = df[df["status"] == "success"].copy()
    if df.empty:
        return None

    r2_col = _resolve_col(df, ["test_outcome_r2", "outcome_r2_mean"])
    balance_col = _resolve_col(df, ["test_balance_score", "balance_score"])
    mediator_col = _resolve_col(df, ["mediator_r2"])

    penalty_flags = ["linearization", "balancing", "ci_penalty", "stability"]
    available_flags = [f for f in penalty_flags if f in df.columns]

    if not available_flags or not r2_col:
        return None

    metric_cols = []
    metric_labels = []
    if r2_col:
        metric_cols.append(r2_col)
        metric_labels.append(r"$\Delta R^2$")
    if balance_col:
        metric_cols.append(balance_col)
        metric_labels.append(r"$\Delta$ Balance")
    if mediator_col:
        metric_cols.append(mediator_col)
        metric_labels.append(r"$\Delta$ Mediator $R^2$")

    marg_rows = []
    for flag in available_flags:
        # Convert to bool in case of string "True"/"False"
        flag_vals = df[flag].astype(str).str.lower().isin(["true", "1"])
        with_flag = df[flag_vals]
        without_flag = df[~flag_vals]

        cells = [_penalty_display(flag)]
        for mc in metric_cols:
            delta = with_flag[mc].mean() - without_flag[mc].mean()
            sign = "+" if delta >= 0 else ""
            cells.append(f"{sign}{delta:.4f}")
        marg_rows.append(" & ".join(cells) + " \\\\")

    marg_header = "Penalty Layer & " + " & ".join(metric_labels) + " \\\\"
    marg_col_spec = "l" + "c" * len(metric_cols)

    latex = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{Marginal contribution of each penalty layer.}" "\n"
        r"$\Delta$ = mean metric with penalty enabled $-$ mean metric with "
        r"penalty disabled," "\n"
        r"averaged across all other penalty combinations and seeds." "\n"
        r"Positive $\Delta R^2$ and $\Delta$ Balance are desirable.}" "\n"
        r"\label{tab:ablation_marginal}" "\n"
        r"\begin{tabular}{" + marg_col_spec + r"}" "\n"
        r"\toprule" "\n"
        + marg_header + "\n"
        r"\midrule" "\n"
        + "\n".join(marg_rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    return latex


def generate_incremental_experiment_tables():
    """Generate LaTeX tables for the incremental data experiment.

    Reads CSV data from ``incremental_data_experiment/data/`` and produces
    up to 4 ``.tex`` files directly in ``paper_visualizations/tables/``:

    - ``table_comprehensive_top10.tex``
    - ``table_arch_optimizer_summary.tex``
    - ``table_ablation_top_configs.tex``
    - ``table_ablation_marginal.tex``

    Returns the number of tables written.
    """
    print("\nGenerating incremental experiment tables from CSV data ...")
    print(f"  Data directory: {_EXPERIMENT_DATA_DIR}")

    exp_data = _load_experiment_csvs()
    if not exp_data:
        print("  No experiment CSV data found; skipping incremental tables.")
        return 0

    _STAGING_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    # --- Architecture comparison tables ---
    arch_df = exp_data.get("architecture")
    if arch_df is not None:
        for name, generator in [
            ("table_comprehensive_top10", _generate_comprehensive_top10_tex),
            ("table_arch_optimizer_summary", _generate_arch_optimizer_summary_tex),
        ]:
            latex = generator(arch_df)
            if latex:
                out = _STAGING_TABLES_DIR / f"{name}.tex"
                _write_latex_table(out, latex)
                print(f"  Saved: {out}")
                count += 1
            else:
                print(f"  Skipping {name}: insufficient data")
    else:
        print("  No architecture comparison CSV found")

    # --- Ablation tables ---
    ablation_df = exp_data.get("ablation")
    if ablation_df is not None:
        for name, generator in [
            ("table_ablation_top_configs", _generate_ablation_top_configs_tex),
            ("table_ablation_marginal", _generate_ablation_marginal_tex),
        ]:
            latex = generator(ablation_df)
            if latex:
                out = _STAGING_TABLES_DIR / f"{name}.tex"
                _write_latex_table(out, latex)
                print(f"  Saved: {out}")
                count += 1
            else:
                print(f"  Skipping {name}: insufficient data")
    else:
        print("  No ablation results CSV found")

    print(f"  Generated {count} incremental experiment tables")
    return count


# =========================================================================
# Composite figure generation (matplotlib, NOT Pillow overlay)
# =========================================================================

# Dose subtitles for LMER composite panels
_LMER_SUBTITLES = ["+15 g Carbohydrates", "+30 g Carbohydrates",
                  "+45 g Carbohydrates"]

# Subtitles for QR composite panels (row-major: 3 doses × 3 taus)
_QR_SUBTITLES = [
    "+15g, \u03c4=0.25", "+15g, \u03c4=0.50", "+15g, \u03c4=0.75",
    "+30g, \u03c4=0.25", "+30g, \u03c4=0.50", "+30g, \u03c4=0.75",
    "+45g, \u03c4=0.25", "+45g, \u03c4=0.50", "+45g, \u03c4=0.75",
]

# Default PCA variance ratios (from the trained model) used in axis labels
# when regenerating the PCA figure from data.
_PCA_VARIANCE_DEFAULTS = {
    "PC1": 0.665,
    "PC2": 0.146,
    "PC3": 0.088,
}

# Set2-derived palette for meal types (matches the original visualize script)
_MEAL_TYPE_COLORS = {
    "Breakfast": "#66c2a5",
    "Dinner":    "#fc8d62",
    "Lunch":     "#8da0cb",
    "Snack":     "#e78ac3",
}

# Possible locations for PCA embeddings CSVs
_EMBEDDINGS_SEARCH_PATHS = [
    "cma_cluster/analysis_data/embeddings",
    "analysis_data/embeddings",
]


def _make_legend_handles():
    """Create legend handles for the mediation effect colours + significance."""
    handles = []
    for _, _, _, color, marker, label in _MEDIATION_EFFECTS:
        h = plt.Line2D([0], [0], color=color, marker=marker, markersize=7,
                       linewidth=2, label=label)
        handles.append(h)
    # Significance star entry
    h_sig = plt.Line2D([0], [0], linestyle="none", marker="*", color="gray",
                       markersize=12, label="Significant (p < 0.05)")
    handles.append(h_sig)
    return handles


def generate_lmer_composites(dpi=300, mediation_df=None):
    """Generate LMER composite figures (1x3 grid) for each meal type.

    **Preferred path** (CSV regeneration): plots from *mediation_df* with a
    common y-axis, single shared x/y labels, and one legend at the bottom.

    **Fallback** (image-based): uses pre-rendered PNGs with title cropping.

    Returns the number of composites generated.
    """
    print("\nGenerating LMER composite figures ...")
    dose_values = [15, 30, 45]
    dose_labels = ["+15 g Carbohydrates", "+30 g Carbohydrates",
                   "+45 g Carbohydrates"]
    count = 0

    for meal in MEAL_TYPE_ORDER:
        out = _STAGING_FIGURES_DIR / meal / f"fig_lmer_composite_{meal}.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- Strategy 1: regenerate from CSV data ---
        if mediation_df is not None and not mediation_df.empty:
            panels = [_get_panel_data(mediation_df, "lmer", meal, d)
                      for d in dose_values]
            if all(p is not None for p in panels):
                ylim = _compute_common_ylim(panels)
                n_cols = len(panels)

                fig, axes = plt.subplots(
                    1, n_cols, figsize=(5.5 * n_cols, 5.5), sharey=True,
                    sharex=True,
                )
                for c, (ax, pdata, lbl, letter) in enumerate(
                    zip(axes, panels, dose_labels, "abc")
                ):
                    _plot_mediation_panel(ax, pdata, ylim)
                    ax.set_title(lbl, fontsize=16, fontweight="bold", pad=10)
                    # Panel letter just outside top-left corner
                    ax.text(
                        -0.03, 1.01, letter, transform=ax.transAxes,
                        fontsize=14, fontweight="bold",
                        va="bottom", ha="right",
                    )

                # Shared axis labels — bold, large
                fig.text(0.5, 0.00, "Minutes Post Meal",
                         ha="center", fontsize=18, fontweight="bold")
                axes[0].set_ylabel("Effect on Glucose (mg/dL)",
                                   fontsize=18, fontweight="bold")

                # Common legend — well below x-title to avoid overlap
                fig.legend(
                    handles=_make_legend_handles(), loc="lower center",
                    bbox_to_anchor=(0.5, -0.12),
                    ncol=4, fontsize=14, frameon=True, fancybox=True,
                    edgecolor="0.6", framealpha=0.9,
                )
                fig.subplots_adjust(
                    left=0.08, right=0.98, bottom=0.08, top=0.88,
                    wspace=0.10,
                )
                fig.savefig(str(out), dpi=dpi, bbox_inches="tight",
                            facecolor="white")
                plt.close(fig)
                print(f"  Saved (from CSV): {out}")
                count += 1
                continue

        # --- Strategy 2: fallback to pre-rendered PNGs ---
        panels_base = RAW_DIRS["mediation_vis"] / "figures" / "pca"
        panel_paths = [
            panels_base / meal / f"fig_effects_lmer_offset{dose}.png"
            for dose in DOSES
        ]
        if not all(p.exists() for p in panel_paths):
            missing = [str(p) for p in panel_paths if not p.exists()]
            print(f"  Skipping {meal}: missing {len(missing)} panel(s)")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img_path, label, subtitle in zip(
            axes, panel_paths, "abc", _LMER_SUBTITLES
        ):
            img = plt.imread(str(img_path))
            img = _crop_title_region(img, 0.08)
            ax.imshow(img)
            ax.axis("off")
            ax.text(
                0.02, 0.98, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.8),
            )
            ax.set_title(subtitle, fontsize=14, fontweight="bold", pad=10)

        fig.tight_layout()
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved (from PNG): {out}")
        count += 1

    print(f"  Generated {count} LMER composite figures")
    return count


def generate_qr_composites(dpi=300, mediation_df=None):
    """Generate QR composite figures (3x3 grid) for each meal type.

    Layout (rows = doses, columns = quantiles)::

             tau=0.25    tau=0.50    tau=0.75
      +15g      a           b           c
      +30g      d           e           f
      +45g      g           h           i

    **Preferred path** (CSV regeneration): common y-axis, single shared axis
    labels, one legend, bold row/column headers.

    **Fallback** (image-based): pre-rendered PNGs with title cropping.

    Returns the number of composites generated.
    """
    print("\nGenerating QR composite figures ...")
    dose_values = [15, 30, 45]
    tau_values = [0.25, 0.50, 0.75]
    row_labels = ["+15 g Carbohydrates", "+30 g Carbohydrates",
                  "+45 g Carbohydrates"]
    col_labels = ["\u03c4 = 0.25", "\u03c4 = 0.50", "\u03c4 = 0.75"]

    count = 0
    for meal in MEAL_TYPE_ORDER:
        out = _STAGING_FIGURES_DIR / meal / f"fig_qr_composite_{meal}.png"
        out.parent.mkdir(parents=True, exist_ok=True)

        # --- Strategy 1: regenerate from CSV data ---
        if mediation_df is not None and not mediation_df.empty:
            # Collect panel data: rows=doses, cols=taus
            all_panels = []
            for dose in dose_values:
                row_panels = []
                for tau in tau_values:
                    row_panels.append(
                        _get_panel_data(mediation_df, "qr", meal, dose, tau)
                    )
                all_panels.append(row_panels)

            flat = [p for row in all_panels for p in row]
            if all(p is not None for p in flat):
                ylim = _compute_common_ylim(flat)
                n_rows, n_cols = 3, 3
                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows),
                    sharey=True, sharex=True,
                )
                labels_iter = iter("abcdefghi")

                for r in range(n_rows):
                    for c in range(n_cols):
                        ax = axes[r, c]
                        _plot_mediation_panel(ax, all_panels[r][c], ylim)
                        # Panel letter just outside top-left corner
                        ax.text(
                            -0.03, 1.01, next(labels_iter),
                            transform=ax.transAxes,
                            fontsize=14, fontweight="bold",
                            va="bottom", ha="right",
                        )

                # Bold column headers
                for c, cl in enumerate(col_labels):
                    axes[0, c].set_title(cl, fontsize=16, fontweight="bold",
                                         pad=10)

                # Row labels on the RIGHT side, close to panel edge
                for r, rl in enumerate(row_labels):
                    axes[r, n_cols - 1].annotate(
                        rl, xy=(1, 0.5),
                        xytext=(10, 0), textcoords="offset points",
                        xycoords="axes fraction",
                        fontsize=14, fontweight="bold",
                        va="center", ha="left", rotation=-90,
                    )

                # Shared Y-axis label — bold, LARGER than row labels
                axes[1, 0].set_ylabel("Effect on Glucose (mg/dL)",
                                      fontsize=20, fontweight="bold")

                # Shared X-axis label — bold
                fig.text(0.5, 0.01, "Minutes Post Meal",
                         ha="center", fontsize=20, fontweight="bold")

                # Common legend
                fig.legend(
                    handles=_make_legend_handles(), loc="lower center",
                    bbox_to_anchor=(0.5, -0.04),
                    ncol=4, fontsize=15, frameon=True, fancybox=True,
                    edgecolor="0.6", framealpha=0.9,
                )
                fig.subplots_adjust(
                    left=0.07, right=0.92, bottom=0.06, top=0.94,
                    hspace=0.15, wspace=0.10,
                )
                fig.savefig(str(out), dpi=dpi, bbox_inches="tight",
                            facecolor="white")
                plt.close(fig)
                print(f"  Saved (from CSV): {out}")
                count += 1
                continue

        # --- Strategy 2: fallback to pre-rendered PNGs ---
        panels_base = RAW_DIRS["mediation_vis"] / "figures" / "pca"
        panel_fnames = []
        for dose in DOSES:
            for tau_str in TAU_STRINGS:
                panel_fnames.append(
                    f"fig_effects_qr_tau{tau_str}_offset{dose}.png"
                )
        panel_paths = [panels_base / meal / fn for fn in panel_fnames]
        if not all(p.exists() for p in panel_paths):
            missing = [str(p) for p in panel_paths if not p.exists()]
            print(f"  Skipping {meal}: missing {len(missing)} panel(s)")
            continue

        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        labels_str = "abcdefghi"
        for idx, (ax, img_path, label) in enumerate(
            zip(axes.flat, panel_paths, labels_str)
        ):
            img = plt.imread(str(img_path))
            img = _crop_title_region(img, 0.08)
            ax.imshow(img)
            ax.axis("off")
            ax.text(
                0.02, 0.98, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.8),
            )

        # Bold column / row headers for fallback
        for c, cl in enumerate(col_labels):
            axes[0, c].set_title(cl, fontsize=15, fontweight="bold", pad=12)
        for r, rl in enumerate(row_labels):
            axes[r, 0].text(
                -0.08, 0.5, rl, transform=axes[r, 0].transAxes,
                fontsize=15, fontweight="bold",
                va="center", ha="center", rotation=90,
            )

        fig.subplots_adjust(hspace=0.12, wspace=0.05)
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved (from PNG): {out}")
        count += 1

    print(f"  Generated {count} QR composite figures")
    return count


def _find_embeddings_csv():
    """Search for PCA embeddings CSV in known locations.

    Returns the path to the most recent ``phi_embeddings_combined_*.csv``
    or *None*.
    """
    for rel in _EMBEDDINGS_SEARCH_PATHS:
        d = PROJECT_ROOT / rel
        if d.is_dir():
            candidates = sorted(d.glob("phi_embeddings_combined_*.csv"),
                                reverse=True)
            if candidates:
                return candidates[0]
    return None


def _draw_confidence_ellipse(ax, x, y, color, n_std=2.4477, **kwargs):
    """Draw a 95 % confidence ellipse for the 2-D data on *ax*.

    The default *n_std* of 2.4477 equals sqrt(chi2.ppf(0.95, df=2)),
    which is the correct scaling for a 2-D confidence region.
    """
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(eigvals)
    ell = mpatches.Ellipse(
        xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
        edgecolor=color, facecolor="none", linewidth=1.5, linestyle="--",
        **kwargs,
    )
    ax.add_patch(ell)


def _regenerate_pca_from_data(csv_path, output_path, dpi=300):
    """Regenerate the PCA meal-type figure from embeddings CSV data.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"    Failed to read {csv_path}: {exc}")
        return False

    # Identify PC columns
    pc_cols = sorted(
        [c for c in df.columns if c.startswith("PC_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if len(pc_cols) < 3 or "meal_type" not in df.columns:
        print("    CSV lacks required PC_1/PC_2/PC_3 or meal_type columns")
        return False

    pc1 = df["PC_1"].values
    pc2 = df["PC_2"].values
    pc3 = df["PC_3"].values

    # Variance labels -- use defaults (may be overridden if stored)
    var = _PCA_VARIANCE_DEFAULTS
    var_labels = [
        f"PC1 ({var['PC1']:.1%})",
        f"PC2 ({var['PC2']:.1%})",
        f"PC3 ({var['PC3']:.1%})",
    ]

    pair_specs = [
        (pc1, pc2, var_labels[0], var_labels[1]),  # PC1 vs PC2
        (pc1, pc3, var_labels[0], var_labels[2]),  # PC1 vs PC3
        (pc2, pc3, var_labels[1], var_labels[2]),  # PC2 vs PC3
    ]

    meal_types = sorted(df["meal_type"].unique())
    colors = {m: _MEAL_TYPE_COLORS.get(m, "#999999") for m in meal_types}

    # 2x2 layout: 3 scatter panels + 1 legend-only panel
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    panel_labels = "abc"
    scatter_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for col_idx, (xdata, ydata, xlabel, ylabel) in enumerate(pair_specs):
        ax = scatter_axes[col_idx]
        for meal in meal_types:
            mask = df["meal_type"] == meal
            ax.scatter(
                xdata[mask], ydata[mask],
                color=colors[meal], label=meal,
                s=22, alpha=0.7, edgecolors="none", rasterized=True,
            )
            _draw_confidence_ellipse(
                ax, xdata[mask], ydata[mask], color=colors[meal],
            )
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        xi_label = xlabel.split(" ")[0]
        yi_label = ylabel.split(" ")[0]
        ax.set_title(f"{xi_label} vs {yi_label}",
                     fontweight="bold", fontsize=15, pad=10)
        ax.grid(True, alpha=0.15)
        ax.tick_params(axis="both", labelsize=12)
        # Panel letter outside top-left
        ax.text(
            -0.02, 1.02, panel_labels[col_idx],
            transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right",
            fontfamily="sans-serif",
        )

    # 4th panel (bottom-right): legend only
    ax_leg = axes[1, 1]
    ax_leg.set_axis_off()
    legend_handles = []
    for meal in meal_types:
        h = plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[meal], markersize=10,
                       label=meal)
        legend_handles.append(h)
    # Confidence ellipse entry
    h_ell = mpatches.Patch(facecolor="none", edgecolor="gray",
                           linestyle="--", linewidth=1.5,
                           label="95% CI Ellipse")
    legend_handles.append(h_ell)

    ax_leg.legend(
        handles=legend_handles, title="Meal Type",
        loc="center", fontsize=14, title_fontsize=16,
        frameon=True, fancybox=True, edgecolor="0.6", framealpha=0.9,
        markerscale=1.5, borderpad=1.5, labelspacing=1.2,
    )

    fig.suptitle(
        "Main PCA \u2014 Pairwise Views by Meal Type",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return True


# ---------- PIL fallback helpers (kept for PCA label fix) ----------

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_bold_font(size):
    """Try common bold font paths; fall back to PIL default."""
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _add_pil_panel_label(img, label, x, y, font_size, bg=True):
    """Draw a bold panel label on a PIL Image at (*x*, *y*) pixel coords."""
    draw = ImageDraw.Draw(img)
    font = _load_bold_font(font_size)
    bbox = draw.textbbox((x, y), label, font=font)
    if bg:
        pad = 2
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill="white",
        )
    draw.text((x, y), label, fill="black", font=font)
    return img


def _pca_pillow_fallback(path):
    """Fallback: white-out old labels and redraw bold ones using Pillow.

    Opens the copied paper_visualizations PNG, paints generous white rectangles over
    the existing small "a", "b", "c" labels, then draws new larger bold
    labels at the same positions.  Returns 1 on success, 0 on failure.
    """
    if not path.exists():
        return 0

    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Approximate label positions (fraction of image dims)
    old_positions = [
        (0.07, 0.03),   # "a"
        (0.37, 0.03),   # "b"
        (0.67, 0.03),   # "c"
    ]

    # White-out old labels with generous rectangles
    for xf, yf in old_positions:
        x0 = int(xf * w) - 6
        y0 = int(yf * h) - 6
        # 50x30 pixel minimum overwrite
        x1 = x0 + max(50, int(0.04 * w)) + 12
        y1 = y0 + max(30, int(0.04 * h)) + 12
        draw.rectangle([x0, y0, x1, y1], fill="white")

    # Redraw new bold labels (~4% of image height, 2x original)
    new_font_size = max(14, int(h * 0.04))
    for idx, (xf, yf) in enumerate(old_positions):
        label = chr(ord("a") + idx)
        x = int(xf * w)
        y = int(yf * h)
        _add_pil_panel_label(img, label, x, y, new_font_size, bg=True)

    img.save(path)
    return 1


def regenerate_pca_figure(dpi=300):
    """Regenerate the PCA embeddings figure with proper bold panel labels.

    Preferred approach: reload PCA-projected data from CSV and regenerate
    the figure from scratch using matplotlib.

    Fallback: if no data CSV is found, use Pillow to white-out the old
    labels on the copied PNG and redraw larger bold labels.

    Returns 1 on success, 0 if the figure could not be produced.
    """
    print("\nRegenerating PCA figure ...")
    output_path = _STAGING_FIGURES_DIR / "phi_pca_meal_type_main.png"

    # Strategy 1: regenerate from data
    csv_path = _find_embeddings_csv()
    if csv_path is not None:
        print(f"  Found embeddings CSV: {csv_path}")
        if _regenerate_pca_from_data(csv_path, output_path, dpi=dpi):
            print(f"  Regenerated from data: {output_path}")
            return 1
        print("  Data-based regeneration failed; trying Pillow fallback ...")

    # Strategy 2: Pillow fallback on the copied figure
    if output_path.exists():
        ok = _pca_pillow_fallback(output_path)
        if ok:
            print(f"  Fixed via Pillow fallback: {output_path}")
            return 1

    print("  Skipping: no data CSV and no copied figure found")
    return 0


# =========================================================================
# Trajectory figure regeneration (legend placement fix)
# =========================================================================

# Meal-type colours matching summarize_meal_windows.py
_TRAJECTORY_MEAL_COLORS = {
    "breakfast": "#EE7733",
    "lunch":     "#009988",
    "dinner":    "#0077BB",
    "snack":     "#CC3311",
}

_TRAJECTORY_MEAL_ORDER = ["breakfast", "lunch", "dinner", "snack"]


def _get_meal_color_traj(meal):
    """Return the colorblind-safe colour for *meal* (case-insensitive)."""
    return _TRAJECTORY_MEAL_COLORS.get(str(meal).lower(), "gray")


def _find_meal_data_csv():
    """Search for the meal-window embeddings CSV used by trajectory plots.

    Returns the path or *None*.
    """
    search_dirs = [
        PROJECT_ROOT / "cma_cluster" / "analysis_data" / "embeddings",
        PROJECT_ROOT / "analysis_data" / "embeddings",
    ]
    for d in search_dirs:
        if d.is_dir():
            candidates = sorted(d.glob("phi_embeddings_combined_*.csv"),
                                reverse=True)
            if candidates:
                return candidates[0]
    return None


def _detect_column(df, candidates):
    """Return the first column from *candidates* present in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _plot_trajectory_panel(ax, df, meal_type_col, y_cols, time_points,
                           legend_loc="lower left"):
    """Plot mean +/- SE trajectory per meal type on *ax* with legend."""
    all_meals = sorted(df[meal_type_col].dropna().unique(),
                       key=lambda m: (_TRAJECTORY_MEAL_ORDER.index(str(m).lower())
                                      if str(m).lower() in _TRAJECTORY_MEAL_ORDER
                                      else 99))

    for meal in all_meals:
        meal_df = df[df[meal_type_col] == meal]
        n_meal = len(meal_df)
        if n_meal == 0:
            continue

        means = [meal_df[c].mean() for c in y_cols]
        sems = [meal_df[c].std() / np.sqrt(n_meal) for c in y_cols]
        color = _get_meal_color_traj(meal)

        ax.plot(time_points, means, color=color, linewidth=2,
                label=f"{str(meal).capitalize()} (N={n_meal})")
        ax.fill_between(time_points,
                        [m - s for m, s in zip(means, sems)],
                        [m + s for m, s in zip(means, sems)],
                        alpha=0.2, color=color)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Minutes Post-Meal", fontsize=11)
    ax.set_ylabel("Delta Glucose (mg/dL)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True, fancybox=True,
              edgecolor="black", facecolor="white", framealpha=1.0,
              loc=legend_loc)


def _regenerate_trajectory_by_cohort(df, output_path, dpi=300):
    """Regenerate fig16 (trajectory by meal type, split by cohort).

    Legend is placed in the **lower left** of each panel.
    Returns ``True`` on success.
    """
    cohort_col = _detect_column(df, ["cohort", "year", "dataset"])
    meal_col = _detect_column(df, ["meal_type", "meal", "meal_category"])
    if not cohort_col or not meal_col:
        return False

    y_cols = sorted(
        [c for c in df.columns if c.startswith("Y_") and c.endswith("min")],
        key=lambda c: int(c.split("_")[1].replace("min", "")),
    )
    if not y_cols:
        return False

    time_points = [int(c.split("_")[1].replace("min", "")) for c in y_cols]
    cohorts = sorted(df[cohort_col].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, cohort in enumerate(cohorts[:2]):
        ax = axes[idx]
        cohort_df = df[df[cohort_col] == cohort]
        ax.set_title(f"Cohort {cohort}", fontweight="bold", loc="left",
                     fontsize=12)
        _plot_trajectory_panel(ax, cohort_df, meal_col, y_cols, time_points,
                               legend_loc="lower left")

    # Synchronise y-axis
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return True


def _regenerate_trajectory_train_test(df, output_path, dpi=300):
    """Regenerate fig19 (trajectory by meal type, split by train/test).

    Legend is placed in the **lower left** of each panel.
    Returns ``True`` on success.
    """
    meal_col = _detect_column(df, ["meal_type", "meal", "meal_category"])
    split_col = _detect_column(df, ["split", "train_test", "fold"])
    if not meal_col or not split_col:
        return False

    y_cols = sorted(
        [c for c in df.columns if c.startswith("Y_") and c.endswith("min")],
        key=lambda c: int(c.split("_")[1].replace("min", "")),
    )
    if not y_cols:
        return False

    time_points = [int(c.split("_")[1].replace("min", "")) for c in y_cols]
    split_order = [("train", "Training Set"), ("test", "Test Set")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (split_val, panel_title) in enumerate(split_order):
        ax = axes[idx]
        split_df = df[df[split_col].astype(str).str.lower() == split_val]
        ax.set_title(panel_title, fontweight="bold", loc="left", fontsize=12)
        _plot_trajectory_panel(ax, split_df, meal_col, y_cols, time_points,
                               legend_loc="lower left")

    # Synchronise y-axis
    ymin = min(ax.get_ylim()[0] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return True


def regenerate_trajectory_figures(dpi=300):
    """Regenerate trajectory figures with legend in the lower-left corner.

    Strategy 1: Reload from CSV and regenerate.
    Strategy 2: Keep existing copied figures unchanged.

    Returns the number of figures regenerated.
    """
    print("\nRegenerating trajectory figures (legend -> lower left) ...")
    csv_path = _find_meal_data_csv()
    if csv_path is None:
        print("  No embeddings CSV found; keeping existing trajectory figures")
        return 0

    print(f"  Found data CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  Failed to read CSV: {exc}")
        return 0

    count = 0

    # fig16: trajectory by cohort
    out16 = RAW_DIRS["data_distribution"] / "figures" / "fig16_trajectory_by_meal_type.png"
    if _regenerate_trajectory_by_cohort(df, out16, dpi=dpi):
        print(f"  Regenerated: {out16.name}")
        count += 1
    else:
        print("  Could not regenerate fig16 (missing columns)")

    # fig19: trajectory train vs test
    out19 = RAW_DIRS["data_distribution"] / "figures" / "fig19_trajectory_train_vs_test.png"
    if _regenerate_trajectory_train_test(df, out19, dpi=dpi):
        print(f"  Regenerated: {out19.name}")
        count += 1
    else:
        print("  Could not regenerate fig19 (missing columns)")

    return count


# =========================================================================
# Data distribution figure regeneration (common legend)
# =========================================================================

# 16-colour Paul Tol palette for subject IDs (from summarize_meal_windows.py)
_SUBJECT_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#332288", "#88CCEE", "#44AA99",
    "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
    "#AA4499",
]


def _format_subject_label(subject_id):
    """Normalise a subject ID into a clean 'S1', 'S2', ... label.

    Strips redundant cohort prefixes like '2018-2018_1' -> 'S1'.
    """
    s = str(subject_id)
    # Handle 'YYYY-YYYY_N' format (e.g. '2018-2018_3')
    import re as _re
    m = _re.match(r"(\d{4})-\d{4}[_-](\d+)", s)
    if m:
        return f"S{m.group(2)}"
    # Handle 'YYYY-SN' or 'YYYY_N'
    m = _re.match(r"\d{4}[_-]S?(\d+)", s)
    if m:
        return f"S{m.group(1)}"
    # Handle 'YYYY-anything'
    parts = s.split("-", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return s


def _get_cohort_from_subject(subject_id):
    """Extract cohort year from a subject_id_unique like '2018-2018_3'."""
    s = str(subject_id)
    parts = s.split("-", 1)
    if parts[0].isdigit() and len(parts[0]) == 4:
        return parts[0]
    return "Unknown"


def _regenerate_treatment_mediator_by_meal(df, output_path, dpi=300):
    """Regenerate fig08 with a single common legend organised by cohort.

    Returns ``True`` on success.
    """
    meal_col = _detect_column(df, ["meal_type", "meal", "meal_category"])
    treat_col = _detect_column(df, ["treat_meal_carbs", "carbs",
                                     "meal_carbs", "carbohydrates"])
    med_col = _detect_column(df, ["mediator_bolus_for_meal", "bolus_dose",
                                   "bolus", "insulin_bolus"])
    patient_col = _detect_column(df, ["subject_id_unique", "subject_id",
                                       "patient_id", "id", "subject"])

    if not all([meal_col, treat_col, med_col, patient_col]):
        return False

    # Ensure unique subject IDs across cohorts
    cohort_col = _detect_column(df, ["cohort", "year", "dataset"])
    if cohort_col and "subject_id_unique" not in df.columns:
        df = df.copy()
        df["subject_id_unique"] = (
            df[cohort_col].astype(str) + "-" + df[patient_col].astype(str)
        )
        patient_col = "subject_id_unique"

    meal_order = ["breakfast", "lunch", "dinner", "snack"]
    actual_meals = sorted(
        df[meal_col].dropna().unique(),
        key=lambda m: (
            meal_order.index(str(m).lower())
            if str(m).lower() in meal_order else 99
        ),
    )

    # Assign consistent colours to each subject across all panels
    all_subjects = sorted(df[patient_col].dropna().unique())
    subject_colors = {}
    for i, subj in enumerate(all_subjects):
        if i < len(_SUBJECT_PALETTE):
            subject_colors[subj] = _SUBJECT_PALETTE[i]
        else:
            subject_colors[subj] = plt.cm.viridis(i / max(len(all_subjects) - 1, 1))

    n_meals = len(actual_meals)
    ncols = min(2, n_meals)
    nrows = (n_meals + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows),
                             squeeze=False)

    panel_labels = list("abcdefghijkl")
    for idx, meal in enumerate(actual_meals):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        meal_df = df[df[meal_col] == meal]

        for subj in all_subjects:
            subj_df = meal_df[meal_df[patient_col] == subj]
            if subj_df.empty:
                continue
            ax.scatter(
                subj_df[treat_col], subj_df[med_col],
                color=subject_colors[subj], s=30, alpha=0.7,
                edgecolors="none",
            )

        # Regression line
        valid = meal_df[[treat_col, med_col]].dropna()
        if len(valid) > 2:
            from numpy.polynomial.polynomial import polyfit
            coeffs = polyfit(valid[treat_col], valid[med_col], 1)
            x_range = np.linspace(valid[treat_col].min(),
                                  valid[treat_col].max(), 50)
            ax.plot(x_range, coeffs[0] + coeffs[1] * x_range,
                    color="black", linestyle="--", linewidth=1)
            corr = valid[treat_col].corr(valid[med_col])
            ax.text(0.95, 0.05, f"r = {corr:.3f}\nN = {len(valid)}",
                    transform=ax.transAxes, fontsize=9,
                    va="bottom", ha="right",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

        ax.set_xlabel("Carbohydrate Intake (g)", fontsize=10)
        ax.set_ylabel("Insulin Bolus (U)", fontsize=10)
        ax.set_title(str(meal).capitalize(), fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.2)

        if idx < len(panel_labels):
            ax.text(
                -0.03, 1.01, panel_labels[idx],
                transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right",
                fontfamily="sans-serif",
            )

    # Hide unused axes
    for idx in range(n_meals, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    # --- Common legend organised by cohort ---
    from collections import OrderedDict
    cohort_subjects = OrderedDict()
    for subj in all_subjects:
        cohort = _get_cohort_from_subject(subj)
        cohort_subjects.setdefault(cohort, []).append(subj)

    # Every entry has a uniform coloured dot + "Cohort YYYY - SN" label.
    # No invisible header entries — avoids alignment issues.
    legend_handles = []
    for cohort, subjects in cohort_subjects.items():
        for subj in subjects:
            legend_handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=subject_colors[subj],
                    markersize=8,
                    label=f"Cohort {cohort} \u2013 {_format_subject_label(subj)}",
                )
            )

    n_items = len(legend_handles)
    ncol_legend = (n_items + 1) // 2
    fig.legend(
        handles=legend_handles,
        title="Subject",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=ncol_legend,
        fontsize=10, title_fontsize=12,
        frameon=True, fancybox=True,
        edgecolor="black", facecolor="white", framealpha=1.0,
        columnspacing=0.8, handletextpad=0.4,
        labelspacing=0.4,
    )

    fig.suptitle(
        "Treatment\u2013Mediator Relationship by Meal Type",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return True


def regenerate_data_distribution_figure(dpi=300):
    """Regenerate the treatment-mediator by meal figure with a common legend.

    Strategy 1: Reload from CSV and regenerate.
    Strategy 2: Keep existing copied figure unchanged.

    Returns 1 on success, 0 otherwise.
    """
    print("\nRegenerating data distribution figure (common legend) ...")
    csv_path = _find_meal_data_csv()
    if csv_path is None:
        print("  No embeddings CSV found; keeping existing figure")
        return 0

    print(f"  Found data CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  Failed to read CSV: {exc}")
        return 0

    out_path = (RAW_DIRS["data_distribution"] / "figures"
                / "fig08_treatment_mediator_by_meal.png")
    if _regenerate_treatment_mediator_by_meal(df, out_path, dpi=dpi):
        print(f"  Regenerated: {out_path.name}")
        return 1

    print("  Could not regenerate fig08 (missing columns)")
    return 0


# =========================================================================
# Assembly functions
# =========================================================================

def copy_all_existing_outputs():
    """Copy pre-existing analysis files to the internal staging area.

    Iterates over the :data:`COPY_MANIFEST` and copies each file whose
    source exists.  Returns ``(copied, missing)`` counts.
    """
    print("Copying raw analysis outputs to staging area ...")
    copied = 0
    missing = 0

    for src_rel, dst_rel in COPY_MANIFEST:
        ok = copy_to_paper_vis(src_rel, dst_rel)
        if ok:
            print(f"  Copied: {dst_rel}")
            copied += 1
        else:
            print(f"  Missing: {src_rel}")
            missing += 1

    print(f"  {copied} copied, {missing} missing")
    return copied, missing


def generate_all_figures(dpi=300):
    """Generate all composite figures into ``paper_visualizations/figures/``.

    Returns the number of figures generated.
    """
    print("Generating composite figures ...")
    count = 0
    for name, cfg in FIGURES.items():
        output = cfg["output"]
        engine = ENGINES[cfg["mode"]]
        print(f"  Composing {name}: {cfg['description']}")
        try:
            engine(cfg, dpi, output)
            print(f"    Saved: {output}")
            count += 1
        except FileNotFoundError as exc:
            print(f"    Skipping: {exc}")
    return count


def generate_clae_config_table():
    """Generate Table 2: CLAE architecture and training configuration.

    This is a static table (no data dependencies) describing the model
    architecture, reconstruction heads, causal penalties, and training
    hyperparameters.

    Returns 1 (always succeeds).
    """
    out_path = _STAGING_TABLES_DIR / "clae_config.tex"

    latex = (
        r"\begin{table*}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{CLAE architecture and training configuration.}}" "\n"
        r"\label{tab:clae_config}" "\n"
        r"\small" "\n"
        r"\begin{tabular}{ll}" "\n"
        r"\toprule" "\n"
        r"\textbf{Component} & \textbf{Specification} \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{2}{l}{\textit{CNN Encoder (selected)}} \\" "\n"
        r"\quad Conv blocks & 3 blocks (32, 64, 128 filters; kernel size 3) \\" "\n"
        r"\quad Pooling & Max pooling (stride 2), global average pooling \\" "\n"
        r"\quad Regularization & $L_2 = 10^{-4}$, dropout $= 0.2$, input noise $\sigma = 0.05$ \\" "\n"
        r"\quad Latent dimension & 8 \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{2}{l}{\textit{Reconstruction heads}} \\" "\n"
        r"\quad Pre-treatment & Predicts baseline feature matrix (weight $= 0.5$) \\" "\n"
        r"\quad Mediator & Predicts insulin bolus (weight $= 0.5$) \\" "\n"
        r"\quad Outcome & Predicts glucose trajectory (weight $= 2.0$) \\" "\n"
        r"\quad Propensity & Predicts treatment (weight $= 0.0$; see text) \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{2}{l}{\textit{Causal penalties}} \\" "\n"
        r"\quad Balancing & $\gamma = 2.0$ \\" "\n"
        r"\quad Linearizability & $\lambda = 0.1$ \\" "\n"
        r"\quad Conditional independence & $\lambda = 0.05$ \\" "\n"
        r"\quad Stability & $\lambda = 0.01$ \\" "\n"
        r"\midrule" "\n"
        r"\multicolumn{2}{l}{\textit{Training}} \\" "\n"
        r"\quad Optimizer & AdamW (lr $= 10^{-3}$, weight decay $= 10^{-5}$) \\" "\n"
        r"\quad Epochs / batch size & 100 / 32 \\" "\n"
        r"\quad Gradient clipping & Norm $\leq 1.0$ \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table*}" "\n"
    )

    _write_latex_table(out_path, latex)
    print(f"  Saved: {out_path}")
    return 1


def generate_qr_summary_table(df):
    """Generate Table 4: QR summary of mediation effects at 120 min.

    Combines LMER and QR (tau = 0.25, 0.50, 0.75) results for Pooled,
    Breakfast, and Dinner at the +30 g contrast into a single summary table.

    Returns 1 if the table was written, 0 otherwise.
    """
    if df is None or df.empty:
        print("  Skipping QR summary table (no data)")
        return 0

    out_path = _STAGING_TABLES_DIR / "qr_summary_main.tex"

    # Meal types for this table (subset of MEAL_TYPE_ORDER)
    summary_meals = [
        ("all", "Pooled"),
        ("breakfast", "Breakfast"),
        ("dinner", "Dinner"),
    ]

    # Models: LMER first, then QR quantiles
    models = [
        ("lmer", None, "LMER"),
        ("qr", 0.25, r"$\tau=0.25$"),
        ("qr", 0.50, r"$\tau=0.50$"),
        ("qr", 0.75, r"$\tau=0.75$"),
    ]

    rows = []

    for meal_idx, (meal_key, meal_display) in enumerate(summary_meals):
        meal_label = "ALL" if meal_key == "all" else meal_key.capitalize()

        meal_rows = []
        for model_type, tau, model_display in models:
            sub = df[
                (df["status"] == "success")
                & (df["model"] == model_type)
            ].copy()

            if "covariate_mode" in sub.columns:
                sub = sub[sub["covariate_mode"] == "pca"]
            if "meal_type" in sub.columns:
                sub = sub[sub["meal_type"] == meal_label]
            if "treat_offset" in sub.columns:
                sub = sub[sub["treat_offset"] == 30]

            # Filter to 120 minutes
            sub = sub[sub["minutes"] == 120]

            # For QR, filter to the specific quantile
            if tau is not None and "quantile_tau" in sub.columns:
                sub = sub[abs(sub["quantile_tau"] - tau) < 0.01]

            if sub.empty:
                continue

            r = sub.iloc[0]

            acme_sig = r["ACME_p"] < 0.05
            ade_sig = r["ADE_p"] < 0.05
            total_sig = r["total_p"] < 0.05

            row_str = (
                f"& {model_display} & "
                f"{_fmt_est(r['ADE'], ade_sig)} & "
                f"{_fmt_p(r['ADE_p'], ade_sig)} & & "
                f"{_fmt_est(r['ACME'], acme_sig)} & "
                f"{_fmt_p(r['ACME_p'], acme_sig)} & & "
                f"{_fmt_est(r['total_effect'], total_sig)} & "
                f"{_fmt_p(r['total_p'], total_sig)} & \\\\"
            )
            meal_rows.append(row_str)

        if not meal_rows:
            continue

        # Add multirow meal label to the first row of this meal group
        meal_rows[0] = (
            rf"\multirow{{{len(meal_rows)}}}{{*}}{{{meal_display}}} "
            + meal_rows[0]
        )

        rows.extend(meal_rows)

        # Midrule between meal groups (not after the last one)
        if meal_idx < len(summary_meals) - 1:
            rows.append("\\midrule")

    if not rows:
        print("  Skipping QR summary table (no matching rows)")
        return 0

    body = "\n".join(rows)

    latex = (
        r"\begin{table*}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{\textbf{\boldmath Summary of quantile regression mediation effects "
        r"at 120 minutes post-meal ($+30$\,g carbohydrate contrast).} "
        r"Entries report point estimates (mg/dL) and $p$-values for the "
        r"average direct effect (ADE), average causal mediation effect "
        r"(ACME), and total effect at three quantiles of the conditional "
        r"glucose response distribution. The LMER mean-level estimate is "
        r"included for comparison. Boldface indicates statistical "
        r"significance ($p < 0.05$).}" "\n"
        r"\label{tab:qr_summary_main}" "\n"
        r"\small" "\n"
        r"\begin{tabular}{ll rrr rrr rrr}" "\n"
        r"\toprule" "\n"
        r"& & \multicolumn{3}{c}{\textbf{ADE}} "
        r"& \multicolumn{3}{c}{\textbf{ACME}} "
        r"& \multicolumn{3}{c}{\textbf{Total}} \\" "\n"
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}" "\n"
        r"\textbf{Meal} & \textbf{Model} "
        r"& Est. & $p$ & & Est. & $p$ & & Est. & $p$ & \\" "\n"
        r"\midrule" "\n"
        + body + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\vspace{0.2cm}" "\n\n"
        r"{\footnotesize All estimates are at $t = 120$ minutes post-meal "
        r"for the $+30$\,g carbohydrate contrast relative to the "
        r"meal-type-specific median. Confidence intervals are 95\% "
        r"quasi-Bayesian from 1000 Monte Carlo simulations. Full quantile "
        r"regression results for all meal types, timepoints, and dose "
        r"levels appear in the Supplementary Information.}" "\n"
        r"\end{table*}" "\n"
    )

    _write_latex_table(out_path, latex)
    print(f"  Saved: {out_path}")
    return 1


def generate_all_tables(mediation_dir):
    """Generate all tables that are produced by this script.

    Generates the CLAE config table (Table 2), per-meal
    ``table_lmer_offsets.tex`` files (for ``\\input``), condensed LMER tables
    (one per meal type), condensed QR tables (one per meal type), the QR
    summary table (Table 4), and incremental experiment tables (from
    pre-computed CSVs).  LMER tables can be built from existing ``.tex``
    files even when CSV data is unavailable.
    Returns the total number of tables generated.
    """
    print(f"\nLoading mediation results from: {mediation_dir}")
    df = load_mediation_results(mediation_dir)

    if df is not None and not df.empty:
        print(f"  Loaded {len(df)} result rows "
              f"({(df['status'] == 'success').sum()} successful)")
    else:
        print("  WARNING: No mediation CSV results found.")
        print(f"  Searched: {mediation_dir}")
        df = None

    total = 0

    print("\nGenerating CLAE config table (Table 2) ...")
    total += generate_clae_config_table()

    print("\nGenerating per-meal LMER tables (table_lmer_offsets.tex) ...")
    total += generate_per_meal_lmer_tables(df)

    print("\nGenerating condensed LMER tables ...")
    total += generate_condensed_lmer_tables(df)

    if df is not None:
        print("\nGenerating condensed QR tables ...")
        total += generate_condensed_qr_tables(df)

        print("\nGenerating QR summary table (Table 4) ...")
        total += generate_qr_summary_table(df)
    else:
        print("\n  Skipping condensed QR tables (no CSV data)")
        print("\n  Skipping QR summary table (no CSV data)")

    # Incremental experiment tables (generated from CSVs, not copied)
    total += generate_incremental_experiment_tables()

    return total


# =========================================================================
# Submission export
# =========================================================================

def _compile_figure1_dag():
    """Compile ``figure1_dag.tex`` into a PDF and copy to paper_visualizations/figures/.

    The ``.tex`` source lives in ``visualization_code/``.  Compilation runs
    in a temporary directory so no auxiliary files (.aux, .log, .pdf) are
    left in the project tree.

    Tries ``pdflatex`` first, then ``lualatex``.  Once an engine is found
    on PATH, only that engine is attempted (if it fails, the error is a TeX
    configuration issue, not a missing-binary issue).

    Returns ``True`` on success, ``False`` on failure.
    """
    import tempfile

    tex_file = VISUALIZATION_CODE_DIR / "figure1_dag.tex"
    dst = SUBMISSION_DIR / "Figure 1.pdf"

    if not tex_file.exists():
        print(f"    Source not found: {tex_file}")
        return False

    # Compile in a temp directory to keep the project tree clean
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tex = Path(tmpdir) / "figure1_dag.tex"
        shutil.copy2(tex_file, tmp_tex)
        pdf_out = Path(tmpdir) / "figure1_dag.pdf"
        log_out = Path(tmpdir) / "figure1_dag.log"

        engines = [
            ["pdflatex", "-interaction=nonstopmode", "figure1_dag.tex"],
            ["lualatex", "-interaction=nonstopmode", "figure1_dag.tex"],
        ]

        compiled = False
        engine_found = False
        for cmd in engines:
            engine_name = cmd[0]
            try:
                subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    check=True,
                    capture_output=True,
                )
                compiled = True
                break
            except FileNotFoundError:
                continue
            except subprocess.CalledProcessError as exc:
                engine_found = True
                if log_out.exists():
                    log_text = log_out.read_text()
                    error_lines = [
                        ln.strip() for ln in log_text.splitlines()
                        if ln.startswith("!") or "File" in ln and "not found" in ln
                    ]
                    if error_lines:
                        print(f"    {engine_name} failed:")
                        for ln in error_lines[:3]:
                            print(f"      {ln}")
                    else:
                        print(f"    {engine_name} failed (exit {exc.returncode})")

                    missing = re.findall(r"File `([^']+)' not found", log_text)
                    if missing:
                        _FILE_TO_PKG = {
                            "standalone.cls": "standalone",
                            "tikz.sty": "pgf",
                            "amsmath.sty": "amsmath",
                        }
                        for fname in missing:
                            pkg = _FILE_TO_PKG.get(fname, fname.split(".")[0])
                            print(f"    Missing: {fname}  ->  tlmgr install {pkg}")
                else:
                    print(f"    {engine_name} failed (exit {exc.returncode})")
                break

        if not compiled:
            if engine_found:
                print("    Compile figure1_dag.tex manually after fixing TeX install")
            else:
                print("    No LaTeX engine found on PATH (tried pdflatex, lualatex)")
                print("    Install TeX Live or compile visualization_code/figure1_dag.tex manually.")
            return False

        if not pdf_out.exists():
            print("    LaTeX ran but figure1_dag.pdf was not produced")
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_out, dst)

    return True


def _fmt_file_size(size_bytes):
    """Return a human-readable size string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def export_submission_figures(dpi=300):
    """Export figures into ``paper_visualizations/figures/`` with journal names.

    Raster figures (PNG) are opened with Pillow and re-saved at the target
    DPI so that the DPI metadata is guaranteed to be >= 300.  No resampling
    is performed -- pixel data is preserved.

    Figure 1 (the causal DAG) is compiled from ``figure1_dag.tex`` via
    pdflatex and kept as a vector PDF.

    Returns the number of figures exported.
    """
    print("\n" + "=" * 70)
    print("SUBMISSION EXPORT")
    print("=" * 70)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = []

    for journal_name, rel_path in SUBMISSION_FIGURES.items():
        dst = SUBMISSION_DIR / journal_name

        # --- Special case: Figure 1 is compiled from TikZ, not copied ---
        if rel_path is None:
            print(f"  Compiling {journal_name} from figure1_dag.tex ...")
            if _compile_figure1_dag():
                size_str = _fmt_file_size(dst.stat().st_size)
                print(f"  {journal_name:<40s}  (vector, {size_str})")
                count += 1
            else:
                errors.append((journal_name, "TikZ compilation failed"))
            continue

        # --- Raster figures: open with Pillow, re-save with DPI tag ---
        src = _STAGING_FIGURES_DIR / rel_path

        if not src.exists():
            errors.append((journal_name, f"source missing: {src}"))
            continue

        try:
            img = Image.open(src)
            img.save(dst, dpi=(dpi, dpi))
            w, h = img.size
            size_str = _fmt_file_size(dst.stat().st_size)
            print(f"  {journal_name:<40s}  ({w} x {h}, {dpi} dpi, {size_str})")
            count += 1
        except Exception as exc:
            errors.append((journal_name, str(exc)))

    if errors:
        print(f"\n  WARNINGS ({len(errors)}):")
        for name, msg in errors:
            print(f"    {name}: {msg}")

    print(f"\nSubmission figures written to {SUBMISSION_DIR}/")
    print(f"  {count} of {len(SUBMISSION_FIGURES)} figures exported")
    print("=" * 70)
    return count


def export_submission_tables():
    """Export tables into ``paper_visualizations/tables/`` with journal names.

    Each ``.tex`` file is copied verbatim (no content modification).
    Returns the number of tables exported.
    """
    print("\n" + "=" * 70)
    print("SUBMISSION TABLE EXPORT")
    print("=" * 70)

    SUBMISSION_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    errors = []

    for journal_name, rel_path in SUBMISSION_TABLES.items():
        src = _STAGING_TABLES_DIR / rel_path
        dst = SUBMISSION_TABLES_DIR / journal_name

        if not src.exists():
            errors.append((journal_name, f"source missing: {src}"))
            continue

        try:
            shutil.copy(src, dst)
            # Derive a short label from the source filename for the printout
            label = Path(rel_path).stem
            size_str = _fmt_file_size(dst.stat().st_size)
            print(f"  {journal_name:<40s}  ({label}, {size_str})")
            count += 1
        except Exception as exc:
            errors.append((journal_name, str(exc)))

    if errors:
        print(f"\n  WARNINGS ({len(errors)}):")
        for name, msg in errors:
            print(f"    {name}: {msg}")

    print(f"\nSubmission tables written to {SUBMISSION_TABLES_DIR}/")
    print(f"  {count} of {len(SUBMISSION_TABLES)} tables exported")
    print("=" * 70)
    return count


def write_visualizations_summary(n_figures, n_tables):
    """Write ``paper_visualizations/visualizations_summary.txt`` with export counts."""
    summary_path = PAPER_VIS_DIR / "visualizations_summary.txt"
    lines = [
        f"Number of figures: {n_figures}",
        f"Number of color figures: {n_figures}",
        f"Number of tables: {n_tables}",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nVisualizations summary written to {summary_path}")


def print_summary():
    """Print a manifest of all files in ``paper_visualizations/`` with sizes."""
    print("\n" + "=" * 70)
    print("PAPER VISUALIZATIONS MANIFEST")
    print("=" * 70)

    if not PAPER_VIS_DIR.exists():
        print("  (paper_visualizations directory does not exist)")
        return

    total_files = 0
    total_bytes = 0

    for section, subdir in [("FIGURES", "figures"), ("TABLES", "tables")]:
        section_dir = PAPER_VIS_DIR / subdir
        if not section_dir.exists():
            print(f"\n{section}: (none)")
            continue

        files = sorted(section_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        print(f"\n{section} ({len(files)} files):")
        for f in files:
            size = f.stat().st_size
            total_bytes += size
            total_files += 1
            rel = f.relative_to(PAPER_VIS_DIR)
            if size >= 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {str(rel):<60s} {size_str:>10s}")

    print(f"\nTotal: {total_files} files, "
          f"{total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 70)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compose publication-ready figures and tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Registered figures:\n" + "\n".join(
            f"  {name:15s} {cfg['description']}"
            for name, cfg in FIGURES.items()
        ),
    )
    parser.add_argument(
        "figures", nargs="*",
        help="Figure name(s) to compose (see list below)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Produce everything: copy, generate, post-process, and summarize",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy raw analysis outputs to paper_visualizations/ only",
    )
    parser.add_argument(
        "--tables", action="store_true",
        help="Generate condensed QR tables for paper_visualizations",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output path (only valid when composing a single figure)",
    )
    parser.add_argument(
        "--submission", action="store_true",
        help="Export journal-ready figures and tables to paper_visualizations/ "
             "(auto-runs all prerequisites)",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to mediation_results directory "
             "(default: cma_cluster/mediation_results)",
    )
    args = parser.parse_args()

    do_all = args.all
    do_copy = args.copy or do_all
    do_tables = args.tables or do_all
    do_submission = args.submission or do_all
    figure_names = list(FIGURES.keys()) if do_all else (args.figures or [])
    do_figures = bool(figure_names)

    if not do_copy and not do_figures and not do_tables and not do_submission:
        parser.print_help()
        return

    if args.output and len(figure_names) > 1:
        parser.error("--output can only be used with a single figure")

    # --submission needs copy + figures + tables + composites as prerequisites.
    # Auto-enable them so --submission works standalone.
    if do_submission and not do_all:
        do_copy = True
        do_tables = True
        if not figure_names:
            figure_names = list(FIGURES.keys())
            do_figures = True

    mediation_dir = (
        Path(args.results_dir) if args.results_dir
        else PROJECT_ROOT / "cma_cluster" / "mediation_results"
    )

    # --- Copy pre-existing raw outputs to staging ---
    if do_copy:
        copy_all_existing_outputs()

    # --- Load mediation CSV data once (shared by tables + figure composites) ---
    med_df = None
    if do_tables or do_figures or do_copy or do_all:
        print(f"\nLoading mediation results from: {mediation_dir}")
        med_df = load_mediation_results(mediation_dir)
        if med_df is not None and not med_df.empty:
            print(f"  Loaded {len(med_df)} result rows "
                  f"({(med_df['status'] == 'success').sum()} successful)")
        else:
            print("  WARNING: No mediation CSV results found — "
                  "composites will fall back to pre-rendered PNGs.")

    # --- Figures ---
    if do_figures:
        fig3_done = False
        if med_df is not None:
            fig3_done = bool(generate_figure3_from_csv(med_df, dpi=args.dpi))
            if fig3_done:
                print("  Figure 3 (mediation grid) generated from CSV data")

        if do_all or do_submission:
            # Generate ALL registered figures (trajectory composite, etc.)
            # but skip figure3 if CSV-based version already succeeded.
            for name, cfg in FIGURES.items():
                if name == "figure3" and fig3_done:
                    continue
                output = cfg["output"]
                engine = ENGINES[cfg["mode"]]
                print(f"  Composing {name}: {cfg['description']}")
                try:
                    engine(cfg, args.dpi, output)
                    print(f"    Saved: {output}")
                except FileNotFoundError as exc:
                    print(f"    Skipping: {exc}")
        else:
            for name in figure_names:
                if name not in FIGURES:
                    parser.error(
                        f"Unknown figure: {name!r}. "
                        f"Available: {', '.join(FIGURES.keys())}"
                    )
                if name == "figure3" and fig3_done:
                    continue
                cfg = FIGURES[name]
                output = Path(args.output) if args.output else cfg["output"]
                engine = ENGINES[cfg["mode"]]

                print(f"Composing {name}: {cfg['description']}")
                engine(cfg, args.dpi, output)
                print(f"  Saved: {output}")

    # --- Tables ---
    if do_tables:
        generate_all_tables(mediation_dir)

    # --- Composite figures (reads panels from original visualizations/ locations) ---
    if do_copy or do_all:
        generate_lmer_composites(dpi=args.dpi, mediation_df=med_df)
        generate_qr_composites(dpi=args.dpi, mediation_df=med_df)
        regenerate_pca_figure(dpi=args.dpi)
        regenerate_trajectory_figures(dpi=args.dpi)
        regenerate_data_distribution_figure(dpi=args.dpi)

    # --- Submission export (staging -> final journal-named files) ---
    if do_submission:
        n_figs = export_submission_figures(dpi=args.dpi)
        n_tables = export_submission_tables()
        # Clean up staging — paper_visualizations/ now has only journal-named files
        if _STAGING_DIR.exists():
            shutil.rmtree(_STAGING_DIR)
            print(f"\nCleaned up staging directory: {_STAGING_DIR}")
        write_visualizations_summary(n_figs, n_tables)

    # --- Summary ---
    if do_all or do_copy:
        print_summary()


if __name__ == "__main__":
    main()
