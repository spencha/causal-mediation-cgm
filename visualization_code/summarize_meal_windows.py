#!/usr/bin/env python3
"""
summarize_meal_windows.py
=========================
Comprehensive summary statistics and visualizations for the meal window data
used in the causal mediation analysis.

This script generates publication-quality tables and figures characterizing:
1. Sample composition (patients, meal events)
2. Treatment distribution (carbohydrate intake) - by meal type and subject
3. Mediator distribution (insulin bolus) - by meal type and subject
4. Outcome characteristics (glucose trajectories)
5. Covariate summaries (latent embeddings)

All figures are saved as individual standalone files.
Tables are stratified by meal type.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# =============================================================================
# COLORBLIND-FRIENDLY PALETTE (Nature Style)
# =============================================================================
# Colors assigned by causal role following journal conventions:
# - Treatment (A): Blue tones - intervention/exposure
# - Mediator (M): Teal/cyan - intermediate mechanism
# - Outcome (Y): Orange/amber - endpoint
# - Confounders (X): Gray tones - adjustment variables
#
# Palette based on Paul Tol's colorblind-safe palette
# https://personal.sron.nl/~pault/#sec:qualitative

COLORS = {
    # Causal role colors (colorblind-friendly)
    'treatment': '#0077BB',     # Blue - carbs (treatment A)
    'mediator': '#33BBEE',      # Cyan - insulin (mediator M)
    'outcome': '#EE7733',       # Orange - glucose (outcome Y)
    'confounder': '#BBBBBB',    # Gray - confounding variables

    # Structural/UI colors
    'primary': '#332288',       # Dark indigo
    'secondary': '#999999',     # Medium gray
    'highlight': '#CC3311',     # Vermillion for emphasis

    # Meal type colors (colorblind-friendly qualitative)
    'breakfast': '#EE7733',     # Orange
    'lunch': '#009988',         # Teal
    'dinner': '#0077BB',        # Blue
    'snack': '#CC3311',         # Vermillion

    # Cohort comparison (colorblind-friendly)
    'cohort_2018': '#0077BB',   # Blue
    'cohort_2020': '#EE7733',   # Orange

    # Train/Test split
    'train': '#0077BB',         # Blue
    'test': '#EE7733',          # Orange

    # Balance assessment
    'unweighted': '#CC3311',    # Vermillion (before weighting)
    'weighted': '#009988',      # Teal (after weighting)
    'balanced': '#009988',      # Teal (good)
    'imbalanced': '#CC3311'     # Vermillion (needs attention)
}

# Directories - add ae_python_code to path for config
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
AE_CODE_DIR = PROJECT_ROOT / "ae_python_code"

if str(AE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(AE_CODE_DIR))

from config import CONFIG
CONFIG.ensure_dirs()

DATA_DIR = CONFIG.ANALYSIS_DATA_DIR
# Output directories live under visualizations/ (separate from code)
FIGURES_DIR = PROJECT_ROOT / "visualizations" / "data_distribution" / "figures"
TABLES_DIR = PROJECT_ROOT / "visualizations" / "data_distribution" / "tables"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Column name mappings
# Note: subject_id_unique is created in load_meal_window_data() to distinguish
# between cohorts (2018 vs 2020 have different people with potentially same IDs)
#
# IMPORTANT: For glucose, we need ACTUAL glucose values (mg/dL), NOT delta glucose.
# Columns like 'delta_glucose_*' contain change values centered around 0.
# Pre-meal glucose should typically be in range 70-400 mg/dL.
COLUMN_MAPPINGS = {
    'treatment': ['treat_meal_carbs', 'carbs', 'meal_carbs', 'carbohydrates'],
    'mediator': ['mediator_bolus_for_meal', 'bolus_dose', 'bolus', 'insulin_bolus'],
    'total_bolus': ['total_bolus', 'bolus_taken'],
    # Priority order: actual glucose columns first, avoid delta columns
    'glucose': ['pre_meal_glucose', 'glucose_at_meal_start', 'bg_at_meal',
                'baseline_glucose', 'glucose_at_meal', 'glucose', 'bg', 'blood_glucose'],
    'patient': ['subject_id_unique', 'subject_id', 'patient_id', 'id', 'subject'],
    'meal_type': ['meal_type', 'meal', 'meal_category'],
    'cohort': ['cohort', 'year', 'dataset'],
    'split': ['split', 'train_test', 'fold']  # train/test split column
}


def get_column(df, col_type):
    """Get the actual column name from the dataframe for a given column type."""
    if col_type not in COLUMN_MAPPINGS:
        return col_type if col_type in df.columns else None

    for candidate in COLUMN_MAPPINGS[col_type]:
        if candidate in df.columns:
            return candidate
    return None


def validate_glucose_column(df, glucose_col):
    """
    Validate that the glucose column contains actual glucose values (mg/dL),
    not delta glucose values (change from baseline).

    Actual glucose values should be in range ~40-400 mg/dL.
    Delta glucose values are typically centered around 0.

    Returns True if values appear to be actual glucose, False otherwise.
    """
    if not glucose_col or glucose_col not in df.columns:
        return False

    glucose = df[glucose_col].dropna()
    if len(glucose) == 0:
        return False

    mean_val = glucose.mean()
    median_val = glucose.median()
    min_val = glucose.min()

    # Check if values look like actual glucose (typically 40-400 mg/dL)
    # vs delta glucose (centered around 0, range typically -200 to +200)
    is_actual_glucose = (
        mean_val > 50 and  # Actual glucose mean should be > 50 mg/dL
        median_val > 50 and  # Median also > 50
        min_val >= 0  # Should be non-negative
    )

    if not is_actual_glucose:
        print(f"WARNING: Column '{glucose_col}' appears to contain DELTA glucose values,")
        print(f"         not actual glucose values (mean={mean_val:.1f}, median={median_val:.1f}, min={min_val:.1f})")
        print(f"         Expected actual glucose in range ~70-400 mg/dL.")
        print(f"         Check that you're using the correct column for pre-meal glucose.")

    return is_actual_glucose


def build_dataset_label(df):
    """Build a human-readable label describing what data is in the dataframe.

    Returns something like:
      'OhioT1DM 2018+2020 | Train+Test (N=1,842)'
      'OhioT1DM 2018+2020 | Train+Test (N=1,842) | Source: phi_embeddings_combined_cnn_...'
    """
    parts = []

    # Cohort info
    cohort_col = get_column(df, 'cohort')
    if cohort_col and cohort_col in df.columns:
        cohorts = sorted(df[cohort_col].dropna().unique())
        cohort_str = "+".join(str(c) for c in cohorts)
        parts.append(f"OhioT1DM {cohort_str}")
    else:
        parts.append("OhioT1DM")

    # Split info
    split_col = get_column(df, 'split')
    if split_col and split_col in df.columns:
        splits = sorted(df[split_col].dropna().unique())
        split_str = "+".join(str(s).capitalize() for s in splits)
        parts.append(split_str)

    # Sample size
    parts.append(f"N={len(df):,}")

    return " | ".join(parts)


def get_meal_color(meal):
    """Get color for a meal type, handling case variations."""
    m_lower = str(meal).lower()
    if 'breakfast' in m_lower:
        return COLORS.get('breakfast', '#FF9800')
    elif 'lunch' in m_lower:
        return COLORS.get('lunch', '#4CAF50')
    elif 'dinner' in m_lower:
        return COLORS.get('dinner', '#2196F3')
    elif 'snack' in m_lower:
        return COLORS.get('snack', '#9C27B0')
    else:
        return 'gray'


def get_sorted_meal_types(df, meal_type_col):
    """Get meal types sorted by preferred order."""
    if not meal_type_col or meal_type_col not in df.columns:
        return []
    
    actual_meals = df[meal_type_col].dropna().unique()
    preferred_order = ['breakfast', 'lunch', 'dinner', 'snack']
    
    def sort_key(m):
        m_lower = str(m).lower()
        for i, pref in enumerate(preferred_order):
            if pref in m_lower:
                return i
        return 100
    
    return sorted(actual_meals, key=sort_key)


def get_meal_boxplot_data(df, value_col, meal_col):
    """Prepare data for boxplot by meal type."""
    if not meal_col or meal_col not in df.columns or not value_col:
        return None, None, None
    
    meal_types = get_sorted_meal_types(df, meal_col)
    
    data_list = []
    labels = []
    colors = []
    
    for m in meal_types:
        meal_data = df[df[meal_col] == m][value_col].dropna()
        if len(meal_data) > 0:
            data_list.append(meal_data.values)
            labels.append(str(m).capitalize())
            colors.append(get_meal_color(m))
    
    if len(data_list) == 0:
        return None, None, None
    
    return data_list, labels, colors


# =============================================================================
# DATA LOADING
# =============================================================================

def load_meal_window_data(embeddings_file=None):
    """Load the meal window data from the phi embeddings file.

    Parameters
    ----------
    embeddings_file : str or Path, optional
        Path to a specific embeddings CSV file. If None, uses the most recent
        phi_embeddings_combined_*.csv file (by modification time).
    """
    print("\n" + "="*70)
    print("LOADING MEAL WINDOW DATA")
    print("="*70)

    # Embeddings are in: cma_cluster/analysis_data/embeddings/
    # File pattern: phi_embeddings_combined_{arch}_{pct}pct_{penalty}_seed{seed}.csv
    # Example: phi_embeddings_combined_cnn_75pct_lin_bal_seed42.csv
    embeddings_dir = DATA_DIR / 'embeddings'

    if embeddings_file is not None:
        # Use specified file
        phi_path = Path(embeddings_file)
        if not phi_path.exists():
            print(f"ERROR: Specified embeddings file not found: {phi_path}")
            return None
    else:
        # Find most recent combined file by modification time
        if not embeddings_dir.exists():
            print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
            print("Run train_and_export_embeddings.py first.")
            return None

        combined_files = list(embeddings_dir.glob('phi_embeddings_combined_*.csv'))

        if not combined_files:
            print(f"ERROR: No combined embeddings found in: {embeddings_dir}")
            print("Expected: phi_embeddings_combined_*.csv")
            print("Run train_and_export_embeddings.py first.")
            return None

        # Sort by modification time (most recent first)
        combined_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        phi_path = combined_files[0]
        print(f"Using most recent embeddings file (by modification time)")

    df = pd.read_csv(phi_path)
    print(f"Loaded {len(df):,} meal events from: {phi_path.name}")

    # IMPORTANT: Create unique subject IDs that distinguish between cohorts
    # Subject IDs 1-6 in 2018 are DIFFERENT PEOPLE from 1-6 in 2020
    if 'cohort' in df.columns and 'subject_id' in df.columns:
        # Guard against double-prefixing: if subject_id already starts with
        # the cohort value (e.g. "2018_1"), don't prepend the cohort again.
        sample_sid = str(df['subject_id'].iloc[0])
        sample_cohort = str(df['cohort'].iloc[0])
        if sample_sid.startswith(sample_cohort + '_'):
            # subject_id already contains cohort prefix – use as-is
            df['subject_id_unique'] = df['subject_id'].astype(str)
        else:
            df['subject_id_unique'] = df['cohort'].astype(str) + '_' + df['subject_id'].astype(str)
        print(f"\nCreated unique subject IDs by cohort:")
        for cohort in sorted(df['cohort'].unique()):
            n_subj = df[df['cohort'] == cohort]['subject_id'].nunique()
            n_events = len(df[df['cohort'] == cohort])
            print(f"  {cohort}: {n_subj} subjects, {n_events} events")
    else:
        print("\nWARNING: No cohort column found - subject IDs may not be unique across cohorts")

    print(f"\nAvailable columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        if i < 15:
            print(f"  - {col}")
    if len(df.columns) > 15:
        print(f"  ... and {len(df.columns) - 15} more")

    return df


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary_statistics(df):
    """Compute comprehensive summary statistics."""
    print("\n" + "="*70)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*70)

    stats_dict = {}

    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    patient_col = get_column(df, 'patient')
    meal_type_col = get_column(df, 'meal_type')
    cohort_col = get_column(df, 'cohort')

    split_col = get_column(df, 'split')

    print(f"\nDetected columns:")
    print(f"  Treatment: {treatment_col}")
    print(f"  Mediator: {mediator_col}")
    print(f"  Glucose (at meal): {glucose_col}")
    print(f"  Patient ID: {patient_col}")
    print(f"  Meal type: {meal_type_col}")
    print(f"  Cohort: {cohort_col}")
    print(f"  Split: {split_col}")

    # Validate glucose column is actual glucose, not delta glucose
    if glucose_col:
        validate_glucose_column(df, glucose_col)
    
    # Sample composition
    stats_dict['sample'] = {
        'n_events': len(df),
        'n_patients': df[patient_col].nunique() if patient_col else np.nan,
    }

    # Cohort breakdown (2018 vs 2020 are DIFFERENT subjects)
    if cohort_col and cohort_col in df.columns:
        stats_dict['cohort'] = {}
        for cohort in sorted(df[cohort_col].unique()):
            cohort_df = df[df[cohort_col] == cohort]
            stats_dict['cohort'][cohort] = {
                'n_events': len(cohort_df),
                'n_patients': cohort_df['subject_id'].nunique() if 'subject_id' in cohort_df.columns else np.nan
            }
        print(f"\nCohort breakdown (different subjects in each!):")
        for cohort, data in stats_dict['cohort'].items():
            print(f"  {cohort}: {data['n_patients']} subjects, {data['n_events']} events")

    if patient_col:
        events_per_patient = df.groupby(patient_col).size()
        stats_dict['sample']['events_per_patient_mean'] = events_per_patient.mean()
        stats_dict['sample']['events_per_patient_sd'] = events_per_patient.std()
        stats_dict['sample']['events_per_patient_min'] = events_per_patient.min()
        stats_dict['sample']['events_per_patient_max'] = events_per_patient.max()
    
    # Treatment
    if treatment_col:
        carbs = df[treatment_col].dropna()
        stats_dict['treatment'] = {
            'column': treatment_col,
            'n': len(carbs),
            'mean': carbs.mean(),
            'sd': carbs.std(),
            'median': carbs.median(),
            'iqr_25': carbs.quantile(0.25),
            'iqr_75': carbs.quantile(0.75),
            'min': carbs.min(),
            'max': carbs.max(),
        }
    
    # Mediator
    if mediator_col:
        bolus = df[mediator_col].dropna()
        stats_dict['mediator'] = {
            'column': mediator_col,
            'n': len(bolus),
            'mean': bolus.mean(),
            'sd': bolus.std(),
            'median': bolus.median(),
            'iqr_25': bolus.quantile(0.25),
            'iqr_75': bolus.quantile(0.75),
            'min': bolus.min(),
            'max': bolus.max(),
            'pct_zero': 100 * (bolus == 0).mean()
        }

    # Glucose at meal (ACTUAL glucose level, NOT delta glucose)
    if glucose_col:
        glucose = df[glucose_col].dropna()
        stats_dict['glucose'] = {
            'column': glucose_col,
            'n': len(glucose),
            'mean': glucose.mean(),
            'sd': glucose.std(),
            'median': glucose.median(),
            'iqr_25': glucose.quantile(0.25),
            'iqr_75': glucose.quantile(0.75),
            'min': glucose.min(),
            'max': glucose.max()
        }
        print(f"\nGlucose at meal statistics (ACTUAL glucose, not delta):")
        print(f"  Mean: {glucose.mean():.1f} mg/dL")
        print(f"  Range: [{glucose.min():.0f}, {glucose.max():.0f}] mg/dL")

    # Meal type distribution
    if meal_type_col:
        meal_counts = df[meal_type_col].value_counts()
        stats_dict['meal_type'] = {
            meal: {'n': count, 'pct': 100 * count / len(df)} 
            for meal, count in meal_counts.items()
        }
    
    return stats_dict


# =============================================================================
# TABLE GENERATION - STRATIFIED BY MEAL TYPE
# =============================================================================

def generate_summary_tables(df, stats_dict, tables_dir):
    """Generate publication-ready summary tables, stratified by meal type."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLES")
    print("="*70)

    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    patient_col = get_column(df, 'patient')
    meal_type_col = get_column(df, 'meal_type')
    
    # =========================================================================
    # Table 1: Overall Sample Characteristics
    # =========================================================================
    sample_data = []
    
    if 'sample' in stats_dict:
        s = stats_dict['sample']
        sample_data.append(['N (meal events)', f"{s['n_events']:,}"])
        if not np.isnan(s.get('n_patients', np.nan)):
            sample_data.append(['N (patients)', f"{s['n_patients']:.0f}"])
            sample_data.append(['Events per patient, mean (SD)', 
                               f"{s['events_per_patient_mean']:.1f} ({s['events_per_patient_sd']:.1f})"])
    
    if 'treatment' in stats_dict:
        t = stats_dict['treatment']
        sample_data.append(['', ''])
        sample_data.append(['Carbohydrate intake (g)', ''])
        sample_data.append(['  Mean (SD)', f"{t['mean']:.1f} ({t['sd']:.1f})"])
        sample_data.append(['  Median [IQR]', f"{t['median']:.1f} [{t['iqr_25']:.1f}, {t['iqr_75']:.1f}]"])
        sample_data.append(['  Range', f"{t['min']:.0f} - {t['max']:.0f}"])
    
    if 'mediator' in stats_dict:
        m = stats_dict['mediator']
        sample_data.append(['', ''])
        sample_data.append(['Insulin bolus (U)', ''])
        sample_data.append(['  Mean (SD)', f"{m['mean']:.2f} ({m['sd']:.2f})"])
        sample_data.append(['  Median [IQR]', f"{m['median']:.2f} [{m['iqr_25']:.2f}, {m['iqr_75']:.2f}]"])
        sample_data.append(['  % with zero bolus', f"{m['pct_zero']:.1f}%"])

    # Glucose at meal (ACTUAL glucose, not delta)
    if 'glucose' in stats_dict:
        g = stats_dict['glucose']
        sample_data.append(['', ''])
        sample_data.append(['Glucose at meal start (mg/dL)', ''])
        sample_data.append(['  Mean (SD)', f"{g['mean']:.1f} ({g['sd']:.1f})"])
        sample_data.append(['  Median [IQR]', f"{g['median']:.1f} [{g['iqr_25']:.1f}, {g['iqr_75']:.1f}]"])
        sample_data.append(['  Range', f"{g['min']:.0f} - {g['max']:.0f}"])

    # Cohort breakdown (2018 vs 2020 are DIFFERENT subjects)
    if 'cohort' in stats_dict:
        sample_data.append(['', ''])
        sample_data.append(['By Cohort (different subjects)', ''])
        for cohort, data in stats_dict['cohort'].items():
            sample_data.append([f'  {cohort} cohort', f"{data['n_patients']} subjects, {data['n_events']} events"])

    table1 = pd.DataFrame(sample_data, columns=['Characteristic', 'Value'])
    table1.to_csv(tables_dir / 'sample_characteristics.csv', index=False)
    print(f"Saved: sample_characteristics.csv")

    # LaTeX version with two-column spanning format
    latex_path = tables_dir / 'sample_characteristics.tex'
    with open(latex_path, 'w') as f:
        f.write(r"""\begin{table*}[ht]
\centering
\caption{\textbf{Sample characteristics.}
Summary statistics for the meal window dataset used in causal mediation analysis.
Treatment = carbohydrate intake (grams); Mediator = insulin bolus dose (units); Outcome = blood glucose (mg/dL).
SD = standard deviation; IQR = interquartile range.}
\label{tab:sample_characteristics}
\begin{tabular}{lr}
\toprule
Characteristic & Value \\
\midrule
""")
        for _, row in table1.iterrows():
            if row['Characteristic'] and row['Value']:  # Skip empty rows
                f.write(f"{row['Characteristic']} & {row['Value']} \\\\\n")
            elif not row['Characteristic'] and not row['Value']:  # Empty separator row
                f.write("\\addlinespace\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
    print(f"Saved: sample_characteristics.tex")
    
    # =========================================================================
    # Table 2: Treatment Statistics by Meal Type
    # =========================================================================
    if treatment_col and meal_type_col:
        meal_types = get_sorted_meal_types(df, meal_type_col)
        
        treatment_by_meal = []
        for meal in meal_types:
            meal_df = df[df[meal_type_col] == meal][treatment_col].dropna()
            if len(meal_df) > 0:
                treatment_by_meal.append({
                    'Meal Type': str(meal).capitalize(),
                    'N': len(meal_df),
                    'Mean': f"{meal_df.mean():.1f}",
                    'SD': f"{meal_df.std():.1f}",
                    'Median': f"{meal_df.median():.1f}",
                    'Q1': f"{meal_df.quantile(0.25):.1f}",
                    'Q3': f"{meal_df.quantile(0.75):.1f}",
                    'Min': f"{meal_df.min():.0f}",
                    'Max': f"{meal_df.max():.0f}"
                })
        
        table2 = pd.DataFrame(treatment_by_meal)
        table2.to_csv(tables_dir / 'treatment_by_meal.csv', index=False)
        print(f"Saved: treatment_by_meal.csv")

        # LaTeX version with two-column spanning format
        latex_path = tables_dir / 'treatment_by_meal.tex'
        with open(latex_path, 'w') as f:
            f.write(r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Carbohydrate intake (treatment) distribution by meal type.}
Summary statistics for carbohydrate intake (grams) stratified by meal category.
N = number of meal events; SD = standard deviation; Q1, Q3 = first and third quartiles.}
\label{tab:treatment_meal}
\begin{tabular}{lccccccc}
\toprule
Meal Type & N & Mean & SD & Median & Q1 & Q3 & Range \\
\midrule
""")
            for _, row in table2.iterrows():
                f.write(f"{row['Meal Type']} & {row['N']} & {row['Mean']} & {row['SD']} & ")
                f.write(f"{row['Median']} & {row['Q1']} & {row['Q3']} & {row['Min']}--{row['Max']} \\\\\n")
            f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
        print(f"Saved: treatment_by_meal.tex")
    
    # =========================================================================
    # Table 3: Mediator Statistics by Meal Type
    # =========================================================================
    if mediator_col and meal_type_col:
        meal_types = get_sorted_meal_types(df, meal_type_col)
        
        mediator_by_meal = []
        for meal in meal_types:
            meal_df = df[df[meal_type_col] == meal][mediator_col].dropna()
            if len(meal_df) > 0:
                mediator_by_meal.append({
                    'Meal Type': str(meal).capitalize(),
                    'N': len(meal_df),
                    'Mean': f"{meal_df.mean():.2f}",
                    'SD': f"{meal_df.std():.2f}",
                    'Median': f"{meal_df.median():.2f}",
                    'Q1': f"{meal_df.quantile(0.25):.2f}",
                    'Q3': f"{meal_df.quantile(0.75):.2f}",
                    '% Zero': f"{100*(meal_df==0).mean():.1f}"
                })
        
        table3 = pd.DataFrame(mediator_by_meal)
        table3.to_csv(tables_dir / 'mediator_by_meal.csv', index=False)
        print(f"Saved: mediator_by_meal.csv")

        # LaTeX version with two-column spanning format
        latex_path = tables_dir / 'mediator_by_meal.tex'
        with open(latex_path, 'w') as f:
            f.write(r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Insulin bolus (mediator) distribution by meal type.}
Summary statistics for insulin bolus dose (units) stratified by meal category.
N = number of meal events; SD = standard deviation; Q1, Q3 = first and third quartiles; \% Zero = percentage of meals with zero bolus.}
\label{tab:mediator_meal}
\begin{tabular}{lcccccccc}
\toprule
Meal Type & N & Mean & SD & Median & Q1 & Q3 & \% Zero \\
\midrule
""")
            for _, row in table3.iterrows():
                f.write(f"{row['Meal Type']} & {row['N']} & {row['Mean']} & {row['SD']} & ")
                f.write(f"{row['Median']} & {row['Q1']} & {row['Q3']} & {row['% Zero']} \\\\\n")
            f.write(r"""\bottomrule
\end{tabular}
\end{table*}
""")
        print(f"Saved: mediator_by_meal.tex")
    
    # =========================================================================
    # Table 4: Statistics by Subject (with cohort information)
    # Note: Subject IDs 1-6 in 2018 are DIFFERENT PEOPLE from 1-6 in 2020
    # =========================================================================
    cohort_col = get_column(df, 'cohort')

    if patient_col:
        subject_stats = []
        for subj in df[patient_col].unique():
            subj_df = df[df[patient_col] == subj]

            # Get cohort for this subject
            cohort = subj_df[cohort_col].iloc[0] if cohort_col and cohort_col in subj_df.columns else 'Unknown'

            # Get the original subject_id (without cohort prefix)
            orig_subj_id = subj_df['subject_id'].iloc[0] if 'subject_id' in subj_df.columns else subj

            row = {
                'Subject_Unique': subj,
                'Cohort': cohort,
                'Subject_ID': orig_subj_id,
                'N Events': len(subj_df)
            }

            if treatment_col:
                carbs = subj_df[treatment_col].dropna()
                row['Carbs Mean'] = f"{carbs.mean():.1f}" if len(carbs) > 0 else 'NA'
                row['Carbs SD'] = f"{carbs.std():.1f}" if len(carbs) > 1 else 'NA'

            if mediator_col:
                bolus = subj_df[mediator_col].dropna()
                row['Bolus Mean'] = f"{bolus.mean():.2f}" if len(bolus) > 0 else 'NA'
                row['Bolus SD'] = f"{bolus.std():.2f}" if len(bolus) > 1 else 'NA'

            if glucose_col:
                glucose = subj_df[glucose_col].dropna()
                row['Glucose Mean'] = f"{glucose.mean():.1f}" if len(glucose) > 0 else 'NA'
                row['Glucose SD'] = f"{glucose.std():.1f}" if len(glucose) > 1 else 'NA'

            subject_stats.append(row)

        table4 = pd.DataFrame(subject_stats)
        # Sort by cohort then by original subject_id (not N Events)
        # Convert Subject_ID to numeric for proper sorting
        table4['_sort_id'] = pd.to_numeric(table4['Subject_ID'], errors='coerce')
        table4 = table4.sort_values(['Cohort', '_sort_id'], ascending=[True, True])
        table4 = table4.drop(columns=['_sort_id', 'Subject_Unique'])  # Remove internal ID column
        table4.to_csv(tables_dir / 'statistics_by_subject.csv', index=False)
        print(f"Saved: statistics_by_subject.csv")
    
    # =========================================================================
    # Table 5: Cross-tabulation Subject x Meal Type
    # =========================================================================
    if patient_col and meal_type_col:
        cross_tab = pd.crosstab(df[patient_col], df[meal_type_col])
        meal_types = get_sorted_meal_types(df, meal_type_col)
        cross_tab = cross_tab[[m for m in meal_types if m in cross_tab.columns]]
        cross_tab.to_csv(tables_dir / 'subject_meal_crosstab.csv')
        print(f"Saved: subject_meal_crosstab.csv")
    
    # =========================================================================
    # Table 6: Correlation Matrix (raw data variables only)
    # NOTE: φ correlations are in generate_embedding_diagnostics.py
    # =========================================================================
    corr_vars = []
    var_names = {}

    if treatment_col:
        corr_vars.append(treatment_col)
        var_names[treatment_col] = 'Carbohydrates (g)'
    if mediator_col:
        corr_vars.append(mediator_col)
        var_names[mediator_col] = 'Bolus Insulin (U)'
    if glucose_col:
        corr_vars.append(glucose_col)
        var_names[glucose_col] = 'Glucose at Meal (mg/dL)'

    if len(corr_vars) >= 2:
        corr_matrix = df[corr_vars].corr()
        corr_matrix = corr_matrix.rename(index=var_names, columns=var_names)
        corr_matrix.to_csv(tables_dir / 'correlation_matrix.csv', float_format='%.3f')
        print(f"Saved: correlation_matrix.csv")

    # =========================================================================
    # Stratified Table: Treatment/Mediator by Meal Type AND Cohort
    # Uses LaTeX multirow for proper publication formatting
    # =========================================================================
    cohort_col = get_column(df, 'cohort')
    if cohort_col and cohort_col in df.columns and treatment_col and meal_type_col:
        cohorts = sorted(df[cohort_col].unique())
        meal_types = get_sorted_meal_types(df, meal_type_col)

        # Build stratified data
        stratified_data = []
        for cohort in cohorts:
            for meal in meal_types:
                subset = df[(df[cohort_col] == cohort) & (df[meal_type_col] == meal)]
                carbs = subset[treatment_col].dropna() if treatment_col else pd.Series()
                bolus = subset[mediator_col].dropna() if mediator_col else pd.Series()

                stratified_data.append({
                    'Cohort': str(cohort),
                    'Meal Type': str(meal).capitalize(),
                    'N': len(subset),
                    'Carbs Mean': f"{carbs.mean():.1f}" if len(carbs) > 0 else '--',
                    'Carbs SD': f"{carbs.std():.1f}" if len(carbs) > 1 else '--',
                    'Bolus Mean': f"{bolus.mean():.2f}" if len(bolus) > 0 else '--',
                    'Bolus SD': f"{bolus.std():.2f}" if len(bolus) > 1 else '--',
                })

        stratified_df = pd.DataFrame(stratified_data)
        stratified_df.to_csv(tables_dir / 'variables_by_cohort_meal.csv', index=False)
        print(f"Saved: variables_by_cohort_meal.csv")

        # LaTeX version with multirow for cohort grouping
        n_meals = len(meal_types)
        latex_content = r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Treatment and mediator distributions stratified by cohort and meal type.}
Summary statistics showing carbohydrate intake (treatment) and insulin bolus (mediator)
stratified by data collection cohort and meal category.
N = number of meal events; Mean and SD in original units (g for carbs, U for insulin).}
\label{tab:stratified_cohort_meal}
\begin{tabular}{llccccc}
\toprule
& & & \multicolumn{2}{c}{Carbohydrate (g)} & \multicolumn{2}{c}{Insulin Bolus (U)} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7}
Cohort & Meal Type & N & Mean & SD & Mean & SD \\
\midrule
"""
        for i, cohort in enumerate(cohorts):
            cohort_rows = stratified_df[stratified_df['Cohort'] == str(cohort)]
            for j, (_, row) in enumerate(cohort_rows.iterrows()):
                # Use multirow for first meal of each cohort
                if j == 0:
                    cohort_cell = f"\\multirow{{{n_meals}}}{{*}}{{{cohort}}}"
                else:
                    cohort_cell = ""
                latex_content += f"{cohort_cell} & {row['Meal Type']} & {row['N']} & "
                latex_content += f"{row['Carbs Mean']} & {row['Carbs SD']} & "
                latex_content += f"{row['Bolus Mean']} & {row['Bolus SD']} \\\\\n"
            # Add midrule between cohorts (but not after last)
            if i < len(cohorts) - 1:
                latex_content += "\\midrule\n"

        latex_content += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        latex_path = tables_dir / 'variables_by_cohort_meal.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"Saved: variables_by_cohort_meal.tex")

    # =========================================================================
    # Stratified Table: Variables by Split (Train/Test)
    # =========================================================================
    split_col = get_column(df, 'split')
    if split_col and split_col in df.columns and treatment_col and meal_type_col:
        splits = sorted(df[split_col].unique())
        meal_types = get_sorted_meal_types(df, meal_type_col)

        split_data = []
        for split in splits:
            for meal in meal_types:
                subset = df[(df[split_col] == split) & (df[meal_type_col] == meal)]
                carbs = subset[treatment_col].dropna() if treatment_col else pd.Series()
                bolus = subset[mediator_col].dropna() if mediator_col else pd.Series()

                split_data.append({
                    'Split': str(split).capitalize(),
                    'Meal Type': str(meal).capitalize(),
                    'N': len(subset),
                    'Carbs Mean': f"{carbs.mean():.1f}" if len(carbs) > 0 else '--',
                    'Carbs SD': f"{carbs.std():.1f}" if len(carbs) > 1 else '--',
                    'Bolus Mean': f"{bolus.mean():.2f}" if len(bolus) > 0 else '--',
                    'Bolus SD': f"{bolus.std():.2f}" if len(bolus) > 1 else '--',
                })

        split_df = pd.DataFrame(split_data)
        split_df.to_csv(tables_dir / 'variables_by_split_meal.csv', index=False)
        print(f"Saved: variables_by_split_meal.csv")

        # LaTeX version with multirow
        n_meals = len(meal_types)
        latex_content = r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Treatment and mediator distributions stratified by data split and meal type.}
Demonstrates that training and test sets have comparable distributions across meal types.
N = number of meal events; Mean and SD in original units.}
\label{tab:stratified_split_meal}
\begin{tabular}{llccccc}
\toprule
& & & \multicolumn{2}{c}{Carbohydrate (g)} & \multicolumn{2}{c}{Insulin Bolus (U)} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7}
Split & Meal Type & N & Mean & SD & Mean & SD \\
\midrule
"""
        for i, split in enumerate(splits):
            split_rows = split_df[split_df['Split'] == str(split).capitalize()]
            for j, (_, row) in enumerate(split_rows.iterrows()):
                if j == 0:
                    split_cell = f"\\multirow{{{n_meals}}}{{*}}{{{str(split).capitalize()}}}"
                else:
                    split_cell = ""
                latex_content += f"{split_cell} & {row['Meal Type']} & {row['N']} & "
                latex_content += f"{row['Carbs Mean']} & {row['Carbs SD']} & "
                latex_content += f"{row['Bolus Mean']} & {row['Bolus SD']} \\\\\n"
            if i < len(splits) - 1:
                latex_content += "\\midrule\n"

        latex_content += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        latex_path = tables_dir / 'variables_by_split_meal.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"Saved: variables_by_split_meal.tex")

    return


# =============================================================================
# INDIVIDUAL FIGURE GENERATION
# =============================================================================

def plot_fig01_treatment_histogram(df, output_dir):
    """Figure 1: Treatment (carbohydrate) distribution histogram."""
    treatment_col = get_column(df, 'treatment')
    if not treatment_col:
        return None
    
    carbs = df[treatment_col].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(carbs, bins=50, color=COLORS['treatment'], alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axvline(carbs.mean(), color='black', linestyle='--', linewidth=2,
              label=f'Mean = {carbs.mean():.1f}g')
    ax.axvline(carbs.median(), color='gray', linestyle='-', linewidth=2,
              label=f'Median = {carbs.median():.1f}g')
    
    ax.set_xlabel('Carbohydrate Intake (g)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Carbohydrate Intake', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    summary_text = f'N = {len(carbs):,}\nSD = {carbs.std():.1f}g'
    ax.text(0.95, 0.85, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig01_treatment_histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig01_treatment_histogram.png")
    plt.close()


def plot_fig02_treatment_by_meal_type(df, output_dir):
    """Figure 2: Treatment distribution by meal type."""
    treatment_col = get_column(df, 'treatment')
    meal_type_col = get_column(df, 'meal_type')
    if not treatment_col:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_meal, labels, colors_list = get_meal_boxplot_data(df, treatment_col, meal_type_col)
    
    if data_by_meal is not None:
        bp = ax.boxplot(data_by_meal, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add jittered points
        for i, (data, color) in enumerate(zip(data_by_meal, colors_list)):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                      alpha=0.3, s=15, color=color, zorder=1)
        
        # Sample sizes
        for i, data in enumerate(data_by_meal):
            ax.text(i + 1, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                   f'n={len(data)}', ha='center', va='top', fontsize=10)
    
    ax.set_ylabel('Carbohydrate Intake (g)', fontsize=12)
    ax.set_xlabel('Meal Type', fontsize=12)
    ax.set_title('Carbohydrate Intake by Meal Type', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig02_treatment_by_meal_type.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig02_treatment_by_meal_type.png")
    plt.close()


def plot_fig03_treatment_by_subject(df, output_dir):
    """Figure 3: Treatment distribution by subject."""
    treatment_col = get_column(df, 'treatment')
    patient_col = get_column(df, 'patient')
    if not treatment_col or not patient_col:
        return None
    
    subject_medians = df.groupby(patient_col)[treatment_col].median().sort_values()
    subjects = subject_medians.index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_subject = [df[df[patient_col] == s][treatment_col].dropna().values for s in subjects]
    
    bp = ax.boxplot(data_by_subject, labels=[f'S{i+1}' for i in range(len(subjects))],
                   patch_artist=True)
    
    norm = plt.Normalize(subject_medians.min(), subject_medians.max())
    cmap = plt.cm.Reds
    for patch, median in zip(bp['boxes'], subject_medians.values):
        patch.set_facecolor(cmap(norm(median)))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Carbohydrate Intake (g)', fontsize=12)
    ax.set_xlabel('Subject (ordered by median intake)', fontsize=12)
    ax.set_title('Carbohydrate Intake by Subject', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Median Carbs (g)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig03_treatment_by_subject.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig03_treatment_by_subject.png")
    plt.close()


def plot_fig04_mediator_histogram(df, output_dir):
    """Figure 4: Mediator (insulin bolus) distribution histogram."""
    mediator_col = get_column(df, 'mediator')
    if not mediator_col:
        return None
    
    bolus = df[mediator_col].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bolus, bins=50, color=COLORS['mediator'], alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axvline(bolus.mean(), color='black', linestyle='--', linewidth=2,
              label=f'Mean = {bolus.mean():.2f}U')
    ax.axvline(bolus.median(), color='gray', linestyle='-', linewidth=2,
              label=f'Median = {bolus.median():.2f}U')
    
    ax.set_xlabel('Insulin Bolus (U)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Insulin Bolus', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    pct_zero = 100 * (bolus == 0).mean()
    summary_text = f'N = {len(bolus):,}\nSD = {bolus.std():.2f}U\n% Zero = {pct_zero:.1f}%'
    ax.text(0.95, 0.85, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig04_mediator_histogram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig04_mediator_histogram.png")
    plt.close()


def plot_fig05_mediator_by_meal_type(df, output_dir):
    """Figure 5: Mediator distribution by meal type."""
    mediator_col = get_column(df, 'mediator')
    meal_type_col = get_column(df, 'meal_type')
    if not mediator_col:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_meal, labels, colors_list = get_meal_boxplot_data(df, mediator_col, meal_type_col)
    
    if data_by_meal is not None:
        bp = ax.boxplot(data_by_meal, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for i, (data, color) in enumerate(zip(data_by_meal, colors_list)):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                      alpha=0.3, s=15, color=color, zorder=1)
        
        for i, data in enumerate(data_by_meal):
            ax.text(i + 1, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                   f'n={len(data)}', ha='center', va='top', fontsize=10)
    
    ax.set_ylabel('Insulin Bolus (U)', fontsize=12)
    ax.set_xlabel('Meal Type', fontsize=12)
    ax.set_title('Insulin Bolus by Meal Type', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05_mediator_by_meal_type.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig05_mediator_by_meal_type.png")
    plt.close()


def plot_fig06_mediator_by_subject(df, output_dir):
    """Figure 6: Mediator distribution by subject."""
    mediator_col = get_column(df, 'mediator')
    patient_col = get_column(df, 'patient')
    if not mediator_col or not patient_col:
        return None
    
    subject_medians = df.groupby(patient_col)[mediator_col].median().sort_values()
    subjects = subject_medians.index.tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_subject = [df[df[patient_col] == s][mediator_col].dropna().values for s in subjects]
    
    bp = ax.boxplot(data_by_subject, labels=[f'S{i+1}' for i in range(len(subjects))],
                   patch_artist=True)
    
    norm = plt.Normalize(subject_medians.min(), subject_medians.max())
    cmap = plt.cm.Blues
    for patch, median in zip(bp['boxes'], subject_medians.values):
        patch.set_facecolor(cmap(norm(median)))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Insulin Bolus (U)', fontsize=12)
    ax.set_xlabel('Subject (ordered by median bolus)', fontsize=12)
    ax.set_title('Insulin Bolus by Subject', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Median Bolus (U)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig06_mediator_by_subject.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig06_mediator_by_subject.png")
    plt.close()


def plot_fig07_treatment_mediator_scatter(df, output_dir):
    """Figure 7: Treatment-mediator relationship scatter."""
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    if not treatment_col or not mediator_col:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    valid_idx = df[treatment_col].notna() & df[mediator_col].notna()
    x = df.loc[valid_idx, treatment_col]
    y = df.loc[valid_idx, mediator_col]
    
    ax.scatter(x, y, alpha=0.4, s=25, color=COLORS['mediator'])
    
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_range = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_range, p(x_range), 'r--', linewidth=2, label='Linear fit')
        r = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x)}', transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Carbohydrate Intake (g)', fontsize=12)
    ax.set_ylabel('Insulin Bolus (U)', fontsize=12)
    ax.set_title('Treatment-Mediator Relationship', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig07_treatment_mediator_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig07_treatment_mediator_scatter.png")
    plt.close()


def plot_fig08_treatment_mediator_by_meal(df, output_dir):
    """Figure 8: Treatment-mediator relationship by meal type (faceted), colored by subject."""
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    meal_type_col = get_column(df, 'meal_type')
    patient_col = get_column(df, 'patient')
    if not treatment_col or not mediator_col or not meal_type_col:
        return None
    
    meal_types = get_sorted_meal_types(df, meal_type_col)
    n_meals = len(meal_types)
    if n_meals == 0:
        return None
    
    # Create a colorblind-friendly palette for subjects
    # Using Paul Tol's qualitative palette (colorblind-safe)
    # https://personal.sron.nl/~pault/#sec:qualitative
    COLORBLIND_QUALITATIVE = [
        '#4477AA',  # Blue
        '#EE6677',  # Rose
        '#228833',  # Green
        '#CCBB44',  # Yellow
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB',  # Grey
        '#332288',  # Indigo
        '#88CCEE',  # Light cyan
        '#44AA99',  # Teal
        '#117733',  # Dark green
        '#999933',  # Olive
        '#DDCC77',  # Sand
        '#CC6677',  # Rose pink
        '#882255',  # Wine
        '#AA4499',  # Magenta
    ]

    if patient_col:
        unique_subjects = sorted(df[patient_col].dropna().unique())
        n_subjects = len(unique_subjects)
        # Use colorblind-friendly qualitative palette
        if n_subjects <= len(COLORBLIND_QUALITATIVE):
            subject_colors = COLORBLIND_QUALITATIVE[:n_subjects]
        else:
            # Fall back to viridis for many subjects (perceptually uniform)
            subject_colors = [plt.cm.viridis(i / n_subjects) for i in range(n_subjects)]
        subject_color_map = {subj: subject_colors[i] for i, subj in enumerate(unique_subjects)}
    else:
        subject_color_map = None
    
    n_cols = min(2, n_meals)
    n_rows = (n_meals + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    if n_meals == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Helper to format subject IDs for the legend
    def format_subject_label(subj_id):
        """Format subject ID for legend (e.g., '2018_1' -> '2018-S1')."""
        s = str(subj_id)
        if '_' in s:
            parts = s.split('_')
            if len(parts) == 2:
                cohort, num = parts
                return f'{cohort}-S{num}'
            elif len(parts) == 3:
                # Handle double-prefixed IDs like '2018_2018_1'
                _, cohort, num = parts
                return f'{cohort}-S{num}'
        return s

    # Build legend handles once (shared across panels)
    if patient_col and subject_color_map and n_subjects <= 15:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=subject_color_map[s],
                                       markersize=8, label=format_subject_label(s))
                          for s in unique_subjects]
    else:
        legend_elements = None

    panel_labels = 'abcdefghijklmnopqrstuvwxyz'

    for i, meal in enumerate(meal_types):
        ax = axes[i]
        meal_df = df[df[meal_type_col] == meal]

        valid_idx = meal_df[treatment_col].notna() & meal_df[mediator_col].notna()
        x = meal_df.loc[valid_idx, treatment_col]
        y = meal_df.loc[valid_idx, mediator_col]

        # Color by subject_id if available
        if patient_col and subject_color_map:
            subjects = meal_df.loc[valid_idx, patient_col]
            colors = [subject_color_map.get(s, 'gray') for s in subjects]
            ax.scatter(x, y, alpha=0.6, s=30, c=colors)
        else:
            ax.scatter(x, y, alpha=0.5, s=30, color='steelblue')

        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_range = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_range, p(x_range), 'k--', linewidth=2)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f'r = {r:.3f}\nn = {len(x)}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Carbohydrate Intake (g)', fontsize=10)
        ax.set_ylabel('Insulin Bolus (U)', fontsize=10)
        ax.set_title(f'{str(meal).capitalize()}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Panel letter label
        ax.text(-0.12, 1.08, panel_labels[i], transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                fontfamily='sans-serif')

        # Per-panel subject legend – placed outside the axes to the right
        if legend_elements is not None:
            ax.legend(handles=legend_elements, title='Subject', fontsize=7,
                      title_fontsize=8, loc='center left',
                      bbox_to_anchor=(1.02, 0.5),
                      frameon=True, fancybox=True, shadow=False,
                      edgecolor='black', facecolor='white', framealpha=1.0)

    for j in range(n_meals, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Treatment-Mediator Relationship by Meal Type\n(Colored by Subject)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    fig.subplots_adjust(wspace=0.55)
    plt.savefig(output_dir / 'fig08_treatment_mediator_by_meal.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig08_treatment_mediator_by_meal.png")
    plt.close()


def plot_fig09_meal_type_distribution(df, output_dir):
    """Figure 9: Meal type distribution bar chart."""
    meal_type_col = get_column(df, 'meal_type')
    if not meal_type_col:
        return None
    
    meal_types = get_sorted_meal_types(df, meal_type_col)
    meal_counts = df[meal_type_col].value_counts()
    meal_counts = meal_counts.reindex(meal_types)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors_list = [get_meal_color(m) for m in meal_counts.index]
    bars = ax.bar(range(len(meal_counts)), meal_counts.values, color=colors_list,
                 alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(meal_counts)))
    ax.set_xticklabels([str(m).capitalize() for m in meal_counts.index], fontsize=11)
    ax.set_ylabel('Number of Meals', fontsize=12)
    ax.set_xlabel('Meal Type', fontsize=12)
    ax.set_title('Meal Type Distribution', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, meal_counts.values):
        pct = 100 * count / meal_counts.sum()
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig09_meal_type_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig09_meal_type_distribution.png")
    plt.close()


def plot_fig10_events_per_subject(df, output_dir):
    """Figure 10: Events per subject distribution."""
    patient_col = get_column(df, 'patient')
    if not patient_col:
        return None
    
    events_per_subject = df.groupby(patient_col).size().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(events_per_subject)), events_per_subject.values,
          color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Subject (ordered by # events)', fontsize=12)
    ax.set_ylabel('Number of Meal Events', fontsize=12)
    ax.set_title('Meal Events per Subject', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean line
    ax.axhline(events_per_subject.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean = {events_per_subject.mean():.1f}')
    ax.legend(loc='upper right')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_events_per_subject.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig10_events_per_subject.png")
    plt.close()


def plot_fig11_subject_meal_heatmap(df, output_dir):
    """Figure 11: Heatmap of meal types by subject."""
    patient_col = get_column(df, 'patient')
    meal_type_col = get_column(df, 'meal_type')
    if not patient_col or not meal_type_col:
        return None
    
    cross_tab = pd.crosstab(df[patient_col], df[meal_type_col])
    meal_types = get_sorted_meal_types(df, meal_type_col)
    cross_tab = cross_tab[[m for m in meal_types if m in cross_tab.columns]]
    cross_tab = cross_tab.loc[cross_tab.sum(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(cross_tab) * 0.5)))
    
    im = ax.imshow(cross_tab.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(cross_tab.columns)))
    ax.set_xticklabels([str(c).capitalize() for c in cross_tab.columns], fontsize=11)
    ax.set_yticks(range(len(cross_tab.index)))
    ax.set_yticklabels([f'S{i+1}' for i in range(len(cross_tab.index))], fontsize=9)
    
    ax.set_xlabel('Meal Type', fontsize=12)
    ax.set_ylabel('Subject', fontsize=12)
    ax.set_title('Meal Events by Subject and Meal Type', fontweight='bold', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Events', fontsize=10)

    for i in range(len(cross_tab.index)):
        for j in range(len(cross_tab.columns)):
            val = cross_tab.iloc[i, j]
            text_color = 'white' if val > cross_tab.values.max() / 2 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center', color=text_color, fontsize=8)

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig11_subject_meal_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig11_subject_meal_heatmap.png")
    plt.close()


# NOTE: φ (phi) distribution figures have been moved to generate_embedding_diagnostics.py
# Data distribution visualizations should focus on raw data variables (carbs, bolus, glucose),
# not the learned latent embeddings. Use the R script create_meal_windows_summary_table.R
# in cma_cluster/ for publication-quality data summary tables.


def plot_fig12_cohort_comparison(df, output_dir):
    """
    Figure 12: Publication-quality cohort comparison with 4 panels.
    A. Treatment Distribution (density by cohort)
    B. Mediator Distribution (density by cohort)
    C. Baseline Glucose Distribution (density by cohort)
    D. Meal Type Distribution (bar chart by cohort)

    Uses proper density estimation with non-negative bounds for carbs/bolus.
    """
    from scipy.stats import gaussian_kde

    cohort_col = get_column(df, 'cohort')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or cohort_col not in df.columns:
        print("No cohort column found - skipping cohort comparison figure")
        return None

    cohorts = sorted(df[cohort_col].unique())
    # Colorblind-friendly cohort colors from COLORS palette
    cohort_colors = {
        '2018': COLORS['cohort_2018'],
        '2020': COLORS['cohort_2020']
    }
    # Handle both string and int cohort labels
    for c in cohorts:
        if str(c) not in cohort_colors:
            cohort_colors[str(c)] = COLORS['cohort_2018'] if cohorts.index(c) == 0 else COLORS['cohort_2020']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def plot_density_panel(ax, col, xlabel, title, panel_letter, min_val=None):
        """Helper to plot density with proper bounds."""
        if not col or col not in df.columns:
            return

        for cohort in cohorts:
            cohort_data = df[df[cohort_col] == cohort][col].dropna()
            if len(cohort_data) < 2:
                continue

            # Calculate KDE
            kde = gaussian_kde(cohort_data)

            # Set x range with proper lower bound
            x_min = min_val if min_val is not None else cohort_data.min()
            x_max = cohort_data.max() * 1.05  # Small buffer on top
            x_range = np.linspace(x_min, x_max, 300)

            # Get density values
            density = kde(x_range)

            color = cohort_colors.get(str(cohort), 'gray')
            ax.plot(x_range, density, color=color, linewidth=2, label=str(cohort))
            ax.fill_between(x_range, density, alpha=0.3, color=color)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Cohort', loc='upper right')
        ax.set_xlim(left=min_val if min_val is not None else None)
        ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                fontfamily='sans-serif')

    # Panel A: Treatment Distribution (carbs >= 0)
    plot_density_panel(axes[0, 0], treatment_col, 'Carbohydrate Intake (g)',
                       'Treatment Distribution', 'a', min_val=0)

    # Panel B: Mediator Distribution (bolus >= 0)
    plot_density_panel(axes[0, 1], mediator_col, 'Insulin Bolus (units)',
                       'Mediator Distribution', 'b', min_val=0)

    # Panel C: Baseline Glucose Distribution (glucose > 0, no artificial lower bound)
    ax = axes[1, 0]
    if glucose_col and glucose_col in df.columns:
        for cohort in cohorts:
            cohort_data = df[df[cohort_col] == cohort][glucose_col].dropna()
            if len(cohort_data) < 2:
                continue
            kde = gaussian_kde(cohort_data)
            x_range = np.linspace(max(0, cohort_data.min() - 20), cohort_data.max() + 20, 300)
            density = kde(x_range)
            color = cohort_colors.get(str(cohort), 'gray')
            ax.plot(x_range, density, color=color, linewidth=2, label=str(cohort))
            ax.fill_between(x_range, density, alpha=0.3, color=color)

        # Add reference lines for normal glucose range (70-180 mg/dL)
        ax.axvline(x=70, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(x=180, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_xlabel('Pre-meal Glucose (mg/dL)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Baseline Glucose Distribution', fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Cohort', loc='upper right')
        ax.set_xlim(left=0)
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # Panel D: Meal Type Distribution (bar chart by cohort)
    ax = axes[1, 1]
    if meal_type_col and meal_type_col in df.columns:
        meal_types = get_sorted_meal_types(df, meal_type_col)

        # Calculate proportions for each cohort
        cohort_props = {}
        for cohort in cohorts:
            cohort_df = df[df[cohort_col] == cohort]
            meal_counts = cohort_df[meal_type_col].value_counts()
            total = len(cohort_df)
            cohort_props[cohort] = {m: meal_counts.get(m, 0) / total for m in meal_types}

        x = np.arange(len(meal_types))
        width = 0.35

        for i, cohort in enumerate(cohorts):
            props = [cohort_props[cohort].get(m, 0) for m in meal_types]
            offset = -width/2 + i * width
            ax.bar(x + offset, props, width, label=str(cohort),
                   color=cohort_colors.get(str(cohort), 'gray'), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m).capitalize() for m in meal_types], fontsize=10)
        ax.set_xlabel('Meal Type', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title('Meal Type Distribution', fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Cohort', loc='upper right')
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig12_cohort_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig12_cohort_comparison.png")
    plt.close()


def plot_fig14_cohort_insulin_carbs(df, output_dir):
    """
    Figure 14: Insulin and carb distributions with cohort-colored points.

    A. Scatter of carbs vs insulin, colored by cohort, with zero-insulin highlighted
    B. Stacked bar showing % zero-insulin by cohort and meal type
    C. Strip/jitter of insulin values by cohort (showing density of zeros)
    D. Strip/jitter of carb values by cohort
    """
    cohort_col = get_column(df, 'cohort')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or not treatment_col or not mediator_col:
        print("Missing required columns - skipping fig14")
        return None

    cohorts = sorted(df[cohort_col].dropna().unique())
    cohort_colors = {
        str(cohorts[0]): COLORS['cohort_2018'],
        str(cohorts[1]): COLORS['cohort_2020'] if len(cohorts) > 1 else COLORS['secondary'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ------------------------------------------------------------------
    # Panel A: Carbs vs Insulin scatter, colored by cohort
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for cohort in cohorts:
        mask = df[cohort_col] == cohort
        carbs = df.loc[mask, treatment_col]
        bolus = df.loc[mask, mediator_col]
        color = cohort_colors[str(cohort)]

        # Plot non-zero bolus
        nz = bolus > 0
        ax.scatter(carbs[nz], bolus[nz], alpha=0.4, s=20, color=color,
                   label=f'{cohort} (bolus > 0)')

        # Highlight zero bolus with distinct marker
        ax.scatter(carbs[~nz], bolus[~nz], alpha=0.7, s=35, color=color,
                   marker='x', linewidths=1.2,
                   label=f'{cohort} (bolus = 0, n={int((~nz).sum())})')

    ax.set_xlabel('Carbohydrate Intake (g)', fontsize=11)
    ax.set_ylabel('Insulin Bolus (U)', fontsize=11)
    ax.set_title('Carbs vs Insulin by Cohort', fontweight='bold', loc='left', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel B: % zero-insulin by cohort and meal type
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    if meal_type_col and meal_type_col in df.columns:
        meal_types = get_sorted_meal_types(df, meal_type_col)
        x = np.arange(len(meal_types))
        width = 0.35

        for i, cohort in enumerate(cohorts):
            cohort_df = df[df[cohort_col] == cohort]
            pcts = []
            counts = []
            for meal in meal_types:
                meal_df = cohort_df[cohort_df[meal_type_col] == meal]
                n_total = len(meal_df)
                n_zero = (meal_df[mediator_col] == 0).sum() if n_total > 0 else 0
                pcts.append(100 * n_zero / n_total if n_total > 0 else 0)
                counts.append(f'{n_zero}/{n_total}')

            offset = -width / 2 + i * width
            color = cohort_colors[str(cohort)]
            bars = ax.bar(x + offset, pcts, width, label=str(cohort),
                          color=color, alpha=0.8)

            for bar, count_str in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        count_str, ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m).capitalize() for m in meal_types], fontsize=10)
        ax.set_xlabel('Meal Type', fontsize=11)
        ax.set_ylabel('% Zero Insulin', fontsize=11)
        ax.set_title('Zero-Insulin Meals by Cohort and Meal Type',
                      fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Cohort', loc='upper right')
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel C: Insulin distribution by cohort (strip plot showing zeros)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    for i, cohort in enumerate(cohorts):
        bolus = df.loc[df[cohort_col] == cohort, mediator_col].dropna()
        color = cohort_colors[str(cohort)]
        jitter = np.random.normal(0, 0.12, len(bolus))
        ax.scatter(np.ones(len(bolus)) * i + jitter, bolus, alpha=0.35, s=12,
                   color=color)
        # Add summary stats
        n_zero = (bolus == 0).sum()
        ax.text(i, bolus.max() * 0.95,
                f'n={len(bolus)}\nzero={n_zero}\n({100*n_zero/len(bolus):.0f}%)',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([str(c) for c in cohorts], fontsize=11)
    ax.set_xlabel('Cohort', fontsize=11)
    ax.set_ylabel('Insulin Bolus (U)', fontsize=11)
    ax.set_title('Insulin Distribution by Cohort', fontweight='bold', loc='left', fontsize=12)
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel D: Carbs distribution by cohort (strip plot)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    for i, cohort in enumerate(cohorts):
        carbs = df.loc[df[cohort_col] == cohort, treatment_col].dropna()
        color = cohort_colors[str(cohort)]
        jitter = np.random.normal(0, 0.12, len(carbs))
        ax.scatter(np.ones(len(carbs)) * i + jitter, carbs, alpha=0.35, s=12,
                   color=color)
        ax.text(i, carbs.max() * 0.95,
                f'n={len(carbs)}\nmean={carbs.mean():.0f}g\nmed={carbs.median():.0f}g',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels([str(c) for c in cohorts], fontsize=11)
    ax.set_xlabel('Cohort', fontsize=11)
    ax.set_ylabel('Carbohydrate Intake (g)', fontsize=11)
    ax.set_title('Carb Distribution by Cohort', fontweight='bold', loc='left', fontsize=12)
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig14_cohort_insulin_carbs.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig14_cohort_insulin_carbs.png")
    plt.close()


def plot_fig15_zero_bolus_glucose(df, output_dir):
    """
    Figure 15: Delta glucose outcomes for zero-bolus meals, by cohort.

    Shows the glucose trajectory for meals where no insulin was taken,
    broken down by cohort and meal type to understand whether the
    zero-bolus pattern differs between 2018 and 2020.

    Panels:
    A. Distribution of final delta glucose (Y_210min) for zero-bolus meals, by cohort
    B. Delta glucose by meal type for zero-bolus meals, by cohort
    C. Glucose trajectory over time (60-210 min) for zero-bolus meals, by cohort
    D. Scatter of carbs vs final delta glucose for zero-bolus meals, by cohort
    """
    cohort_col = get_column(df, 'cohort')
    mediator_col = get_column(df, 'mediator')
    treatment_col = get_column(df, 'treatment')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or not mediator_col:
        print("Missing required columns - skipping fig15")
        return None

    # Find available Y timepoint columns
    y_cols = sorted([c for c in df.columns if c.startswith('Y_') and c.endswith('min')],
                    key=lambda c: int(c.split('_')[1].replace('min', '')))
    if not y_cols:
        print("No Y_*min outcome columns found - skipping fig15")
        return None

    final_y_col = y_cols[-1]  # e.g. Y_210min
    final_minutes = int(final_y_col.split('_')[1].replace('min', ''))

    # Filter to truly zero-bolus meals: require BOTH mediator_bolus_for_meal AND
    # total_bolus to be zero. mediator_bolus_for_meal covers -120 to +60 min,
    # while total_bolus covers the full window (-120 to +240 min). Filtering on
    # only the mediator would include meals where a late bolus was taken (>60 min
    # post-meal), masking the true no-insulin glucose response.
    total_bolus_col = 'total_bolus' if 'total_bolus' in df.columns else None
    if total_bolus_col:
        zero_bolus = df[(df[mediator_col] == 0) & (df[total_bolus_col] == 0)].copy()
        filter_desc = "mediator_bolus_for_meal=0 AND total_bolus=0"
    else:
        print("WARNING: 'total_bolus' column not found, filtering on mediator_bolus_for_meal only.")
        print("  Re-run train_and_export_embeddings.py to export total_bolus.")
        zero_bolus = df[df[mediator_col] == 0].copy()
        filter_desc = "mediator_bolus_for_meal=0 only (total_bolus unavailable)"
    n_zero = len(zero_bolus)
    if n_zero == 0:
        print("No zero-bolus meals found - skipping fig15")
        return None
    print(f"  Zero-bolus filter: {filter_desc} -> {n_zero} meals")

    cohorts = sorted(df[cohort_col].dropna().unique())
    cohort_colors = {
        str(cohorts[0]): COLORS['cohort_2018'],
        str(cohorts[1]): COLORS['cohort_2020'] if len(cohorts) > 1 else COLORS['secondary'],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ------------------------------------------------------------------
    # Panel A: Histogram of final delta glucose for zero-bolus, by cohort
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for cohort in cohorts:
        cohort_data = zero_bolus[zero_bolus[cohort_col] == cohort][final_y_col].dropna()
        if len(cohort_data) == 0:
            continue
        color = cohort_colors[str(cohort)]
        ax.hist(cohort_data, bins=30, alpha=0.5, color=color, edgecolor='black',
                linewidth=0.3, label=f'{cohort} (N={len(cohort_data)}, '
                f'mean={cohort_data.mean():.1f})')

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel(f'Delta Glucose at {final_minutes} min (mg/dL)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Delta Glucose at {final_minutes} min (Zero-Bolus Meals)',
                  fontweight='bold', loc='left', fontsize=12)
    ax.legend(fontsize=9)
    ax.text(-0.12, 1.08, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel B: Boxplot of final delta glucose by meal type, by cohort
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    if meal_type_col and meal_type_col in zero_bolus.columns:
        meal_types = get_sorted_meal_types(zero_bolus, meal_type_col)
        x = np.arange(len(meal_types))
        width = 0.35

        for i, cohort in enumerate(cohorts):
            cohort_df = zero_bolus[zero_bolus[cohort_col] == cohort]
            data_by_meal = []
            for meal in meal_types:
                vals = cohort_df.loc[cohort_df[meal_type_col] == meal, final_y_col].dropna()
                data_by_meal.append(vals.values)

            positions = x + (-width / 2 + i * width)
            color = cohort_colors[str(cohort)]
            bp = ax.boxplot(data_by_meal, positions=positions, widths=width * 0.8,
                            patch_artist=True, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')

            # Annotate sample sizes
            for j, vals in enumerate(data_by_meal):
                ax.text(positions[j], ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -5,
                        f'n={len(vals)}', ha='center', va='top', fontsize=7)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([str(m).capitalize() for m in meal_types], fontsize=10)
        ax.set_xlabel('Meal Type', fontsize=11)
        ax.set_ylabel(f'Delta Glucose at {final_minutes} min (mg/dL)', fontsize=11)
        ax.set_title('Delta Glucose by Meal Type (Zero-Bolus)',
                      fontweight='bold', loc='left', fontsize=12)
        # Manual legend
        from matplotlib.patches import Patch
        legend_patches = [Patch(facecolor=cohort_colors[str(c)], alpha=0.7, label=str(c))
                          for c in cohorts]
        ax.legend(handles=legend_patches, title='Cohort', loc='upper right', fontsize=9)
    ax.text(-0.12, 1.08, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel C: Mean glucose trajectory over time for zero-bolus, by cohort
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    time_points = [int(c.split('_')[1].replace('min', '')) for c in y_cols]

    for cohort in cohorts:
        cohort_df = zero_bolus[zero_bolus[cohort_col] == cohort]
        n_cohort = len(cohort_df)
        if n_cohort == 0:
            continue
        means = [cohort_df[c].mean() for c in y_cols]
        sems = [cohort_df[c].std() / np.sqrt(n_cohort) for c in y_cols]
        color = cohort_colors[str(cohort)]
        ax.plot(time_points, means, color=color, linewidth=2,
                label=f'{cohort} (N={n_cohort})')
        ax.fill_between(time_points,
                         [m - s for m, s in zip(means, sems)],
                         [m + s for m, s in zip(means, sems)],
                         alpha=0.2, color=color)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Minutes Post-Meal', fontsize=11)
    ax.set_ylabel('Delta Glucose (mg/dL)', fontsize=11)
    ax.set_title('Mean Glucose Trajectory (Zero-Bolus Meals)',
                  fontweight='bold', loc='left', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # ------------------------------------------------------------------
    # Panel D: Carbs vs final delta glucose for zero-bolus, by cohort
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if treatment_col:
        for cohort in cohorts:
            cohort_df = zero_bolus[zero_bolus[cohort_col] == cohort]
            valid = cohort_df[treatment_col].notna() & cohort_df[final_y_col].notna()
            carbs = cohort_df.loc[valid, treatment_col]
            y_val = cohort_df.loc[valid, final_y_col]
            color = cohort_colors[str(cohort)]
            ax.scatter(carbs, y_val, alpha=0.5, s=25, color=color,
                       label=f'{cohort} (N={len(carbs)})')

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Carbohydrate Intake (g)', fontsize=11)
        ax.set_ylabel(f'Delta Glucose at {final_minutes} min (mg/dL)', fontsize=11)
        ax.set_title('Carbs vs Delta Glucose (Zero-Bolus)',
                      fontweight='bold', loc='left', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    fig.text(0.5, -0.02, f'Truly zero-bolus meals (N={n_zero}, {filter_desc}) | {build_dataset_label(df)}',
             ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig15_zero_bolus_glucose.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig15_zero_bolus_glucose.png")
    plt.close()


def _plot_trajectory_by_meal_type(df, output_dir, fig_num, filename, title_suffix,
                                  split_filter=None, ylim=None):
    """
    Shared implementation for trajectory-by-meal-type figures.

    Parameters
    ----------
    df : DataFrame
        Full dataset (will be filtered internally if split_filter is set).
    output_dir : Path
        Where to save the PNG.
    fig_num : int
        Figure number for log messages.
    filename : str
        Output filename (e.g. 'fig16_trajectory_by_meal_type.png').
    title_suffix : str
        Text appended to the suptitle (e.g. '' or ' (Test Set)').
    split_filter : str or None
        If set, restrict data to rows where the split column matches this
        value (case-insensitive).  None means use all data.
    ylim : tuple of (float, float) or None
        If provided, force this (ymin, ymax) on all panels.  When None the
        limits are computed from the plotted data so that both panels share
        the same y-axis range.
    """
    cohort_col = get_column(df, 'cohort')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or cohort_col not in df.columns:
        print(f"No cohort column found - skipping fig{fig_num}")
        return None
    if not meal_type_col or meal_type_col not in df.columns:
        print(f"No meal_type column found - skipping fig{fig_num}")
        return None

    # Optional split filtering
    if split_filter is not None:
        split_col = get_column(df, 'split')
        if not split_col or split_col not in df.columns:
            print(f"No split column found - skipping fig{fig_num}")
            return None
        # Case-insensitive match
        df = df[df[split_col].astype(str).str.lower() == split_filter.lower()]
        if len(df) == 0:
            print(f"No rows for split='{split_filter}' - skipping fig{fig_num}")
            return None

    # Discover Y timepoint columns (e.g. Y_60min, Y_90min, ... Y_210min)
    y_cols = sorted([c for c in df.columns if c.startswith('Y_') and c.endswith('min')],
                    key=lambda c: int(c.split('_')[1].replace('min', '')))
    if not y_cols:
        print(f"No Y_*min outcome columns found - skipping fig{fig_num}")
        return None

    time_points = [int(c.split('_')[1].replace('min', '')) for c in y_cols]
    cohorts = sorted(df[cohort_col].dropna().unique())
    meal_types = get_sorted_meal_types(df, meal_type_col)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, cohort in enumerate(cohorts):
        ax = axes[panel_idx]
        cohort_df = df[df[cohort_col] == cohort]

        for meal in meal_types:
            meal_df = cohort_df[cohort_df[meal_type_col] == meal]
            n_meal = len(meal_df)
            if n_meal == 0:
                continue

            means = [meal_df[c].mean() for c in y_cols]
            sems = [meal_df[c].std() / np.sqrt(n_meal) for c in y_cols]
            color = get_meal_color(meal)

            ax.plot(time_points, means, color=color, linewidth=2,
                    label=f'{str(meal).capitalize()} (N={n_meal})')
            ax.fill_between(time_points,
                            [m - s for m, s in zip(means, sems)],
                            [m + s for m, s in zip(means, sems)],
                            alpha=0.2, color=color)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Minutes Post-Meal', fontsize=11)
        ax.set_ylabel('Delta Glucose (mg/dL)', fontsize=11)
        ax.set_title(f'Cohort {cohort}', fontweight='bold', loc='left', fontsize=12)
        ax.legend(fontsize=9, frameon=True, fancybox=True,
                  edgecolor='black', facecolor='white', framealpha=1.0,
                  loc='lower left')
        ax.grid(True, alpha=0.3)

    # Synchronise y-axis across panels
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    else:
        all_ymin = min(ax.get_ylim()[0] for ax in axes)
        all_ymax = max(ax.get_ylim()[1] for ax in axes)
        ylim = (all_ymin, all_ymax)
        for ax in axes:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300,
                bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}")
    plt.close()
    return ylim


def plot_fig16_trajectory_by_meal_type(df, output_dir, ylim=None):
    """Figure 16: Mean postprandial delta glucose trajectory by meal type (all data)."""
    return _plot_trajectory_by_meal_type(
        df, output_dir, fig_num=16,
        filename='fig16_trajectory_by_meal_type.png',
        title_suffix='', ylim=ylim)


def plot_fig17_trajectory_by_meal_type_test(df, output_dir, ylim=None):
    """Figure 17: Mean postprandial delta glucose trajectory by meal type (test set only)."""
    return _plot_trajectory_by_meal_type(
        df, output_dir, fig_num=17,
        filename='fig17_trajectory_by_meal_type_test.png',
        title_suffix=' (Test Set)',
        split_filter='test', ylim=ylim)


def plot_fig18_trajectory_by_meal_type_train(df, output_dir, ylim=None):
    """Figure 18: Mean postprandial delta glucose trajectory by meal type (training set only)."""
    return _plot_trajectory_by_meal_type(
        df, output_dir, fig_num=18,
        filename='fig18_trajectory_by_meal_type_train.png',
        title_suffix=' (Training Set)',
        split_filter='train', ylim=ylim)


def plot_fig19_trajectory_train_vs_test(df, output_dir, ylim=None):
    """
    Figure 19: Mean postprandial delta glucose trajectory by meal type,
    with panels split by train/test (pooling both cohorts).

    Panels:
    a. Training set – mean ± SE trajectory per meal type (2018 + 2020)
    b. Test set     – mean ± SE trajectory per meal type (2018 + 2020)

    Parameters
    ----------
    ylim : tuple of (float, float) or None
        If provided, force this (ymin, ymax) on all panels.  When None the
        limits are computed from the plotted data so that both panels share
        the same y-axis range.
    """
    meal_type_col = get_column(df, 'meal_type')
    split_col = get_column(df, 'split')

    if not meal_type_col or meal_type_col not in df.columns:
        print("No meal_type column found - skipping fig19")
        return None
    if not split_col or split_col not in df.columns:
        print("No split column found - skipping fig19")
        return None

    # Discover Y timepoint columns
    y_cols = sorted([c for c in df.columns if c.startswith('Y_') and c.endswith('min')],
                    key=lambda c: int(c.split('_')[1].replace('min', '')))
    if not y_cols:
        print("No Y_*min outcome columns found - skipping fig19")
        return None

    time_points = [int(c.split('_')[1].replace('min', '')) for c in y_cols]
    meal_types = get_sorted_meal_types(df, meal_type_col)

    # Normalise split values for matching
    df = df.copy()
    df['_split_lower'] = df[split_col].astype(str).str.lower()

    split_order = [('train', 'Training Set'), ('test', 'Test Set')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for panel_idx, (split_val, panel_title) in enumerate(split_order):
        ax = axes[panel_idx]
        split_df = df[df['_split_lower'] == split_val]

        for meal in meal_types:
            meal_df = split_df[split_df[meal_type_col] == meal]
            n_meal = len(meal_df)
            if n_meal == 0:
                continue

            means = [meal_df[c].mean() for c in y_cols]
            sems = [meal_df[c].std() / np.sqrt(n_meal) for c in y_cols]
            color = get_meal_color(meal)

            ax.plot(time_points, means, color=color, linewidth=2,
                    label=f'{str(meal).capitalize()} (N={n_meal})')
            ax.fill_between(time_points,
                            [m - s for m, s in zip(means, sems)],
                            [m + s for m, s in zip(means, sems)],
                            alpha=0.2, color=color)

        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Minutes Post-Meal', fontsize=11)
        ax.set_ylabel('Delta Glucose (mg/dL)', fontsize=11)
        ax.set_title(panel_title, fontweight='bold', loc='left', fontsize=12)
        ax.legend(fontsize=9, frameon=True, fancybox=True,
                  edgecolor='black', facecolor='white', framealpha=1.0,
                  loc='lower left')
        ax.grid(True, alpha=0.3)

    # Synchronise y-axis across panels
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    else:
        all_ymin = min(ax.get_ylim()[0] for ax in axes)
        all_ymax = max(ax.get_ylim()[1] for ax in axes)
        ylim = (all_ymin, all_ymax)
        for ax in axes:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig19_trajectory_train_vs_test.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    print("Saved: fig19_trajectory_train_vs_test.png")
    plt.close()
    return ylim


def generate_cohort_characteristics_table(df, tables_dir):
    """
    Generate LaTeX table for cohort characteristics.
    Matches the format from the user's screenshot.
    """
    cohort_col = get_column(df, 'cohort')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or cohort_col not in df.columns:
        print("No cohort column found - skipping cohort characteristics table")
        return

    cohorts = sorted(df[cohort_col].unique())
    if len(cohorts) != 2:
        print(f"Expected 2 cohorts, found {len(cohorts)} - skipping table")
        return

    c1, c2 = cohorts[0], cohorts[1]
    df1 = df[df[cohort_col] == c1]
    df2 = df[df[cohort_col] == c2]

    rows = []

    # Number of meal events
    rows.append(f"Number of meal events & {len(df1)} & {len(df2)} & -- \\\\")

    # Number of subjects
    n_subj1 = df1['subject_id'].nunique() if 'subject_id' in df1.columns else 'NA'
    n_subj2 = df2['subject_id'].nunique() if 'subject_id' in df2.columns else 'NA'
    rows.append(f"Number of subjects & {n_subj1} & {n_subj2} & -- \\\\")

    # Carbohydrate intake
    if treatment_col:
        carbs1 = df1[treatment_col].dropna()
        carbs2 = df2[treatment_col].dropna()
        stat, p_val = stats.mannwhitneyu(carbs1, carbs2, alternative='two-sided')
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
        rows.append(f"Carbohydrate intake (g) & {carbs1.mean():.1f} +/- {carbs1.std():.1f} & {carbs2.mean():.1f} +/- {carbs2.std():.1f} & {p_str} \\\\")

    # Insulin bolus
    if mediator_col:
        bolus1 = df1[mediator_col].dropna()
        bolus2 = df2[mediator_col].dropna()
        stat, p_val = stats.mannwhitneyu(bolus1, bolus2, alternative='two-sided')
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
        rows.append(f"Insulin bolus (units) & {bolus1.mean():.1f} +/- {bolus1.std():.1f} & {bolus2.mean():.1f} +/- {bolus2.std():.1f} & {p_str} \\\\")

    # Pre-meal glucose
    if glucose_col:
        gluc1 = df1[glucose_col].dropna()
        gluc2 = df2[glucose_col].dropna()
        stat, p_val = stats.mannwhitneyu(gluc1, gluc2, alternative='two-sided')
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
        rows.append(f"Pre-meal glucose (mg/dL) & {gluc1.mean():.1f} +/- {gluc1.std():.1f} & {gluc2.mean():.1f} +/- {gluc2.std():.1f} & {p_str} \\\\")

    # Meal type percentages
    if meal_type_col:
        meal_types = get_sorted_meal_types(df, meal_type_col)
        for meal in meal_types:
            pct1 = 100 * (df1[meal_type_col] == meal).mean()
            pct2 = 100 * (df2[meal_type_col] == meal).mean()
            rows.append(f"{str(meal).capitalize()} (\\%) & {pct1:.1f} & {pct2:.1f} & -- \\\\")

        # Chi-square test for meal distribution
        contingency = pd.crosstab(df[cohort_col], df[meal_type_col])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
        rows.append(f"Meal type distribution (chi-sq test) & -- & -- & {p_str} \\\\")

    latex_content = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Cohort characteristics comparison.}
Summary statistics for the two data collection cohorts (""" + str(c1) + r""" and """ + str(c2) + r""").
Continuous variables are presented as mean $\pm$ standard deviation.
Categorical variables (meal types) are presented as percentages.
P-values are from Mann-Whitney U test (continuous) or chi-squared test (categorical).}
\label{tab:cohort_characteristics}
\resizebox{0.8\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
Variable & """ + str(c1) + r""" & """ + str(c2) + r""" & p-value \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    latex_path = Path(tables_dir) / 'cohort_characteristics.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"Saved: {latex_path}")


def plot_fig13_train_test_comparison(df, output_dir):
    """
    Figure 13: Train/Test split comparison with 4 panels.
    Demonstrates that training and test sets have similar distributions.

    A. Treatment Distribution (density by split)
    B. Mediator Distribution (density by split)
    C. Baseline Glucose Distribution (density by split)
    D. Meal Type Distribution (bar chart by split)

    Uses colorblind-friendly palette.
    """
    from scipy.stats import gaussian_kde

    split_col = get_column(df, 'split')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not split_col or split_col not in df.columns:
        print("No split column found - skipping train/test comparison figure")
        return None

    splits = sorted(df[split_col].unique())
    if len(splits) < 2:
        print(f"Need at least 2 splits for comparison, found {len(splits)} - skipping")
        return None

    # Colorblind-friendly split colors
    split_colors = {
        'train': COLORS['train'],
        'test': COLORS['test'],
        'training': COLORS['train'],
        'validation': COLORS['outcome']  # Orange/amber for validation
    }
    # Handle any other split names
    for i, s in enumerate(splits):
        if str(s).lower() not in split_colors:
            split_colors[str(s).lower()] = COLORS['train'] if i == 0 else COLORS['test']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def plot_density_panel(ax, col, xlabel, title, panel_letter, min_val=None):
        """Helper to plot density with proper bounds."""
        if not col or col not in df.columns:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold', loc='left', fontsize=12)
            ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
                    fontsize=18, fontweight='bold', va='top', ha='left',
                    fontfamily='sans-serif')
            return

        for split in splits:
            split_data = df[df[split_col] == split][col].dropna()
            if len(split_data) < 2:
                continue

            # Calculate KDE
            kde = gaussian_kde(split_data)

            # Set x range with proper lower bound
            x_min = min_val if min_val is not None else split_data.min()
            x_max = split_data.max() * 1.05
            x_range = np.linspace(x_min, x_max, 300)

            # Get density values
            density = kde(x_range)

            color = split_colors.get(str(split).lower(), COLORS['secondary'])
            label = str(split).capitalize()
            ax.plot(x_range, density, color=color, linewidth=2, label=label)
            ax.fill_between(x_range, density, alpha=0.3, color=color)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Split', loc='upper right')
        ax.set_xlim(left=min_val if min_val is not None else None)
        ax.text(-0.12, 1.08, panel_letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='left',
                fontfamily='sans-serif')

    # Panel A: Treatment Distribution (carbs >= 0)
    plot_density_panel(axes[0, 0], treatment_col, 'Carbohydrate Intake (g)',
                       'Treatment Distribution', 'a', min_val=0)

    # Panel B: Mediator Distribution (bolus >= 0)
    plot_density_panel(axes[0, 1], mediator_col, 'Insulin Bolus (units)',
                       'Mediator Distribution', 'b', min_val=0)

    # Panel C: Baseline Glucose Distribution
    ax = axes[1, 0]
    if glucose_col and glucose_col in df.columns:
        # Validate glucose is actual values, not delta
        is_valid = validate_glucose_column(df, glucose_col)

        for split in splits:
            split_data = df[df[split_col] == split][glucose_col].dropna()
            if len(split_data) < 2:
                continue
            kde = gaussian_kde(split_data)
            x_range = np.linspace(max(0, split_data.min() - 20), split_data.max() + 20, 300)
            density = kde(x_range)
            color = split_colors.get(str(split).lower(), COLORS['secondary'])
            ax.plot(x_range, density, color=color, linewidth=2, label=str(split).capitalize())
            ax.fill_between(x_range, density, alpha=0.3, color=color)

        # Add reference lines for normal glucose range (70-180 mg/dL) only if valid glucose
        if is_valid:
            ax.axvline(x=70, color=COLORS['highlight'], linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=180, color=COLORS['highlight'], linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_xlabel('Pre-meal Glucose (mg/dL)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Baseline Glucose Distribution', fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Split', loc='upper right')
        ax.set_xlim(left=0)
    else:
        ax.text(0.5, 0.5, 'Glucose data not available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Baseline Glucose Distribution', fontweight='bold', loc='left', fontsize=12)
    ax.text(-0.12, 1.08, 'c', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    # Panel D: Meal Type Distribution (bar chart by split)
    ax = axes[1, 1]
    if meal_type_col and meal_type_col in df.columns:
        meal_types = get_sorted_meal_types(df, meal_type_col)

        # Calculate proportions for each split
        split_props = {}
        for split in splits:
            split_df = df[df[split_col] == split]
            meal_counts = split_df[meal_type_col].value_counts()
            total = len(split_df)
            split_props[split] = {m: meal_counts.get(m, 0) / total for m in meal_types}

        x = np.arange(len(meal_types))
        width = 0.35

        for i, split in enumerate(splits):
            props = [split_props[split].get(m, 0) for m in meal_types]
            offset = -width/2 + i * width
            color = split_colors.get(str(split).lower(), COLORS['secondary'])
            ax.bar(x + offset, props, width, label=str(split).capitalize(),
                   color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(m).capitalize() for m in meal_types], fontsize=10)
        ax.set_xlabel('Meal Type', fontsize=11)
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title('Meal Type Distribution', fontweight='bold', loc='left', fontsize=12)
        ax.legend(title='Split', loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Meal type data not available', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Meal Type Distribution', fontweight='bold', loc='left', fontsize=12)
    ax.text(-0.12, 1.08, 'd', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')

    fig.text(0.5, -0.02, build_dataset_label(df), ha='center', fontsize=9, style='italic', color='gray')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig13_train_test_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig13_train_test_comparison.png")
    plt.close()


def generate_train_test_characteristics_table(df, tables_dir):
    """
    Generate LaTeX table for train/test split characteristics.
    Shows that training and test sets are comparable.
    """
    split_col = get_column(df, 'split')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not split_col or split_col not in df.columns:
        print("No split column found - skipping train/test characteristics table")
        return

    splits = sorted(df[split_col].unique())
    if len(splits) < 2:
        print(f"Need at least 2 splits, found {len(splits)} - skipping table")
        return

    # Use first two splits
    s1, s2 = splits[0], splits[1]
    df1 = df[df[split_col] == s1]
    df2 = df[df[split_col] == s2]

    rows = []

    # Number of meal events
    rows.append(f"Number of meal events & {len(df1)} & {len(df2)} & -- \\\\")

    # Number of subjects
    patient_col = get_column(df, 'patient')
    if patient_col:
        n_subj1 = df1[patient_col].nunique()
        n_subj2 = df2[patient_col].nunique()
        rows.append(f"Number of subjects & {n_subj1} & {n_subj2} & -- \\\\")

    # Carbohydrate intake
    if treatment_col and treatment_col in df.columns:
        carbs1 = df1[treatment_col].dropna()
        carbs2 = df2[treatment_col].dropna()
        if len(carbs1) > 0 and len(carbs2) > 0:
            stat, p_val = stats.mannwhitneyu(carbs1, carbs2, alternative='two-sided')
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
            rows.append(f"Carbohydrate intake (g) & {carbs1.mean():.1f} $\\pm$ {carbs1.std():.1f} & {carbs2.mean():.1f} $\\pm$ {carbs2.std():.1f} & {p_str} \\\\")

    # Insulin bolus
    if mediator_col and mediator_col in df.columns:
        bolus1 = df1[mediator_col].dropna()
        bolus2 = df2[mediator_col].dropna()
        if len(bolus1) > 0 and len(bolus2) > 0:
            stat, p_val = stats.mannwhitneyu(bolus1, bolus2, alternative='two-sided')
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
            rows.append(f"Insulin bolus (units) & {bolus1.mean():.2f} $\\pm$ {bolus1.std():.2f} & {bolus2.mean():.2f} $\\pm$ {bolus2.std():.2f} & {p_str} \\\\")

    # Pre-meal glucose
    if glucose_col and glucose_col in df.columns:
        gluc1 = df1[glucose_col].dropna()
        gluc2 = df2[glucose_col].dropna()
        if len(gluc1) > 0 and len(gluc2) > 0:
            stat, p_val = stats.mannwhitneyu(gluc1, gluc2, alternative='two-sided')
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
            rows.append(f"Pre-meal glucose (mg/dL) & {gluc1.mean():.1f} $\\pm$ {gluc1.std():.1f} & {gluc2.mean():.1f} $\\pm$ {gluc2.std():.1f} & {p_str} \\\\")

    # Meal type percentages
    if meal_type_col and meal_type_col in df.columns:
        meal_types = get_sorted_meal_types(df, meal_type_col)
        for meal in meal_types:
            pct1 = 100 * (df1[meal_type_col] == meal).mean()
            pct2 = 100 * (df2[meal_type_col] == meal).mean()
            rows.append(f"{str(meal).capitalize()} (\\%) & {pct1:.1f} & {pct2:.1f} & -- \\\\")

        # Chi-square test for meal distribution
        contingency = pd.crosstab(df[split_col], df[meal_type_col])
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
        rows.append(f"Meal type distribution & -- & -- & {p_str} \\\\")

    s1_label = str(s1).capitalize()
    s2_label = str(s2).capitalize()

    latex_content = r"""\begin{table*}[ht]
\centering
\caption{\textbf{Training and test set characteristics.}
Comparison of variable distributions between """ + s1_label + r""" and """ + s2_label + r""" sets.
Continuous variables presented as mean $\pm$ SD.
Categorical variables (meal types) as percentages.
P-values from Mann-Whitney U test (continuous) or chi-squared test (categorical).
Non-significant p-values ($p > 0.05$) indicate comparable distributions.}
\label{tab:train_test_characteristics}
\resizebox{0.85\textwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
Variable & """ + s1_label + r""" & """ + s2_label + r""" & p-value \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}
"""

    latex_path = Path(tables_dir) / 'train_test_characteristics.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"Saved: {latex_path}")

    # Also save CSV version
    csv_rows = []
    for row in rows:
        # Parse LaTeX row into columns
        parts = row.replace('\\\\', '').split('&')
        if len(parts) >= 3:
            csv_rows.append({
                'Variable': parts[0].strip().replace('\\%', '%').replace('$\\pm$', '±'),
                s1_label: parts[1].strip().replace('$\\pm$', '±'),
                s2_label: parts[2].strip().replace('$\\pm$', '±'),
                'p-value': parts[3].strip() if len(parts) > 3 else '--'
            })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = Path(tables_dir) / 'train_test_characteristics.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


# =============================================================================
# NEW PUBLICATION TABLES: Split x Meal Type and Cohort x Meal Type
# =============================================================================

def generate_split_meal_summary_table(df, tables_dir):
    """
    Generate a publication table stratified by train/test split AND meal type.

    For each (split, meal_type) combination, reports:
      - N  (number of meal events)
      - Meal carbs: Mean +/- SD
      - Bolus: Mean +/- SD
      - % zero bolus
      - Pre-meal glucose: Mean +/- SD
    """
    split_col = get_column(df, 'split')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not split_col or split_col not in df.columns:
        print("No split column found - skipping split x meal summary table")
        return
    if not meal_type_col or meal_type_col not in df.columns:
        print("No meal_type column found - skipping split x meal summary table")
        return

    tables_dir = Path(tables_dir)
    splits = sorted(df[split_col].unique())
    meal_types = get_sorted_meal_types(df, meal_type_col)

    # ---- build rows ----
    csv_rows = []
    for split in splits:
        for meal in meal_types:
            subset = df[(df[split_col] == split) & (df[meal_type_col] == meal)]
            n = len(subset)

            carbs = subset[treatment_col].dropna() if treatment_col else pd.Series(dtype=float)
            bolus = subset[mediator_col].dropna() if mediator_col else pd.Series(dtype=float)
            gluc = subset[glucose_col].dropna() if glucose_col else pd.Series(dtype=float)

            csv_rows.append({
                'Split': str(split).capitalize(),
                'Meal Type': str(meal).capitalize(),
                'N': n,
                'Carbs Mean': f"{carbs.mean():.1f}" if len(carbs) > 0 else '--',
                'Carbs SD': f"{carbs.std():.1f}" if len(carbs) > 1 else '--',
                'Bolus Mean': f"{bolus.mean():.2f}" if len(bolus) > 0 else '--',
                'Bolus SD': f"{bolus.std():.2f}" if len(bolus) > 1 else '--',
                '% Zero Bolus': f"{100 * (bolus == 0).mean():.1f}" if len(bolus) > 0 else '--',
                'Glucose Mean': f"{gluc.mean():.1f}" if len(gluc) > 0 else '--',
                'Glucose SD': f"{gluc.std():.1f}" if len(gluc) > 1 else '--',
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = tables_dir / 'split_meal_summary.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ---- LaTeX ----
    n_meals = len(meal_types)
    latex = r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Data characteristics by data split and meal type.}
Summary statistics for carbohydrate intake (treatment), insulin bolus (mediator),
and pre-meal glucose stratified by training/test split and meal category.
N = number of meal observations used in the analysis; SD = standard deviation; values are mean $\pm$ SD;
\% Zero Bolus = percentage of meals with no insulin bolus.}
\label{tab:split_meal_summary}
\begin{tabular}{llccccccc}
\toprule
& & & \multicolumn{2}{c}{Carbs (g)} & \multicolumn{2}{c}{Bolus (U)} & \% Zero & Glucose (mg/dL) \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){9-9}
Split & Meal Type & N & Mean & SD & Mean & SD & Bolus & Mean $\pm$ SD \\
\midrule
"""
    for i, split in enumerate(splits):
        split_label = str(split).capitalize()
        rows_for_split = csv_df[csv_df['Split'] == split_label]
        for j, (_, row) in enumerate(rows_for_split.iterrows()):
            split_cell = f"\\multirow{{{n_meals}}}{{*}}{{{split_label}}}" if j == 0 else ""
            gluc_str = f"{row['Glucose Mean']} $\\pm$ {row['Glucose SD']}" if row['Glucose Mean'] != '--' else '--'
            latex += (f"{split_cell} & {row['Meal Type']} & {row['N']} & "
                      f"{row['Carbs Mean']} & {row['Carbs SD']} & "
                      f"{row['Bolus Mean']} & {row['Bolus SD']} & "
                      f"{row['% Zero Bolus']} & {gluc_str} \\\\\n")
        if i < len(splits) - 1:
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    latex_path = tables_dir / 'split_meal_summary.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {latex_path}")


def generate_cohort_meal_summary_table(df, tables_dir):
    """
    Generate a publication table stratified by cohort (2018/2020) AND meal type.

    For each (cohort, meal_type) combination, reports:
      - N  (number of meal events)
      - Meal carbs: Mean +/- SD
      - Bolus: Mean +/- SD
      - % zero bolus
      - Pre-meal glucose: Mean +/- SD
    """
    cohort_col = get_column(df, 'cohort')
    treatment_col = get_column(df, 'treatment')
    mediator_col = get_column(df, 'mediator')
    glucose_col = get_column(df, 'glucose')
    meal_type_col = get_column(df, 'meal_type')

    if not cohort_col or cohort_col not in df.columns:
        print("No cohort column found - skipping cohort x meal summary table")
        return
    if not meal_type_col or meal_type_col not in df.columns:
        print("No meal_type column found - skipping cohort x meal summary table")
        return

    tables_dir = Path(tables_dir)
    cohorts = sorted(df[cohort_col].unique())
    meal_types = get_sorted_meal_types(df, meal_type_col)

    # ---- build rows ----
    csv_rows = []
    for cohort in cohorts:
        for meal in meal_types:
            subset = df[(df[cohort_col] == cohort) & (df[meal_type_col] == meal)]
            n = len(subset)

            carbs = subset[treatment_col].dropna() if treatment_col else pd.Series(dtype=float)
            bolus = subset[mediator_col].dropna() if mediator_col else pd.Series(dtype=float)
            gluc = subset[glucose_col].dropna() if glucose_col else pd.Series(dtype=float)

            csv_rows.append({
                'Cohort': str(cohort),
                'Meal Type': str(meal).capitalize(),
                'N': n,
                'Carbs Mean': f"{carbs.mean():.1f}" if len(carbs) > 0 else '--',
                'Carbs SD': f"{carbs.std():.1f}" if len(carbs) > 1 else '--',
                'Bolus Mean': f"{bolus.mean():.2f}" if len(bolus) > 0 else '--',
                'Bolus SD': f"{bolus.std():.2f}" if len(bolus) > 1 else '--',
                '% Zero Bolus': f"{100 * (bolus == 0).mean():.1f}" if len(bolus) > 0 else '--',
                'Glucose Mean': f"{gluc.mean():.1f}" if len(gluc) > 0 else '--',
                'Glucose SD': f"{gluc.std():.1f}" if len(gluc) > 1 else '--',
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = tables_dir / 'cohort_meal_summary.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ---- LaTeX ----
    n_meals = len(meal_types)
    latex = r"""\begin{table*}[ht]
\centering
\small
\caption{\textbf{Data characteristics by cohort and meal type.}
Summary statistics for carbohydrate intake (treatment), insulin bolus (mediator),
and pre-meal glucose stratified by data collection cohort and meal category.
N = number of meal observations used in the analysis; SD = standard deviation; values are mean $\pm$ SD;
\% Zero Bolus = percentage of meals with no insulin bolus.}
\label{tab:cohort_meal_summary}
\begin{tabular}{llccccccc}
\toprule
& & & \multicolumn{2}{c}{Carbs (g)} & \multicolumn{2}{c}{Bolus (U)} & \% Zero & Glucose (mg/dL) \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){9-9}
Cohort & Meal Type & N & Mean & SD & Mean & SD & Bolus & Mean $\pm$ SD \\
\midrule
"""
    for i, cohort in enumerate(cohorts):
        cohort_label = str(cohort)
        rows_for_cohort = csv_df[csv_df['Cohort'] == cohort_label]
        for j, (_, row) in enumerate(rows_for_cohort.iterrows()):
            cohort_cell = f"\\multirow{{{n_meals}}}{{*}}{{{cohort_label}}}" if j == 0 else ""
            gluc_str = f"{row['Glucose Mean']} $\\pm$ {row['Glucose SD']}" if row['Glucose Mean'] != '--' else '--'
            latex += (f"{cohort_cell} & {row['Meal Type']} & {row['N']} & "
                      f"{row['Carbs Mean']} & {row['Carbs SD']} & "
                      f"{row['Bolus Mean']} & {row['Bolus SD']} & "
                      f"{row['% Zero Bolus']} & {gluc_str} \\\\\n")
        if i < len(cohorts) - 1:
            latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    latex_path = tables_dir / 'cohort_meal_summary.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {latex_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(embeddings_file=None):
    """Run complete data summary analysis.

    Parameters
    ----------
    embeddings_file : str or Path, optional
        Path to a specific embeddings CSV file. If None, uses the most recent.
    """
    print("\n" + "="*70)
    print("MEAL WINDOW DATA SUMMARY")
    print("="*70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_meal_window_data(embeddings_file=embeddings_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return None

    stats_dict = compute_summary_statistics(df)

    # Print quick summary
    print("\n" + "-"*50)
    print("QUICK SUMMARY")
    print("-"*50)
    if 'sample' in stats_dict:
        print(f"  N events: {stats_dict['sample']['n_events']:,}")
        if not np.isnan(stats_dict['sample'].get('n_patients', np.nan)):
            print(f"  N subjects: {stats_dict['sample']['n_patients']:.0f}")
    if 'treatment' in stats_dict:
        t = stats_dict['treatment']
        print(f"  Treatment (carbs): {t['mean']:.1f} +/- {t['sd']:.1f} g")
    if 'mediator' in stats_dict:
        m = stats_dict['mediator']
        print(f"  Mediator (bolus): {m['mean']:.2f} +/- {m['sd']:.2f} U")

    # Generate tables
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)
    generate_summary_tables(df, stats_dict, TABLES_DIR)

    # New publication summary tables (split x meal, cohort x meal)
    print("\n--- Split x Meal Summary Table ---")
    generate_split_meal_summary_table(df, TABLES_DIR)
    print("\n--- Cohort x Meal Summary Table ---")
    generate_cohort_meal_summary_table(df, TABLES_DIR)

    # Generate figures
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    print("\n--- Treatment Figures ---")
    plot_fig01_treatment_histogram(df, FIGURES_DIR)
    plot_fig02_treatment_by_meal_type(df, FIGURES_DIR)
    plot_fig03_treatment_by_subject(df, FIGURES_DIR)

    print("\n--- Mediator Figures ---")
    plot_fig04_mediator_histogram(df, FIGURES_DIR)
    plot_fig05_mediator_by_meal_type(df, FIGURES_DIR)
    plot_fig06_mediator_by_subject(df, FIGURES_DIR)

    print("\n--- Relationship Figures ---")
    plot_fig07_treatment_mediator_scatter(df, FIGURES_DIR)
    plot_fig08_treatment_mediator_by_meal(df, FIGURES_DIR)

    print("\n--- Distribution Figures ---")
    plot_fig09_meal_type_distribution(df, FIGURES_DIR)
    plot_fig10_events_per_subject(df, FIGURES_DIR)
    plot_fig11_subject_meal_heatmap(df, FIGURES_DIR)

    print("\n--- Cohort Comparison Figures ---")
    plot_fig12_cohort_comparison(df, FIGURES_DIR)
    plot_fig14_cohort_insulin_carbs(df, FIGURES_DIR)
    plot_fig15_zero_bolus_glucose(df, FIGURES_DIR)
    # Two-pass generation so all trajectory panels share exactly the same y-axis.
    # Pass 1: generate each figure to discover its natural y-axis range.
    ylim16 = plot_fig16_trajectory_by_meal_type(df, FIGURES_DIR)
    ylim19 = plot_fig19_trajectory_train_vs_test(df, FIGURES_DIR)
    # Compute the union of both ranges and regenerate.
    if ylim16 and ylim19:
        shared_ylim = (min(ylim16[0], ylim19[0]), max(ylim16[1], ylim19[1]))
        plot_fig16_trajectory_by_meal_type(df, FIGURES_DIR, ylim=shared_ylim)
        plot_fig19_trajectory_train_vs_test(df, FIGURES_DIR, ylim=shared_ylim)
    plot_fig17_trajectory_by_meal_type_test(df, FIGURES_DIR,
                                            ylim=shared_ylim if ylim16 and ylim19 else None)
    plot_fig18_trajectory_by_meal_type_train(df, FIGURES_DIR,
                                             ylim=shared_ylim if ylim16 and ylim19 else None)
    generate_cohort_characteristics_table(df, TABLES_DIR)

    print("\n--- Train/Test Comparison Figures ---")
    plot_fig13_train_test_comparison(df, FIGURES_DIR)
    generate_train_test_characteristics_table(df, TABLES_DIR)

    # NOTE: φ distribution figures are in generate_embedding_diagnostics.py

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {FIGURES_DIR.absolute()}")
    print(f"Tables saved to: {TABLES_DIR.absolute()}")

    print("\nGenerated figures:")
    for f in sorted(FIGURES_DIR.glob('fig*.png')):
        print(f"  - {f.name}")

    print("\nGenerated tables:")
    for f in sorted(TABLES_DIR.glob('table*.csv')):
        print(f"  - {f.name}")

    return df, stats_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate summary statistics and visualizations for meal window data."
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Path to a specific phi_embeddings_combined_*.csv file. "
             "If not specified, uses the most recent file by modification time."
    )
    args = parser.parse_args()

    df, stats = main(embeddings_file=args.embeddings_file)