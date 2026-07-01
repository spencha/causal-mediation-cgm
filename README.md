# Causal Mediation Analysis of Postprandial Glucose Response Using Autoencoder Embeddings

> Code for the paper: **"Causal Mediation Pathways in Continuous Postprandial Glucose Monitoring for Type 1 Diabetes Patients"**

A framework for causal mediation analysis (CMA) of continuous glucose monitoring (CGM) data from the OhioT1DM dataset. The pipeline learns low-dimensional representations of pre-meal CGM trajectories using causal-constrained autoencoders, then uses these representations as confounders in a mediation analysis estimating how meal carbohydrates affect postprandial glucose response through insulin bolus decisions.

## Overview

In type 1 diabetes, meal carbohydrate intake influences postprandial glucose both directly (via digestion) and indirectly through the insulin bolus a patient delivers. Estimating these causal pathways requires controlling for pre-meal physiological state, which is high-dimensional (multi-channel CGM time series). This project addresses that challenge by:

1. **Learning low-dimensional representations (phi)** of pre-meal CGM trajectories using causal linear-friendly autoencoders (CLAE) with penalties that encourage linearity, conditional independence, and balanceability
2. **Estimating balancing weights** via non-parametric covariate balancing propensity scores (npCBPS) to satisfy the sequential ignorability assumptions required for mediation analysis
3. **Running causal mediation analysis** using mixed-effects models (lmer) and quantile regression (QR) across multiple post-meal time horizons, meal types, and treatment offsets

### Causal Framework

| Role | Variable | Description |
|------|----------|-------------|
| Treatment (A) | Meal carbohydrate content | Grams of carbs consumed |
| Mediator (M) | Insulin bolus for meal | Total bolus (pre-meal + 60 min post-meal) |
| Outcome (Y) | Delta glucose at time t | Change in glucose from meal time to t minutes post-meal |
| Confounders (phi) | Learned CGM embeddings | Low-dimensional summary of pre-meal glucose, steps, basal insulin, heart rate, and meal history |

The pipeline estimates:
- **ACME** (Average Causal Mediation Effect): the indirect effect of carbs on glucose *through* insulin bolus
- **ADE** (Average Direct Effect): the direct effect of carbs on glucose *not through* bolus
- **ATE** (Average Total Effect): ACME + ADE

## Repository Structure

```
causal-mediation-cgm/
├── ae_python_code/                        # Autoencoder training & embedding export
│   ├── config.py                          # Centralized Python path configuration
│   ├── causal_linear_ae.py                # CLAE architecture (CNN & LSTM encoders)
│   ├── resid_ae_utils.py                  # Data loading, windowing, residualization
│   ├── train_and_export_embeddings.py     # Train best config & export phi embeddings
│   ├── train_horizon_specific_embeddings.py  # Train per-horizon (30-180 min) models
│   ├── glycemic_event_prediction.py       # Evaluate phi on clinical prediction tasks
│   ├── experiments/                       # Model selection experiments
│   │   ├── run_comprehensive_ae_comparison.py   # Architecture × optimizer × penalty sweep
│   │   ├── run_incremental_data_experiment.py   # Penalization ablation on combined data
│   │   ├── ablation_penalization_layers.py      # Fine-grained penalty layer ablation
│   │   ├── compare_optimizers.py                # Optimizer comparison
│   │   ├── incremental_penalization_study.py    # Incremental penalization sweep
│   │   ├── visualize_architecture_comparison.py # Experiment visualization
│   │   ├── legacy_covproj_encoder.py            # Legacy: covariance-projection encoder
│   │   └── legacy_single_dataset_training.py    # Legacy: single-dataset training
│   ├── meal_windows_2018/                 # 2018 cohort meal window CSVs (not tracked)
│   └── meal_windows_combined/             # Combined 2018+2020 train/test (not tracked)
│
├── cma_cluster/                           # Causal mediation analysis code (R) — code only
│   ├── config.R                           # Centralized R path configuration
│   ├── npcbps_weights.R                   # npCBPS weight estimation
│   ├── run_mixed_effects_mediation.R      # Mixed-effects mediation (lmer + QR)
│   └── run_all_timepoints_lmer.R          # Wrapper: run CMA across all timepoints
│
├── analysis_data/                         # Phi embeddings, weights, RData, diagnostics (not tracked)
├── mediation_results/                     # HPC mediation output CSVs; */figures/ tables+figures tracked
│
├── data_processing/                       # Raw data preprocessing (R)
│   ├── data_pre_processing_2018.Rmd       # R Markdown: 2018 XML preprocessing
│   ├── data_pre_processing_2020.Rmd       # R Markdown: 2020 XML preprocessing
│   ├── z_meal_mediation_analysis_data_2018_5min.R  # Create 2018 meal windows
│   ├── z_meal_mediation_analysis_data_2020_5min.R  # Create 2020 meal windows
│   ├── combine_2018_2020_datasets.R       # Merge cohorts into combined dataset
│   ├── export_meal_windows_for_autoencoder.R  # Export CSVs for Python AE training
│   ├── realdata-preprocess-functions.R    # OhioT1DM XML parsing helpers
│   └── mediation_analysis_preprocessing_functions.R  # Meal window extraction helpers
│
├── visualization_code/                    # All visualization & figure generation scripts
│   ├── compose_paper_visualizations.py    # Orchestration: assemble paper figures & tables
│   ├── summarize_meal_windows.py          # 19 figures: distributions, cohort comparison,
│   │                                      #   train/test comparison, trajectories by meal type
│   ├── bolus_timing_distribution.R        # Bolus timing relative to meal onset
│   ├── generate_embedding_diagnostics.py  # Phi distributions, correlations, PCA, t-SNE
│   ├── generate_balance_diagnostics.py    # Balance checks, overlap, weight quality
│   ├── generate_mediation_outputs.py      # Publication figures & tables (per meal type,
│   │                                      #   covariate mode, model type, treatment offset)
│   └── generate_architecture_comparisons.py  # Architecture & penalization comparison figures
│
├── visualizations/                        # All generated outputs (figures, tables, data)
│   ├── paper_visualizations/              # Paper-ready figures & tables (generated)
│   │   ├── figures/
│   │   └── tables/
│   ├── data_distribution/                 # Generated PNGs & LaTeX/CSV tables
│   ├── ae_embeddings/                     # Generated figures & tables
│   ├── npcbps_balance/                    # Generated figures & tables
│   ├── mediation_visualizations/          # Generated figures & tables
│   └── incremental_data_experiment/       # Generated figures, tables, & experiment CSVs
│
├── OhioT1DM/                             # Raw OhioT1DM data processing
│   └── process_source.R                   # XML to R data conversion
│
├── config_local.yaml.template             # Template for local Python config
├── config_local.R.template                # Template for local R config
├── requirements.txt                       # Python dependencies
├── install_r_packages.R                   # R dependency installer
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone and configure

```bash
git clone <repository-url>
cd causal-mediation-cgm

# Configure local paths (git-ignored)
cp config_local.yaml.template config_local.yaml   # Python
cp config_local.R.template config_local.R          # R
# Edit both files to set your local base directory
```

### 2. Install dependencies

```bash
# Python (3.9+)
pip install -r requirements.txt

# R (4.2+)
Rscript install_r_packages.R
```

### 3. Verify setup

```bash
python ae_python_code/config.py
Rscript -e "source('cma_cluster/config.R'); print_config(CONFIG)"
```

## Pipeline

The analysis runs in six stages. Each stage depends on the output of the previous one.

### Stage 0: Preprocess Raw Data

Convert OhioT1DM XML files into meal windows suitable for analysis.

```bash
# Preprocess raw OhioT1DM XML data into 5-minute interval RData files
# (run from the data_processing/ohiot1dm/ directory)
cd data_processing/ohiot1dm
Rscript -e "rmarkdown::render('data_pre_processing_2018.Rmd')"
Rscript -e "rmarkdown::render('data_pre_processing_2020.Rmd')"
cd ../..

# Create meal windows for each cohort (produces FULL, TRAIN, TEST splits)
Rscript data_processing/ohiot1dm/z_meal_mediation_analysis_data_2018_5min.R
Rscript data_processing/ohiot1dm/z_meal_mediation_analysis_data_2020_5min.R

# Combine cohorts and export CSVs for Python
Rscript data_processing/ohiot1dm/combine_2018_2020_datasets.R
Rscript data_processing/ohiot1dm/export_meal_windows_for_autoencoder.R
```

**Output:** `ae_python_code/meal_windows_combined/{train,test}/` containing per-window CSVs with columns for glucose, steps, basal, meal, heart rate, and bolus at 5-minute resolution.

### Stage 1: Model Selection (Experiments)

Select the best autoencoder architecture, optimizer, and penalization configuration. All experiments train on the combined 2018+2020 training set and evaluate on the held-out test set.

```bash
# Architecture x optimizer x penalty sweep
python ae_python_code/experiments/run_comprehensive_ae_comparison.py

# Penalization ablation study
python ae_python_code/experiments/run_incremental_data_experiment.py

# Visualize experiment results
python visualization_code/generate_architecture_comparisons.py
```

### Stage 2: Train Autoencoder & Export Embeddings

Train the selected configuration and export phi embeddings for downstream CMA.

```bash
# Train and export phi embeddings
python ae_python_code/train_and_export_embeddings.py \
    --arch cnn --penalty lin_bal --seed 42

# Optional: train horizon-specific models (30-180 min)
python ae_python_code/train_horizon_specific_embeddings.py

# Optional: evaluate phi on glycemic event prediction
python ae_python_code/glycemic_event_prediction.py
```

**Output:** `analysis_data/phi_embeddings_{train,test}.csv` with phi features, treatment, mediator, outcomes, and metadata.

### Stage 3: Estimate Balancing Weights

Compute npCBPS weights that balance phi features across treatment levels.

```bash
Rscript cma_cluster/ohiot1dm/npcbps_weights.R
```

**Output:** `analysis_data/weights/` containing treatment and mediator balancing weights.

### Stage 4: Run Causal Mediation Analysis

Run mediation analysis across all combinations of time points, meal types, model specifications, and treatment offsets.

```bash
# Run all timepoints with mixed-effects model
Rscript cma_cluster/ohiot1dm/run_all_timepoints_lmer.R

# Run specific configuration
Rscript cma_cluster/ohiot1dm/run_mixed_effects_mediation.R \
    --timepoint 90 \
    --model lmer \
    --meal ALL \
    --offset 30 \
    --bootstrap 1000 \
    --covariate-mode phi

# Run all models, quantiles, and offsets
Rscript cma_cluster/ohiot1dm/run_all_timepoints_lmer.R --run-all
```

### Stage 5: Generate Visualizations

```bash
# Data distribution figures (figs 1-19)
python visualization_code/summarize_meal_windows.py

# Autoencoder embedding diagnostics
python visualization_code/generate_embedding_diagnostics.py

# npCBPS covariate balance verification
python visualization_code/generate_balance_diagnostics.py

# Mediation results (ACME, ADE, ATE figures and tables)
python visualization_code/generate_mediation_outputs.py
```

## Visualization Inventory

### Data Distribution (`visualization_code/summarize_meal_windows.py`)

| Figure | Description |
|--------|-------------|
| fig1-4 | Carbohydrate, bolus, and glucose distributions (histograms, boxplots) |
| fig5-6 | Pre-meal glucose trajectory heatmaps |
| fig7-9 | Bivariate relationships (carbs vs bolus, carbs vs glucose, bolus vs glucose) |
| fig10 | Events per subject |
| fig11 | Subject x meal type heatmap |
| fig12 | Cohort comparison (2018 vs 2020) |
| fig13 | Train/test split comparison |
| fig14 | Cohort insulin and carb distributions |
| fig15 | Zero-bolus meal glucose trajectories |
| fig16 | Mean delta glucose trajectory by meal type (all data, per cohort) |
| fig17 | Mean delta glucose trajectory by meal type (test set only, per cohort) |
| fig18 | Mean delta glucose trajectory by meal type (training set only, per cohort) |
| fig19 | Mean delta glucose trajectory by meal type (train vs test panels, pooled cohorts) |

### Autoencoder Embeddings (`visualization_code/generate_embedding_diagnostics.py`)
- Phi feature distributions, correlation matrices, PCA/t-SNE projections
- Feature importance and outcome predictability diagnostics

### npCBPS Balance (`visualization_code/generate_balance_diagnostics.py`)
- Standardized mean differences before/after weighting
- Propensity score overlap (positivity check)
- Effective sample size and weight distribution

### Mediation Results (`visualization_code/generate_mediation_outputs.py`)
- ACME, ADE, ATE trajectory plots over post-meal time
- Per-meal-type comparison tables (Breakfast, Lunch, Dinner, Snack)
- Covariate-mode comparison (phi vs PCA)
- Multi-model results (lmer, quantile regression at various quantiles)
- Individual panel figures for journal submission

## Configuration

The project uses a three-tier configuration system (in order of precedence):

1. **Environment variables** (for HPC/cluster use)
2. **Local config files** (`config_local.yaml` / `config_local.R`, git-ignored)
3. **Default paths** relative to the project root

| Environment Variable | Description |
|---------------------|-------------|
| `CAUSAL_AE_BASE_DIR` | Project root directory |
| `CAUSAL_AE_DATA_DIR` | Analysis data directory (embeddings, weights) |
| `CAUSAL_AE_MEDIATION_RESULTS_DIR` | Mediation output directory |
| `CAUSAL_AE_VISUALIZATIONS_DIR` | Visualizations root directory |
| `CAUSAL_AE_RAW_DATA_DIR` | Raw OhioT1DM data directory |

## Requirements

### Python (>= 3.9)

| Package | Purpose |
|---------|---------|
| tensorflow >= 2.10 | Autoencoder training |
| numpy, pandas, scipy | Core scientific computing |
| scikit-learn | PCA, regression, evaluation metrics |
| matplotlib, seaborn | Visualization |
| statsmodels | Statistical tests |
| pyyaml | Configuration file parsing |
| tqdm | Progress bars |

### R (>= 4.2)

| Package | Purpose |
|---------|---------|
| mediation | Causal mediation analysis |
| CBPS | Covariate balancing propensity scores |
| lme4 | Mixed-effects models (lmer) |
| quantreg | Quantile regression |
| mgcv | Generalized additive models |
| dplyr, tidyr, readr, purrr | Data manipulation |
| ggplot2 | Visualization |
| optparse | Command-line argument parsing |

## Data

Data files are not included in this repository due to size and privacy constraints. The project uses the [OhioT1DM dataset](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) (2018 and 2020 cohorts).

**Train/test split strategy:** The original OhioT1DM study-day-based train/test split is used for both the 2018 and 2020 cohorts. Each subject's recording period is divided into training days and testing days as defined by the dataset. The combined training set (2018 TRAIN + 2020 TRAIN) is used for autoencoder training, while the combined test set is used for causal mediation analysis.

Expected data locations:
- `ae_python_code/meal_windows_combined/{train,test}/` -- per-window CSVs for AE training
- `analysis_data/` -- phi embeddings, RData files, balancing weights

## Methods

### Causal Linear-Friendly Autoencoder (CLAE)

The autoencoder learns representations phi that are optimized for use in linear and GAM models for causal mediation analysis. The architecture supports both CNN and LSTM encoders with the following causal constraint penalties:

- **Linearizability penalty:** Encourages phi to have linear relationships with outcomes (Y)
- **Conditional independence penalty:** Encourages A (treatment) to be independent of M (mediator) given phi
- **Balance penalty:** Encourages phi features to be balanced across treatment levels

Input channels: glucose, steps, basal insulin, meal, heart rate. Bolus is intentionally excluded from the encoder to prevent mediator leakage (the mediator is computed separately as total bolus within a meal window).

### Mediation Models

- **lmer (mixed-effects):** Random intercepts by subject to account for within-subject correlation across repeated meal events
- **Quantile regression:** Estimates mediation effects at different quantiles of the glucose response distribution (e.g., median, 75th, 90th percentile)

Both model types support npCBPS weighting and bootstrap-based confidence intervals.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Spencer Hilligoss
University of California, Irvine
shilligo@uci.edu
