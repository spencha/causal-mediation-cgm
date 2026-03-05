#!/usr/bin/env Rscript
# ================================================================
# MIXED-EFFECTS CAUSAL MEDIATION ANALYSIS
# ================================================================
# Uses lmer for random intercepts by subject to account for
# within-subject correlation. This is more appropriate for
# repeated-measures data like CGM meal responses.
#
# Key advantages:
# - Accounts for subject-level variation
# - Improves precision of effect estimates
# - Handles unbalanced data (different # meals per subject)
# ================================================================

library(lme4)
library(mediation)
library(survival)
library(dplyr)
library(readr)
library(optparse)

# Try to load MuMIn for R-squared calculation (optional)
has_MuMIn <- suppressWarnings(require(MuMIn, quietly = TRUE))
if (!has_MuMIn) {
  cat("Note: MuMIn package not installed. R-squared values will not be computed.\n")
  cat("Install with: install.packages('MuMIn')\n\n")
}

# Try to load quantreg for quantile regression (optional)
has_quantreg <- suppressWarnings(require(quantreg, quietly = TRUE))
if (!has_quantreg) {
  cat("Note: quantreg package not installed. QR models will not be available.\n")
  cat("Install with: install.packages('quantreg')\n\n")
}

# --- START CONFIG BLOCK ---
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

config_locations <- c(
  file.path(script_dir, "config.R"),
  file.path(script_dir, "..", "cma_cluster", "config.R"),
  file.path(getwd(), "cma_cluster", "config.R")
)

config_loaded <- FALSE
for (cfg_path in config_locations) {
  if (file.exists(cfg_path)) {
    source(cfg_path)
    config_loaded <- TRUE
    break
  }
}

if (!config_loaded) {
  stop("Could not find config.R. Please run from project root or set CAUSAL_AE_BASE_DIR")
}
# --- END CONFIG BLOCK ---

# ================================================================
# COMMAND LINE ARGUMENTS
# ================================================================

option_list <- list(
  make_option(c("-t", "--minutes"), type="integer", default=60,
              help="Minutes after meal to analyze (30, 60, 90, 120, 180), default=60"),
  make_option(c("-o", "--offset"), type="integer", default=30,
              help="Treatment offset in grams"),
  make_option(c("-e", "--meal"), type="character", default="ALL",
              help="Meal type: Breakfast, Lunch, Dinner, Snack, or ALL"),
  make_option(c("-w", "--weights"), type="character", default="embedded",
              help="Weights source: 'embedded', path to file, or 'none'"),
  make_option(c("-b", "--bootstrap"), type="integer", default=1000,
              help="Number of simulations for quasi-Bayesian confidence intervals"),
  make_option(c("-d", "--dataset"), type="character", default="2020_TEST",
              help="Dataset to use: 2018, 2020, 2020_TEST, 2020_TRAIN, combined"),
  make_option(c("--phi-file"), type="character", default=NULL,
              help="Custom path to phi embeddings CSV file"),
  make_option(c("--output-dir"), type="character", default=NULL,
              help="Output directory for results (default: cma_cluster/mediation_results)"),
  make_option(c("-n", "--n-phi"), type="integer", default=3,
              help="Number of phi/PC features to use as covariates in mediation models (default: 3)"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PC_ columns instead of phi_ columns for covariates (default: FALSE)"),
  make_option(c("-m", "--model"), type="character", default="lmer",
              help="Outcome model type: 'lmer' (mixed-effects) or 'qr' (quantile regression) [default: lmer]"),
  make_option(c("-q", "--quantile"), type="numeric", default=0.5,
              help="Quantile for QR model (0-1), only used when --model=qr [default: 0.5]"),
  make_option(c("--cohort"), type="character", default=NULL,
              help="Filter to specific cohort: '2018', '2020', or NULL for all (default: NULL)")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Strip any stray quotes from string arguments (can happen from SLURM --export quoting)
opt$meal <- gsub('^["\']|["\']$', '', opt$meal)
opt$dataset <- gsub('^["\']|["\']$', '', opt$dataset)

# Validate model type
if (!opt$model %in% c("lmer", "qr")) {
  stop("--model must be 'lmer' or 'qr'")
}

# Validate QR requirements
if (opt$model == "qr" && !has_quantreg) {
  stop("quantreg package required for --model=qr. Install with: install.packages('quantreg')")
}

# Set default output directory if not specified
if (is.null(opt$`output-dir`)) {
  opt$`output-dir` <- CONFIG$MEDIATION_RESULTS_DIR
}

# Save into phi/ or pca/ subdirectory
covariate_subdir <- if (opt$`use-pca`) "pca" else "phi"
opt$`output-dir` <- file.path(opt$`output-dir`, covariate_subdir)

# Save into meal-specific subdirectory (e.g., phi/ALL/, phi/Breakfast/, etc.)
meal_subdir <- opt$meal
opt$`output-dir` <- file.path(opt$`output-dir`, meal_subdir)

# Create output directory
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

# Print configuration
cat("\n================================================================\n")
cat("CAUSAL MEDIATION ANALYSIS\n")
cat("================================================================\n")
cat(sprintf("Dataset: %s\n", opt$dataset))
cat(sprintf("Outcome model: %s\n", toupper(opt$model)))
if (opt$model == "qr") {
  cat(sprintf("Quantile (tau): %.2f\n", opt$quantile))
}
cat(sprintf("Outcome: glucose change at %d minutes after meal\n", opt$minutes))
cat(sprintf("Meal type: %s\n", opt$meal))
cat(sprintf("Treatment offset: %d grams\n", opt$offset))
cat(sprintf("Bootstrap sims: %d\n", opt$bootstrap))
cat(sprintf("Weights: %s\n", opt$weights))
cat(sprintf("Phi features: %d\n", opt$`n-phi`))
cat(sprintf("Use PCA: %s\n", ifelse(opt$`use-pca`, "YES (PC_ columns)", "NO (phi_ columns)")))
cat("================================================================\n\n")

# ================================================================
# LOAD DATA
# ================================================================

cat("Loading data...\n")

dataset_choice <- tolower(opt$dataset)

rdata_file_map <- list(
  "2018" = "z_meal_mediation_analysis_data_2018_5min.RData",
  "2018_test" = "z_meal_mediation_analysis_data_2018_TEST_5min.RData",
  "2018_train" = "z_meal_mediation_analysis_data_2018_TRAIN_5min.RData",
  "2020" = "z_meal_mediation_analysis_data_2020_5min.RData",
  "2020_test" = "z_meal_mediation_analysis_data_2020_TEST_5min.RData",
  "2020_train" = "z_meal_mediation_analysis_data_2020_TRAIN_5min.RData",
  "combined" = "z_meal_mediation_analysis_data_combined_5min.RData"
)

# Legacy phi file map (for backward compatibility)
phi_file_map <- list(
  "2018" = "z_meal_y_delta_glucose_phi_embeddings_causal.csv",
  "2020" = "z_meal_y_delta_glucose_phi_embeddings_2020_causal.csv",
  "2020_test" = "z_meal_y_delta_glucose_phi_embeddings_2020_test_causal.csv",
  "2020_train" = "z_meal_y_delta_glucose_phi_embeddings_2020_train_causal.csv",
  "combined" = "z_meal_y_delta_glucose_phi_embeddings_combined_causal.csv"
)

# Helper function to find embeddings in new location
find_embeddings_file <- function(base_dir) {
  embeddings_dir <- file.path(base_dir, "embeddings")
  # Look for combined embeddings (preferred for CMA)
  combined_files <- list.files(embeddings_dir,
                               pattern = "^phi_embeddings_combined_.*\\.csv$",
                               full.names = TRUE)
  if (length(combined_files) > 0) {
    # Return most recently modified
    file_info <- file.info(combined_files)
    return(combined_files[which.max(file_info$mtime)])
  }
  return(NULL)
}

rdata_filename <- rdata_file_map[[dataset_choice]]
if (is.null(rdata_filename)) {
  stop(sprintf("Unknown dataset: %s", opt$dataset))
}
rdata_path <- file.path(CONFIG$ANALYSIS_DATA_DIR, rdata_filename)

rdata_available <- file.exists(rdata_path)
if (!rdata_available) {
  cat(sprintf("Note: RData file not found: %s\n", rdata_path))
  cat("  Will use outcome columns from phi embeddings CSV instead.\n")
}

# Determine phi file
if (!is.null(opt$`phi-file`)) {
  phi_file <- opt$`phi-file`
} else {
  # First, try to find embeddings in new location
  phi_file <- find_embeddings_file(CONFIG$ANALYSIS_DATA_DIR)

  if (is.null(phi_file) || !file.exists(phi_file)) {
    # Fall back to legacy location
    phi_filename <- phi_file_map[[dataset_choice]]
    if (is.null(phi_filename)) {
      phi_filename <- "z_meal_y_delta_glucose_phi_embeddings_causal.csv"
    }
    phi_file <- file.path(CONFIG$ANALYSIS_DATA_DIR, phi_filename)
  }
}

if (!file.exists(phi_file)) {
  stop(sprintf("Phi embeddings file not found: %s", phi_file))
}

cat(sprintf("Phi embeddings: %s\n", phi_file))
cat(sprintf("RData file: %s\n", rdata_path))

# Load phi embeddings
phi_df <- read_csv(phi_file, show_col_types = FALSE)

cat(sprintf("Loaded phi embeddings: %d observations\n", nrow(phi_df)))

# Check if glucose_at_meal is already in phi_df (exported embeddings include it)
if (!"glucose_at_meal" %in% names(phi_df)) {
  cat("  glucose_at_meal not in phi_df, will try to load from RData\n")
  need_rdata <- TRUE
} else {
  cat("  glucose_at_meal found in phi_df\n")
  need_rdata <- FALSE
}

# Check if subject_id is already in phi_df
if (!"subject_id" %in% names(phi_df)) {
  cat("  subject_id not in phi_df, will try to load from RData\n")
  need_rdata <- TRUE
} else {
  cat("  subject_id found in phi_df\n")
}

# Check if outcome columns are in phi_df (from new export format)
outcome_cols <- grep("^Y_\\d+min$", names(phi_df), value = TRUE)
if (length(outcome_cols) > 0) {
  cat(sprintf("Found outcome columns in phi_df: %s\n", paste(outcome_cols, collapse = ", ")))
}

# Always try to load RData for y_seq_change as fallback
# (needed if phi_df doesn't have the specific time point requested)
if (file.exists(rdata_path)) {
  load(rdata_path)
  cat(sprintf("Loaded RData for fallback: %d observations\n", nrow(all_cleaned_data)))

  # Check dimensions match
  if (nrow(phi_df) != nrow(all_cleaned_data)) {
    cat(sprintf("WARNING: phi_df (%d rows) and all_cleaned_data (%d rows) have different sizes!\n",
                nrow(phi_df), nrow(all_cleaned_data)))
    cat("  This may indicate mismatched data sources.\n")
  }

  # Only add columns if they're missing from phi_df
  if (!"glucose_at_meal" %in% names(phi_df) && "glucose_at_meal" %in% names(all_cleaned_data)) {
    phi_df$glucose_at_meal <- all_cleaned_data$glucose_at_meal
  }
  if (!"subject_id" %in% names(phi_df) && "subject_id" %in% names(all_cleaned_data)) {
    phi_df$subject_id <- all_cleaned_data$subject_id
  }
} else {
  cat(sprintf("Note: RData file not found: %s\n", rdata_path))
  if (length(outcome_cols) == 0) {
    cat("  WARNING: No Y columns in phi_df and no RData fallback available.\n")
  }
}

# ================================================================
# FILTER TO TEST DATA ONLY
# ================================================================
# The combined embeddings contain both train and test data.
# For CMA, we only want to analyze the test set to get unbiased estimates.
# ================================================================

# First, add original row index BEFORE filtering (for RData matrix indexing)
phi_df$original_row_index <- 1:nrow(phi_df)

# Check if we have outcome columns in phi_df (from exported CSV)
# If so, we don't need the RData matrices
outcome_cols_in_phi <- grep("^Y_\\d+min$", names(phi_df), value = TRUE)
has_outcomes_in_csv <- length(outcome_cols_in_phi) > 0

if ("split" %in% names(phi_df)) {
  n_before <- nrow(phi_df)
  train_n <- sum(phi_df$split == "train", na.rm = TRUE)
  test_n <- sum(phi_df$split == "test", na.rm = TRUE)
  cat(sprintf("Data split: %d train, %d test (total: %d)\n", train_n, test_n, n_before))

  # Check cohort breakdown if available
  if ("cohort" %in% names(phi_df)) {
    cohort_counts <- table(phi_df$cohort, phi_df$split)
    cat("Cohort breakdown:\n")
    print(cohort_counts)
  }

  # Filter to match exactly the observations in the RData test file
  # NOTE: The 'split' column in the embeddings CSV comes from the Python AE pipeline
  # and has different counts than the RData files. The RData file contains the
  # authoritative test set (79 observations from 2020 subjects).
  #
  # We filter by matching global_window_id to ensure we use exactly the same
  # observations that are in the RData test file.

  # Track row indices for RData matrix filtering (needed if outcomes not in CSV)
  test_row_indices <- NULL

  if (exists("all_cleaned_data") && "global_window_id" %in% names(all_cleaned_data) &&
      "global_window_id" %in% names(phi_df)) {
    # Get the window IDs from the RData file (authoritative test set)
    rdata_window_ids <- all_cleaned_data$global_window_id

    # Track which rows we're keeping (for RData matrix filtering)
    test_row_indices <- which(phi_df$global_window_id %in% rdata_window_ids)

    # Filter phi_df to only include windows that are in the RData
    phi_df <- phi_df[phi_df$global_window_id %in% rdata_window_ids, ]
    cat(sprintf("Filtered to match RData test set (by global_window_id): %d observations\n", nrow(phi_df)))

    # Verify the match
    if (nrow(phi_df) != nrow(all_cleaned_data)) {
      cat(sprintf("WARNING: After filtering, phi_df has %d rows but RData has %d rows\n",
                  nrow(phi_df), nrow(all_cleaned_data)))
      cat("  Some window IDs may be missing from the embeddings CSV.\n")
    }
  } else if ("cohort" %in% names(phi_df)) {
    # Fallback: use split column (include all cohorts in combined analysis)
    test_row_indices <- which(phi_df$split == "test")
    phi_df <- phi_df[phi_df$split == "test", ]
    cat(sprintf("Filtered to test set across all cohorts (by split column): %d observations\n", nrow(phi_df)))
    if ("cohort" %in% names(phi_df)) {
      cat(sprintf("  Cohort breakdown: %s\n", paste(names(table(phi_df$cohort)), table(phi_df$cohort), sep="=", collapse=", ")))
    }
    cat("Warning: Using split column - may not match RData test set exactly\n")
  } else {
    test_row_indices <- which(phi_df$split == "test")
    phi_df <- phi_df[phi_df$split == "test", ]
    cat(sprintf("Filtered to test set: %d observations\n", nrow(phi_df)))
    cat("Warning: No matching method available - using split column\n")
  }

  # Only filter RData matrices if we need them (when outcomes NOT in CSV)
  # AND if the RData size matches the CSV (same data source)
  if (!has_outcomes_in_csv) {
    cat("No outcome columns in CSV, will use RData fallback if available\n")
    # Filter RData matrices if they exist AND are the right size AND we have indices
    if (!is.null(test_row_indices) && length(test_row_indices) > 0) {
      if (exists("y_seq_change") && nrow(y_seq_change) == n_before) {
        y_seq_change <- y_seq_change[test_row_indices, , drop = FALSE]
        cat(sprintf("Filtered y_seq_change to test rows: %d x %d\n", nrow(y_seq_change), ncol(y_seq_change)))
      }
      if (exists("y_seq") && nrow(y_seq) == n_before) {
        y_seq <- y_seq[test_row_indices, , drop = FALSE]
      }
      if (exists("all_cleaned_data") && nrow(all_cleaned_data) == n_before) {
        all_cleaned_data <- all_cleaned_data[test_row_indices, ]
      }
    } else {
      cat("Warning: Could not determine test row indices, RData matrices may not match phi_df\n")
    }
  } else {
    cat(sprintf("Using outcome columns from CSV: %s\n", paste(outcome_cols_in_phi, collapse = ", ")))
    # Clear RData matrices to avoid confusion since they don't match the CSV
    if (exists("y_seq_change")) rm(y_seq_change)
    if (exists("y_seq")) rm(y_seq)
  }

  # Reset row indices after filtering (now 1:n for the filtered data)
  phi_df$original_row_index <- 1:nrow(phi_df)
} else {
  cat("Warning: 'split' column not found in phi_df. Using all data.\n")
}

# Filter by cohort if specified
if (!is.null(opt$cohort)) {
  if ("cohort" %in% names(phi_df)) {
    n_before <- nrow(phi_df)
    phi_df <- phi_df[phi_df$cohort == opt$cohort, ]
    cat(sprintf("Filtered to cohort '%s': %d -> %d observations\n",
                opt$cohort, n_before, nrow(phi_df)))
    phi_df$original_row_index <- 1:nrow(phi_df)
  } else {
    cat("Warning: --cohort specified but 'cohort' column not found in phi_df\n")
  }
}

# Ensure subject_id is available
if (!"subject_id" %in% names(phi_df)) {
  if ("subject_id" %in% names(all_cleaned_data)) {
    phi_df$subject_id <- all_cleaned_data$subject_id
  } else {
    stop("subject_id not found in phi_df or all_cleaned_data")
  }
}

# Make subject_id a factor for lmer
phi_df$subject_id <- as.factor(phi_df$subject_id)

cat(sprintf("Loaded %d observations from %d subjects\n",
            nrow(phi_df), length(unique(phi_df$subject_id))))

# Load weights if specified
if (opt$weights != "none") {
  if (opt$weights == "embedded") {
    # Load weights from weights directory
    weights_file <- file.path(CONFIG$WEIGHTS_DIR, "npCBPS_weights.csv")
    if (file.exists(weights_file)) {
      cat(sprintf("Loading weights from %s\n", weights_file))
      weights_df <- read_csv(weights_file, show_col_types = FALSE)

      # Determine join columns - use split if available to avoid duplicates
      # Also select stratified_weight if available (for meal-specific analysis)
      has_stratified <- "stratified_weight" %in% names(weights_df)

      if ("split" %in% names(weights_df) && "split" %in% names(phi_df)) {
        join_cols <- c("global_window_id", "split")
        select_cols <- c("global_window_id", "split", "treatment_weight")
        if (has_stratified) select_cols <- c(select_cols, "stratified_weight")
        weights_df <- weights_df %>%
          dplyr::select(all_of(select_cols)) %>%
          dplyr::distinct(global_window_id, split, .keep_all = TRUE)
      } else {
        join_cols <- "global_window_id"
        select_cols <- c("global_window_id", "treatment_weight")
        if (has_stratified) select_cols <- c(select_cols, "stratified_weight")
        weights_df <- weights_df %>%
          dplyr::select(all_of(select_cols)) %>%
          dplyr::distinct(global_window_id, .keep_all = TRUE)
      }

      n_before <- nrow(phi_df)
      phi_df <- phi_df %>%
        left_join(weights_df, by = join_cols)

      # Use stratified_weight for meal-specific analysis, treatment_weight for ALL
      # Fall back to pooled weights when meal-specific sample is too small
      # (stratified npCBPS overfits on small strata, destroying ESS)
      MIN_N_FOR_STRATIFIED <- 50
      meal_n <- if (opt$meal != "ALL" && "meal_type" %in% names(phi_df)) {
        sum(tolower(phi_df$meal_type) == tolower(opt$meal), na.rm = TRUE)
      } else {
        nrow(phi_df)
      }

      if (opt$meal != "ALL" && has_stratified && "stratified_weight" %in% names(phi_df) &&
          meal_n >= MIN_N_FOR_STRATIFIED) {
        cat(sprintf("Using stratified (meal-specific) weights (n=%d >= %d)\n",
                    meal_n, MIN_N_FOR_STRATIFIED))
        phi_df <- phi_df %>%
          mutate(weight = ifelse(is.na(stratified_weight), 1, stratified_weight))
      } else {
        if (opt$meal != "ALL" && meal_n < MIN_N_FOR_STRATIFIED) {
          cat(sprintf("Using pooled weights (meal n=%d < %d threshold for stratified)\n",
                      meal_n, MIN_N_FOR_STRATIFIED))
        }
        phi_df <- phi_df %>%
          mutate(weight = ifelse(is.na(treatment_weight), 1, treatment_weight))
      }
      n_after <- nrow(phi_df)

      if (n_after != n_before) {
        warning(sprintf("Row count changed after weight join: %d -> %d. Deduplicating...", n_before, n_after))
        phi_df <- phi_df %>% dplyr::distinct()
      }
    } else {
      cat("No weights file found, using uniform weights\n")
      phi_df$weight <- 1
    }
  } else if (file.exists(opt$weights)) {
    weights_df <- read_csv(opt$weights, show_col_types = FALSE)
    phi_df <- phi_df %>%
      left_join(weights_df, by = "global_window_id") %>%
      mutate(weight = ifelse(is.na(treatment_weight), 1, treatment_weight))
  } else {
    phi_df$weight <- 1
  }
} else {
  phi_df$weight <- 1
}

# ================================================================
# EXCLUDE LATE-BOLUS OBSERVATIONS
# ================================================================
# Remove observations where total_bolus > 0 but mediator_bolus_for_meal == 0.
# These are cases where insulin was taken AFTER the mediator window (-120 to +60
# min), violating the A -> M -> Y temporal ordering required for CMA.
# ~58 observations total (~30 in 2018, ~28 in 2020).
# ================================================================

if ("total_bolus" %in% names(phi_df) && "mediator_bolus_for_meal" %in% names(phi_df)) {
  late_bolus_mask <- (phi_df$total_bolus > 0) & (phi_df$mediator_bolus_for_meal == 0)
  n_late_bolus <- sum(late_bolus_mask, na.rm = TRUE)

  if (n_late_bolus > 0) {
    cat(sprintf("\nExcluding %d late-bolus observations (total_bolus > 0 but mediator_bolus_for_meal == 0)\n",
                n_late_bolus))
    if ("cohort" %in% names(phi_df)) {
      cat(sprintf("  By cohort: %s\n",
                  paste(names(table(phi_df$cohort[late_bolus_mask])),
                        table(phi_df$cohort[late_bolus_mask]),
                        sep = "=", collapse = ", ")))
    }
    phi_df <- phi_df[!late_bolus_mask, ]
    cat(sprintf("  Remaining observations: %d\n", nrow(phi_df)))
  } else {
    cat("\nNo late-bolus observations to exclude.\n")
  }
} else {
  cat("\nNote: total_bolus column not found in embeddings. Late-bolus filtering skipped.\n")
  cat("  Re-run train_and_export_embeddings.py to include total_bolus.\n")
}

# ================================================================
# MIXED-EFFECTS MEDIATION FUNCTION
# ================================================================

run_mixed_mediation <- function(data, minutes, treat_offset = 30, n_phi_max = 3,
                                use_pca = FALSE, model_type = "lmer", quantile_tau = 0.5) {

  # Look for outcome column in data (from exported embeddings)
  # Column names are Y_60min, Y_65min, ..., Y_210min (5-minute intervals)
  outcome_col <- sprintf("Y_%dmin", minutes)

  if (outcome_col %in% names(data)) {
    # Use outcome from exported CSV
    data$Y <- data[[outcome_col]]
    cat(sprintf("Using outcome from column: %s\n", outcome_col))
  } else if (exists("y_seq_change") || exists("y_seq")) {
    # Fallback to RData if available
    # IMPORTANT: y_seq starts at 60 minutes post-meal
    # y_seq[, i] = glucose at (60 + (i-1)*5) minutes (R uses 1-based indexing)
    # So for minutes M, we need index = (M - 60) / 5 + 1
    if (minutes < 60) {
      stop(sprintf("Cannot analyze %d minutes - y_seq only has data starting at 60 min post-meal", minutes))
    }
    time_point <- (minutes - 60) / 5 + 1  # +1 for R's 1-based indexing

    if (exists("y_seq_change")) {
      # y_seq_change already contains glucose changes from baseline
      if (time_point > ncol(y_seq_change)) {
        stop(sprintf("Time point %d (%.0f min) exceeds y_seq_change columns (%d)",
                     time_point, minutes, ncol(y_seq_change)))
      }
      data$Y <- y_seq_change[data$original_row_index, time_point]
      cat(sprintf("Using outcome from y_seq_change[, %d] (minutes=%d)\n", time_point, minutes))
    } else if (exists("y_seq")) {
      # y_seq contains raw glucose values - compute change from glucose_at_meal
      if (time_point > ncol(y_seq)) {
        stop(sprintf("Time point %d (%.0f min) exceeds y_seq columns (%d)",
                     time_point, minutes, ncol(y_seq)))
      }
      glucose_at_time <- y_seq[data$original_row_index, time_point]
      data$Y <- glucose_at_time - data$glucose_at_meal
      cat(sprintf("Using outcome from y_seq[, %d] - glucose_at_meal (minutes=%d)\n", time_point, minutes))
    }
  } else {
    stop(sprintf("Outcome not found. Need either '%s' column in phi_df, or y_seq/y_seq_change from RData.",
                 outcome_col))
  }

  # Remove missing values
  complete_idx <- !is.na(data$Y) & !is.na(data$mediator_bolus_for_meal) &
                  !is.na(data$treat_meal_carbs) & !is.na(data$subject_id)
  data <- data[complete_idx, ]

  n_obs <- nrow(data)
  n_subjects <- length(unique(data$subject_id))

  cat(sprintf("\nAnalysis sample: %d observations from %d subjects\n", n_obs, n_subjects))

  if (n_obs == 0 || n_subjects < 2) {
    return(list(
      status = "insufficient_data",
      n_obs = n_obs,
      n_subjects = n_subjects,
      error_msg = "No observations or fewer than 2 subjects"
    ))
  }

  # Get phi/PC columns based on use_pca flag
  if (use_pca) {
    phi_cols <- grep("^PC_", names(data), value = TRUE)
    # Sort numerically by PC number (PC_1, PC_2, ..., PC_10)
    phi_cols <- phi_cols[order(as.numeric(gsub("PC_", "", phi_cols)))]
    feature_type <- "PC"
  } else {
    phi_cols <- grep("^phi_", names(data), value = TRUE)
    # Sort numerically by phi number (phi_1, phi_2, ..., phi_16)
    phi_cols <- phi_cols[order(as.numeric(gsub("phi_", "", phi_cols)))]
    feature_type <- "phi"
  }
  n_phi <- min(n_phi_max, length(phi_cols))

  cat(sprintf("Using %d %s features as covariates (requested: %d, available: %d)\n",
              n_phi, feature_type, n_phi_max, length(phi_cols)))

  # Prepare phi columns (dynamically for any n_phi)
  for (j in seq_len(n_phi)) {
    data[[paste0("phi", j)]] <- data[[phi_cols[j]]]
  }

  tryCatch({
    # ----------------------------------------------------------------
    # BUILD MODEL FORMULAS
    # ----------------------------------------------------------------

    # Covariates portion of formula (phi/PC features only)
    phi_terms <- if (n_phi == 0) "" else paste(paste0("phi", 1:n_phi), collapse = " + ")
    base_covs <- phi_terms

    # NOTE: cohort is used in npCBPS balancing but NOT in the mediator/outcome
    # models. Including it here caused issues with 2018 results — the phi
    # embeddings already capture cohort-level differences in glucose dynamics.
    #
    # NOTE: glucose_at_meal is also kept OUT of the mediator/outcome models.
    # It is a strong predictor of both M and Y, and conditioning on it in both
    # models distorts the M->Y pathway estimate (producing positive ACME, i.e.,
    # more insulin -> glucose UP, which is clinically nonsensical). The PC
    # embeddings partially capture pre-meal glucose level (PC2 r=-0.55), and
    # npCBPS balances on glucose_at_meal directly.

    # ----------------------------------------------------------------
    # MEDIATOR MODEL: Tobit (survreg) — left-censored at zero
    # ----------------------------------------------------------------
    # Insulin bolus is left-censored at zero: ~12% of observations have
    # zero bolus, and positive values are right-skewed. OLS/lmer treats
    # these zeros as coming from the same continuous distribution,
    # biasing the A→M coefficient. Tobit models a latent variable
    # M* = Xβ + ε where M = max(0, M*), properly accounting for the
    # point mass at zero.
    #
    # survreg mediator + lm/lmer outcome is supported by mediate()
    # with quasi-Bayesian inference (no bootstrap needed).
    # ----------------------------------------------------------------
    mediator_formula <- as.formula(
      paste("Surv(mediator_bolus_for_meal, mediator_bolus_for_meal > 0, type='left') ~ treat_meal_carbs +", base_covs)
    )

    # Outcome formula depends on model type
    if (model_type == "lmer") {
      outcome_formula_lmer <- as.formula(
        paste("Y ~ treat_meal_carbs + mediator_bolus_for_meal +", base_covs, "+ (1|subject_id)")
      )
      outcome_formula_lm <- as.formula(
        paste("Y ~ treat_meal_carbs + mediator_bolus_for_meal +", base_covs)
      )
    } else {
      # QR: no random effects (quantreg::rq does not support them)
      outcome_formula_lmer <- as.formula(
        paste("Y ~ treat_meal_carbs + mediator_bolus_for_meal +", base_covs)
      )
      outcome_formula_lm <- outcome_formula_lmer
    }

    # ----------------------------------------------------------------
    # FIT MODELS
    # ----------------------------------------------------------------

    # --- Mediator model (Tobit via survreg) ---
    cat("\nFitting mediator model (Tobit / survreg)...\n")
    n_zero <- sum(data$mediator_bolus_for_meal == 0, na.rm = TRUE)
    pct_zero <- 100 * n_zero / nrow(data)
    cat(sprintf("  Zero-inflation: %d/%d observations (%.1f%%) have zero bolus\n",
                n_zero, nrow(data), pct_zero))
    model.m <- survreg(mediator_formula, data = data, weights = weight, dist = "gaussian")

    # --- Outcome model (lm or QR) ---
    # mediate() requires matching group structures. survreg has no
    # random effects, so the outcome must also be non-mixed (lm).
    # We still fit lmer first to report the ICC for diagnostics,
    # then use lm for the actual mediation.
    use_lm_outcome <- FALSE
    icc_y <- NA

    if (model_type == "lmer") {
      # Fit lmer to compute ICC for diagnostics only
      cat("Fitting outcome model (lmer for ICC diagnostics)...\n")
      model.y_lmer <- tryCatch(
        lmer(outcome_formula_lmer, data = data, weights = weight, REML = FALSE),
        error = function(e) NULL
      )
      if (!is.null(model.y_lmer)) {
        vc_y <- as.data.frame(VarCorr(model.y_lmer))
        icc_y <- vc_y$vcov[1] / sum(vc_y$vcov)
        cat(sprintf("  Outcome ICC = %.3f (%.1f%% between-subject variance)\n", icc_y, icc_y * 100))
      }

      # Use lm for mediation (survreg mediator has no groups)
      cat("Fitting outcome model (lm — required for survreg mediator compatibility)...\n")
      model.y <- lm(outcome_formula_lm, data = data, weights = weight)
      use_lm_outcome <- TRUE
    } else {
      cat(sprintf("Fitting outcome model (QR, tau=%.2f)...\n", quantile_tau))
      model.y <- rq(outcome_formula_lmer, tau = quantile_tau, data = data,
                     weights = data$weight, model = TRUE)
    }

    outcome_formula <- if (use_lm_outcome) outcome_formula_lm else outcome_formula_lmer

    cat(sprintf("\nFinal mediator model: survreg (Tobit, dist=gaussian)\n"))
    cat(sprintf("Final outcome model:  %s\n",
                if (model_type == "qr") "qr" else "lm"))
    cat(sprintf("Mediator formula: %s\n", deparse(mediator_formula)))
    cat(sprintf("Outcome formula:  %s\n", deparse(outcome_formula)))

    # Print model summaries
    cat("\n--- Mediator Model Summary (Tobit) ---\n")
    surv_summary <- summary(model.m)
    print(surv_summary$table)
    cat(sprintf("  Log(scale) = %.4f  =>  sigma = %.4f\n",
                model.m$icoef[length(model.m$icoef)], model.m$scale))

    cat("\n--- Outcome Model Summary ---\n")
    print(summary(model.y)$coefficients)

    # ----------------------------------------------------------------
    # MODEL DIAGNOSTICS
    # ----------------------------------------------------------------

    cat("\n--- Model Diagnostics ---\n")

    # Mediator model residuals (Tobit / survreg)
    # survreg residuals: response type = observed - linear predictor (Xβ)
    m_resid <- residuals(model.m, type = "response")
    m_fitted <- predict(model.m, type = "response")
    m_resid_mean <- mean(m_resid)
    m_resid_sd <- sd(m_resid)
    m_resid_skew <- mean((m_resid - m_resid_mean)^3) / m_resid_sd^3
    m_resid_kurt <- mean((m_resid - m_resid_mean)^4) / m_resid_sd^4 - 3
    m_shapiro_p <- tryCatch({
      # Shapiro-Wilk on sample (max 5000 obs)
      idx <- if (length(m_resid) > 5000) sample(length(m_resid), 5000) else seq_along(m_resid)
      shapiro.test(m_resid[idx])$p.value
    }, error = function(e) NA)

    cat(sprintf("Mediator residuals: mean=%.4f, sd=%.4f, skew=%.3f, kurt=%.3f, Shapiro p=%.4f\n",
                m_resid_mean, m_resid_sd, m_resid_skew, m_resid_kurt,
                ifelse(is.na(m_shapiro_p), -1, m_shapiro_p)))

    # Outcome model residuals
    y_resid <- residuals(model.y)
    y_fitted <- fitted(model.y)
    y_resid_mean <- mean(y_resid)
    y_resid_sd <- sd(y_resid)
    y_resid_skew <- mean((y_resid - y_resid_mean)^3) / y_resid_sd^3
    y_resid_kurt <- mean((y_resid - y_resid_mean)^4) / y_resid_sd^4 - 3
    y_shapiro_p <- tryCatch({
      idx <- if (length(y_resid) > 5000) sample(length(y_resid), 5000) else seq_along(y_resid)
      shapiro.test(y_resid[idx])$p.value
    }, error = function(e) NA)

    cat(sprintf("Outcome residuals:  mean=%.4f, sd=%.4f, skew=%.3f, kurt=%.3f, Shapiro p=%.4f\n",
                y_resid_mean, y_resid_sd, y_resid_skew, y_resid_kurt,
                ifelse(is.na(y_shapiro_p), -1, y_shapiro_p)))

    # ----------------------------------------------------------------
    # BUILD RESIDUALS DATA FRAME FOR EXPORT
    # ----------------------------------------------------------------

    residuals_df <- data.frame(
      global_window_id  = if ("global_window_id" %in% names(data)) data$global_window_id else seq_len(nrow(data)),
      subject_id        = as.character(data$subject_id),
      mediator_actual   = data$mediator_bolus_for_meal,
      mediator_fitted   = as.numeric(m_fitted),
      mediator_residual = as.numeric(m_resid),
      outcome_actual    = data$Y,
      outcome_fitted    = as.numeric(y_fitted),
      outcome_residual  = as.numeric(y_resid),
      treat_meal_carbs  = data$treat_meal_carbs,
      stringsAsFactors  = FALSE
    )
    if ("meal_type" %in% names(data)) {
      residuals_df$meal_type <- data$meal_type
    }
    if ("cohort" %in% names(data)) {
      residuals_df$cohort <- data$cohort
    }

    cat(sprintf("Residuals data frame: %d rows\n", nrow(residuals_df)))

    # ----------------------------------------------------------------
    # RUN MEDIATION ANALYSIS
    # ----------------------------------------------------------------

    cat("\nRunning mediation analysis (bootstrap)...\n")

    # Treatment values
    control_value <- median(data$treat_meal_carbs)
    treat_value <- control_value + treat_offset

    cat(sprintf("Control value: %.1f g, Treatment value: %.1f g\n",
                control_value, treat_value))

    # Run mediation with quasi-Bayesian approximation (boot = FALSE)
    # Note: boot = FALSE is required for both lmer (weighted) and rq models.
    # The mediation package does not support boot = TRUE with rq objects.
    med <- mediate(
      model.m = model.m,
      model.y = model.y,
      treat = "treat_meal_carbs",
      mediator = "mediator_bolus_for_meal",
      control.value = control_value,
      treat.value = treat_value,
      boot = FALSE,
      sims = opt$bootstrap
    )

    # ----------------------------------------------------------------
    # EXTRACT RANDOM EFFECTS / ICC INFO
    # ----------------------------------------------------------------

    cat(sprintf("\nIntra-class correlation (ICC):\n"))

    # Mediator model: survreg has no random effects
    icc_m <- NA
    cat("  Mediator model: N/A (Tobit / survreg has no random effects)\n")

    # Outcome model ICC (computed from diagnostic lmer; actual model is lm)
    if (!is.na(icc_y)) {
      cat(sprintf("  Outcome model:  %.3f (%.1f%% between-subject; lm used for mediation)\n",
                  icc_y, icc_y * 100))
    } else if (model_type == "qr") {
      cat("  Outcome model:  N/A (QR has no random effects)\n")
    } else {
      cat("  Outcome model:  N/A (lmer failed to converge)\n")
    }

    # ----------------------------------------------------------------
    # CALCULATE R-SQUARED
    # ----------------------------------------------------------------

    # Mediator R² (survreg: use McFadden pseudo-R² = 1 - logLik(full)/logLik(null))
    r2_m <- tryCatch({
      ll_full <- logLik(model.m)
      null_formula <- as.formula(
        "Surv(mediator_bolus_for_meal, mediator_bolus_for_meal > 0, type='left') ~ 1"
      )
      model_null <- survreg(null_formula, data = data, weights = weight, dist = "gaussian")
      ll_null <- logLik(model_null)
      pseudo_r2 <- as.numeric(1 - ll_full / ll_null)
      c(R2m = pseudo_r2, R2c = pseudo_r2)  # No random effects
    }, error = function(e) {
      c(R2m = NA, R2c = NA)
    })

    # Outcome R²
    if (model_type == "lmer") {
      # Outcome is always lm (for survreg compatibility)
      r2_y <- tryCatch({
        r2 <- summary(model.y)$r.squared
        c(R2m = r2, R2c = r2)
      }, error = function(e) {
        c(R2m = NA, R2c = NA)
      })
    } else {
      # For QR: use pseudo-R1 (Koenker & Machado 1999)
      # rho_tau(u) = u * (tau - I(u<0))
      # R1 = 1 - V(tau) / V0(tau) where V is weighted sum of check losses
      rq_r1 <- tryCatch({
        rq_null <- rq(Y ~ 1, tau = quantile_tau, data = data, weights = data$weight)
        1 - model.y$rho / rq_null$rho
      }, error = function(e) NA)
      r2_y <- c(R2m = rq_r1, R2c = NA)
    }

    # ----------------------------------------------------------------
    # RETURN RESULTS
    # ----------------------------------------------------------------

    return(list(
      status = "success",
      n_obs = n_obs,
      n_subjects = n_subjects,

      # Mediation effects
      ACME = med$d0,
      ACME_lower = med$d0.ci[1],
      ACME_upper = med$d0.ci[2],
      ACME_p = med$d0.p,

      ADE = med$z0,
      ADE_lower = med$z0.ci[1],
      ADE_upper = med$z0.ci[2],
      ADE_p = med$z0.p,

      total_effect = med$tau.coef,
      total_lower = med$tau.ci[1],
      total_upper = med$tau.ci[2],
      total_p = med$tau.p,

      prop_mediated = ifelse(abs(med$tau.coef) > 1e-10,
                             med$d0 / med$tau.coef, NA),

      # Model fit
      mediator_r2_marginal = r2_m[1],
      mediator_r2_conditional = r2_m[2],
      outcome_r2_marginal = r2_y[1],
      outcome_r2_conditional = r2_y[2],

      # Random effects
      icc_mediator = icc_m,
      icc_outcome = icc_y,

      # Diagnostics: mediator model residuals
      m_resid_mean = m_resid_mean,
      m_resid_sd = m_resid_sd,
      m_resid_skew = m_resid_skew,
      m_resid_kurt = m_resid_kurt,
      m_shapiro_p = m_shapiro_p,

      # Diagnostics: outcome model residuals
      y_resid_mean = y_resid_mean,
      y_resid_sd = y_resid_sd,
      y_resid_skew = y_resid_skew,
      y_resid_kurt = y_resid_kurt,
      y_shapiro_p = y_shapiro_p,

      n_phi = n_phi,
      used_weights = sum(data$weight != 1) > 0,
      use_pca = use_pca,
      model_type = model_type,
      quantile_tau = if (model_type == "qr") quantile_tau else NA,

      # Tobit mediator diagnostics
      tobit_sigma = model.m$scale,
      pct_zero_mediator = pct_zero,

      # Actual residuals for export
      residuals_df = residuals_df
    ))

  }, error = function(e) {
    return(list(
      status = "error",
      n_obs = n_obs,
      n_subjects = n_subjects,
      error_msg = substr(e$message, 1, 200)
    ))
  })
}

# ================================================================
# RUN ANALYSIS
# ================================================================

analysis_data <- phi_df

# Filter by meal type if specified
if (opt$meal != "ALL") {
  if ("meal_type" %in% names(analysis_data)) {
    # Show available meal types
    available_meals <- unique(analysis_data$meal_type)
    cat(sprintf("Available meal types: %s\n", paste(available_meals, collapse=", ")))

    # Case-insensitive matching
    meal_lower <- tolower(opt$meal)
    analysis_data <- analysis_data[tolower(analysis_data$meal_type) == meal_lower, ]
    cat(sprintf("Filtered to %s meals: %d observations\n", opt$meal, nrow(analysis_data)))
  }
}

# Run mediation
result <- run_mixed_mediation(
  data = analysis_data,
  minutes = opt$minutes,
  treat_offset = opt$offset,
  n_phi_max = opt$`n-phi`,
  use_pca = opt$`use-pca`,
  model_type = opt$model,
  quantile_tau = opt$quantile
)

# ================================================================
# SAVE RESULTS
# ================================================================

# Helper to safely extract result fields
safe <- function(field) if (!is.null(result[[field]])) result[[field]] else NA

result_df <- data.frame(
  dataset = opt$dataset,
  model = opt$model,
  quantile_tau = if (opt$model == "qr") opt$quantile else NA,
  time_point = opt$minutes / 5,
  minutes = opt$minutes,
  meal_type = opt$meal,
  treat_offset = opt$offset,
  status = result$status,
  n_obs = result$n_obs,
  n_subjects = safe("n_subjects"),

  ACME = safe("ACME"),
  ACME_lower = safe("ACME_lower"),
  ACME_upper = safe("ACME_upper"),
  ACME_p = safe("ACME_p"),

  ADE = safe("ADE"),
  ADE_lower = safe("ADE_lower"),
  ADE_upper = safe("ADE_upper"),
  ADE_p = safe("ADE_p"),

  total_effect = safe("total_effect"),
  total_lower = safe("total_lower"),
  total_upper = safe("total_upper"),
  total_p = safe("total_p"),

  prop_mediated = safe("prop_mediated"),

  mediator_r2_marginal = safe("mediator_r2_marginal"),
  mediator_r2_conditional = safe("mediator_r2_conditional"),
  outcome_r2_marginal = safe("outcome_r2_marginal"),
  outcome_r2_conditional = safe("outcome_r2_conditional"),

  icc_mediator = safe("icc_mediator"),
  icc_outcome = safe("icc_outcome"),

  # Model diagnostics: residuals
  m_resid_mean = safe("m_resid_mean"),
  m_resid_sd = safe("m_resid_sd"),
  m_resid_skew = safe("m_resid_skew"),
  m_resid_kurt = safe("m_resid_kurt"),
  m_shapiro_p = safe("m_shapiro_p"),
  y_resid_mean = safe("y_resid_mean"),
  y_resid_sd = safe("y_resid_sd"),
  y_resid_skew = safe("y_resid_skew"),
  y_resid_kurt = safe("y_resid_kurt"),
  y_shapiro_p = safe("y_shapiro_p"),

  n_phi = safe("n_phi"),
  used_weights = if (!is.null(result$used_weights)) result$used_weights else FALSE,
  use_pca = if (!is.null(result$use_pca)) result$use_pca else FALSE,

  # Tobit mediator diagnostics
  tobit_sigma = safe("tobit_sigma"),
  pct_zero_mediator = safe("pct_zero_mediator"),

  error_msg = safe("error_msg"),
  stringsAsFactors = FALSE
)

# Save results
if (opt$model == "qr") {
  model_suffix <- sprintf("qr_tau%s", gsub("\\.", "", sprintf("%.2f", opt$quantile)))
} else {
  model_suffix <- "lmer"
}

covariate_suffix <- if (opt$`use-pca`) "pca" else "phi"
output_file <- file.path(opt$`output-dir`,
                         sprintf("mediation_%s_%s_%s_%dmin_%s_offset%d.csv",
                                 model_suffix,
                                 covariate_suffix,
                                 tolower(gsub("_", "", opt$dataset)),
                                 opt$minutes,
                                 tolower(opt$meal),
                                 opt$offset))

write_csv(result_df, output_file)

# Save actual residuals alongside the main results
if (!is.null(result$residuals_df)) {
  residuals_file <- sub("\\.csv$", "_residuals.csv", output_file)
  write_csv(result$residuals_df, residuals_file)
  cat(sprintf("Residuals saved to: %s\n", residuals_file))
}

# ================================================================
# PRINT SUMMARY
# ================================================================

cat("\n================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("================================================================\n")
cat(sprintf("Status: %s\n", result$status))

if (result$status == "success") {
  cat(sprintf("\nMEDIATION EFFECTS (treatment offset: +%d g carbs):\n", opt$offset))
  cat(sprintf("  ACME (indirect): %.4f [%.4f, %.4f] (p=%.4f)\n",
              result$ACME, result$ACME_lower, result$ACME_upper, result$ACME_p))
  cat(sprintf("  ADE (direct):    %.4f [%.4f, %.4f] (p=%.4f)\n",
              result$ADE, result$ADE_lower, result$ADE_upper, result$ADE_p))
  cat(sprintf("  Total effect:    %.4f [%.4f, %.4f] (p=%.4f)\n",
              result$total_effect, result$total_lower, result$total_upper, result$total_p))
  cat(sprintf("  Prop. mediated:  %.1f%%\n", 100 * result$prop_mediated))

  cat(sprintf("\nINTERPRETATION:\n"))
  cat(sprintf("  A +%d g increase in carbs leads to:\n", opt$offset))
  cat(sprintf("    - Direct effect on glucose: %.2f mg/dL (at %d min)\n",
              result$ADE, opt$minutes))
  cat(sprintf("    - Indirect effect (via insulin): %.2f mg/dL\n", result$ACME))
  cat(sprintf("    - Total effect: %.2f mg/dL\n", result$total_effect))

  cat(sprintf("\nMODEL FIT:\n"))
  cat(sprintf("  Mediator model: Tobit (survreg, sigma=%.4f, %.1f%% zeros)\n",
              result$tobit_sigma, result$pct_zero_mediator))
  icc_y_val <- if (!is.null(result$icc_outcome) && !is.na(result$icc_outcome)) {
    sprintf("%.1f%%", result$icc_outcome * 100)
  } else {
    "N/A"
  }
  cat(sprintf("  Outcome ICC (between-subject variance): %s\n", icc_y_val))

} else if (result$status == "error") {
  cat(sprintf("Error: %s\n", result$error_msg))
}

cat(sprintf("\nResults saved to: %s\n", output_file))
