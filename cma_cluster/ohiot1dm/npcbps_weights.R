#!/usr/bin/env Rscript
# ================================================================
# Non-parametric Covariate Balancing Propensity Score (npCBPS)
# ================================================================
# Purpose: Estimate weights that balance phi features between treatment groups
# This ensures A ⊥ M | φ assumption is better satisfied
# ================================================================
.libPaths("~/R/4.4.2-library")

library(CBPS)
library(dplyr)
library(readr)
library(tidyr)

# --- START CONFIG BLOCK ---
# Source configuration
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

# Find config.R (check multiple locations)
config_locations <- c(
  file.path(script_dir, "..", "config.R"),
  file.path(script_dir, "..", "..", "cma_cluster", "config.R"),
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

# Configuration
CONTINUOUS_TREATMENT <- TRUE  # Meal carbs is continuous
MEDIATOR_BALANCE <- TRUE      # Whether to balance for mediator model too

# ================================================================
# MAIN WEIGHT ESTIMATION FUNCTION
# ================================================================

estimate_balancing_weights <- function(phi_df,
                                       treatment_var = "treat_meal_carbs",
                                       mediator_var = "mediator_bolus_for_meal",
                                       phi_cols = NULL,
                                       output_dir = CONFIG$WEIGHTS_DIR,
                                       n_phi_features = 6,
                                       use_pca = FALSE) {

  cat("\n========================================\n")
  cat("COVARIATE BALANCING WEIGHT ESTIMATION\n")
  cat("========================================\n\n")

  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Identify phi/PC columns if not specified
  if (is.null(phi_cols)) {
    if (use_pca) {
      phi_cols <- grep("^PC_", names(phi_df), value = TRUE)
      cat("Using PCA components for balancing (orthogonal features)\n")
    } else {
      phi_cols <- grep("^phi_", names(phi_df), value = TRUE)
    }
  }

  # Select specified number of phi features to avoid overfitting
  n_phi_use <- min(n_phi_features, length(phi_cols))
  phi_cols_use <- phi_cols[1:n_phi_use]

  cat(sprintf("Using %d phi features for balancing (requested: %d, available: %d)\n",
              n_phi_use, n_phi_features, length(phi_cols)))
  
  # Check if cohort column is available (for combined datasets)
  has_cohort <- "cohort" %in% names(phi_df) && length(unique(phi_df$cohort)) > 1
  if (has_cohort) {
    cat("Cohort column detected - including as balancing covariate\n")
    phi_df$cohort <- as.factor(phi_df$cohort)
  }

  # Prepare data - ensure complete cases for all covariates
  select_cols <- c(treatment_var, mediator_var, phi_cols_use,
                   "glucose_at_meal", "meal_type", "global_window_id")
  if (has_cohort) select_cols <- c(select_cols, "cohort")

  data_for_weights <- phi_df %>%
    dplyr::select(all_of(select_cols)) %>%
    filter(!is.na(!!sym(treatment_var)),
           !is.na(!!sym(mediator_var)),
           !is.na(glucose_at_meal)) %>%
    # Remove rows with NA in any phi columns
    filter(complete.cases(.[phi_cols_use]))

  n_obs <- nrow(data_for_weights)
  cat(sprintf("Total observations after removing NAs: %d\n\n", n_obs))

  # ----------------------------------------------------------------
  # TREATMENT MODEL WEIGHTS (for outcome model)
  # ----------------------------------------------------------------

  cat("Estimating treatment balancing weights...\n")

  # Create formula for npCBPS (include cohort if available)
  covariates <- c(phi_cols_use, "glucose_at_meal")
  if (has_cohort) covariates <- c(covariates, "cohort")
  formula_str <- paste(treatment_var, "~", paste(covariates, collapse = " + "))
  cbps_formula <- as.formula(formula_str)
  
  # Estimate weights using npCBPS for continuous treatment
  tryCatch({
    if (CONTINUOUS_TREATMENT) {
      # For continuous treatment, use npCBPS
      cbps_fit <- npCBPS(cbps_formula, 
                         data = data_for_weights,
                         method = "exact",  # Use exact moment matching
                         print.level = 0)
      
      treatment_weights <- cbps_fit$weights
      
    } else {
      # For binary treatment (if discretized)
      data_for_weights$treat_binary <- as.numeric(
        data_for_weights[[treatment_var]] > median(data_for_weights[[treatment_var]])
      )
      
      binary_formula <- update(cbps_formula, treat_binary ~ .)
      cbps_fit <- CBPS(binary_formula,
                       data = data_for_weights,
                       ATT = 0,  # ATE not ATT
                       method = "exact")
      
      treatment_weights <- cbps_fit$weights
    }
    
    # Normalize weights
    treatment_weights <- treatment_weights * n_obs / sum(treatment_weights)
    
    # ADDED: Cap extreme weights to improve ESS
    # Winsorize at 99th percentile
    weight_cap <- quantile(treatment_weights, 0.99)
    treatment_weights <- pmin(treatment_weights, weight_cap)
    
    # Re-normalize after capping
    treatment_weights <- treatment_weights * n_obs / sum(treatment_weights)
    
    cat(sprintf("  Weight range: [%.3f, %.3f]\n", 
                min(treatment_weights), max(treatment_weights)))
    cat(sprintf("  Weight SD: %.3f\n", sd(treatment_weights)))
    
    # Check effective sample size
    ess_treatment <- sum(treatment_weights)^2 / sum(treatment_weights^2)
    cat(sprintf("  Effective sample size: %.1f (%.1f%% of original)\n\n", 
                ess_treatment, 100 * ess_treatment / n_obs))
    
  }, error = function(e) {
    cat("  Warning: npCBPS failed, using uniform weights\n")
    cat(sprintf("  Error: %s\n", e$message))
    treatment_weights <- rep(1, n_obs)
  })
  
  # ----------------------------------------------------------------
  # MEDIATOR MODEL WEIGHTS (optional)
  # ----------------------------------------------------------------
  
  mediator_weights <- rep(1, n_obs)
  
  if (MEDIATOR_BALANCE) {
    cat("Estimating mediator balancing weights...\n")
    cat("(This helps ensure M model is well-specified)\n")
    
    # For mediator model, we want to balance phi with respect to A
    # This helps with the first stage of mediation
    mediator_covs <- c(phi_cols_use, "glucose_at_meal")
    if (has_cohort) mediator_covs <- c(mediator_covs, "cohort")
    mediator_formula_str <- paste(mediator_var, "~",
                                  paste(mediator_covs, collapse = " + "))
    mediator_formula <- as.formula(mediator_formula_str)
    
    tryCatch({
      # Use simpler CBPS for mediator weights to avoid overfitting
      mediator_cbps <- npCBPS(mediator_formula,
                              data = data_for_weights,
                              method = "over",  # Over-identified for stability
                              print.level = 0)
      
      mediator_weights <- mediator_cbps$weights
      
      # Cap extreme weights
      weight_cap_med <- quantile(mediator_weights, 0.99)
      mediator_weights <- pmin(mediator_weights, weight_cap_med)
      
      mediator_weights <- mediator_weights * n_obs / sum(mediator_weights)
      
      ess_mediator <- sum(mediator_weights)^2 / sum(mediator_weights^2)
      cat(sprintf("  Effective sample size: %.1f (%.1f%% of original)\n\n", 
                  ess_mediator, 100 * ess_mediator / n_obs))
      
    }, error = function(e) {
      cat("  Warning: Mediator weight estimation failed, using uniform\n")
      mediator_weights <- rep(1, n_obs)
    })
  }
  
  # ----------------------------------------------------------------
  # CHECK BALANCE IMPROVEMENT - FIXED VERSION
  # ----------------------------------------------------------------
  
  cat("Checking covariate balance...\n")
  
  # FIXED Function to compute standardized mean difference
  compute_smd <- function(covariate, treatment, weights = NULL) {
    # Remove NAs from all vectors together
    complete_idx <- !is.na(covariate) & !is.na(treatment)
    cov_clean <- covariate[complete_idx]
    treat_clean <- treatment[complete_idx]
    
    if (is.null(weights)) {
      weights_clean <- rep(1, length(cov_clean))
    } else {
      weights_clean <- weights[complete_idx]
    }
    
    # Check if we have enough data
    if (length(cov_clean) < 2) {
      return(NA)
    }
    
    # Weighted correlation for continuous treatment
    tryCatch({
      weighted_corr <- cov.wt(cbind(cov_clean, treat_clean), 
                              wt = weights_clean, cor = TRUE)$cor[1,2]
      return(abs(weighted_corr))
    }, error = function(e) {
      return(NA)
    })
  }
  
  # Check balance before and after weighting
  balance_results <- data.frame(
    covariate = covariates,
    corr_before = NA,
    corr_after_treatment = NA,
    corr_after_mediator = NA
  )
  
  for (i in seq_along(covariates)) {
    cov_values <- data_for_weights[[covariates[i]]]
    treat_values <- data_for_weights[[treatment_var]]
    
    balance_results$corr_before[i] <- compute_smd(cov_values, treat_values)
    balance_results$corr_after_treatment[i] <- compute_smd(cov_values, treat_values, 
                                                           treatment_weights)
    if (MEDIATOR_BALANCE) {
      balance_results$corr_after_mediator[i] <- compute_smd(cov_values, treat_values,
                                                            mediator_weights)
    }
  }
  
  # Remove NAs from summary statistics
  balance_results_clean <- balance_results[!is.na(balance_results$corr_before), ]
  
  # Print balance summary
  cat("\nBalance Summary (absolute correlations with treatment):\n")
  cat("----------------------------------------------------\n")
  cat(sprintf("%-20s %10s %10s\n", "Covariate", "Before", "After"))
  cat("----------------------------------------------------\n")
  
  for (i in 1:nrow(balance_results_clean)) {
    cat(sprintf("%-20s %10.3f %10.3f\n",
                balance_results_clean$covariate[i],
                balance_results_clean$corr_before[i],
                balance_results_clean$corr_after_treatment[i]))
  }
  
  cat("----------------------------------------------------\n")
  cat(sprintf("%-20s %10.3f %10.3f\n",
              "Mean |correlation|",
              mean(balance_results_clean$corr_before, na.rm = TRUE),
              mean(balance_results_clean$corr_after_treatment, na.rm = TRUE)))
  
  improvement <- 100 * (mean(balance_results_clean$corr_before, na.rm = TRUE) - 
                          mean(balance_results_clean$corr_after_treatment, na.rm = TRUE)) / 
    mean(balance_results_clean$corr_before, na.rm = TRUE)
  
  cat(sprintf("\nBalance improvement: %.1f%%\n", improvement))
  
  # ----------------------------------------------------------------
  # WARNING ABOUT LOW ESS
  # ----------------------------------------------------------------
  
  if (ess_treatment < 0.3 * n_obs) {
    cat("\n⚠️  WARNING: Effective sample size is very low!\n")
    cat("   Consider:\n")
    cat("   1. Using fewer covariates for balancing\n")
    cat("   2. Using a less aggressive balancing method\n")
    cat("   3. Trimming extreme weights\n")
  }
  
  # ----------------------------------------------------------------
  # SAVE WEIGHTS
  # ----------------------------------------------------------------

  # Combine all weights with original data
  # Include split column if available to ensure unique matching
  weights_df <- data.frame(
    global_window_id = data_for_weights$global_window_id,
    meal_type = data_for_weights$meal_type,
    treatment_weight = treatment_weights,
    mediator_weight = mediator_weights
  )

  # Add split column if available (important for combined train+test data)
  if ("split" %in% names(data_for_weights)) {
    weights_df$split <- data_for_weights$split
  }
  
  # Save weights
  output_file <- file.path(output_dir, "npCBPS_weights.csv")
  write_csv(weights_df, output_file)
  cat(sprintf("\nWeights saved to: %s\n", output_file))
  
  # Save balance diagnostics
  balance_file <- file.path(output_dir, "balance_diagnostics.csv")
  write_csv(balance_results, balance_file)
  cat(sprintf("Balance diagnostics saved to: %s\n", balance_file))
  
  # ----------------------------------------------------------------
  # STRATIFIED WEIGHTS (by meal type)
  # ----------------------------------------------------------------
  
  if (length(unique(data_for_weights$meal_type)) > 1) {
    cat("\n========================================\n")
    cat("STRATIFIED WEIGHT ESTIMATION\n")
    cat("========================================\n\n")
    
    stratified_weights <- list()
    
    for (meal in unique(data_for_weights$meal_type)) {
      cat(sprintf("\nMeal type: %s\n", meal))
      cat("------------------------\n")
      
      meal_data <- data_for_weights %>% filter(meal_type == meal)
      n_meal <- nrow(meal_data)
      
      if (n_meal < 25) {
        cat(sprintf("  Insufficient data (n=%d), using uniform weights\n", n_meal))
        meal_weights <- rep(1, n_meal)
      } else {
        tryCatch({
          # Use even simpler formula for stratified weights
          strat_covariates <- c(phi_cols_use, "glucose_at_meal")
          if (has_cohort) strat_covariates <- c(strat_covariates, "cohort")
          formula_str <- paste(treatment_var, "~", paste(strat_covariates, collapse = " + "))
          cbps_formula <- as.formula(formula_str)
          
          meal_cbps <- npCBPS(cbps_formula, 
                              data = meal_data,
                              method = "exact",
                              #coprior = .1/nrow(meal_data),
                              print.level = 0)
          
          meal_weights <- meal_cbps$weights
          
          # Cap extreme weights
          weight_cap_meal <- quantile(meal_weights, 0.95)  # More aggressive capping for strata
          meal_weights <- pmin(meal_weights, weight_cap_meal)
          
          meal_weights <- meal_weights * n_meal / sum(meal_weights)
          
          ess_meal <- sum(meal_weights)^2 / sum(meal_weights^2)
          cat(sprintf("  n=%d, ESS=%.1f (%.1f%%)\n", 
                      n_meal, ess_meal, 100*ess_meal/n_meal))
          
        }, error = function(e) {
          cat("  Weight estimation failed, using uniform\n")
          meal_weights <- rep(1, n_meal)
        })
      }
      
      stratified_weights[[meal]] <- data.frame(
        global_window_id = meal_data$global_window_id,
        stratified_weight = meal_weights
      )
    }

    # Combine stratified weights
    stratified_df <- bind_rows(stratified_weights)

    # Merge with main weights - stratified_weight is used for meal-specific analysis
    weights_df <- weights_df %>%
      left_join(stratified_df, by = "global_window_id")

    # Save updated weights with stratified column
    write_csv(weights_df, output_file)
    cat(sprintf("\nUpdated weights with stratification saved\n"))
  }
  
  return(weights_df)
}

# ================================================================
# COMMAND LINE ARGUMENTS
# ================================================================

library(optparse)

option_list <- list(
  make_option(c("-p", "--phi-file"), type="character", default=NULL,
              help="Path to phi embeddings CSV file"),
  make_option(c("-o", "--output-dir"), type="character", default=NULL,
              help="Output directory for weights (default: CONFIG$WEIGHTS_DIR)"),
  make_option(c("-s", "--split"), type="character", default="test",
              help="Filter to specific split before weight estimation: 'train', 'test', or 'all' (default: test)"),
  make_option(c("-c", "--cohort"), type="character", default=NULL,
              help="Filter to specific cohort before weight estimation: '2018', '2020', or NULL for all"),
  make_option(c("-n", "--n-phi"), type="integer", default=6,
              help="Number of phi/PC features to use for balancing (default: 6)"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PC_ columns instead of phi_ columns for balancing (default: FALSE)")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# ================================================================
# RUN WEIGHT ESTIMATION
# ================================================================

# Set paths from configuration
ANALYSIS_DIR <- CONFIG$ANALYSIS_DATA_DIR
RESULTS_DIR <- file.path(CONFIG$BASE_DIR, "cma_results")

# Load data
cat("Loading data...\n")

# Determine phi file path
if (!is.null(opt$`phi-file`) && file.exists(opt$`phi-file`)) {
  phi_file <- opt$`phi-file`
  cat(sprintf("Using custom phi file: %s\n", phi_file))
} else {
  # Look for embeddings in the new location first
  embeddings_dir <- file.path(CONFIG$ANALYSIS_DATA_DIR, "embeddings")

  # Search for combined embedding files (preferred for CMA)
  combined_files <- list.files(embeddings_dir,
                               pattern = "^phi_embeddings_combined_.*\\.csv$",
                               full.names = TRUE)

  if (length(combined_files) > 0) {
    # Use the most recently modified file
    file_info <- file.info(combined_files)
    phi_file <- combined_files[which.max(file_info$mtime)]
    cat(sprintf("Found embedding file: %s\n", phi_file))
  } else {
    # Fallback to old location/naming
    phi_file <- file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_y_delta_glucose_phi_embeddings_causal.csv")
    cat(sprintf("Using legacy phi file: %s\n", phi_file))
  }
}

phi_df <- read_csv(phi_file, show_col_types = FALSE)
cat(sprintf("Loaded %d observations from phi file\n", nrow(phi_df)))

# ================================================================
# FILTER DATA BASED ON COMMAND LINE OPTIONS
# ================================================================

# Filter by split if specified (default is 'test')
if (!is.null(opt$split) && tolower(opt$split) != "all") {
  if (!"split" %in% names(phi_df)) {
    stop("--split option specified but 'split' column not found in phi file")
  }

  n_before <- nrow(phi_df)
  phi_df <- phi_df[phi_df$split == opt$split, ]
  n_after <- nrow(phi_df)

  cat(sprintf("Filtered to split='%s': %d -> %d observations\n",
              opt$split, n_before, n_after))

  if (n_after == 0) {
    stop(sprintf("No observations remaining after filtering to split='%s'", opt$split))
  }
} else if (tolower(opt$split) == "all") {
  cat("Using all observations (no split filter)\n")
}

# Filter by cohort if specified
if (!is.null(opt$cohort)) {
  if (!"cohort" %in% names(phi_df)) {
    stop("--cohort option specified but 'cohort' column not found in phi file")
  }

  n_before <- nrow(phi_df)
  phi_df <- phi_df[phi_df$cohort == opt$cohort, ]
  n_after <- nrow(phi_df)

  cat(sprintf("Filtered to cohort='%s': %d -> %d observations\n",
              opt$cohort, n_before, n_after))

  if (n_after == 0) {
    stop(sprintf("No observations remaining after filtering to cohort='%s'", opt$cohort))
  }
}

# Print final sample size
cat(sprintf("\nFinal sample for weight estimation: %d observations\n", nrow(phi_df)))

# Show breakdown by split/cohort if available
if ("split" %in% names(phi_df)) {
  cat("  Split breakdown: ")
  cat(paste(names(table(phi_df$split)), "=", table(phi_df$split), collapse = ", "))
  cat("\n")
}
if ("cohort" %in% names(phi_df)) {
  cat("  Cohort breakdown: ")
  cat(paste(names(table(phi_df$cohort)), "=", table(phi_df$cohort), collapse = ", "))
  cat("\n")
}

# ================================================================
# DETERMINE OUTPUT DIRECTORY
# ================================================================

if (!is.null(opt$`output-dir`)) {
  output_dir <- opt$`output-dir`
} else {
  # Default to weights directory (separate from embeddings)
  output_dir <- CONFIG$WEIGHTS_DIR
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("Saving weights to: %s\n", output_dir))
}

# Make sure we have glucose_at_meal
if (!"glucose_at_meal" %in% names(phi_df)) {
  # Try to load from RData file
  rdata_file <- file.path(ANALYSIS_DIR, "z_meal_mediation_analysis_data_2018_5min.RData")
  if (file.exists(rdata_file)) {
    load(rdata_file)
    idx <- match(phi_df$global_window_id, all_cleaned_data$window_id)
    phi_df$glucose_at_meal <- all_cleaned_data$glucose_at_meal[idx]
  } else {
    cat("Warning: glucose_at_meal not found in phi_df and RData file not available\n")
    cat("Setting glucose_at_meal to 0 (this may affect balance)\n")
    phi_df$glucose_at_meal <- 0
  }
}

# ================================================================
# ESTIMATE WEIGHTS
# ================================================================

# Get number of phi features to use from command line option
n_phi_use <- opt$`n-phi`
use_pca <- opt$`use-pca`

if (use_pca) {
  cat(sprintf("\nUsing %d PC features for balancing (orthogonal, from --use-pca)\n", n_phi_use))
} else {
  cat(sprintf("\nUsing %d phi features for balancing (from --n-phi option)\n", n_phi_use))
}

weights_df <- estimate_balancing_weights(
  phi_df,
  output_dir = output_dir,
  n_phi_features = n_phi_use,
  use_pca = use_pca
)

cat("\n✅ Weight estimation complete!\n")