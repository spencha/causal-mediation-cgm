#!/usr/bin/env Rscript
# =============================================================================
# export_meal_windows_for_autoencoder.R
# =============================================================================
# Exports meal windows to CSV format for Python autoencoder training.
#
# Combines 2018 and 2020 data using the original OhioT1DM train/test splits:
#   - Combined TRAIN = 2018 TRAIN + 2020 TRAIN -> meal_windows_combined/train/
#   - Combined TEST  = 2018 TEST  + 2020 TEST  -> meal_windows_combined/test/
#
# Usage:
#   Rscript export_meal_windows_for_autoencoder.R
# =============================================================================

library(zoo)
library(dplyr)

# --- START CONFIG BLOCK ---
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

config_locations <- c(
  file.path(script_dir, "config.R"),
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

base_dir <- CONFIG$AE_CODE_DIR

# =============================================================================
# Helper Function
# =============================================================================

dump_windows <- function(rdata_path, out_dir, prefix, description) {
  if (!file.exists(rdata_path)) {
    warning(sprintf("RData file not found: %s", rdata_path))
    return(0)
  }

  load(rdata_path, verbose = FALSE)

  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  n_windows <- length(aligned_data_list)

  if (n_windows == 0) {
    warning(sprintf("No windows found in %s", rdata_path))
    return(0)
  }

  # Export each window
  for (wid in 1:n_windows) {
    wd <- aligned_data_list[[wid]]

    # Get global window ID
    if ("window_id" %in% names(all_cleaned_data)) {
      global_wid <- all_cleaned_data[wid, "window_id"]
    } else {
      global_wid <- all_cleaned_data[wid, 1]
    }

    # Add global_window_id column
    wd$global_window_id <- global_wid

    write.csv(
      wd,
      file = file.path(out_dir, sprintf("%s_window_%05d.csv", prefix, global_wid)),
      row.names = FALSE
    )
  }

  cat(sprintf("  %s: Wrote %d windows to %s\n", description, n_windows, out_dir))
  return(n_windows)
}

# =============================================================================
# Export Combined TRAIN (2018 TRAIN + 2020 TRAIN)
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("Exporting Combined Meal Windows for Autoencoder\n")
cat("=============================================================================\n")

# --- Combined TRAIN ---
cat("\n--- Combined TRAIN (2018 TRAIN + 2020 TRAIN) ---\n")

combined_train_dir <- CONFIG$MEAL_WINDOWS_COMBINED_TRAIN_DIR

# Clear existing files
if (dir.exists(combined_train_dir)) {
  existing_files <- list.files(combined_train_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(existing_files) > 0) {
    file.remove(existing_files)
    cat(sprintf("  Removed %d existing CSV files\n", length(existing_files)))
  }
}

n_2018_train <- dump_windows(
  rdata_path  = file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2018_TRAIN_5min.RData"),
  out_dir     = combined_train_dir,
  prefix      = "meal_combined",
  description = "2018 TRAIN"
)

n_2020_train <- dump_windows(
  rdata_path  = file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2020_TRAIN_5min.RData"),
  out_dir     = combined_train_dir,
  prefix      = "meal_combined",
  description = "2020 TRAIN"
)

cat(sprintf("  Combined TRAIN total: %d windows (%d from 2018, %d from 2020)\n",
            n_2018_train + n_2020_train, n_2018_train, n_2020_train))

# =============================================================================
# Export Combined TEST (2018 TEST + 2020 TEST)
# =============================================================================

cat("\n--- Combined TEST (2018 TEST + 2020 TEST) ---\n")

combined_test_dir <- CONFIG$MEAL_WINDOWS_COMBINED_TEST_DIR

# Clear existing files
if (dir.exists(combined_test_dir)) {
  existing_files <- list.files(combined_test_dir, pattern = "\\.csv$", full.names = TRUE)
  if (length(existing_files) > 0) {
    file.remove(existing_files)
    cat(sprintf("  Removed %d existing CSV files\n", length(existing_files)))
  }
}

n_2018_test <- dump_windows(
  rdata_path  = file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2018_TEST_5min.RData"),
  out_dir     = combined_test_dir,
  prefix      = "meal_combined",
  description = "2018 TEST"
)

n_2020_test <- dump_windows(
  rdata_path  = file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2020_TEST_5min.RData"),
  out_dir     = combined_test_dir,
  prefix      = "meal_combined",
  description = "2020 TEST"
)

cat(sprintf("  Combined TEST total: %d windows (%d from 2018, %d from 2020)\n",
            n_2018_test + n_2020_test, n_2018_test, n_2020_test))

# =============================================================================
# Summary
# =============================================================================

cat("\n")
cat("=============================================================================\n")
cat("Export Complete\n")
cat("=============================================================================\n")
cat(sprintf("Combined TRAIN: %d windows -> %s\n", n_2018_train + n_2020_train, combined_train_dir))
cat(sprintf("  - 2018: %d windows\n", n_2018_train))
cat(sprintf("  - 2020: %d windows\n", n_2020_train))
cat(sprintf("Combined TEST:  %d windows -> %s\n", n_2018_test + n_2020_test, combined_test_dir))
cat(sprintf("  - 2018: %d windows\n", n_2018_test))
cat(sprintf("  - 2020: %d windows\n", n_2020_test))
cat(sprintf("\nGrand total:    %d windows\n",
            n_2018_train + n_2020_train + n_2018_test + n_2020_test))
cat("=============================================================================\n")
