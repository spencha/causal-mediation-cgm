#!/usr/bin/env Rscript
# ================================================================
# RUN CAUSAL MEDIATION ANALYSIS FOR ALL TIME POINTS
# ================================================================
# Wrapper that calls run_mixed_effects_mediation.R for each time point.
# Supports lmer (mixed-effects) and qr (quantile regression) models.
#
# Usage examples:
#   Rscript run_all_timepoints_lmer.R                    # lmer, +30g, every 5 min
#   Rscript run_all_timepoints_lmer.R --model qr --quantile 0.5
#   Rscript run_all_timepoints_lmer.R --offsets 15,30,45
#   Rscript run_all_timepoints_lmer.R --run-all          # all models, quantiles, offsets
# ================================================================

library(optparse)

# --- START CONFIG BLOCK ---
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    normalizePath(dirname(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
})

config_path <- file.path(script_dir, "..", "config.R")
if (!file.exists(config_path)) {
  config_path <- file.path(getwd(), "cma_cluster", "config.R")
}
if (file.exists(config_path)) {
  source(config_path)
} else {
  stop("Could not find config.R")
}
# --- END CONFIG BLOCK ---

# Command line arguments
option_list <- list(
  make_option(c("--phi-file"), type="character", default=NULL,
              help="Path to phi embeddings CSV file"),
  make_option(c("--weights"), type="character", default="embedded",
              help="Weights: 'embedded', path to file, or 'none'"),
  make_option(c("--output-dir"), type="character", default=NULL,
              help="Output directory for results (default: mediation_results)"),
  make_option(c("--sims"), type="integer", default=1000,
              help="Number of quasi-Bayesian simulations [default: 1000]"),
  make_option(c("--meal"), type="character", default="ALL",
              help="Meal type: 'ALL', 'breakfast', 'lunch', or 'dinner' [default: ALL]"),
  make_option(c("-n", "--n-phi"), type="integer", default=3,
              help="Number of phi/PC features to use as covariates [default: 3]"),
  make_option(c("--use-pca"), action="store_true", default=FALSE,
              help="Use PC_ columns instead of phi_ columns [default: FALSE]"),
  make_option(c("-m", "--model"), type="character", default="lmer",
              help="Outcome model: 'lmer' or 'qr' [default: lmer]"),
  make_option(c("-q", "--quantile"), type="numeric", default=0.5,
              help="Quantile for QR model (0-1) [default: 0.5]"),
  make_option(c("--offsets"), type="character", default="30",
              help="Comma-separated treatment offsets in grams [default: 30]"),
  make_option(c("--dataset"), type="character", default="2020_TEST",
              help="Dataset: 2018, 2020, 2020_TEST, 2020_TRAIN, combined [default: 2020_TEST]"),
  make_option(c("--run-all"), action="store_true", default=FALSE,
              help="Run all combinations: lmer + qr(0.25,0.5,0.75,0.95), offsets 15,30,45"),
  make_option(c("--cohort"), type="character", default=NULL,
              help="Filter to specific cohort: '2018', '2020', or NULL for all (default: NULL)")
)

opt <- parse_args(OptionParser(option_list=option_list))

# Strip any stray quotes from string arguments (can happen from SLURM --export quoting)
opt$meal <- gsub('^["\']|["\']$', '', opt$meal)
opt$dataset <- gsub('^["\']|["\']$', '', opt$dataset)

# Set default output directory if not specified
if (is.null(opt$`output-dir`)) {
  opt$`output-dir` <- CONFIG$MEDIATION_RESULTS_DIR
}

# Create output directory
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

# Define time points: every 5 minutes from 60 to 210
TIME_POINTS <- seq(60, 210, by = 5)

# Parse offsets
offsets <- as.integer(strsplit(opt$offsets, ",")[[1]])

# Build list of model configurations to run
if (opt$`run-all`) {
  # Run all combinations
  configs <- list(
    list(model = "lmer", quantile = NA),
    list(model = "qr", quantile = 0.25),
    list(model = "qr", quantile = 0.50),
    list(model = "qr", quantile = 0.75),
    list(model = "qr", quantile = 0.95)
  )
  offsets <- c(15, 30, 45)
  cat(">>> --run-all mode: running all model/quantile/offset combinations\n")
} else {
  configs <- list(
    list(model = opt$model, quantile = if (opt$model == "qr") opt$quantile else NA)
  )
}

# Print configuration
cat("\n================================================================\n")
cat("CAUSAL MEDIATION ANALYSIS - ALL TIME POINTS\n")
cat("================================================================\n")
cat(sprintf("Phi file: %s\n", ifelse(is.null(opt$`phi-file`), "default", opt$`phi-file`)))
cat(sprintf("Weights: %s\n", opt$weights))
cat(sprintf("Dataset: %s\n", opt$dataset))
cat(sprintf("Output dir: %s\n", opt$`output-dir`))
cat(sprintf("Simulations: %d\n", opt$sims))
cat(sprintf("Meal type: %s\n", opt$meal))
cat(sprintf("Phi features: %d\n", opt$`n-phi`))
cat(sprintf("Use PCA: %s\n", ifelse(opt$`use-pca`, "YES", "NO")))
cat(sprintf("Time points: %d-%d min (every 5 min, %d total)\n",
            min(TIME_POINTS), max(TIME_POINTS), length(TIME_POINTS)))
cat(sprintf("Offsets: %s g\n", paste(offsets, collapse = ", ")))
cat(sprintf("Model configs: %d\n", length(configs)))
for (i in seq_along(configs)) {
  cfg <- configs[[i]]
  if (cfg$model == "qr") {
    cat(sprintf("  [%d] QR (tau=%.2f)\n", i, cfg$quantile))
  } else {
    cat(sprintf("  [%d] LMER\n", i))
  }
}

# Count total jobs
n_total_jobs <- length(TIME_POINTS) * length(offsets) * length(configs)
cat(sprintf("Total jobs: %d timepoints x %d offsets x %d models = %d\n",
            length(TIME_POINTS), length(offsets), length(configs), n_total_jobs))
cat("================================================================\n\n")

# Get the directory of this script (robust method)
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg))))
  }
  script_path <- tryCatch({
    normalizePath(sys.frame(1)$ofile)
  }, error = function(e) NULL)
  if (!is.null(script_path)) {
    return(dirname(script_path))
  }
  candidates <- c(file.path(getwd(), "cma_cluster"), getwd())
  for (dir in candidates) {
    if (file.exists(file.path(dir, "run_mixed_effects_mediation.R"))) {
      return(dir)
    }
  }
  return(getwd())
}

script_dir <- get_script_dir()
base_cmd <- file.path(script_dir, "run_mixed_effects_mediation.R")

if (!file.exists(base_cmd)) {
  stop(sprintf("Cannot find run_mixed_effects_mediation.R at: %s", base_cmd))
}

cat(sprintf("Using mediation script: %s\n\n", base_cmd))

# Run all configurations
results <- list()
job_num <- 0

for (cfg in configs) {
  for (offset in offsets) {
    for (minutes in TIME_POINTS) {
      job_num <- job_num + 1

      if (cfg$model == "qr") {
        label <- sprintf("QR(tau=%.2f) offset=%dg t=%dmin", cfg$quantile, offset, minutes)
      } else {
        label <- sprintf("LMER offset=%dg t=%dmin", offset, minutes)
      }

      cat(sprintf("\n>>> [%d/%d] %s\n", job_num, n_total_jobs, label))

      # Build command
      cmd <- sprintf("Rscript %s --minutes %d --bootstrap %d --output-dir %s --n-phi %d --offset %d --dataset %s --model %s",
                     shQuote(base_cmd), minutes, opt$sims, shQuote(opt$`output-dir`),
                     opt$`n-phi`, offset, opt$dataset, cfg$model)

      if (cfg$model == "qr") {
        cmd <- paste(cmd, sprintf("--quantile %.2f", cfg$quantile))
      }

      if (!is.null(opt$`phi-file`)) {
        cmd <- paste(cmd, "--phi-file", shQuote(opt$`phi-file`))
      }

      if (opt$weights != "embedded") {
        cmd <- paste(cmd, "--weights", shQuote(opt$weights))
      }

      if (opt$meal != "ALL") {
        cmd <- paste(cmd, "--meal", shQuote(opt$meal))
      }

      if (opt$`use-pca`) {
        cmd <- paste(cmd, "--use-pca")
      }

      if (!is.null(opt$cohort)) {
        cmd <- paste(cmd, "--cohort", opt$cohort)
      }

      # Run the command
      exit_code <- system(cmd)

      key <- sprintf("%s_%s_offset%d_%dmin",
                     cfg$model,
                     if (cfg$model == "qr") sprintf("tau%.2f", cfg$quantile) else "",
                     offset, minutes)

      results[[key]] <- list(
        model = cfg$model,
        quantile = cfg$quantile,
        offset = offset,
        minutes = minutes,
        exit_code = exit_code,
        success = exit_code == 0
      )

      if (exit_code == 0) {
        cat(sprintf(">>> Completed successfully\n"))
      } else {
        cat(sprintf(">>> WARNING: failed with exit code %d\n", exit_code))
      }
    }
  }
}

# Summary
cat("\n================================================================\n")
cat("SUMMARY\n")
cat("================================================================\n")

n_success <- sum(sapply(results, function(x) x$success))
cat(sprintf("\nCompleted: %d/%d jobs\n", n_success, length(results)))

# Group by model config
for (cfg in configs) {
  for (offset in offsets) {
    if (cfg$model == "qr") {
      label <- sprintf("QR(tau=%.2f) +%dg", cfg$quantile, offset)
    } else {
      label <- sprintf("LMER +%dg", offset)
    }

    cfg_results <- Filter(function(x) {
      x$model == cfg$model && x$offset == offset &&
        (is.na(cfg$quantile) || identical(x$quantile, cfg$quantile))
    }, results)

    n_ok <- sum(sapply(cfg_results, function(x) x$success))
    n_tot <- length(cfg_results)
    cat(sprintf("  %s: %d/%d successful\n", label, n_ok, n_tot))
  }
}

# Combine results into CSV files
cat("\n>>> Combining results...\n")

library(readr)
library(dplyr)

# Build the correct subdirectory path (must match run_mixed_effects_mediation.R)
# Structure: mediation_results/<cov_mode>/<meal>/
covariate_subdir <- if (opt$`use-pca`) "pca" else "phi"
combined_output_dir <- file.path(opt$`output-dir`, covariate_subdir, opt$meal)
dir.create(combined_output_dir, recursive = TRUE, showWarnings = FALSE)

# Find all result files (both lmer and qr) — search recursively from base dir
meal_pattern <- tolower(opt$meal)
file_pattern <- sprintf("mediation_.*_%s_offset.*\\.csv$", meal_pattern)

result_files <- list.files(opt$`output-dir`,
                           pattern = file_pattern,
                           full.names = TRUE,
                           recursive = TRUE)

# Exclude combined files and residuals
result_files <- result_files[!grepl("all_timepoints|_residuals", result_files)]

cat(sprintf("Found %d result files\n", length(result_files)))

if (length(result_files) > 0) {
  combined <- bind_rows(lapply(result_files, read_csv, show_col_types = FALSE))

  # Deduplicate by (model, quantile_tau, minutes, treat_offset)
  combined <- combined %>%
    arrange(model, quantile_tau, treat_offset, minutes) %>%
    group_by(model, quantile_tau, minutes, treat_offset) %>%
    slice_tail(n = 1) %>%
    ungroup()

  # Save offset-specific combined files into the correct subdirectory
  unique_offsets <- sort(unique(combined$treat_offset))
  for (off in unique_offsets) {
    offset_df <- combined %>% filter(treat_offset == off)
    offset_file <- file.path(combined_output_dir,
                             sprintf("mediation_all_timepoints_%s_offset%dg.csv", tolower(opt$meal), off))
    write_csv(offset_df, offset_file)
    cat(sprintf("Saved offset-specific results: %s (%d rows)\n", offset_file, nrow(offset_df)))
  }

  # Also save combined file with all offsets
  combined_file <- file.path(combined_output_dir,
                             sprintf("mediation_all_timepoints_%s.csv", tolower(opt$meal)))
  write_csv(combined, combined_file)
  cat(sprintf("\nCombined results saved to: %s\n", combined_file))
  cat(sprintf("  Total rows: %d\n", nrow(combined)))

  # Print summary table for each model config
  for (cfg in configs) {
    for (offset in offsets) {
      if (cfg$model == "qr") {
        label <- sprintf("QR (tau=%.2f), +%dg carbs", cfg$quantile, offset)
        sub_df <- combined %>%
          filter(model == "qr",
                 abs(quantile_tau - cfg$quantile) < 0.01,
                 treat_offset == offset)
      } else {
        label <- sprintf("LMER, +%dg carbs", offset)
        sub_df <- combined %>%
          filter(model == "lmer", treat_offset == offset)
      }

      if (nrow(sub_df) == 0) next

      cat(sprintf("\n--- %s ---\n", label))
      cat(sprintf("%-8s %-6s %-15s %-15s %-15s %-8s\n",
                  "Minutes", "N", "ACME", "ADE", "Total", "% Med"))
      cat(paste(rep("-", 70), collapse=""), "\n")

      for (i in 1:nrow(sub_df)) {
        row <- sub_df[i, ]
        if (row$status == "success") {
          prop_med <- if (!is.na(row$prop_mediated)) sprintf("%.1f%%", row$prop_mediated * 100) else "NA"
          cat(sprintf("%-8d %-6d %-15.4f %-15.4f %-15.4f %-8s\n",
                      row$minutes, row$n_obs, row$ACME, row$ADE, row$total_effect, prop_med))
        }
      }
    }
  }
} else {
  cat("No result files found to combine.\n")
}

cat("\n================================================================\n")
cat("ALL JOBS COMPLETE\n")
cat("================================================================\n")
