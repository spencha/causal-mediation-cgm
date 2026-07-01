#!/usr/bin/env Rscript
# DiaTrend: run the primary mediation analysis plus the three Section
# 8.5-8.6 sensitivity arms in one invocation.
#
# Expected inputs (paths produced by `train_and_export_embeddings.py
# --dataset diatrend` runs):
#   --primary-phi-file     <path>   univariate CLAE, IOB-adjusted (cohort 2)
#   --multivariate-phi-file <path>  CLAE with (glucose, meal, bolus) channels
#                                   (Section 8.6 sensitivity)
#
# Arms:
#   primary       cohort 2, IOB on,  univariate CLAE
#   no-iob        cohort 2, IOB off, univariate CLAE
#   full-bob      cohorts 1+2, IOB on (kernel BOB for cohort 1),
#                 univariate CLAE
#   multivariate  cohort 2, IOB on,  multivariate CLAE input
#
# Each arm produces an .rds + .csv under
# CONFIG$DIATREND_MEDIATION_RESULTS_DIR. The arms share random-seed
# initialisation only — they are otherwise independent.

suppressPackageStartupMessages({
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

  config_locations <- c(
    file.path(script_dir, "..", "config.R"),
    file.path(script_dir, "..", "..", "cma_cluster", "config.R"),
    file.path(getwd(), "cma_cluster", "config.R")
  )
  for (cfg_path in config_locations) {
    if (file.exists(cfg_path)) {
      source(cfg_path)
      break
    }
  }
  if (!exists("CONFIG")) {
    stop("Could not find config.R.")
  }

  library(optparse)
})


option_list <- list(
  make_option(c("--primary-phi-file"), type = "character", default = NULL,
    help = "Phi embeddings CSV from the univariate CLAE (primary spec)."
  ),
  make_option(c("--multivariate-phi-file"), type = "character", default = NULL,
    help = "Phi embeddings CSV from the multivariate CLAE (Section 8.6)."
  ),
  make_option(c("--timepoint"), type = "integer", default = 90,
    help = "Outcome timepoint in minutes post-meal. [default: 90]"
  ),
  make_option(c("--sims"), type = "integer", default = 1000,
    help = "Monte Carlo simulations for mediation. [default: 1000]"
  ),
  make_option(c("--n-phi"), type = "integer", default = 6,
    help = "Number of covariate components as covariates (count of <prefix>_k). [default: 6]"
  ),
  make_option(c("--covariate-prefix"), type = "character", default = "phi",
    help = paste(
      "Column prefix for CLAE covariates ('phi' or 'PC'). Manuscript spec:",
      "--covariate-prefix PC --n-phi 3 on an 8-dim CLAE. [default: phi]"
    )
  ),
  make_option(c("--skip"), type = "character", default = "",
    help = "Comma-separated arms to skip (primary,no-iob,full-bob,multivariate)."
  ),
  make_option(c("--use-cbps"), type = "logical", default = FALSE,
    help = paste(
      "Compute npCBPS weights per arm (via npcbps_weights.R) and run the",
      "mediation weighted. FALSE runs unweighted (unit weights). [default: FALSE]"
    )
  ),
  make_option(c("--demographics"), type = "logical", default = FALSE,
    help = paste(
      "Include subject demographics (age, sex, HbA1c) as confounders in the",
      "propensity, mediator, and outcome models. Requires the phi CSVs to",
      "carry demo_* columns (run merge_demographics.py first). [default: FALSE]"
    )
  ),
  make_option(c("--weights-dir"), type = "character", default = NULL,
    help = "Directory for npCBPS weight CSVs. Defaults to CONFIG$DIATREND_WEIGHTS_DIR."
  ),
  make_option(c("--output-dir"), type = "character", default = NULL,
    help = "Output directory. Defaults to CONFIG$DIATREND_MEDIATION_RESULTS_DIR."
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`primary-phi-file`)) {
  stop("--primary-phi-file is required.")
}
if (is.null(opt$`output-dir`)) {
  opt$`output-dir` <- CONFIG$DIATREND_MEDIATION_RESULTS_DIR
}
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

skip_arms <- strsplit(opt$skip, ",")[[1]]
mediation_script <- file.path(script_dir, "run_mixed_effects_mediation.R")
if (!file.exists(mediation_script)) {
  stop(sprintf("run_mixed_effects_mediation.R not found at %s", mediation_script))
}
weights_script <- file.path(script_dir, "npcbps_weights.R")
if (isTRUE(opt$`use-cbps`) && !file.exists(weights_script)) {
  stop(sprintf("npcbps_weights.R not found at %s", weights_script))
}

weights_dir <- if (is.null(opt$`weights-dir`)) CONFIG$DIATREND_WEIGHTS_DIR else opt$`weights-dir`
if (isTRUE(opt$`use-cbps`)) {
  dir.create(weights_dir, recursive = TRUE, showWarnings = FALSE)
}
# Shared batch stamp so each arm's weight file has a stable, reconstructable
# name. npcbps_weights.R names its output npcbps_weights_c<cohorts>_<iobtag>_<run-id>.csv.
batch_run_id <- format(Sys.time(), "%Y-%m-%d_%H%M%S")

# Compute npCBPS weights for one arm and return the path to the weighted CSV.
cbps_weights_file <- function(phi_file, cohort, use_iob, arm) {
  run_id <- paste0(batch_run_id, "_", arm)
  args <- c(weights_script,
    "--phi-file", phi_file,
    "--cohort", cohort,
    "--use-iob", as.character(use_iob),
    "--n-phi", as.character(opt$`n-phi`),
    "--covariate-prefix", opt$`covariate-prefix`,
    "--demographics", as.character(opt$demographics),
    "--output-dir", weights_dir,
    "--run-id", run_id
  )
  cat("$ Rscript", paste(args, collapse = " "), "\n")
  status <- system2("Rscript", args)
  if (status != 0) {
    stop(sprintf("npCBPS weighting failed for arm '%s' (status %d)", arm, status))
  }
  cohort_tag <- paste(as.integer(strsplit(cohort, ",")[[1]]), collapse = "")
  iob_tag <- if (isTRUE(use_iob)) "iob" else "noiob"
  file.path(weights_dir,
            sprintf("npcbps_weights_c%s_%s_%s.csv", cohort_tag, iob_tag, run_id))
}

run_arm <- function(name, phi_file, cohort, use_iob) {
  if (name %in% skip_arms) {
    cat("\n[skip]", name, "\n")
    return(invisible(NULL))
  }
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("ARM:", name, "\n")
  cat(strrep("=", 60), "\n", sep = "")

  if (isTRUE(opt$`use-cbps`)) {
    wfile <- cbps_weights_file(phi_file, cohort, use_iob, name)
    input_args <- c("--weights-file", wfile, "--use-weights", "TRUE")
  } else {
    input_args <- c("--phi-file", phi_file, "--use-weights", "FALSE")
  }
  args <- c(input_args,
    "--cohort", cohort,
    "--use-iob", as.character(use_iob),
    "--demographics", as.character(opt$demographics),
    "--timepoint", as.character(opt$timepoint),
    "--sims", as.character(opt$sims),
    "--n-phi", as.character(opt$`n-phi`),
    "--covariate-prefix", opt$`covariate-prefix`,
    "--output-dir", opt$`output-dir`,
    "--arm-tag", name
  )
  cat("$ Rscript", mediation_script, paste(args, collapse = " "), "\n")
  status <- system2("Rscript", c(mediation_script, args))
  if (status != 0) {
    warning(sprintf("Arm '%s' exited with status %d", name, status))
  }
  invisible(status)
}


run_arm("primary",  opt$`primary-phi-file`, "2",   TRUE)
run_arm("no-iob",   opt$`primary-phi-file`, "2",   FALSE)
run_arm("full-bob", opt$`primary-phi-file`, "1,2", TRUE)

if (!is.null(opt$`multivariate-phi-file`)) {
  run_arm("multivariate", opt$`multivariate-phi-file`, "2", TRUE)
} else {
  cat("\n[note] --multivariate-phi-file not provided; skipping the Section 8.6 arm.\n")
  cat("       Build the multivariate embeddings via:\n")
  cat("         python ae_python_code/train_and_export_embeddings.py \\\n")
  cat("           --dataset diatrend --diatrend-features glucose,meal,bolus\n")
}

cat("\nDone. Results in:", opt$`output-dir`, "\n")
