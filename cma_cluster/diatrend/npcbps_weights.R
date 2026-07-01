#!/usr/bin/env Rscript
# DiaTrend: estimate non-parametric CBPS weights to balance phi
# covariates across treatment levels.
#
# Reads a phi-embeddings CSV produced by
#   `python ae_python_code/train_and_export_embeddings.py --dataset diatrend`
# and writes a balanced-weights CSV that the mediation script consumes.
#
# DiaTrend-specific choices (see handoff Section 8):
#   - Treatment: meal carbs (treat_meal_carbs), continuous.
#   - Covariates: phi_1..phi_<n_phi>, glucose_at_meal, cohort,
#     and (when --use-iob TRUE) iob_at_meal. Meal type is included
#     as a fixed effect downstream; it is NOT in the propensity model.
#   - Cohort filter follows the analysis arm: --cohort 2 for the
#     37-subject IOB-adjusted primary; --cohort 1,2 for the full
#     54-subject robustness arm.

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
    stop("Could not find config.R. Run from project root or set CAUSAL_AE_BASE_DIR.")
  }

  library(optparse)
  library(dplyr)
  library(CBPS)
})


option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Path to the phi embeddings CSV (output of train_and_export_embeddings.py --dataset diatrend)."
  ),
  make_option(c("--cohort"), type = "character", default = "2",
    help = "Cohort filter. '2' for primary 37-subject arm; '1,2' for full 54-subject arm. [default: 2]"
  ),
  make_option(c("--balance-cohort"), type = "logical", default = TRUE,
    help = paste(
      "When >1 cohort, add a 'cohort' indicator to the propensity model.",
      "Set FALSE for the full-cohort demographics arm: cohort is nearly",
      "collinear with age/HbA1c (cohort 2 = young, A1c>7.5%), so balancing on",
      "both is singular -- the demographics already capture the cohort",
      "difference. [default: TRUE]"
    )
  ),
  make_option(c("--split"), type = "character", default = "all",
    help = paste(
      "Filter to one within-subject temporal split: 'test' for the",
      "OhioT1DM-matched held-out analysis, 'train', or 'all' (no filter).",
      "Requires a 'split' column in the embeddings CSV. [default: all]"
    )
  ),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Include iob_at_meal in the propensity model. TRUE for primary / full-sample BOB; FALSE for no-IOB robustness. [default: TRUE]"
  ),
  make_option(c("--n-phi"), type = "integer", default = 6,
    help = "Number of covariate components to balance on (count of <prefix>_k). [default: 6]"
  ),
  make_option(c("--covariate-prefix"), type = "character", default = "phi",
    help = paste(
      "Column prefix for the CLAE covariates: 'phi' for raw latent dims,",
      "'PC' for PCA components. The manuscript uses PCA top-3 (--covariate-prefix",
      "PC --n-phi 3) on an 8-dim CLAE. [default: phi]"
    )
  ),
  make_option(c("--demographics"), type = "logical", default = FALSE,
    help = paste(
      "Include subject demographics (demo_age, factor(demo_sex), demo_hba1c)",
      "in the propensity model so the weights balance them too. Requires the",
      "input CSV to carry those columns (run merge_demographics.py). [default: FALSE]"
    )
  ),
  make_option(c("--output-dir"), type = "character", default = NULL,
    help = "Output directory. Defaults to CONFIG$DIATREND_WEIGHTS_DIR."
  ),
  make_option(c("--run-id"), type = "character", default = NULL,
    help = "Identifier for the output filename. Defaults to a timestamp."
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`phi-file`)) {
  stop("--phi-file is required.")
}
if (is.null(opt$`output-dir`)) {
  opt$`output-dir` <- CONFIG$DIATREND_WEIGHTS_DIR
}
if (is.null(opt$`run-id`)) {
  opt$`run-id` <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
}
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

cat("Loading phi embeddings:", opt$`phi-file`, "\n")
df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
cat("  Loaded", nrow(df), "episodes,", ncol(df), "columns.\n")

cohorts <- as.integer(strsplit(opt$cohort, ",")[[1]])
df <- df[df$cohort %in% cohorts, ]
cat("  After cohort filter (", paste(cohorts, collapse = ","), "):",
    nrow(df), "episodes.\n", sep = "")

if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df)) {
    stop(sprintf("--split %s requested but the embeddings CSV has no 'split' column. Re-export with --diatrend-test-frac > 0.", opt$split))
  }
  df <- df[df$split == opt$split, ]
  cat("  After split filter (", opt$split, "): ", nrow(df), " episodes.\n", sep = "")
}

if (isTRUE(opt$`use-iob`)) {
  n_before <- nrow(df)
  df <- df[!is.na(df$iob_at_meal), ]
  cat("  After IOB-availability filter:", nrow(df), "episodes (dropped",
      n_before - nrow(df), ").\n")
}

if (nrow(df) < 30) {
  stop(sprintf(
    "Only %d episodes survived filtering; CBPS requires more. Check inputs.",
    nrow(df)
  ))
}

phi_cols <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`n-phi`))
missing_phi <- setdiff(phi_cols, colnames(df))
if (length(missing_phi) > 0) {
  stop(sprintf(
    "Embeddings CSV is missing covariate columns: %s",
    paste(missing_phi, collapse = ", ")
  ))
}

covariate_cols <- c(phi_cols, "glucose_at_meal")
if (length(cohorts) > 1 && isTRUE(opt$`balance-cohort`)) {
  covariate_cols <- c(covariate_cols, "cohort")
  df$cohort <- as.factor(df$cohort)
}
if (isTRUE(opt$`use-iob`)) {
  covariate_cols <- c(covariate_cols, "iob_at_meal")
}
if (isTRUE(opt$demographics)) {
  demo_needed <- c("demo_age", "demo_sex", "demo_hba1c")
  missing_demo <- setdiff(demo_needed, colnames(df))
  if (length(missing_demo) > 0) {
    stop(sprintf(
      "--demographics set but columns missing: %s. Run merge_demographics.py first.",
      paste(missing_demo, collapse = ", ")
    ))
  }
  # npCBPS cannot fit with NA covariates; drop incomplete-demographic
  # episodes here (the same rows the mediation script drops downstream).
  n_before <- nrow(df)
  df <- df[stats::complete.cases(df[, demo_needed]), ]
  if (nrow(df) < n_before) {
    cat(sprintf("  Dropped %d episode(s) with missing demographics.\n",
                n_before - nrow(df)))
  }
  # Harmonize banded cohort-1 demographics (age/HbA1c stored as "40-49", ">9")
  # to numeric, so npCBPS balances ONE scalar moment per variable instead of a
  # ~per-subject factor (cohort-1 bands + cohort-2 numeric strings = ~one level
  # per subject, which over-fits and inflates weight variance). demo_sex stays
  # categorical. Matches decompose_demographics.R's parse_num.
  parse_num <- function(x) {
    if (is.numeric(x)) return(x)
    vapply(as.character(x), function(s) {
      nums <- as.numeric(unlist(regmatches(s, gregexpr("[0-9.]+", s))))
      if (length(nums) == 0) NA_real_ else mean(nums)
    }, numeric(1), USE.NAMES = FALSE)
  }
  df$demo_age <- parse_num(df$demo_age)
  df$demo_hba1c <- parse_num(df$demo_hba1c)
  df$demo_sex <- as.factor(df$demo_sex)
  covariate_cols <- c(covariate_cols, "demo_age", "demo_sex", "demo_hba1c")
}

rhs <- paste(covariate_cols, collapse = " + ")
fmla <- as.formula(paste("treat_meal_carbs ~", rhs))
cat("\nCBPS formula:\n  "); print(fmla)

cat("\nFitting non-parametric CBPS (continuous treatment)...\n")
t0 <- Sys.time()
cbps_fit <- npCBPS(fmla, data = df, corprior = 0.01)
cat("  npCBPS done in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s.\n")

df$cbps_weight <- cbps_fit$weights

# Normalize so weights sum to n_obs (matches OhioT1DM npcbps_weights.R:146).
n_obs <- nrow(df)
df$cbps_weight <- df$cbps_weight * n_obs / sum(df$cbps_weight)

# Cap extreme weights at the 99th percentile, matching OhioT1DM
# (npcbps_weights.R:148-154). This bounds the influence of extreme
# treatment values (e.g., rare 666 g meals) on the weighted regression
# without removing the underlying observations. Observations whose
# uncapped weight exceeds the 99th percentile retain influence equal
# to that of a "typical extreme" observation rather than dominating
# the fit. Re-normalize so weights still sum to n_obs.
weight_cap <- quantile(df$cbps_weight, 0.99)
n_capped <- sum(df$cbps_weight > weight_cap)
df$cbps_weight <- pmin(df$cbps_weight, weight_cap)
df$cbps_weight <- df$cbps_weight * n_obs / sum(df$cbps_weight)
cat(sprintf("  Capped %d weight(s) at the 99th percentile (cap = %.4f).\n",
            n_capped, weight_cap))
cat(sprintf("  Weight range after capping: [%.4f, %.4f], SD = %.4f.\n",
            min(df$cbps_weight), max(df$cbps_weight), sd(df$cbps_weight)))

# Effective sample size after capping.
ess <- sum(df$cbps_weight)^2 / sum(df$cbps_weight^2)
cat(sprintf("  Effective sample size: %.1f (%.1f%% of n_obs).\n",
            ess, 100 * ess / n_obs))

# Quick balance diagnostic: weighted vs. unweighted correlation of each
# covariate with the treatment. After balancing, weighted correlations
# should be near zero.
balance_rows <- lapply(covariate_cols, function(name) {
  # Coerce any non-numeric covariate (factor or character, e.g. banded
  # demo_age in cohort 1) to an integer code so the correlation is defined.
  x <- df[[name]]
  if (!is.numeric(x)) x <- as.integer(as.factor(x))
  unwt_r <- suppressWarnings(cor(x, df$treat_meal_carbs))
  wt_r <- suppressWarnings(cov.wt(cbind(x, df$treat_meal_carbs),
                                  wt = df$cbps_weight, cor = TRUE)$cor[1, 2])
  data.frame(covariate = name, unweighted_r = unwt_r, weighted_r = wt_r)
})
balance_df <- do.call(rbind, balance_rows)
cat("\nBalance diagnostics (correlation with treatment):\n")
print(balance_df, row.names = FALSE)

cohort_tag <- paste(cohorts, collapse = "")
iob_tag <- if (isTRUE(opt$`use-iob`)) "iob" else "noiob"
out_csv <- file.path(
  opt$`output-dir`,
  sprintf("npcbps_weights_c%s_%s_%s.csv", cohort_tag, iob_tag, opt$`run-id`)
)
out_balance <- file.path(
  opt$`output-dir`,
  sprintf("balance_c%s_%s_%s.csv", cohort_tag, iob_tag, opt$`run-id`)
)
write.csv(df, out_csv, row.names = FALSE)
write.csv(balance_df, out_balance, row.names = FALSE)

cat("\nWrote:\n")
cat("  Weighted episodes: ", out_csv, "\n")
cat("  Balance diagnostics:", out_balance, "\n")
cat("Next: Rscript cma_cluster/diatrend/run_mixed_effects_mediation.R \\\n")
cat("        --weights-file ", out_csv, " ...\n", sep = "")
