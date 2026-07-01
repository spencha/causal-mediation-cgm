#!/usr/bin/env Rscript
# DiaTrend causal mediation — sequential-ignorability sensitivity analysis.
#
# The ACME identified by run_mixed_effects_mediation.R rests on the
# (untestable) assumption of no unmeasured mediator-outcome confounding.
# This script quantifies how strong such confounding would have to be to
# explain away the estimated ACME, via Imai, Keele & Yamamoto's sensitivity
# parameter rho (the correlation between the mediator and outcome error
# terms), using mediation::medsens().
#
# medsens() ONLY supports lm/glm mediator and outcome models -- it cannot
# consume the lmer / survreg / quantile-regression fits used by the primary
# analysis. So this arm deliberately fits the OhioT1DM-literal specification:
#   * Mediator: lm(mediator ~ treat_centered + covariates)
#   * Outcome:  lm(Y ~ treat_centered + mediator + covariates)
# both with the same npCBPS weights, the same PC + IOB [+ demographics]
# covariate set, and the same group-median treatment contrast as the
# primary script. It is meant to be run on the held-out TEST split for the
# primary endpoint (e.g. pooled, 120 min, +30 g), not swept over the grid.
#
# Output: the rho at which ACME = 0, the corresponding R^2 product
# interpretations, the full summary text, and a CSV of the ACME-vs-rho
# sensitivity curve.

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
  library(mediation)
})


option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Phi/PC embeddings CSV (unit weights). Used when --weights-file is not set."),
  make_option(c("--weights-file"), type = "character", default = NULL,
    help = "Weighted CSV produced by npcbps_weights.R (cohort/IOB/split filter already applied)."),
  make_option(c("--cohort"), type = "character", default = "2",
    help = "Cohort filter when --phi-file is supplied. [default: 2]"),
  make_option(c("--split"), type = "character", default = "test",
    help = "Within-subject temporal split: 'test' (default), 'train', or 'all'."),
  make_option(c("--meal"), type = "character", default = "ALL",
    help = "Restrict to one meal type (breakfast/lunch/dinner/snack) or ALL. [default: ALL]"),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Include iob_at_meal as a (pre-treatment) covariate. [default: TRUE]"),
  make_option(c("--demographics"), type = "logical", default = FALSE,
    help = "Add demo_age + factor(demo_sex) + demo_hba1c to the models. [default: FALSE]"),
  make_option(c("--timepoint"), type = "integer", default = 120,
    help = "Outcome timepoint in minutes (column Y_<t>min). [default: 120]"),
  make_option(c("--offset"), type = "integer", default = 30,
    help = "Treatment contrast: +offset g above the anchor median. [default: 30]"),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype",
    help = "Median anchor: global / mealtype / subject / subject-mealtype. [default: mealtype]"),
  make_option(c("--n-phi"), type = "integer", default = 3,
    help = "Number of CLAE components in the models. [default: 3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC",
    help = "Column prefix for CLAE covariates ('PC' or 'phi'). [default: PC]"),
  make_option(c("--use-weights"), type = "logical", default = TRUE,
    help = "Use the cbps_weight column when fitting the models. [default: TRUE]"),
  make_option(c("--sims"), type = "integer", default = 1000,
    help = "Monte Carlo simulations for mediate(). [default: 1000]"),
  make_option(c("--rho-by"), type = "double", default = 0.05,
    help = "Step size of the rho sensitivity grid in medsens(). [default: 0.05]"),
  make_option(c("--output-dir"), type = "character", default = NULL,
    help = "Output directory. Defaults to CONFIG$DIATREND_MEDIATION_RESULTS_DIR."),
  make_option(c("--run-id"), type = "character", default = NULL,
    help = "Filename identifier. Defaults to a timestamp."),
  make_option(c("--arm-tag"), type = "character", default = "",
    help = "Tag added to filenames (e.g. 'base', 'demographics').")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`phi-file`) && is.null(opt$`weights-file`)) {
  stop("Provide either --phi-file or --weights-file.")
}
if (!opt$`contrast-anchor` %in% c("global", "mealtype", "subject", "subject-mealtype")) {
  stop("--contrast-anchor must be global / mealtype / subject / subject-mealtype.")
}
if (is.null(opt$`output-dir`)) opt$`output-dir` <- CONFIG$DIATREND_MEDIATION_RESULTS_DIR
if (is.null(opt$`run-id`)) opt$`run-id` <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)


# ---------------------------------------------------------------------------
# LOAD + FILTER (mirrors run_mixed_effects_mediation.R)
# ---------------------------------------------------------------------------
if (!is.null(opt$`weights-file`)) {
  cat("Loading weighted CSV:", opt$`weights-file`, "\n")
  df <- read.csv(opt$`weights-file`, stringsAsFactors = FALSE)
  if (!"cbps_weight" %in% colnames(df)) {
    warning("--weights-file has no cbps_weight column; using unit weights.")
    df$cbps_weight <- 1
  }
} else {
  cat("Loading phi embeddings:", opt$`phi-file`, "\n")
  df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
  cohorts <- as.integer(strsplit(opt$cohort, ",")[[1]])
  df <- df[df$cohort %in% cohorts, ]
  if (isTRUE(opt$`use-iob`)) df <- df[!is.na(df$iob_at_meal), ]
  df$cbps_weight <- 1
}

if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df)) {
    stop(sprintf("--split %s requested but the input CSV has no 'split' column. Re-export with --diatrend-test-frac > 0.", opt$split))
  }
  df <- df[df$split == opt$split, ]
  cat(sprintf("  After split filter (%s): %d episodes.\n", opt$split, nrow(df)))
}

if (toupper(opt$meal) != "ALL") {
  df <- df[tolower(df$meal_type) == tolower(opt$meal), ]
  cat(sprintf("  After meal filter (%s): %d episodes.\n", opt$meal, nrow(df)))
  if (nrow(df) < 30) stop(sprintf("Only %d episodes for meal '%s'; too few.", nrow(df), opt$meal))
}

outcome_col <- sprintf("Y_%dmin", opt$timepoint)
if (!outcome_col %in% colnames(df)) stop(sprintf("Outcome column '%s' not found.", outcome_col))
df$Y <- df[[outcome_col]]

phi_cols <- if (opt$`n-phi` > 0) paste0(opt$`covariate-prefix`, "_", seq_len(opt$`n-phi`)) else character(0)
missing_phi <- setdiff(phi_cols, colnames(df))
if (length(missing_phi) > 0) stop(sprintf("Covariate columns missing: %s", paste(missing_phi, collapse = ", ")))

covariate_terms <- phi_cols
if (isTRUE(opt$`use-iob`)) covariate_terms <- c(covariate_terms, "iob_at_meal")
if (isTRUE(opt$demographics)) {
  demo_needed <- c("demo_age", "demo_sex", "demo_hba1c")
  missing_demo <- setdiff(demo_needed, colnames(df))
  if (length(missing_demo) > 0) {
    stop(sprintf("--demographics set but columns missing: %s.", paste(missing_demo, collapse = ", ")))
  }
  df <- df[stats::complete.cases(df[, demo_needed]), ]
  covariate_terms <- c(covariate_terms, "demo_age", "factor(demo_sex)", "demo_hba1c")
}

# ---------------------------------------------------------------------------
# TREATMENT CONTRAST (group-median centering; identical to primary script)
# ---------------------------------------------------------------------------
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
anchor <- opt$`contrast-anchor`
anchor_med <- switch(anchor,
  global             = rep(median(df$treat_meal_carbs, na.rm = TRUE), nrow(df)),
  mealtype           = med_by(df$treat_meal_carbs, df$meal_type),
  subject            = med_by(df$treat_meal_carbs, df$subject_id),
  `subject-mealtype` = med_by(df$treat_meal_carbs, interaction(df$subject_id, df$meal_type))
)
df$treat_centered <- df$treat_meal_carbs - anchor_med
weights_arg <- if (isTRUE(opt$`use-weights`)) df$cbps_weight else rep(1, nrow(df))

# ---------------------------------------------------------------------------
# lm MEDIATOR + lm OUTCOME (medsens-compatible), mediate(), then medsens()
# ---------------------------------------------------------------------------
rhs <- paste(covariate_terms, collapse = " + ")
mediator_formula <- as.formula(paste("mediator_bolus_for_meal ~ treat_centered +", rhs))
outcome_formula  <- as.formula(paste("Y ~ treat_centered + mediator_bolus_for_meal +", rhs))
cat("Mediator model (lm): "); print(mediator_formula)
cat("Outcome model (lm):  "); print(outcome_formula)

model.m <- lm(mediator_formula, data = df, weights = weights_arg)
model.y <- lm(outcome_formula,  data = df, weights = weights_arg)

cat(sprintf("\nRunning mediate() (sims=%d, contrast 0 -> +%d g)...\n", opt$sims, opt$offset))
med_result <- mediate(
  model.m = model.m, model.y = model.y,
  treat = "treat_centered", mediator = "mediator_bolus_for_meal",
  control.value = 0, treat.value = opt$offset, boot = FALSE, sims = opt$sims
)
cat(sprintf("  ACME = %.4f [%.4f, %.4f]  p = %.4f\n",
            as.numeric(med_result$d0), as.numeric(med_result$d0.ci[1]),
            as.numeric(med_result$d0.ci[2]), as.numeric(med_result$d0.p)))

cat(sprintf("\nRunning medsens() (rho grid step = %.3f)...\n", opt$`rho-by`))
t0 <- Sys.time()
sens <- medsens(med_result, rho.by = opt$`rho-by`, effect.type = "indirect", sims = opt$sims)
cat("  medsens() done in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s.\n")

sens_summary <- summary(sens)
cat("\n", strrep("=", 64), "\n", sep = "")
cat("SEQUENTIAL-IGNORABILITY SENSITIVITY (ACME via rho)\n")
cat(strrep("=", 64), "\n", sep = "")
print(sens_summary)

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
cohort_tag <- if (!is.null(opt$`weights-file`)) "w" else gsub(",", "", opt$cohort)
meal_tag <- tolower(opt$meal)
arm_tag <- if (nzchar(opt$`arm-tag`)) paste0("_", opt$`arm-tag`) else ""
base <- sprintf("medsens_diatrend_c%s_%s_%s_t%d_off%d%s_%s",
                cohort_tag, opt$split, meal_tag, opt$timepoint, opt$offset, arm_tag, opt$`run-id`)
txt_file <- file.path(opt$`output-dir`, paste0(base, "_summary.txt"))
curve_file <- file.path(opt$`output-dir`, paste0(base, "_rho_curve.csv"))

writeLines(capture.output(print(sens_summary)), txt_file)

# ACME-vs-rho sensitivity curve. Field names vary slightly across mediation
# versions; guard each extraction so the curve still writes if some are absent.
curve <- tryCatch({
  data.frame(
    rho = sens$rho,
    acme = sens$d0,
    acme_lo = if (!is.null(sens$lower.d0)) sens$lower.d0 else NA_real_,
    acme_hi = if (!is.null(sens$upper.d0)) sens$upper.d0 else NA_real_
  )
}, error = function(e) {
  warning("Could not assemble rho curve: ", conditionMessage(e)); NULL
})
if (!is.null(curve)) write.csv(curve, curve_file, row.names = FALSE)

# Rho / R^2 at which ACME = 0 (the headline sensitivity numbers).
err_cr <- tryCatch(sens_summary$err.cr, error = function(e) NA_real_)
r2_tilde <- tryCatch(sens_summary$R2star.prod, error = function(e) NA_real_)

cat("\nOutputs:\n")
cat("  ", txt_file, "\n", sep = "")
if (!is.null(curve)) cat("  ", curve_file, "\n", sep = "")
cat(sprintf("\nInterpretation: the ACME is nullified only if the mediator-outcome\n"))
cat(sprintf("error correlation rho reaches the value(s) in the summary above;\n"))
cat(sprintf("a larger |rho*| means the mediation finding is more robust to\n"))
cat(sprintf("unmeasured mediator-outcome confounding.\n"))
