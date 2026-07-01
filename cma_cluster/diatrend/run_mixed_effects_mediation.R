#!/usr/bin/env Rscript
# DiaTrend causal mediation — faithful port of the OhioT1DM pipeline.
#
# Decomposes the effect of meal carbs (treatment Z) on the postprandial
# glucose excursion DG(t) = G(t) - G(0) into ACME (mediated through the
# meal bolus M) and ADE (direct), following Imai et al. via the
# `mediation` package.
#
# Model structure MATCHES the manuscript / OhioT1DM code:
#   * Mediator: Tobit (survreg), left-censored at 0 — ~12% of meals have
#     zero bolus, so a Gaussian/lmer mediator is mis-specified.
#   * Outcome (primary, --model lm): linear model used for mediate(); an
#     lmer with random subject intercept is fit alongside for ICC and the
#     manuscript's LMER coefficient table (mediate() cannot pair a survreg
#     mediator with an lmer outcome).
#   * Outcome (robustness, --model qr): quantile regression at --quantile.
#   * Model covariates: the first --n-phi PCs of phi PLUS iob_at_meal
#     (pre-treatment) [+ demographics]. glucose_at_meal, meal_type, and
#     cohort are deliberately EXCLUDED from the models — they are balanced
#     in npCBPS instead; conditioning on glucose_at_meal (the baseline of
#     the change-score outcome) flips ACME positive (collider bias).
#   * Treatment contrast: carbs are centered at a group median
#     (--contrast-anchor: global / mealtype / subject / subject-mealtype),
#     so mediate(control.value=0, treat.value=offset) evaluates a
#     "+offset g above the group's typical meal" contrast. Effects do NOT
#     scale linearly (the Tobit mediator is non-linear), so each offset is
#     evaluated through mediate(), not by scaling.

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
  library(survival)   # survreg (Tobit mediator)
  library(quantreg)   # rq (quantile-regression outcome)
  library(lme4)       # lmer (ICC + manuscript LMER coefficient table only)
  library(mediation)
})


option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Phi/PC embeddings CSV (unit weights). Used when --weights-file is not set."
  ),
  make_option(c("--weights-file"), type = "character", default = NULL,
    help = "Weighted CSV produced by npcbps_weights.R (cohort/IOB filter already applied)."
  ),
  make_option(c("--cohort"), type = "character", default = "2",
    help = "Cohort filter when --phi-file is supplied. [default: 2]"
  ),
  make_option(c("--meal"), type = "character", default = "ALL",
    help = "Restrict to one meal type (breakfast/lunch/dinner/snack) or ALL. [default: ALL]"
  ),
  make_option(c("--split"), type = "character", default = "all",
    help = paste(
      "Restrict to one within-subject temporal split: 'test' for the",
      "OhioT1DM-matched held-out analysis, 'train', or 'all' (no filter).",
      "Requires a 'split' column in the input CSV. [default: all]"
    )
  ),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Include iob_at_meal as a (pre-treatment) covariate in the models. [default: TRUE]"
  ),
  make_option(c("--demographics"), type = "logical", default = FALSE,
    help = "Add demo_age + factor(demo_sex) + demo_hba1c to the models. [default: FALSE]"
  ),
  make_option(c("--timepoint"), type = "integer", default = 90,
    help = "Outcome timepoint in minutes (column Y_<t>min). [default: 90]"
  ),
  make_option(c("--offset"), type = "integer", default = 30,
    help = "Treatment contrast: +offset g above the anchor median. [default: 30]"
  ),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype",
    help = "Median anchor for centering carbs: global / mealtype / subject / subject-mealtype. [default: mealtype]"
  ),
  make_option(c("--model"), type = "character", default = "lmer",
    help = paste(
      "Model family for mediate(): 'lmer' (lmer mediator + lmer outcome,",
      "random subject intercept — DiaTrend-appropriate, ~0% zero bolus),",
      "'lm' (Tobit mediator + lm outcome, OhioT1DM-literal), or 'qr'",
      "(Tobit mediator + quantile-regression outcome). [default: lmer]"
    )
  ),
  make_option(c("--quantile"), type = "double", default = 0.5,
    help = "Quantile tau for --model qr. [default: 0.5]"
  ),
  make_option(c("--n-phi"), type = "integer", default = 3,
    help = "Number of CLAE components in the MODELS (manuscript: 3 PCs). [default: 3]"
  ),
  make_option(c("--covariate-prefix"), type = "character", default = "PC",
    help = "Column prefix for CLAE covariates ('PC' or 'phi'). [default: PC]"
  ),
  make_option(c("--sims"), type = "integer", default = 1000,
    help = "Monte Carlo simulations for mediate(). [default: 1000]"
  ),
  make_option(c("--use-weights"), type = "logical", default = TRUE,
    help = "Use the cbps_weight column when fitting the models. [default: TRUE]"
  ),
  make_option(c("--output-dir"), type = "character", default = NULL,
    help = "Output directory. Defaults to CONFIG$DIATREND_MEDIATION_RESULTS_DIR."
  ),
  make_option(c("--run-id"), type = "character", default = NULL,
    help = "Filename identifier. Defaults to a timestamp."
  ),
  make_option(c("--arm-tag"), type = "character", default = "",
    help = "Tag added to filenames (e.g. 'base', 'demographics')."
  )
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$`phi-file`) && is.null(opt$`weights-file`)) {
  stop("Provide either --phi-file or --weights-file.")
}
if (!opt$model %in% c("lmer", "lm", "qr")) stop("--model must be 'lmer', 'lm', or 'qr'.")
if (!opt$`contrast-anchor` %in% c("global", "mealtype", "subject", "subject-mealtype")) {
  stop("--contrast-anchor must be global / mealtype / subject / subject-mealtype.")
}
if (is.null(opt$`output-dir`)) opt$`output-dir` <- CONFIG$DIATREND_MEDIATION_RESULTS_DIR
if (is.null(opt$`run-id`)) opt$`run-id` <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)


# ---------------------------------------------------------------------------
# LOAD + FILTER
# ---------------------------------------------------------------------------
load_and_filter <- function(opt) {
  if (!is.null(opt$`weights-file`)) {
    cat("Loading weighted CSV:", opt$`weights-file`, "\n")
    df <- read.csv(opt$`weights-file`, stringsAsFactors = FALSE)
    if (!"cbps_weight" %in% colnames(df)) {
      warning("--weights-file has no cbps_weight column; using unit weights.")
      df$cbps_weight <- 1
    }
    return(df)
  }
  cat("Loading phi embeddings:", opt$`phi-file`, "\n")
  df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
  cohorts <- as.integer(strsplit(opt$cohort, ",")[[1]])
  df <- df[df$cohort %in% cohorts, ]
  cat("  After cohort filter (", paste(cohorts, collapse = ","), "): ",
      nrow(df), " episodes.\n", sep = "")
  if (isTRUE(opt$`use-iob`)) {
    df <- df[!is.na(df$iob_at_meal), ]
    cat("  After IOB filter:", nrow(df), "episodes.\n")
  }
  df$cbps_weight <- 1
  df
}

df <- load_and_filter(opt)

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
if (!outcome_col %in% colnames(df)) {
  stop(sprintf("Outcome column '%s' not found in input.", outcome_col))
}
df$Y <- df[[outcome_col]]

phi_cols <- if (opt$`n-phi` > 0) {
  paste0(opt$`covariate-prefix`, "_", seq_len(opt$`n-phi`))
} else {
  character(0)
}
missing_phi <- setdiff(phi_cols, colnames(df))
if (length(missing_phi) > 0) {
  stop(sprintf("Covariate columns missing: %s", paste(missing_phi, collapse = ", ")))
}

# Model covariates: PC + IOB [+ demographics]. NOTE: glucose_at_meal,
# meal_type, and cohort are intentionally absent — balanced via npCBPS only.
covariate_terms <- phi_cols
if (isTRUE(opt$`use-iob`)) covariate_terms <- c(covariate_terms, "iob_at_meal")
if (isTRUE(opt$demographics)) {
  demo_needed <- c("demo_age", "demo_sex", "demo_hba1c")
  missing_demo <- setdiff(demo_needed, colnames(df))
  if (length(missing_demo) > 0) {
    stop(sprintf("--demographics set but columns missing: %s. Run merge_demographics.py first.",
                 paste(missing_demo, collapse = ", ")))
  }
  n_before <- nrow(df)
  df <- df[stats::complete.cases(df[, demo_needed]), ]
  if (nrow(df) < n_before) {
    cat(sprintf("  Dropped %d episode(s) with missing demographics.\n", n_before - nrow(df)))
  }
  covariate_terms <- c(covariate_terms, "demo_age", "factor(demo_sex)", "demo_hba1c")
}

# ---------------------------------------------------------------------------
# TREATMENT CONTRAST: center carbs at a group median, contrast 0 -> +offset.
# ---------------------------------------------------------------------------
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
anchor <- opt$`contrast-anchor`
anchor_med <- switch(anchor,
  global           = rep(median(df$treat_meal_carbs, na.rm = TRUE), nrow(df)),
  mealtype         = med_by(df$treat_meal_carbs, df$meal_type),
  subject          = med_by(df$treat_meal_carbs, df$subject_id),
  `subject-mealtype` = med_by(df$treat_meal_carbs, interaction(df$subject_id, df$meal_type))
)
df$treat_centered <- df$treat_meal_carbs - anchor_med
control_value <- 0
treat_value <- opt$offset
cat(sprintf("\nContrast anchor: %s | control=0 (group median), treat=+%d g (centered)\n",
            anchor, opt$offset))

weights_arg <- if (isTRUE(opt$`use-weights`)) df$cbps_weight else rep(1, nrow(df))

# ---------------------------------------------------------------------------
# MEDIATOR + OUTCOME MODELS (branch on --model)
#   lmer : lmer mediator + lmer outcome, both with a random subject intercept.
#          DiaTrend-appropriate: ~0% zero bolus (no Tobit needed) and large
#          between-subject variance (ICC ~ 0.13) that lm would leave as
#          confounding, flipping the mediator sign.
#   lm   : Tobit (survreg) mediator + lm outcome (OhioT1DM-literal mean level);
#          a companion lmer is fit for ICC + LMER coefficients.
#   qr   : Tobit mediator + quantile-regression outcome at --quantile.
# ---------------------------------------------------------------------------
rhs <- paste(covariate_terms, collapse = " + ")
n_zero <- sum(df$mediator_bolus_for_meal == 0, na.rm = TRUE)
cat(sprintf("\nMediator zero-inflation: %d/%d (%.1f%%) zero bolus\n",
            n_zero, nrow(df), 100 * n_zero / nrow(df)))

icc_y <- NA_real_
lmer_outcome <- NULL

if (opt$model == "lmer") {
  mediator_formula <- as.formula(paste(
    "mediator_bolus_for_meal ~ treat_centered +", rhs, "+ (1 | subject_id)"))
  outcome_formula <- as.formula(paste(
    "Y ~ treat_centered + mediator_bolus_for_meal +", rhs, "+ (1 | subject_id)"))
  cat("Mediator model (LMER): "); print(mediator_formula)
  cat("Outcome model (LMER):  "); print(outcome_formula)
  model.m <- lme4::lmer(mediator_formula, data = df, weights = weights_arg, REML = FALSE)
  model.y <- lme4::lmer(outcome_formula, data = df, weights = weights_arg, REML = FALSE)
  vc <- as.data.frame(lme4::VarCorr(model.y))
  icc_y <- vc$vcov[vc$grp != "Residual"][1] / sum(vc$vcov)
} else {
  mediator_formula <- as.formula(paste(
    "Surv(mediator_bolus_for_meal, mediator_bolus_for_meal > 0, type='left') ~ treat_centered +", rhs))
  outcome_formula <- as.formula(paste("Y ~ treat_centered + mediator_bolus_for_meal +", rhs))
  cat("Mediator model (Tobit): "); print(mediator_formula)
  cat("Outcome model: "); print(outcome_formula)
  model.m <- survreg(mediator_formula, data = df, weights = weights_arg, dist = "gaussian")
  if (opt$model == "lm") {
    model.y <- lm(outcome_formula, data = df, weights = weights_arg)
    lmer_formula <- as.formula(paste(
      "Y ~ treat_centered + mediator_bolus_for_meal +", rhs, "+ (1 | subject_id)"))
    lmer_outcome <- tryCatch(
      lme4::lmer(lmer_formula, data = df, weights = weights_arg, REML = FALSE),
      error = function(e) NULL)
    if (!is.null(lmer_outcome)) {
      vc <- as.data.frame(lme4::VarCorr(lmer_outcome))
      icc_y <- vc$vcov[vc$grp != "Residual"][1] / sum(vc$vcov)
    }
  } else {  # qr
    model.y <- rq(outcome_formula, tau = opt$quantile, data = df,
                  weights = weights_arg, model = TRUE)
  }
}

# ---------------------------------------------------------------------------
# DIAGNOSTICS: coefficient significance + R^2 per model
# ---------------------------------------------------------------------------
extract_coefs <- function(model) {
  cl <- class(model)[1]
  if (cl == "survreg") {
    tab <- summary(model)$table
    tab <- tab[rownames(tab) != "Log(scale)", , drop = FALSE]
  } else if (cl == "lm") {
    tab <- summary(model)$coefficients
  } else if (cl == "rq") {
    tab <- summary(model, se = "nid")$coefficients
  } else if (cl %in% c("lmerMod", "lmerModLmerTest")) {
    ct <- summary(model)$coefficients
    tval <- ct[, "t value"]
    return(data.frame(term = rownames(ct), estimate = ct[, "Estimate"],
                      std_error = ct[, "Std. Error"], stat = tval,
                      p_value = 2 * stats::pnorm(-abs(tval)), row.names = NULL))
  } else {
    stop("unsupported model class: ", cl)
  }
  data.frame(term = rownames(tab), estimate = tab[, 1], std_error = tab[, 2],
             stat = tab[, 3], p_value = tab[, 4], row.names = NULL)
}

model_r2 <- function(model, tau = NULL) {
  cl <- class(model)[1]
  if (cl == "lm") {
    s <- summary(model); c(r2 = unname(s$r.squared), adj_r2 = unname(s$adj.r.squared))
  } else if (cl == "rq") {
    rho <- function(u, t) sum(u * (t - (u < 0)))
    y <- model$y; res <- y - model$fitted.values
    c(pseudo_r2 = 1 - rho(res, tau) / rho(y - as.numeric(quantile(y, tau)), tau), adj_r2 = NA_real_)
  } else if (cl %in% c("lmerMod", "lmerModLmerTest")) {
    vc <- as.data.frame(lme4::VarCorr(model))
    var_f <- stats::var(as.vector(stats::model.matrix(model) %*% lme4::fixef(model)))
    var_r <- sum(vc$vcov[vc$grp != "Residual"]); var_e <- vc$vcov[vc$grp == "Residual"]
    tot <- var_f + var_r + var_e
    c(marginal = unname(var_f / tot), conditional = unname((var_f + var_r) / tot))
  } else {
    c(r2 = NA_real_)
  }
}

print_diag <- function(label, coefs, r2, extra = "") {
  cat("\n", strrep("-", 64), "\n", sep = "")
  cat(label, "\n", sep = "")
  if (length(r2)) {
    cat("  ", paste(sprintf("%s=%.4f", names(r2), r2), collapse = " | "), "\n", sep = "")
  }
  if (nzchar(extra)) cat("  ", extra, "\n", sep = "")
  d <- coefs
  d$signif <- ifelse(d$p_value < 0.001, "***",
              ifelse(d$p_value < 0.01, "**",
              ifelse(d$p_value < 0.05, "*",
              ifelse(d$p_value < 0.1, ".", ""))))
  d$estimate <- round(d$estimate, 4); d$std_error <- round(d$std_error, 4)
  d$stat <- round(d$stat, 3); d$p_value <- signif(d$p_value, 3)
  print(d, row.names = FALSE)
}

med_coefs <- extract_coefs(model.m)
out_coefs <- extract_coefs(model.y)
out_r2 <- model_r2(model.y, tau = opt$quantile)

is_tobit <- class(model.m)[1] == "survreg"
med_r2 <- if (!is_tobit) model_r2(model.m) else c()
print_diag(if (is_tobit) "Mediator model (Tobit / survreg)" else "Mediator model (LMER)",
           med_coefs, med_r2,
           if (is_tobit) sprintf("scale = %.4f", model.m$scale) else "")

out_label <- if (opt$model == "qr") {
  sprintf("Outcome model (QR tau=%.2f)", opt$quantile)
} else if (opt$model == "lmer") {
  "Outcome model (LMER)"
} else {
  "Outcome model (lm)"
}
print_diag(out_label, out_coefs, out_r2,
           if (!is.na(icc_y)) sprintf("subject ICC = %.3f", icc_y) else "")

lmer_coefs <- NULL; lmer_r2 <- NULL
if (!is.null(lmer_outcome)) {
  lmer_coefs <- extract_coefs(lmer_outcome)
  lmer_r2 <- model_r2(lmer_outcome)
  print_diag("Outcome model (companion LMER, random subject intercept)", lmer_coefs, lmer_r2)
}

# ---------------------------------------------------------------------------
# MEDIATION
# ---------------------------------------------------------------------------
cat(sprintf("\nRunning mediation (model=%s, sims=%d, contrast 0 -> +%d g)...\n",
            opt$model, opt$sims, opt$offset))
t0 <- Sys.time()
med_result <- mediate(
  model.m = model.m,
  model.y = model.y,
  treat = "treat_centered",
  mediator = "mediator_bolus_for_meal",
  control.value = control_value,
  treat.value = treat_value,
  boot = FALSE,
  sims = opt$sims
)
cat("  mediate() done in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s.\n")

# No treat x mediator interaction -> INT=FALSE, so use d0/z0/n0/tau.coef and
# strip the named CI scalars ("2.5%"/"97.5%") via as.numeric.
results <- list(
  arm_tag = opt$`arm-tag`,
  meal = opt$meal,
  split = opt$split,
  model = opt$model,
  quantile_tau = if (opt$model == "qr") opt$quantile else NA_real_,
  contrast_anchor = anchor,
  offset_g = opt$offset,
  n_episodes = nrow(df),
  n_subjects = length(unique(df$subject_id)),
  cohorts = sort(unique(df$cohort)),
  use_iob = isTRUE(opt$`use-iob`),
  demographics = isTRUE(opt$demographics),
  use_weights = isTRUE(opt$`use-weights`),
  timepoint_min = opt$timepoint,
  n_phi = opt$`n-phi`,
  sims = opt$sims,
  outcome_icc = icc_y,
  acme = c(est = as.numeric(med_result$d0), lo = as.numeric(med_result$d0.ci[1]),
           hi = as.numeric(med_result$d0.ci[2]), p = as.numeric(med_result$d0.p)),
  ade = c(est = as.numeric(med_result$z0), lo = as.numeric(med_result$z0.ci[1]),
          hi = as.numeric(med_result$z0.ci[2]), p = as.numeric(med_result$z0.p)),
  total = c(est = as.numeric(med_result$tau.coef), lo = as.numeric(med_result$tau.ci[1]),
            hi = as.numeric(med_result$tau.ci[2]), p = as.numeric(med_result$tau.p)),
  prop_mediated = c(est = as.numeric(med_result$n0), lo = as.numeric(med_result$n0.ci[1]),
                    hi = as.numeric(med_result$n0.ci[2]), p = as.numeric(med_result$n0.p)),
  mediator_model_coefs = med_coefs,
  mediator_model_r2 = med_r2,
  outcome_model_coefs = out_coefs,
  outcome_model_r2 = out_r2,
  lmer_outcome_coefs = lmer_coefs,
  lmer_outcome_r2 = lmer_r2,
  summary_text = capture.output(summary(med_result))
)

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
cohort_tag <- paste(results$cohorts, collapse = "")
iob_tag <- if (results$use_iob) "iob" else "noiob"
model_tag <- if (opt$model == "qr") sprintf("qr%02d", round(opt$quantile * 100)) else opt$model
meal_tag <- tolower(opt$meal)
arm_tag <- if (nzchar(opt$`arm-tag`)) paste0("_", opt$`arm-tag`) else ""
base <- sprintf("mediation_diatrend_c%s_%s_%s_%s_t%d_off%d%s_%s",
                cohort_tag, meal_tag, iob_tag, model_tag, opt$timepoint, opt$offset, arm_tag, opt$`run-id`)
rds_file <- file.path(opt$`output-dir`, paste0(base, ".rds"))
csv_file <- file.path(opt$`output-dir`, paste0(base, ".csv"))
coefs_file <- file.path(opt$`output-dir`, paste0(base, "_model_coefs.csv"))

saveRDS(results, rds_file)

bind_coefs <- function(tag, coefs) if (is.null(coefs)) NULL else cbind(model = tag, coefs)
all_coefs <- do.call(rbind, Filter(Negate(is.null), list(
  bind_coefs("mediator_tobit", med_coefs),
  bind_coefs(paste0("outcome_", model_tag), out_coefs),
  bind_coefs("outcome_lmer", lmer_coefs)
)))
write.csv(all_coefs, coefs_file, row.names = FALSE)

summary_df <- data.frame(
  quantity = c("ACME", "ADE", "Total effect", "Prop. mediated"),
  estimate = c(results$acme["est"], results$ade["est"], results$total["est"], results$prop_mediated["est"]),
  ci_lower = c(results$acme["lo"], results$ade["lo"], results$total["lo"], results$prop_mediated["lo"]),
  ci_upper = c(results$acme["hi"], results$ade["hi"], results$total["hi"], results$prop_mediated["hi"]),
  p_value  = c(results$acme["p"], results$ade["p"], results$total["p"], results$prop_mediated["p"])
)
write.csv(summary_df, csv_file, row.names = FALSE)

cat("\n", strrep("=", 64), "\n", sep = "")
cat("DIATREND MEDIATION RESULTS\n")
cat(strrep("=", 64), "\n", sep = "")
cat(sprintf("  Arm / model:   %s / %s%s\n",
            if (nzchar(opt$`arm-tag`)) opt$`arm-tag` else "(unlabeled)",
            opt$model, if (opt$model == "qr") sprintf(" (tau=%.2f)", opt$quantile) else ""))
cat(sprintf("  Cohorts:       %s | IOB in model: %s | demographics: %s\n",
            cohort_tag, results$use_iob, results$demographics))
cat(sprintf("  Episodes:      %d (%d subjects)\n", results$n_episodes, results$n_subjects))
cat(sprintf("  Contrast:      +%d g above %s median | timepoint %d min\n",
            opt$offset, anchor, opt$timepoint))
cat(sprintf("  ACME:          %.4f [%.4f, %.4f]  p = %.4f\n",
            results$acme["est"], results$acme["lo"], results$acme["hi"], results$acme["p"]))
cat(sprintf("  ADE:           %.4f [%.4f, %.4f]  p = %.4f\n",
            results$ade["est"], results$ade["lo"], results$ade["hi"], results$ade["p"]))
cat(sprintf("  Total effect:  %.4f [%.4f, %.4f]  p = %.4f\n",
            results$total["est"], results$total["lo"], results$total["hi"], results$total["p"]))
cat(sprintf("  Prop. mediated: %.4f [%.4f, %.4f]  p = %.4f\n",
            results$prop_mediated["est"], results$prop_mediated["lo"],
            results$prop_mediated["hi"], results$prop_mediated["p"]))
cat("\nOutputs:\n")
cat("  ", rds_file, "\n", sep = "")
cat("  ", csv_file, "\n", sep = "")
cat("  ", coefs_file, "\n", sep = "")
