#!/usr/bin/env Rscript
# Exploratory: does the DiaTrend mediation miss insulin-effect HETEROGENEITY /
# INTERACTIONS? The production spec forces ONE bolus->glucose slope for everyone
# and no treatment x mediator interaction, yet per-subject profiles show insulin
# effectiveness varies wildly. For each timepoint this fits four outcome models
# on the SAME weighted data + (timepoint-independent) mediator model and compares
# the mediation:
#
#   A. baseline   Y ~ treat + M + X + (1 | subject)              [reproduces prod]
#   B. ranslope   Y ~ treat + M + X + (1 + M | subject)          [subject-varying b]
#   C. interact   Y ~ treat * M + X + (1 | subject)              [Z x M interaction]
#   D. both       Y ~ treat * M + X + (1 + M | subject)
#
# Instant tells (per timepoint, printed even with --skip-mediate):
#   * random-slope SD of M  -> spread of insulin effectiveness across subjects
#   * treat:M interaction coef + p -> does meal size modify the bolus effect
#   * fixed b (M->Y) + p -> does it sharpen vs baseline
# mediate() is quasi-Bayesian (no refit per sim) so the whole timepoint sweep is
# ~minutes. Writes one tidy CSV (rows = timepoint x model).
#
# Usage:
#   Rscript cma_cluster/diatrend/run_interaction_mediation.R \
#     --phi-file analysis_data/diatrend/embeddings_full/phi_embeddings_diatrend_full_demo.csv \
#     --weights-file analysis_data/diatrend/weights/npcbps_weights_c12_noiob_fullcohort_demowt.csv \
#     --cohort 1,2 --split test --use-iob FALSE --covariate-prefix PC --n-phi 3 \
#     --timepoints 60,90,120,150,180,210 --offset 30 --sims 1000 --run-id fullcohort_interaction
suppressPackageStartupMessages({
  library(optparse); library(lme4); library(mediation)
})

option_list <- list(
  make_option("--phi-file", type = "character", default = NULL),
  make_option("--weights-file", type = "character", default = NULL),
  make_option("--cohort", type = "character", default = "1,2"),
  make_option("--split", type = "character", default = "test"),
  make_option("--meal", type = "character", default = "ALL"),
  make_option("--use-iob", type = "logical", default = FALSE),
  make_option("--demographics", type = "logical", default = FALSE),
  make_option("--timepoints", type = "character", default = "60,90,120,150,180,210"),
  make_option("--offset", type = "integer", default = 30),
  make_option("--contrast-anchor", type = "character", default = "mealtype"),
  make_option("--n-phi", type = "integer", default = 3),
  make_option("--covariate-prefix", type = "character", default = "PC"),
  make_option("--use-weights", type = "logical", default = TRUE),
  make_option("--sims", type = "integer", default = 200),
  make_option("--skip-mediate", action = "store_true", default = FALSE),
  make_option("--output-dir", type = "character", default = "mediation_results/diatrend"),
  make_option("--run-id", type = "character", default = "interaction")
)
opt <- parse_args(OptionParser(option_list = option_list))
timepoints <- as.integer(strsplit(opt$timepoints, ",")[[1]])

# ---- load + filter (mirrors run_mixed_effects_mediation.R) ----
if (!is.null(opt$`weights-file`)) {
  df <- read.csv(opt$`weights-file`, stringsAsFactors = FALSE)
  if (!"cbps_weight" %in% colnames(df)) df$cbps_weight <- 1
} else {
  df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
  df <- df[df$cohort %in% as.integer(strsplit(opt$cohort, ",")[[1]]), ]
  if (isTRUE(opt$`use-iob`)) df <- df[!is.na(df$iob_at_meal), ]
  df$cbps_weight <- 1
}
if (tolower(opt$split) != "all") df <- df[df$split == opt$split, ]
if (toupper(opt$meal) != "ALL") df <- df[tolower(df$meal_type) == tolower(opt$meal), ]

phi_cols <- if (opt$`n-phi` > 0) paste0(opt$`covariate-prefix`, "_", seq_len(opt$`n-phi`)) else character(0)
covariate_terms <- phi_cols
if (isTRUE(opt$`use-iob`)) covariate_terms <- c(covariate_terms, "iob_at_meal")
if (isTRUE(opt$demographics)) {
  df <- df[stats::complete.cases(df[, c("demo_age", "demo_sex", "demo_hba1c")]), ]
  covariate_terms <- c(covariate_terms, "demo_age", "factor(demo_sex)", "demo_hba1c")
}
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
anchor_med <- switch(opt$`contrast-anchor`,
  global = rep(median(df$treat_meal_carbs, na.rm = TRUE), nrow(df)),
  mealtype = med_by(df$treat_meal_carbs, df$meal_type),
  subject = med_by(df$treat_meal_carbs, df$subject_id),
  `subject-mealtype` = med_by(df$treat_meal_carbs, interaction(df$subject_id, df$meal_type)))
df$treat_centered <- df$treat_meal_carbs - anchor_med
df$M <- df$mediator_bolus_for_meal
w <- if (isTRUE(opt$`use-weights`)) df$cbps_weight else rep(1, nrow(df))
rhs <- paste(covariate_terms, collapse = " + ")
cat(sprintf("n=%d episodes, %d subjects (split=%s, meal=%s); covariates: %s\n",
            nrow(df), length(unique(df$subject_id)), opt$split, opt$meal, rhs))

# ---- mediator model (shared; M does not depend on the outcome timepoint) ----
mfit <- lmer(as.formula(paste("M ~ treat_centered +", rhs, "+ (1 | subject_id)")),
             data = df, weights = w, REML = FALSE)
cat(sprintf("Mediator (a: carbs->bolus) = %.4f  | timepoints: %s\n",
            fixef(mfit)["treat_centered"], paste(timepoints, collapse = ", ")))

specs <- list(
  A = list(lbl = "baseline", f = "Y ~ treat_centered + M + %s + (1 | subject_id)", int = FALSE),
  B = list(lbl = "ranslope", f = "Y ~ treat_centered + M + %s + (1 + M | subject_id)", int = FALSE),
  C = list(lbl = "interact", f = "Y ~ treat_centered * M + %s + (1 | subject_id)", int = TRUE),
  D = list(lbl = "both",     f = "Y ~ treat_centered * M + %s + (1 + M | subject_id)", int = TRUE)
)
get <- function(r, nm) tryCatch(as.numeric(r[[nm]]), error = function(e) NA_real_)

rows <- list()
for (tp in timepoints) {
  df$Y <- df[[sprintf("Y_%dmin", tp)]]
  cat(sprintf("\n================= timepoint %d min =================\n", tp))
  for (k in names(specs)) {
    s <- specs[[k]]
    fit <- tryCatch(lmer(as.formula(sprintf(s$f, rhs)), data = df, weights = w, REML = FALSE),
                    error = function(e) {cat(sprintf("[t%d %s] FIT ERROR: %s\n", tp, k, conditionMessage(e))); NULL})
    if (is.null(fit)) next
    ct <- summary(fit)$coefficients
    b <- ct["M", "Estimate"]; b_p <- 2 * pnorm(-abs(ct["M", "t value"]))
    vc <- as.data.frame(VarCorr(fit))
    ss <- vc$sdcor[vc$grp == "subject_id" & vc$var1 == "M" & is.na(vc$var2)]
    slope_sd <- if (length(ss)) ss else NA_real_
    inter_c <- inter_p <- NA_real_
    if (s$int && "treat_centered:M" %in% rownames(ct)) {
      inter_c <- ct["treat_centered:M", "Estimate"]
      inter_p <- 2 * pnorm(-abs(ct["treat_centered:M", "t value"]))
    }
    d0 <- d1 <- d0p <- d1p <- z0 <- tau <- taup <- NA_real_
    if (!isTRUE(opt$`skip-mediate`)) {
      r <- tryCatch(mediate(mfit, fit, treat = "treat_centered", mediator = "M",
                            control.value = 0, treat.value = opt$offset, sims = opt$sims),
                    error = function(e) {cat(sprintf("[t%d %s] mediate ERROR: %s\n", tp, k, conditionMessage(e))); NULL})
      if (!is.null(r)) {
        d0 <- get(r, "d0"); d1 <- get(r, "d1"); d0p <- get(r, "d0.p"); d1p <- get(r, "d1.p")
        z0 <- get(r, "z0"); tau <- get(r, "tau.coef"); taup <- get(r, "tau.p")
      }
    }
    sd_txt <- if (!is.na(slope_sd)) sprintf(" slopeSD=%.2f", slope_sd) else ""
    in_txt <- if (!is.na(inter_p)) sprintf(" trtxM=%+.4f(p=%.3f)", inter_c, inter_p) else ""
    if (isTRUE(opt$`skip-mediate`)) {
      cat(sprintf("  [%s] b=%+.3f(p=%.3f)%s%s\n", s$lbl, b, b_p, sd_txt, in_txt))
    } else if (s$int) {
      cat(sprintf("  [%s] ACME d0=%+.2f(p=%.3f) d1=%+.2f(p=%.3f) ADE=%+.2f Tot=%+.2f(p=%.3f)%s%s\n",
                  s$lbl, d0, d0p, d1, d1p, z0, tau, taup, sd_txt, in_txt))
    } else {
      cat(sprintf("  [%s] ACME=%+.2f(p=%.3f) ADE=%+.2f Tot=%+.2f(p=%.3f)%s\n",
                  s$lbl, d0, d0p, z0, tau, taup, sd_txt))
    }
    rows[[length(rows) + 1]] <- data.frame(
      timepoint = tp, model = k, label = s$lbl, b = b, b_p = b_p, slope_sd = slope_sd,
      inter_coef = inter_c, inter_p = inter_p, acme_d0 = d0, acme_d0_p = d0p,
      acme_d1 = d1, acme_d1_p = d1p, ade = z0, total = tau, total_p = taup)
  }
}

if (length(rows)) {
  out <- do.call(rbind, rows)
  dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
  f <- file.path(opt$`output-dir`, sprintf("interaction_mediation_%s.csv", opt$`run-id`))
  write.csv(out, f, row.names = FALSE)
  cat(sprintf("\nWrote %s (%d rows)\n", f, nrow(out)))
}
