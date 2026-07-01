#!/usr/bin/env Rscript
# Decompose the base -> demographics ACME/ADE attenuation into its two causes:
#   (1) RE-WEIGHTING   - npCBPS additionally balances age/sex/HbA1c, and
#   (2) MODEL ADJUSTMENT - the LMERs additionally include age/sex/HbA1c as
#                          (subject-constant) fixed effects.
#
# The two production arms differ in BOTH at once *and* in sample size
# (1094 vs 989). This script holds the SAMPLE fixed (the demographics-complete
# test episodes) and runs a 2x2: {base weights, demo weights} x {no demo, demo}
# in the models, at one endpoint. The corners reproduce the two arms; the
# off-corners isolate each mechanism. Emits aggregate numbers only.
#
# Usage (on the cluster, demographics-MERGED embeddings CSV):
#   Rscript cma_cluster/diatrend/decompose_demographics.R \
#       --phi-file analysis_data/diatrend/embeddings/phi_embeddings_diatrend_demo.csv \
#       --split test --timepoint 120 --offset 30 --meal ALL

suppressPackageStartupMessages({
  library(optparse)
  library(lme4)
  library(mediation)
  library(CBPS)
})

option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Demographics-MERGED embeddings CSV (run merge_demographics.py first)."),
  make_option(c("--cohort"), type = "character", default = "2", help = "Cohort filter. [2]"),
  make_option(c("--split"), type = "character", default = "test",
    help = "Within-subject temporal split: test/train/all. [test]"),
  make_option(c("--meal"), type = "character", default = "ALL", help = "Meal subset or ALL. [ALL]"),
  make_option(c("--timepoint"), type = "integer", default = 120, help = "Outcome timepoint min. [120]"),
  make_option(c("--offset"), type = "integer", default = 30, help = "Carb offset g. [30]"),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype",
    help = "global/mealtype/subject/subject-mealtype. [mealtype]"),
  make_option(c("--balance-n-phi"), type = "integer", default = 6, help = "PCs in npCBPS. [6]"),
  make_option(c("--model-n-phi"), type = "integer", default = 3, help = "PCs in the LMERs. [3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC", help = "PC or phi. [PC]"),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Include iob_at_meal as a covariate/weight term. FALSE for the full cohort (cohort 1 has no IOB). [TRUE]"),
  make_option(c("--sims"), type = "integer", default = 1000, help = "mediate() sims. [1000]")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`phi-file`)) stop("--phi-file (demographics-merged) is required.")

# ---- Load + fix the sample (demographics-complete, so all 4 cells are comparable) ----
df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
cohorts <- as.integer(strsplit(opt$cohort, ",")[[1]])
df <- df[df$cohort %in% cohorts, ]
if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df)) stop("No 'split' column; re-export with --diatrend-test-frac > 0.")
  df <- df[df$split == opt$split, ]
}
# Only require IOB when it is actually used (cohort 1 has no IOB, so requiring it
# would silently drop the whole cohort and make this a cohort-2-only analysis).
if (isTRUE(opt$`use-iob`)) df <- df[!is.na(df$iob_at_meal), ]
demo_needed <- c("demo_age", "demo_sex", "demo_hba1c")
if (!all(demo_needed %in% colnames(df))) {
  stop("Missing demographic columns; pass the merge_demographics.py output.")
}
df <- df[stats::complete.cases(df[, demo_needed]), ]
if (toupper(opt$meal) != "ALL") df <- df[tolower(df$meal_type) == tolower(opt$meal), ]
# Harmonize demographics to numeric. Cohort 1 stores age/HbA1c as BANDS
# ("40-49", ">9"), which makes the whole column character even for cohort-2
# numeric strings; map each value to the mean of the numbers it contains
# (band -> midpoint, numeric string -> itself) so the models see a clean scalar.
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

outcome_col <- sprintf("Y_%dmin", opt$timepoint)
if (!outcome_col %in% colnames(df)) stop(sprintf("Outcome column %s missing.", outcome_col))
df$Y <- df[[outcome_col]]

cat(sprintf("Fixed sample: %d episodes, %d subjects (cohort %s, split %s, meal %s, demo-complete).\n",
            nrow(df), length(unique(df$subject_id)), opt$cohort, opt$split, opt$meal))

# ---- Treatment contrast: center carbs at the group median (same as production) ----
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
anchor_med <- switch(opt$`contrast-anchor`,
  global             = rep(median(df$treat_meal_carbs, na.rm = TRUE), nrow(df)),
  mealtype           = med_by(df$treat_meal_carbs, df$meal_type),
  subject            = med_by(df$treat_meal_carbs, df$subject_id),
  `subject-mealtype` = med_by(df$treat_meal_carbs, interaction(df$subject_id, df$meal_type)))
df$treat_centered <- df$treat_meal_carbs - anchor_med

pc_bal <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`balance-n-phi`))
pc_mod <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`model-n-phi`))

# ---- npCBPS weights (same recipe as npcbps_weights.R: normalize, cap@99, renorm) ----
fit_weights <- function(cov_cols, label) {
  fmla <- as.formula(paste("treat_meal_carbs ~", paste(cov_cols, collapse = " + ")))
  fit <- npCBPS(fmla, data = df, corprior = 0.01)
  w <- fit$weights
  n <- nrow(df)
  w <- w * n / sum(w)
  cap <- quantile(w, 0.99); w <- pmin(w, cap); w <- w * n / sum(w)
  ess <- sum(w)^2 / sum(w^2)
  cat(sprintf("  [%s] ESS = %.1f (%.1f%%), weight SD = %.3f, range [%.3f, %.3f]\n",
              label, ess, 100 * ess / n, sd(w), min(w), max(w)))
  w
}
cat("\nFitting npCBPS weights on the fixed sample...\n")
iob_term <- if (isTRUE(opt$`use-iob`)) "iob_at_meal" else character(0)
cov_base <- c(pc_bal, "glucose_at_meal", iob_term)
cov_demo <- c(cov_base, "demo_age", "demo_sex", "demo_hba1c")
w_base <- fit_weights(cov_base, "base weights")
w_demo <- fit_weights(cov_demo, "demo weights")

# How balanced are the demographics under each weight set (weighted corr w/ treat)?
wcor <- function(x, w) {
  if (!is.numeric(x)) x <- as.integer(as.factor(x))   # factor OR character -> code
  ok <- is.finite(x) & is.finite(df$treat_meal_carbs) & is.finite(w)
  suppressWarnings(cov.wt(cbind(x[ok], df$treat_meal_carbs[ok]), wt = w[ok], cor = TRUE)$cor[1, 2])
}
cat("\nDemographic balance (weighted corr with treatment; near 0 = balanced):\n")
cat(sprintf("  %-12s  base_w=%+.3f  demo_w=%+.3f\n", "demo_age",   wcor(df$demo_age, w_base),   wcor(df$demo_age, w_demo)))
cat(sprintf("  %-12s  base_w=%+.3f  demo_w=%+.3f\n", "demo_sex",   wcor(df$demo_sex, w_base),   wcor(df$demo_sex, w_demo)))
cat(sprintf("  %-12s  base_w=%+.3f  demo_w=%+.3f\n", "demo_hba1c", wcor(df$demo_hba1c, w_base), wcor(df$demo_hba1c, w_demo)))

# ---- Mediation: 2x2 of {weights} x {model demo covariates} ----
run_med <- function(weights, use_demo) {
  rhs <- c(pc_mod, iob_term)
  if (use_demo) rhs <- c(rhs, "demo_age", "demo_sex", "demo_hba1c")
  rhs <- paste(rhs, collapse = " + ")
  mf <- as.formula(paste("mediator_bolus_for_meal ~ treat_centered +", rhs, "+ (1 | subject_id)"))
  yf <- as.formula(paste("Y ~ treat_centered + mediator_bolus_for_meal +", rhs, "+ (1 | subject_id)"))
  m <- lme4::lmer(mf, data = df, weights = weights, REML = FALSE)
  y <- lme4::lmer(yf, data = df, weights = weights, REML = FALSE)
  r <- mediate(m, y, treat = "treat_centered", mediator = "mediator_bolus_for_meal",
               control.value = 0, treat.value = opt$offset, boot = FALSE, sims = opt$sims)
  c(acme = as.numeric(r$d0), acme_p = as.numeric(r$d0.p),
    ade  = as.numeric(r$z0), ade_p  = as.numeric(r$z0.p),
    tot  = as.numeric(r$tau.coef), tot_p = as.numeric(r$tau.p))
}

cat(sprintf("\nRunning 2x2 mediation at %d min, +%d g, meal=%s (sims=%d)...\n",
            opt$timepoint, opt$offset, opt$meal, opt$sims))
cells <- list(
  a = list(w = w_base, demo = FALSE, lab = "(a) base wts + no demo cov   [= base arm]"),
  b = list(w = w_base, demo = TRUE,  lab = "(b) base wts + demo cov      [fixed-effects only]"),
  c = list(w = w_demo, demo = FALSE, lab = "(c) demo wts + no demo cov   [re-weighting only]"),
  d = list(w = w_demo, demo = TRUE,  lab = "(d) demo wts + demo cov      [= demographics arm]"))

cat("\n", strrep("=", 78), "\n", sep = "")
cat(sprintf("%-44s %8s %8s %8s\n", "config", "ACME", "ADE", "Total"))
cat(strrep("-", 78), "\n", sep = "")
res <- list()
star <- function(p) if (p < 0.05) "*" else " "
for (nm in names(cells)) {
  cc <- cells[[nm]]
  r <- run_med(cc$w, cc$demo)
  res[[nm]] <- r
  cat(sprintf("%-44s %7.2f%s %7.2f%s %7.2f%s\n", cc$lab,
              r["acme"], star(r["acme_p"]), r["ade"], star(r["ade_p"]), r["tot"], star(r["tot_p"])))
}
cat(strrep("=", 78), "\n", sep = "")
cat("(* = p < 0.05; all on the same fixed sample)\n")
cat(sprintf("\nDecomposition of ACME (a -> d): total change = %+.2f\n", res$d["acme"] - res$a["acme"]))
cat(sprintf("  re-weighting alone   (a -> c): %+.2f\n", res$c["acme"] - res$a["acme"]))
cat(sprintf("  fixed-effects alone  (a -> b): %+.2f\n", res$b["acme"] - res$a["acme"]))
cat(sprintf("\nDecomposition of ADE (a -> d): total change = %+.2f\n", res$d["ade"] - res$a["ade"]))
cat(sprintf("  re-weighting alone   (a -> c): %+.2f\n", res$c["ade"] - res$a["ade"]))
cat(sprintf("  fixed-effects alone  (a -> b): %+.2f\n", res$b["ade"] - res$a["ade"]))
