#!/usr/bin/env Rscript
# Leave-one-subject-out (LOSO) jackknife for the pooled DiaTrend mediation.
#
# Motivation: the decomposition showed the pooled ACME's significance is
# fragile to sample composition (dropping 2 demographics-missing subjects
# more than halved it). This quantifies how much each subject moves the
# pooled ACME/ADE at one endpoint. For each subject it drops that subject,
# RE-fits the npCBPS weights and the LMER mediation on the remainder, and
# records the leave-one-out estimate. Reports aggregate influence only
# (a dfbeta-style table); no per-episode data is emitted.
#
# Mirrors the production spec via the same flags, so run it on whichever
# arm is primary:
#   # primary (demographics balanced in weights, random-intercept model):
#   Rscript cma_cluster/diatrend/influence_diagnostics.R \
#       --phi-file analysis_data/diatrend/embeddings/phi_embeddings_diatrend_demo.csv \
#       --split test --timepoint 120 --offset 30 --meal ALL \
#       --demographics-weights TRUE --demographics-models FALSE
#   # no-demographics ohio spec (the full 1094-episode sample where the
#   # fragility first appeared):
#   Rscript cma_cluster/diatrend/influence_diagnostics.R \
#       --phi-file <ohio embeddings CSV> --split test --timepoint 120 --offset 30

suppressPackageStartupMessages({
  library(optparse)
  library(lme4)
  library(mediation)
  library(CBPS)
})

option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Embeddings CSV (demographics-merged if any --demographics-* is TRUE)."),
  make_option(c("--cohort"), type = "character", default = "2", help = "Cohort filter. [2]"),
  make_option(c("--split"), type = "character", default = "test", help = "test/train/all. [test]"),
  make_option(c("--meal"), type = "character", default = "ALL", help = "Meal subset or ALL. [ALL]"),
  make_option(c("--timepoint"), type = "integer", default = 120, help = "Outcome timepoint. [120]"),
  make_option(c("--offset"), type = "integer", default = 30, help = "Carb offset g. [30]"),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype",
    help = "global/mealtype/subject/subject-mealtype. [mealtype]"),
  make_option(c("--balance-n-phi"), type = "integer", default = 6, help = "PCs in npCBPS. [6]"),
  make_option(c("--model-n-phi"), type = "integer", default = 3, help = "PCs in the LMERs. [3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC", help = "PC or phi. [PC]"),
  make_option(c("--demographics-weights"), type = "logical", default = FALSE,
    help = "Balance age/sex/HbA1c in npCBPS. [FALSE]"),
  make_option(c("--demographics-models"), type = "logical", default = FALSE,
    help = "Add age/sex/HbA1c as LMER fixed effects. [FALSE]"),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Use iob_at_meal (covariate + filter). FALSE for no-IOB / full-cohort. [TRUE]"),
  make_option(c("--sims"), type = "integer", default = 300,
    help = "mediate() sims (point estimates are ~deterministic; sims drive p only). [300]"),
  make_option(c("--top"), type = "integer", default = 10, help = "How many influential subjects to list. [10]")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`phi-file`)) stop("--phi-file is required.")
demo_w <- isTRUE(opt$`demographics-weights`)
demo_m <- isTRUE(opt$`demographics-models`)
use_iob <- isTRUE(opt$`use-iob`)

# ---- Load + filter ----
df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
cohorts <- as.integer(strsplit(opt$cohort, ",")[[1]])
df <- df[df$cohort %in% cohorts, ]
if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df)) stop("No 'split' column; re-export with --diatrend-test-frac > 0.")
  df <- df[df$split == opt$split, ]
}
if (use_iob) df <- df[!is.na(df$iob_at_meal), ]
if (demo_w || demo_m) {
  demo_needed <- c("demo_age", "demo_sex", "demo_hba1c")
  if (!all(demo_needed %in% colnames(df))) stop("Missing demographic columns; pass merge_demographics.py output.")
  df <- df[stats::complete.cases(df[, demo_needed]), ]
  df$demo_sex <- as.factor(df$demo_sex)
}
if (toupper(opt$meal) != "ALL") df <- df[tolower(df$meal_type) == tolower(opt$meal), ]
outcome_col <- sprintf("Y_%dmin", opt$timepoint)
if (!outcome_col %in% colnames(df)) stop(sprintf("Outcome column %s missing.", outcome_col))
df$Y <- df[[outcome_col]]

pc_bal <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`balance-n-phi`))
pc_mod <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`model-n-phi`))
bal_cov <- c(pc_bal, "glucose_at_meal", if (use_iob) "iob_at_meal", if (demo_w) c("demo_age", "demo_sex", "demo_hba1c"))
mod_cov <- c(pc_mod, if (use_iob) "iob_at_meal", if (demo_m) c("demo_age", "demo_sex", "demo_hba1c"))

cat(sprintf("Spec: demographics in weights=%s, in models=%s | balance PCs=%d, model PCs=%d\n",
            demo_w, demo_m, opt$`balance-n-phi`, opt$`model-n-phi`))
cat(sprintf("Sample: %d episodes, %d subjects (cohort %s, split %s, meal %s).\n",
            nrow(df), length(unique(df$subject_id)), opt$cohort, opt$split, opt$meal))

med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))

# Center treatment, fit npCBPS weights, run mediation -> ACME/ADE on a data subset.
process <- function(d) {
  anchor_med <- switch(opt$`contrast-anchor`,
    global             = rep(median(d$treat_meal_carbs, na.rm = TRUE), nrow(d)),
    mealtype           = med_by(d$treat_meal_carbs, d$meal_type),
    subject            = med_by(d$treat_meal_carbs, d$subject_id),
    `subject-mealtype` = med_by(d$treat_meal_carbs, interaction(d$subject_id, d$meal_type)))
  d$treat_centered <- d$treat_meal_carbs - anchor_med

  fmla <- as.formula(paste("treat_meal_carbs ~", paste(bal_cov, collapse = " + ")))
  w <- npCBPS(fmla, data = d, corprior = 0.01)$weights
  n <- nrow(d); w <- w * n / sum(w)
  cap <- quantile(w, 0.99); w <- pmin(w, cap); w <- w * n / sum(w)

  rhs <- paste(mod_cov, collapse = " + ")
  mf <- as.formula(paste("mediator_bolus_for_meal ~ treat_centered +", rhs, "+ (1 | subject_id)"))
  yf <- as.formula(paste("Y ~ treat_centered + mediator_bolus_for_meal +", rhs, "+ (1 | subject_id)"))
  m <- lme4::lmer(mf, data = d, weights = w, REML = FALSE)
  y <- lme4::lmer(yf, data = d, weights = w, REML = FALSE)
  r <- mediate(m, y, treat = "treat_centered", mediator = "mediator_bolus_for_meal",
               control.value = 0, treat.value = opt$offset, boot = FALSE, sims = opt$sims)
  c(acme = as.numeric(r$d0), acme_p = as.numeric(r$d0.p),
    ade = as.numeric(r$z0), ade_p = as.numeric(r$z0.p))
}

cat(sprintf("\nBaseline (full sample), %d min, +%d g...\n", opt$timepoint, opt$offset))
base <- process(df)
sig0 <- base["acme_p"] < 0.05
cat(sprintf("  ACME = %.2f (p=%.3f)%s   ADE = %.2f (p=%.3f)\n",
            base["acme"], base["acme_p"], if (sig0) " *" else "", base["ade"], base["ade_p"]))

subs <- sort(unique(df$subject_id))
cat(sprintf("\nLeave-one-subject-out over %d subjects (re-fitting weights + models each)...\n", length(subs)))
rows <- lapply(subs, function(s) {
  d <- df[df$subject_id != s, ]
  r <- tryCatch(process(d), error = function(e) c(acme = NA, acme_p = NA, ade = NA, ade_p = NA))
  data.frame(subject = s, n_drop = sum(df$subject_id == s),
             acme = r["acme"], acme_p = r["acme_p"], ade = r["ade"], ade_p = r["ade_p"],
             row.names = NULL)
})
J <- do.call(rbind, rows)
J$dfbeta_acme <- J$acme - base["acme"]
J$dfbeta_ade  <- J$ade  - base["ade"]
J$flips_acme  <- (J$acme_p < 0.05) != sig0

cat("\n", strrep("=", 72), "\n", sep = "")
cat("LOSO SUMMARY (ACME)\n")
cat(strrep("-", 72), "\n", sep = "")
cat(sprintf("  baseline ACME            : %.2f (p=%.3f, %s)\n",
            base["acme"], base["acme_p"], if (sig0) "significant" else "n.s."))
cat(sprintf("  leave-one-out ACME range : [%.2f, %.2f]\n", min(J$acme, na.rm = TRUE), max(J$acme, na.rm = TRUE)))
cat(sprintf("  max |dfbeta| (one subject moves ACME by up to): %.2f\n", max(abs(J$dfbeta_acme), na.rm = TRUE)))
cat(sprintf("  # subjects whose removal FLIPS ACME significance: %d / %d\n",
            sum(J$flips_acme, na.rm = TRUE), nrow(J)))

ord <- order(-abs(J$dfbeta_acme))
top <- head(J[ord, ], opt$`top`)
cat("\nMost influential subjects (by |change in ACME| when removed):\n")
cat(sprintf("  %-14s %6s | %8s %8s %6s | %8s %8s\n",
            "subject", "n_ep", "ACME_-s", "dACME", "flip?", "ADE_-s", "dADE"))
for (i in seq_len(nrow(top))) {
  r <- top[i, ]
  cat(sprintf("  %-14s %6d | %8.2f %+8.2f %6s | %8.2f %+8.2f\n",
              as.character(r$subject), r$n_drop, r$acme, r$dfbeta_acme,
              if (isTRUE(r$flips_acme)) "YES" else "-", r$ade, r$dfbeta_ade))
}
cat(strrep("=", 72), "\n", sep = "")
cat("\nInterpretation: a large |dACME| or any 'flip=YES' means the pooled\n")
cat("ACME hinges on individual subjects -> report this fragility alongside\n")
cat("medsens. A stable ACME (small range, no flips) is the reassuring case.\n")
