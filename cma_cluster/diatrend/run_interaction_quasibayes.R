#!/usr/bin/env Rscript
# Quasi-Bayesian (the manuscript's inference) arm-specific mediation for the
# meal-size x bolus INTERACTION model, across timepoints. Uses mediation::mediate
# (boot=FALSE, 1000 Monte Carlo sims) directly -- the SAME method as the main
# Ohio/DiaTrend tables/figures -- so the interaction causal-effect figures match
# the house inference (not the subject-cluster bootstrap).
#
# Per cell (meal x timepoint x model): fit the mediator model + the interaction
# outcome model, then mediate() returns the arm-specific ACME (d0 control / d1
# treated/+offset), ADE (z0), and total, each with quasi-Bayesian CIs. Output
# schema matches run_interaction_bootstrap.R so diatrend_interaction_to_canonical.py
# can reshape it for the house generator.
#
#   --model lmer : lmer mediator + lmer(Y ~ treat*M + PC + (1|subject)) outcome.
#   --model qr   : Tobit mediator + rq(Y ~ treat*M + PC, tau) outcome per --taus.
#
# Usage (per meal x model):
#   Rscript cma_cluster/diatrend/run_interaction_quasibayes.R --weights-file <w> \
#     --split test --meal dinner --covariate-prefix PC --n-phi 3 \
#     --timepoints 60,65,...,210 --offset 30 --model qr --taus 0.25,0.5,0.75 \
#     --sims 1000 --cores 48 --run-id qb_fc_int_qr_dinner
suppressPackageStartupMessages({
  library(optparse); library(lme4); library(parallel); library(survival)
  library(quantreg); library(mediation)
})

option_list <- list(
  make_option("--weights-file", type = "character", default = NULL),
  make_option("--split", type = "character", default = "test"),
  make_option("--meal", type = "character", default = "ALL"),
  make_option("--timepoints", type = "character", default = "60,90,120,150,180,210"),
  make_option("--offset", type = "integer", default = 30),
  make_option("--contrast-anchor", type = "character", default = "mealtype"),
  make_option("--covariate-prefix", type = "character", default = "PC"),
  make_option("--n-phi", type = "integer", default = 3),
  make_option("--model", type = "character", default = "lmer"),
  make_option("--taus", type = "character", default = "0.25,0.5,0.75"),
  make_option("--sims", type = "integer", default = 1000),
  make_option("--cores", type = "integer", default = 48),
  make_option("--seed", type = "integer", default = 42),
  make_option("--output-dir", type = "character", default = "mediation_results/diatrend"),
  make_option("--run-id", type = "character", default = "qb_interaction")
)
opt <- parse_args(OptionParser(option_list = option_list))
tps <- as.integer(strsplit(opt$timepoints, ",")[[1]])
is_qr <- identical(opt$model, "qr")
taus <- if (is_qr) as.numeric(strsplit(opt$taus, ",")[[1]]) else NA_real_

df <- read.csv(opt$`weights-file`, stringsAsFactors = FALSE)
if (!"cbps_weight" %in% colnames(df)) df$cbps_weight <- 1
if (tolower(opt$split) != "all") df <- df[df$split == opt$split, ]
if (toupper(opt$meal) != "ALL") df <- df[tolower(df$meal_type) == tolower(opt$meal), ]
pc <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`n-phi`))
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
anchor_med <- switch(opt$`contrast-anchor`,
  global = rep(median(df$treat_meal_carbs, na.rm = TRUE), nrow(df)),
  mealtype = med_by(df$treat_meal_carbs, df$meal_type),
  subject = med_by(df$treat_meal_carbs, df$subject_id))
df$treat_centered <- df$treat_meal_carbs - anchor_med
df$M <- df$mediator_bolus_for_meal
df$wt <- df$cbps_weight
rhs <- paste(pc, collapse = " + ")

# mediator model (does not depend on timepoint/tau): fit once.
if (is_qr) {
  model.m <- survreg(as.formula(paste("Surv(M, M > 0, type='left') ~ treat_centered +", rhs)),
                     data = df, weights = wt, dist = "gaussian")
} else {
  model.m <- lmer(as.formula(paste("M ~ treat_centered +", rhs, "+ (1 | subject_id)")),
                  data = df, weights = wt, REML = FALSE)
}
cat(sprintf("meal=%s model=%s  n=%d episodes, %d subjects | sims=%d | %d tp x %d tau\n",
            opt$meal, opt$model, nrow(df), length(unique(df$subject_id)), opt$sims,
            length(tps), length(if (is_qr) taus else 1)))

cells <- expand.grid(tau = if (is_qr) taus else NA_real_, tp = tps, stringsAsFactors = FALSE)
set.seed(opt$seed)

one_cell <- function(i) {
  tp <- cells$tp[i]; tau <- cells$tau[i]
  f <- sprintf("Y_%dmin ~ treat_centered * M + %s", tp, rhs)
  res <- tryCatch({
    if (is_qr) {
      model.y <- rq(as.formula(f), tau = tau, data = df, weights = df$wt)
    } else {
      model.y <- lmer(as.formula(paste(f, "+ (1 | subject_id)")), data = df, weights = wt, REML = FALSE)
    }
    th3 <- tryCatch({
      ct <- if (is_qr) summary(model.y, se = "nid")$coefficients else summary(model.y)$coefficients
      ct["treat_centered:M", 1]
    }, error = function(e) NA_real_)
    r <- mediate(model.m, model.y, treat = "treat_centered", mediator = "M",
                 control.value = 0, treat.value = opt$offset, sims = opt$sims)
    data.frame(meal = opt$meal, model = opt$model, tau = tau, timepoint = tp,
               th3 = th3, th3_lo = NA_real_, th3_hi = NA_real_, th3_p = NA_real_,
               d0 = r$d0, d0_lo = r$d0.ci[1], d0_hi = r$d0.ci[2], d0_p = r$d0.p,
               d1 = r$d1, d1_lo = r$d1.ci[1], d1_hi = r$d1.ci[2], d1_p = r$d1.p,
               ade = r$z0, ade_lo = r$z0.ci[1], ade_hi = r$z0.ci[2], ade_p = r$z0.p,
               tot = r$tau.coef, tot_lo = r$tau.ci[1], tot_hi = r$tau.ci[2], tot_p = r$tau.p)
  }, error = function(e) { cat(sprintf("  [tp=%d tau=%s] ERROR: %s\n", tp, tau, conditionMessage(e))); NULL })
  res
}

rows <- mclapply(seq_len(nrow(cells)), one_cell, mc.cores = opt$cores, mc.preschedule = FALSE)
out <- do.call(rbind, Filter(Negate(is.null), rows))
out <- out[order(out$tau, out$timepoint), ]
dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
f <- file.path(opt$`output-dir`, sprintf("interaction_bootstrap_%s.csv", opt$`run-id`))
write.csv(out, f, row.names = FALSE)
for (tp in c(180, 210)) {
  s <- out[out$timepoint == tp, ]
  for (j in seq_len(nrow(s))) cat(sprintf("%-4s t=%3d d0=%+.2f(p=%.3f) d1=%+.2f(p=%.3f) ADE=%+.2f(p=%.3f) Tot=%+.2f(p=%.3f)\n",
    if (is_qr) paste0("q", s$tau[j]) else "mean", tp, s$d0[j], s$d0_p[j], s$d1[j], s$d1_p[j], s$ade[j], s$ade_p[j], s$tot[j], s$tot_p[j]))
}
cat(sprintf("\nWrote %s (%d rows)\n", f, nrow(out)))
