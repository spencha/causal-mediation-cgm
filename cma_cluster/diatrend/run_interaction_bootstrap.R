#!/usr/bin/env Rscript
# FAST subject-cluster bootstrap of the meal-size x bolus INTERACTION model,
# returning the ARM-SPECIFIC mediation decomposition across timepoints, for the
# LMER (mean) outcome OR quantile-regression outcomes at chosen taus:
#   theta3 = treat_centered:M interaction coefficient
#   d0     = ACME at the control arm           = a * delta * b2
#   d1     = ACME at the treated (+offset) arm = a * delta * (b2 + b3*delta)
#   ADE    = average direct effect             = delta * (b1 + b3*Mbar0)
#   total  = d1 + ADE
# a (carbs->bolus) + Mbar0 (mean predicted bolus at treat=0) come from the
# mediator model; b1/b2/b3 from the interaction outcome model. These closed forms
# match mediation::mediate() point estimates for a linear/quantile outcome (the
# outcome is linear in M); CIs come from the OUTER subject-cluster bootstrap.
#
#   --model lmer : lmer mediator + lmer outcome, both with (1|subject); the mean.
#   --model qr   : Tobit (survreg, left-censored) mediator + rq outcome at each
#                  --taus; NO random effect (clustering via the bootstrap), which
#                  matches the production QR spec.
#
# Weights (precomputed npCBPS) held FIXED so each resample only refits the cheap
# models -> ~minutes for B=1000 across 31 timepoints. Run once per meal (--meal).
#
# Usage:
#   Rscript cma_cluster/diatrend/run_interaction_bootstrap.R --weights-file <w> \
#     --split test --meal dinner --covariate-prefix PC --n-phi 3 \
#     --timepoints 60,65,...,210 --offset 30 --model qr --taus 0.25,0.5,0.75 \
#     --B 1000 --cores 48 --run-id fc_int_qr_dinner
suppressPackageStartupMessages({
  library(optparse); library(lme4); library(parallel); library(survival); library(quantreg)
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
  make_option("--B", type = "integer", default = 1000),
  make_option("--cores", type = "integer", default = 48),
  make_option("--seed", type = "integer", default = 42),
  make_option("--output-dir", type = "character", default = "mediation_results/diatrend"),
  make_option("--run-id", type = "character", default = "interaction")
)
opt <- parse_args(OptionParser(option_list = option_list))
tps <- as.integer(strsplit(opt$timepoints, ",")[[1]])
delta <- opt$offset
is_qr <- identical(opt$model, "qr")
taus <- if (is_qr) as.numeric(strsplit(opt$taus, ",")[[1]]) else NA_real_
tau_labs <- if (is_qr) paste0("q", taus) else "mean"

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
rhs <- paste(pc, collapse = " + ")
QS <- c("th3", "d0", "d1", "ade", "tot")
keyset <- character(0)   # same nesting order decomp() appends: tau -> tp -> quantity
for (lab in tau_labs) for (tp in tps) for (q in QS) keyset <- c(keyset, paste0(q, "_", lab, "_t", tp))

subs <- unique(df$subject_id)
cat(sprintf("meal=%s model=%s  n=%d episodes, %d subjects | B=%d on %d cores | %d timepoints x %d taus\n",
            opt$meal, opt$model, nrow(df), length(subs), opt$B, opt$cores, length(tps), length(tau_labs)))

decomp <- function(d) {
  na_out <- setNames(rep(NA_real_, length(keyset)), keyset)
  if (is_qr) {
    m <- tryCatch(survreg(as.formula(paste("Surv(M, M > 0, type='left') ~ treat_centered +", rhs)),
                          data = d, weights = cbps_weight, dist = "gaussian"), error = function(e) NULL)
    if (is.null(m)) return(na_out)
    a <- as.numeric(coef(m)["treat_centered"])
    lp <- as.numeric(predict(m, type = "lp")); Mbar0 <- mean(lp - a * d$treat_centered)
  } else {
    m <- tryCatch(lmer(as.formula(paste("M ~ treat_centered +", rhs, "+ (1 | cl)")),
                       data = d, weights = cbps_weight, REML = FALSE), error = function(e) NULL)
    if (is.null(m)) return(na_out)
    a <- as.numeric(fixef(m)["treat_centered"]); Mbar0 <- mean(fitted(m) - a * d$treat_centered)
  }
  out <- numeric(0)
  for (ti in seq_along(tau_labs)) {
    lab <- tau_labs[ti]; tau <- taus[ti]
    for (tp in tps) {
      v <- tryCatch({
        if (is_qr) {
          cf <- coef(rq(as.formula(sprintf("Y_%dmin ~ treat_centered * M + %s", tp, rhs)),
                        tau = tau, data = d, weights = d$cbps_weight))
          b1 <- cf["treat_centered"]; b2 <- cf["M"]; b3 <- cf["treat_centered:M"]
        } else {
          ct <- summary(lmer(as.formula(sprintf("Y_%dmin ~ treat_centered * M + %s + (1 | cl)", tp, rhs)),
                             data = d, weights = cbps_weight, REML = FALSE))$coefficients
          b1 <- ct["treat_centered", "Estimate"]; b2 <- ct["M", "Estimate"]
          b3 <- if ("treat_centered:M" %in% rownames(ct)) ct["treat_centered:M", "Estimate"] else NA_real_
        }
        d0 <- a * delta * b2; d1 <- a * delta * (b2 + b3 * delta)
        ade <- delta * (b1 + b3 * Mbar0)
        setNames(as.numeric(c(b3, d0, d1, ade, d1 + ade)), QS)
      }, error = function(e) setNames(rep(NA_real_, length(QS)), QS))
      out <- c(out, setNames(as.numeric(v), paste0(QS, "_", lab, "_t", tp)))
    }
  }
  out[keyset]
}

df$cl <- df$subject_id
point <- decomp(df)

set.seed(opt$seed)
seeds <- sample.int(.Machine$integer.max, opt$B)
boot <- mclapply(seq_len(opt$B), function(b) {
  set.seed(seeds[b])
  samp <- sample(subs, length(subs), replace = TRUE)
  ii <- do.call(rbind, lapply(seq_along(samp), function(i)
    data.frame(row = which(df$subject_id == samp[i]), cl = paste0("b", i))))
  d <- df[ii$row, ]; d$cl <- ii$cl
  suppressWarnings(decomp(d))
}, mc.cores = opt$cores, mc.preschedule = FALSE)
B <- do.call(rbind, boot)

pct <- function(v) quantile(v, c(0.025, 0.975), na.rm = TRUE)
bp  <- function(v) { v <- v[is.finite(v)]; if (!length(v)) NA_real_ else 2 * min(mean(v <= 0), mean(v >= 0)) }

cat("\n=========== interaction arm-specific decomposition (subject bootstrap) ===========\n")
rows <- list()
for (ti in seq_along(tau_labs)) {
  lab <- tau_labs[ti]
  for (tp in tps) {
    r <- list(meal = opt$meal, model = opt$model, tau = if (is_qr) taus[ti] else NA_real_, timepoint = tp)
    for (q in QS) {
      k <- paste0(q, "_", lab, "_t", tp); v <- B[, k]; ci <- pct(v)
      r[[q]] <- as.numeric(point[k]); r[[paste0(q, "_lo")]] <- ci[1]
      r[[paste0(q, "_hi")]] <- ci[2]; r[[paste0(q, "_p")]] <- bp(v)
    }
    rows[[length(rows) + 1]] <- as.data.frame(r, stringsAsFactors = FALSE)
    if (tp %in% c(180, 210))
      cat(sprintf("%-5s t=%3d  d1=%+.2f(p=%.3f) ADE=%+.2f(p=%.3f) Tot=%+.2f(p=%.3f) th3=%+.4f(p=%.3f)\n",
                  lab, tp, r$d1, r$d1_p, r$ade, r$ade_p, r$tot, r$tot_p, r$th3, r$th3_p))
  }
}
out <- do.call(rbind, rows)
dir.create(opt$`output-dir`, showWarnings = FALSE, recursive = TRUE)
f <- file.path(opt$`output-dir`, sprintf("interaction_bootstrap_%s.csv", opt$`run-id`))
write.csv(out, f, row.names = FALSE)
cat(sprintf("\nWrote %s (%d rows)\n", f, nrow(out)))
