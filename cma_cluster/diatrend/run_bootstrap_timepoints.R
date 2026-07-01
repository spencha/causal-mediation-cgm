#!/usr/bin/env Rscript
# DiaTrend: subject-cluster bootstrap swept over timepoints x offsets x meals,
# emitting a grid CSV in the SAME schema as run_all_timepoints.R so the existing
# Ohio-style plotter renders it directly with BOOTSTRAP confidence intervals:
#
#   python visualization_code/generate_diatrend_mediation_outputs.py \
#       --grid mediation_results/diatrend/grid_diatrend_<run-id>_bootstrap.csv \
#       --out  mediation_results/diatrend/figures_bootstrap \
#       --table3-model lmer --table4-timepoint 120 --table4-offset 30
#
# This is the final-inference analog of run_all_timepoints.R (which uses the
# quasi-Bayesian mediate() path): the estimation engine is identical to
# cluster_bootstrap_mediation.R -- it resamples SUBJECTS with replacement,
# re-fits the npCBPS weights and the LMER mediator/outcome models on each
# resample, and reports percentile CIs and bootstrap two-sided p-values. The
# Every outcome model is linear in the mediator, so the mediation effect is the
# product of coefficients -- exactly what mediate() returns for this structure:
#   ACME = a * b_M * delta   ADE = cprime * delta   Total = (a*b_M + cprime)*delta
# This holds per replicate from the fitted coefficients (no mediate()/sims call).
#
# Two model families are produced, mirroring run_all_timepoints.R:
#   * LMER (mean): lmer mediator + lmer outcome (random intercept).
#   * QR  (tau):   left-censored Tobit (survreg) mediator + weighted rq outcome
#                  (no random intercept), one spec per --taus quantile.
# So the grid carries model="lmer" (tau=NA) and model="qr" (tau in --taus), and
# the plotter shows the LMER + QR columns just like the quasi-Bayesian figures.
#
# Runtime note: this is meals x timepoints bootstrap fits. npCBPS weights and the
# mediator model do NOT depend on the timepoint, so each resample fits them ONCE
# per meal and only re-fits the (cheap) outcome model per timepoint. Effects are
# linear in the carb offset, so all offsets are scaled from one unit fit. Even so,
# expect this to be several times the cost of one headline cell -- scope it with
# --B / --timepoints / --meals on a shared node.
#
# Defaults MATCH run_all_timepoints.R's base arm (the OhioT1DM-literal spec:
# cohort 2, IOB in the model, NO demographic covariates) so this is a true
# robustness overlay of the quasi-Bayesian base figures -- same analysis, just
# subject-cluster bootstrap CIs instead of quasi-Bayesian ones. (For the
# separate full-cohort/demographics-weighted headline bootstrap, see
# run_bootstrap_grid.sh + cluster_bootstrap_mediation.R.)
#
# Usage:
#   Rscript cma_cluster/diatrend/run_bootstrap_timepoints.R --B 1000 --cores 48

suppressPackageStartupMessages({
  library(optparse); library(lme4); library(CBPS); library(parallel)
  library(survival); library(quantreg)   # Tobit mediator + rq outcome for the QR specs
})

option_list <- list(
  make_option(c("--phi-file"), type = "character",
    default = "analysis_data/diatrend/embeddings/phi_embeddings_diatrend_demo.csv",
    help = "Embeddings CSV. [base-arm demo file]"),
  make_option(c("--arm-tag"), type = "character", default = "base",
    help = "Arm label for the CSV's `arm` column + filename. [base]"),
  make_option(c("--cohort"), type = "character", default = "2", help = "Cohort filter. [2]"),
  make_option(c("--split"), type = "character", default = "test", help = "test/train/all. [test]"),
  make_option(c("--meals"), type = "character", default = "ALL,breakfast,lunch,dinner,snack",
    help = "Comma-separated meal subsets. [ALL,breakfast,lunch,dinner,snack]"),
  make_option(c("--timepoints"), type = "character",
    default = paste(seq(60, 210, by = 5), collapse = ","),
    help = paste(
      "Comma-separated outcome timepoints (min). Default is every 5 min from 60",
      "to 210 (31 points), matching the OhioT1DM sweep and run_all_timepoints.R.",
      "Cheap to extend: npCBPS weights + the mediator fit are timepoint-invariant",
      "(fit once per resample); only the outcome model is refit per timepoint.")),
  make_option(c("--offsets"), type = "character", default = "30",
    help = "Comma-separated carb offsets (g). [30]"),
  make_option(c("--taus"), type = "character", default = "0.25,0.5,0.75",
    help = paste("Comma-separated quantiles for the QR specs (Tobit mediator +",
                 "weighted rq outcome). Empty string = LMER only. [0.25,0.5,0.75]")),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype", help = "[mealtype]"),
  make_option(c("--balance-n-phi"), type = "integer", default = 6, help = "PCs in npCBPS. [6]"),
  make_option(c("--model-n-phi"), type = "integer", default = 3, help = "PCs in LMERs. [3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC", help = "[PC]"),
  make_option(c("--demographics-weights"), type = "logical", default = FALSE, help = "Balance demos in npCBPS. [FALSE = base arm]"),
  make_option(c("--demographics-models"), type = "logical", default = FALSE, help = "Demos as LMER fixed FX. [FALSE]"),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Use iob_at_meal (covariate + filter). TRUE = base arm (OhioT1DM-literal). [TRUE]"),
  make_option(c("--B"), type = "integer", default = 1000, help = "Bootstrap resamples. [1000]"),
  make_option(c("--seed"), type = "integer", default = 42, help = "RNG seed. [42]"),
  make_option(c("--cores"), type = "integer", default = 0,
    help = "Parallel workers (0 = detectCores()-2). [0]"),
  make_option(c("--output-dir"), type = "character", default = "mediation_results/diatrend",
    help = "Grid CSV output dir. [mediation_results/diatrend]"),
  make_option(c("--run-id"), type = "character", default = NULL,
    help = "Batch id for the filename. Defaults to a timestamp.")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`phi-file`)) stop("--phi-file is required.")
if (is.null(opt$`run-id`)) opt$`run-id` <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
set.seed(opt$seed)
demo_w <- isTRUE(opt$`demographics-weights`); demo_m <- isTRUE(opt$`demographics-models`)
use_iob <- isTRUE(opt$`use-iob`)
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

meals <- trimws(strsplit(opt$meals, ",")[[1]])
timepoints <- as.integer(trimws(strsplit(opt$timepoints, ",")[[1]]))
offsets <- as.integer(trimws(strsplit(opt$offsets, ",")[[1]]))
taus <- { t <- trimws(strsplit(opt$taus, ",")[[1]]); as.numeric(t[nzchar(t)]) }

# One "spec" per grid column: the LMER mean spec, then one QR spec per tau. Each
# (meal, timepoint, offset) produces n_spec rows. lmer carries tau=NA.
specs <- c(list(list(model = "lmer", tau = NA_real_)),
           lapply(taus, function(t) list(model = "qr", tau = t)))
n_spec <- length(specs)
need_lmer <- any(vapply(specs, function(s) s$model == "lmer", logical(1)))
need_qr   <- any(vapply(specs, function(s) s$model == "qr", logical(1)))

# ---- Load + filter (once; baseline filters are timepoint-independent) ----
df0 <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
df0 <- df0[df0$cohort %in% as.integer(strsplit(opt$cohort, ",")[[1]]), ]
if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df0)) stop("No 'split' column.")
  df0 <- df0[df0$split == opt$split, ]
}
if (use_iob) df0 <- df0[!is.na(df0$iob_at_meal), ]
if (demo_w || demo_m) {
  need <- c("demo_age", "demo_sex", "demo_hba1c")
  if (!all(need %in% colnames(df0))) stop("Missing demographic columns.")
  df0 <- df0[stats::complete.cases(df0[, need]), ]
  df0$demo_sex <- as.factor(df0$demo_sex)
}
for (tp in timepoints) {
  oc <- sprintf("Y_%dmin", tp)
  if (!oc %in% colnames(df0)) stop(sprintf("Outcome column %s missing.", oc))
}

pc_bal <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`balance-n-phi`))
pc_mod <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`model-n-phi`))
bal_cov <- c(pc_bal, "glucose_at_meal", if (use_iob) "iob_at_meal", if (demo_w) c("demo_age", "demo_sex", "demo_hba1c"))
mod_cov <- c(pc_mod, if (use_iob) "iob_at_meal", if (demo_m) c("demo_age", "demo_sex", "demo_hba1c"))

# npCBPS / model.matrix silently DROP rows with NA in any covariate, which
# desyncs the weights vector from the data and makes the fit fail on the FULL
# sample (while resamples that omit the offending subject still fit -> CI but no
# point estimate). Drop incomplete-covariate rows up front, like npcbps_weights.R,
# so the point fit and every resample use the same complete-case sample.
model_cov <- intersect(unique(c(bal_cov, mod_cov, "treat_meal_carbs",
                                "mediator_bolus_for_meal")), colnames(df0))
keep <- stats::complete.cases(df0[, model_cov])
if (any(!keep)) cat(sprintf("Dropped %d/%d rows with NA in a model covariate (%s).\n",
                            sum(!keep), nrow(df0), paste(model_cov, collapse = ", ")))
df0 <- df0[keep, ]

med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))

# Unit (per +1 g) ACME/ADE/Total for every timepoint AND every spec, from ONE
# npCBPS + the (timepoint-invariant) mediator fits + per-timepoint outcome fits.
# Returns an array[timepoints, n_spec, 3] (effect order acme/ade/total) per +1 g;
# the caller scales by the carb offset. NA on any failure.
#   LMER spec: lmer mediator (a) + lmer outcome (b_M, c').
#   QR spec:   Tobit/survreg mediator (a) + weighted rq outcome per tau (b_M, c').
# Mediator fits are timepoint-invariant; only the outcome models refit per tp.
fit_meal_unit <- function(d, verbose = FALSE) {
  na <- array(NA_real_, dim = c(length(timepoints), n_spec, 3))
  if (length(unique(d$subject_id)) < 3 || nrow(d) < 40) {
    if (verbose) message("  [point] too few subjects/rows: ", length(unique(d$subject_id)), " / ", nrow(d))
    return(na)
  }
  anchor <- switch(opt$`contrast-anchor`,
    global = rep(median(d$treat_meal_carbs, na.rm = TRUE), nrow(d)),
    mealtype = med_by(d$treat_meal_carbs, d$meal_type),
    subject = med_by(d$treat_meal_carbs, d$subject_id),
    `subject-mealtype` = med_by(d$treat_meal_carbs, interaction(d$subject_id, d$meal_type)))
  d$tc <- d$treat_meal_carbs - anchor
  fm <- as.formula(paste("treat_meal_carbs ~", paste(bal_cov, collapse = " + ")))
  w <- tryCatch({ utils::capture.output(.w <- npCBPS(fm, data = d, corprior = 0.01)$weights); .w },
                error = function(e) { if (verbose) message("  [point] npCBPS failed: ", conditionMessage(e)); NULL })
  if (is.null(w) || length(w) != nrow(d)) {   # NA-covariate row dropping desyncs weights
    if (verbose) message("  [point] npCBPS weights unusable (null or length mismatch)")
    return(na)
  }
  n <- nrow(d); w <- w * n / sum(w); cap <- quantile(w, 0.99); w <- pmin(w, cap); w <- w * n / sum(w)
  rhs <- paste(mod_cov, collapse = " + ")

  # Mediator treatment effect a, per family (timepoint-invariant).
  a_lmer <- NA_real_; a_tobit <- NA_real_
  if (need_lmer) {
    m <- tryCatch(lme4::lmer(as.formula(paste("mediator_bolus_for_meal ~ tc +", rhs, "+ (1|subject_id)")),
                             data = d, weights = w, REML = FALSE),
                  error = function(e) { if (verbose) message("  [point] mediator lmer failed: ", conditionMessage(e)); NULL })
    if (!is.null(m)) a_lmer <- as.numeric(lme4::fixef(m)["tc"])
  }
  if (need_qr) {
    mt <- tryCatch(survreg(as.formula(paste(
            "Surv(mediator_bolus_for_meal, mediator_bolus_for_meal > 0, type = 'left') ~ tc +", rhs)),
            data = d, weights = w, dist = "gaussian"),
          error = function(e) { if (verbose) message("  [point] Tobit mediator failed: ", conditionMessage(e)); NULL })
    if (!is.null(mt)) a_tobit <- as.numeric(stats::coef(mt)["tc"])
  }

  out <- na
  for (k in seq_along(timepoints)) {
    yv <- d[[sprintf("Y_%dmin", timepoints[k])]]
    ok <- !is.na(yv)                       # drop NA-outcome rows AND their weights together
    dk <- d[ok, , drop = FALSE]; dk$Y <- yv[ok]; wk <- w[ok]
    if (length(unique(dk$subject_id)) < 3 || nrow(dk) < 40) {
      if (verbose) message(sprintf("  [point] t=%d: %d rows after NA-Y drop -> skip", timepoints[k], nrow(dk)))
      next
    }
    # LMER outcome (mean) once per timepoint; reused by the lmer spec.
    y_lmer <- NULL
    if (need_lmer && is.finite(a_lmer)) {
      y_lmer <- tryCatch(lme4::lmer(as.formula(paste("Y ~ tc + mediator_bolus_for_meal +", rhs, "+ (1|subject_id)")),
                                    data = dk, weights = wk, REML = FALSE),
                         error = function(e) { if (verbose) message(sprintf("  [point] outcome lmer failed @t=%d: %s", timepoints[k], conditionMessage(e))); NULL })
    }
    for (si in seq_len(n_spec)) {
      sp <- specs[[si]]
      if (sp$model == "lmer") {
        if (is.null(y_lmer)) next
        b <- as.numeric(lme4::fixef(y_lmer)["mediator_bolus_for_meal"])
        cp <- as.numeric(lme4::fixef(y_lmer)["tc"])
        out[k, si, ] <- c(a_lmer * b, cp, a_lmer * b + cp)
      } else {                               # qr spec at sp$tau
        if (!is.finite(a_tobit)) next
        yq <- tryCatch(rq(as.formula(paste("Y ~ tc + mediator_bolus_for_meal +", rhs)),
                          tau = sp$tau, data = dk, weights = wk),
                       error = function(e) { if (verbose) message(sprintf("  [point] rq(tau=%.2f) failed @t=%d: %s", sp$tau, timepoints[k], conditionMessage(e))); NULL })
        if (is.null(yq)) next
        bq <- as.numeric(stats::coef(yq)["mediator_bolus_for_meal"])
        cpq <- as.numeric(stats::coef(yq)["tc"])
        out[k, si, ] <- c(a_tobit * bq, cpq, a_tobit * bq + cpq)
      }
    }
  }
  out
}

boot_resample <- function(d, ss) {
  picks <- sample(ss, length(ss), replace = TRUE)
  do.call(rbind, lapply(seq_along(picks), function(i) {
    sd <- d[d$subject_id == picks[i], ]; sd$subject_id <- paste0(picks[i], "_b", i); sd
  }))
}

summ <- function(pt, bcol) {
  v <- bcol[is.finite(bcol)]
  if (length(v) < 20) return(c(lo = NA, hi = NA, p = NA))
  ci <- quantile(v, c(0.025, 0.975))
  p <- min(1, 2 * min(mean(v <= 0), mean(v >= 0)))
  c(lo = as.numeric(ci[1]), hi = as.numeric(ci[2]), p = p)
}

ncores <- if (opt$cores > 0) opt$cores else max(1, parallel::detectCores() - 2)
cat(sprintf("\nDiaTrend bootstrap timepoint grid (arm: %s)\n", opt$`arm-tag`))
cat(sprintf("  spec: cohort=%s split=%s use_iob=%s demo_w=%s demo_m=%s | B=%d on %d cores\n",
            opt$cohort, opt$split, use_iob, demo_w, demo_m, opt$B, ncores))
cat(sprintf("  sweep: %d meals x %d timepoints x %d offsets x %d specs (LMER + QR taus: %s)\n",
            length(meals), length(timepoints), length(offsets), n_spec,
            if (length(taus)) paste(taus, collapse = ",") else "none"))

rows <- list()
for (meal in meals) {
  d <- if (toupper(meal) == "ALL") df0 else df0[tolower(df0$meal_type) == tolower(meal), ]
  n_ep <- nrow(d); n_sub <- length(unique(d$subject_id))
  cat(sprintf("\n[meal=%s] %d episodes, %d subjects -- point fit + %d subject resamples...\n",
              meal, n_ep, n_sub, opt$B))

  point_unit <- fit_meal_unit(d, verbose = TRUE)   # timepoints x 3, per +1 g; report any failure
  subs <- unique(d$subject_id)

  na_rep <- array(NA_real_, dim = c(length(timepoints), n_spec, 3))
  one_rep <- function(bi) {
    tryCatch(fit_meal_unit(boot_resample(d, subs)), error = function(e) na_rep)
  }
  RNGkind("L'Ecuyer-CMRG"); set.seed(opt$seed)
  reps <- parallel::mclapply(seq_len(opt$B), one_rep, mc.cores = ncores,
                             mc.set.seed = TRUE, mc.preschedule = FALSE)
  reps <- lapply(reps, function(r) if (is.array(r) && all(dim(r) == dim(na_rep))) r else na_rep)
  # boot_unit[timepoint, spec, effect, rep]
  boot_unit <- array(unlist(reps), dim = c(length(timepoints), n_spec, 3, length(reps)))

  for (off in offsets) {
    for (k in seq_along(timepoints)) {
      tp <- timepoints[k]
      for (si in seq_len(n_spec)) {
        sp <- specs[[si]]
        eff <- list()
        for (j in seq_len(3)) {              # 1=acme, 2=ade, 3=total
          # as.numeric() strips any name array/quantile extraction keeps; without
          # it c(est = pt) becomes "est.acme" and acme["est"] returns NA (point
          # blank while CI is fine -- the symptom we chased earlier).
          pt <- as.numeric(point_unit[k, si, j]) * off
          bc <- as.numeric(boot_unit[k, si, j, ]) * off
          s <- summ(pt, bc)
          eff[[j]] <- c(est = pt, lo = as.numeric(s["lo"]),
                        hi = as.numeric(s["hi"]), p = as.numeric(s["p"]))
        }
        acme <- eff[[1]]; ade <- eff[[2]]; total <- eff[[3]]
        prop <- if (is.finite(total["est"]) && abs(total["est"]) > 1e-8)
                  as.numeric(acme["est"] / total["est"]) else NA_real_
        rows[[length(rows) + 1]] <- data.frame(
          arm = opt$`arm-tag`, meal = meal, split = opt$split, model = sp$model, tau = sp$tau,
          offset_g = off, timepoint = tp, n_episodes = n_ep, n_subjects = n_sub,
          acme = acme["est"], acme_lo = acme["lo"], acme_hi = acme["hi"], acme_p = acme["p"],
          ade = ade["est"], ade_lo = ade["lo"], ade_hi = ade["hi"], ade_p = ade["p"],
          total = total["est"], total_lo = total["lo"], total_hi = total["hi"], total_p = total["p"],
          prop_mediated = prop, prop_p = NA_real_, row.names = NULL)
      }
    }
  }
}

grid <- do.call(rbind, rows)
grid <- grid[order(grid$meal, grid$offset_g, grid$timepoint), ]
grid_file <- file.path(opt$`output-dir`,
                       sprintf("grid_diatrend_%s_%s.csv", opt$`run-id`, opt$`arm-tag`))
write.csv(grid, grid_file, row.names = FALSE)

cat(sprintf("\nDone. %d cells -> grid:\n  %s\n", nrow(grid), grid_file))
cat("\nPlot (Ohio-style panels + tables, bootstrap CIs):\n")
cat(sprintf(paste0("  python visualization_code/generate_diatrend_mediation_outputs.py \\\n",
                   "      --grid %s \\\n",
                   "      --out mediation_results/diatrend/figures --inference bootstrap \\\n",
                   "      --table3-model lmer --table4-timepoint 120 --table4-offset 30\n"), grid_file))
cat("\nSignificant cells (any effect p < 0.05):\n")
sig <- grid[which(grid$acme_p < 0.05 | grid$ade_p < 0.05 | grid$total_p < 0.05), ]
if (nrow(sig) == 0) cat("  (none)\n") else
  print(sig[, c("meal", "offset_g", "timepoint", "acme", "acme_p", "ade", "ade_p", "total", "total_p")],
        row.names = FALSE)
