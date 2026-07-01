#!/usr/bin/env Rscript
# Subject-cluster bootstrap for the DiaTrend mediation headline cells, with
# optional demographic moderation (stratified ACME/ADE + between-stratum
# difference). This is the rigorous inference engine: it resamples SUBJECTS
# (not episodes) with replacement, re-fits the npCBPS weights and the LMER
# mediator/outcome models on each resample, and reports percentile CIs and
# bootstrap p-values -- the right interval given clustered data, few subjects,
# and the influence we documented.
#
# For the LMER (linear, no treat x mediator interaction) primary spec the
# estimands are closed-form from the fitted coefficients:
#   ACME  = a * b * delta      (a = treat->mediator, b = mediator->outcome)
#   ADE   = cprime * delta      (cprime = treat->outcome | mediator)
#   Total = (a*b + cprime) * delta
# so no mediate()/sims call is needed per replicate -- only two lmer fits.
#
# Usage:
#   # pooled primary endpoint, primary spec:
#   Rscript cma_cluster/diatrend/cluster_bootstrap_mediation.R \
#       --phi-file analysis_data/diatrend/embeddings/phi_embeddings_diatrend_demo.csv \
#       --split test --timepoint 120 --offset 30 --meal ALL \
#       --demographics-weights TRUE --demographics-models FALSE --B 1000
#
#   # moderation by HbA1c tertile (or --by sex):
#   ...same... --by hba1c

suppressPackageStartupMessages({
  library(optparse); library(lme4); library(CBPS); library(parallel)
})

option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL, help = "Embeddings CSV."),
  make_option(c("--cohort"), type = "character", default = "2", help = "Cohort filter. [2]"),
  make_option(c("--split"), type = "character", default = "test", help = "test/train/all. [test]"),
  make_option(c("--meal"), type = "character", default = "ALL", help = "Meal subset or ALL. [ALL]"),
  make_option(c("--timepoint"), type = "integer", default = 120, help = "Outcome timepoint. [120]"),
  make_option(c("--offset"), type = "integer", default = 30, help = "Carb offset g. [30]"),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype", help = "[mealtype]"),
  make_option(c("--balance-n-phi"), type = "integer", default = 6, help = "PCs in npCBPS. [6]"),
  make_option(c("--model-n-phi"), type = "integer", default = 3, help = "PCs in LMERs. [3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC", help = "[PC]"),
  make_option(c("--demographics-weights"), type = "logical", default = TRUE, help = "Balance demos in npCBPS. [TRUE]"),
  make_option(c("--demographics-models"), type = "logical", default = FALSE, help = "Demos as LMER fixed FX. [FALSE]"),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = "Use iob_at_meal (covariate + filter). FALSE for the no-IOB / full-cohort arms. [TRUE]"),
  make_option(c("--by"), type = "character", default = "none", help = "Moderator: none / sex / hba1c. [none]"),
  make_option(c("--B"), type = "integer", default = 1000, help = "Bootstrap resamples. [1000]"),
  make_option(c("--seed"), type = "integer", default = 42, help = "RNG seed. [42]"),
  make_option(c("--cores"), type = "integer", default = 0,
    help = "Parallel workers for the bootstrap (0 = detectCores()-2). [0]")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`phi-file`)) stop("--phi-file is required.")
set.seed(opt$seed)
demo_w <- isTRUE(opt$`demographics-weights`); demo_m <- isTRUE(opt$`demographics-models`)
use_iob <- isTRUE(opt$`use-iob`)
by <- tolower(opt$by)
if (!by %in% c("none", "sex", "hba1c")) stop("--by must be none / sex / hba1c.")

# ---- Load + filter ----
df <- read.csv(opt$`phi-file`, stringsAsFactors = FALSE)
df <- df[df$cohort %in% as.integer(strsplit(opt$cohort, ",")[[1]]), ]
if (tolower(opt$split) != "all") {
  if (!"split" %in% colnames(df)) stop("No 'split' column.")
  df <- df[df$split == opt$split, ]
}
if (use_iob) df <- df[!is.na(df$iob_at_meal), ]
if (demo_w || demo_m || by != "none") {
  need <- c("demo_age", "demo_sex", "demo_hba1c")
  if (!all(need %in% colnames(df))) stop("Missing demographic columns.")
  df <- df[stats::complete.cases(df[, need]), ]
  df$demo_sex <- as.factor(df$demo_sex)
}
if (toupper(opt$meal) != "ALL") df <- df[tolower(df$meal_type) == tolower(opt$meal), ]
outcome_col <- sprintf("Y_%dmin", opt$timepoint)
if (!outcome_col %in% colnames(df)) stop(sprintf("Outcome column %s missing.", outcome_col))
df$Y <- df[[outcome_col]]

# Moderator stratum (subject-level; HbA1c tertiles fixed from full sample) ----
if (by == "sex") {
  df$stratum <- as.character(df$demo_sex)
} else if (by == "hba1c") {
  subj_a1c <- tapply(df$demo_hba1c, df$subject_id, function(v) v[1])
  br <- quantile(subj_a1c, c(1/3, 2/3), na.rm = TRUE)
  cat(sprintf("HbA1c tertile breaks (subject-level): %.1f, %.1f\n", br[1], br[2]))
  df$stratum <- cut(df$demo_hba1c, breaks = c(-Inf, br, Inf), labels = c("Low", "Mid", "High"))
  df$stratum <- as.character(df$stratum)
} else {
  df$stratum <- "all"
}

pc_bal <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`balance-n-phi`))
pc_mod <- paste0(opt$`covariate-prefix`, "_", seq_len(opt$`model-n-phi`))
bal_cov <- c(pc_bal, "glucose_at_meal", if (use_iob) "iob_at_meal", if (demo_w) c("demo_age", "demo_sex", "demo_hba1c"))
mod_cov <- c(pc_mod, if (use_iob) "iob_at_meal", if (demo_m) c("demo_age", "demo_sex", "demo_hba1c"))
# When stratifying by sex, demo_sex is constant within each stratum: a one-level
# factor breaks npCBPS / model.matrix. Drop it from the covariate sets. (HbA1c
# is continuous and still varies within tertiles, so it is kept.)
if (by == "sex") {
  bal_cov <- setdiff(bal_cov, "demo_sex")
  mod_cov <- setdiff(mod_cov, "demo_sex")
}
med_by <- function(x, g) ave(x, g, FUN = function(v) median(v, na.rm = TRUE))
delta <- opt$offset

# ACME/ADE/Total for one (already-subset) data frame via product-of-coefficients.
fit_cell <- function(d) {
  if (length(unique(d$subject_id)) < 3 || nrow(d) < 40) return(c(acme = NA, ade = NA, total = NA))
  anchor <- switch(opt$`contrast-anchor`,
    global = rep(median(d$treat_meal_carbs, na.rm = TRUE), nrow(d)),
    mealtype = med_by(d$treat_meal_carbs, d$meal_type),
    subject = med_by(d$treat_meal_carbs, d$subject_id),
    `subject-mealtype` = med_by(d$treat_meal_carbs, interaction(d$subject_id, d$meal_type)))
  d$tc <- d$treat_meal_carbs - anchor
  fm <- as.formula(paste("treat_meal_carbs ~", paste(bal_cov, collapse = " + ")))
  # capture.output() suppresses npCBPS's "Estimating npCBPS..." print so 48
  # parallel workers don't flood the log; the fitted weights are unaffected.
  w <- tryCatch({ utils::capture.output(.w <- npCBPS(fm, data = d, corprior = 0.01)$weights); .w },
                error = function(e) NULL)
  if (is.null(w)) return(c(acme = NA, ade = NA, total = NA))
  n <- nrow(d); w <- w * n / sum(w); cap <- quantile(w, 0.99); w <- pmin(w, cap); w <- w * n / sum(w)
  rhs <- paste(mod_cov, collapse = " + ")
  m <- tryCatch(lme4::lmer(as.formula(paste("mediator_bolus_for_meal ~ tc +", rhs, "+ (1|subject_id)")),
                           data = d, weights = w, REML = FALSE), error = function(e) NULL)
  y <- tryCatch(lme4::lmer(as.formula(paste("Y ~ tc + mediator_bolus_for_meal +", rhs, "+ (1|subject_id)")),
                           data = d, weights = w, REML = FALSE), error = function(e) NULL)
  if (is.null(m) || is.null(y)) return(c(acme = NA, ade = NA, total = NA))
  a <- lme4::fixef(m)["tc"]; b <- lme4::fixef(y)["mediator_bolus_for_meal"]; cp <- lme4::fixef(y)["tc"]
  acme <- as.numeric(a * b * delta); ade <- as.numeric(cp * delta)
  c(acme = acme, ade = ade, total = acme + ade)
}

# Estimates for every stratum (and, if moderating, the High-Low difference) ----
strata <- sort(unique(df$stratum))
estimate_all <- function(d) {
  out <- list()
  for (s in strata) out[[s]] <- fit_cell(d[d$stratum == s, ])
  if (by == "hba1c" && all(c("Low", "High") %in% strata)) {
    out[["High_minus_Low"]] <- out[["High"]] - out[["Low"]]
  } else if (by == "sex" && length(strata) == 2) {
    out[[paste0(strata[2], "_minus_", strata[1])]] <- out[[strata[2]]] - out[[strata[1]]]
  }
  out
}

cat(sprintf("\nSpec: demo in weights=%s, models=%s | moderator=%s | meal=%s, %d min, +%d g\n",
            demo_w, demo_m, by, opt$meal, opt$timepoint, opt$offset))
for (s in strata) cat(sprintf("  stratum %-6s: %d episodes, %d subjects\n",
                              s, sum(df$stratum == s), length(unique(df$subject_id[df$stratum == s]))))

point <- estimate_all(df)

# ---- Subject-cluster bootstrap ----
subs <- unique(df$subject_id)
boot_resample <- function(d, ss) {
  picks <- sample(ss, length(ss), replace = TRUE)
  do.call(rbind, lapply(seq_along(picks), function(i) {
    sd <- d[d$subject_id == picks[i], ]; sd$subject_id <- paste0(picks[i], "_b", i); sd
  }))
}
labels <- names(point)
ncol_b <- length(labels) * 3
col_b <- as.vector(t(outer(labels, c("acme", "ade", "total"), paste, sep = ".")))
ncores <- if (opt$cores > 0) opt$cores else max(1, parallel::detectCores() - 2)
cat(sprintf("\nBootstrapping %d subject-resamples on %d cores (re-fitting weights + models each)...\n",
            opt$B, ncores))

# One resample -> a length-ncol_b numeric row; any failure (singular npCBPS,
# killed worker) degrades to an all-NA row that is dropped at summary time.
one_rep <- function(bi) {
  tryCatch({
    est <- estimate_all(boot_resample(df, subs))
    v <- unlist(lapply(labels, function(l) est[[l]][c("acme", "ade", "total")]))
    if (length(v) == ncol_b) as.numeric(v) else rep(NA_real_, ncol_b)
  }, error = function(e) rep(NA_real_, ncol_b))
}

# L'Ecuyer-CMRG gives each resample an independent, reproducible RNG stream
# (mc.preschedule=FALSE => one stream per task, so results are stable).
RNGkind("L'Ecuyer-CMRG"); set.seed(opt$seed)
reps <- parallel::mclapply(seq_len(opt$B), one_rep, mc.cores = ncores,
                           mc.set.seed = TRUE, mc.preschedule = FALSE)
reps <- lapply(reps, function(r) if (is.numeric(r) && length(r) == ncol_b) r else rep(NA_real_, ncol_b))
boot <- do.call(rbind, reps)
colnames(boot) <- col_b

# ---- Report: point, percentile 95% CI, bootstrap two-sided p ----
summ <- function(pt, bcol) {
  v <- bcol[is.finite(bcol)]
  if (length(v) < 20) return(sprintf("%8.2f  [   n/a    ]  p=  n/a", pt))
  ci <- quantile(v, c(0.025, 0.975))
  p <- 2 * min(mean(v <= 0), mean(v >= 0)); p <- min(1, p)
  sprintf("%8.2f  [%7.2f,%7.2f]  p=%.3f", pt, ci[1], ci[2], p)
}
cat("\n", strrep("=", 78), "\n", sep = "")
cat(sprintf("%-18s %-8s %s\n", "estimand", "effect", "  point      95% CI (subject bootstrap)     p"))
cat(strrep("-", 78), "\n", sep = "")
for (l in labels) {
  tag <- if (grepl("minus", l)) paste0(l, "  <-- moderation") else l
  for (e in c("acme", "ade", "total")) {
    cat(sprintf("%-18s %-8s %s\n", if (e == "acme") tag else "", toupper(e),
                summ(point[[l]][e], boot[, paste(l, e, sep = ".")])))
  }
  cat(strrep("-", 78), "\n", sep = "")
}
cat("Inference = percentile bootstrap over subjects; p = 2*min(Pr<=0, Pr>=0).\n")
if (by != "none") cat("A '..._minus_...' CI excluding 0 = evidence the effect differs across strata (moderation).\n")
