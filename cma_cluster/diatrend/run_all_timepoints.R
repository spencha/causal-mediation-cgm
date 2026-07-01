#!/usr/bin/env Rscript
# DiaTrend: sweep the mediation analysis over timepoints x offsets x models
# x meal types for ONE arm, producing a Table-3/4-style grid.
#
# Computes the arm's npCBPS weights ONCE (balance on the first --balance-n-phi
# PCs + glucose_at_meal + iob [+ cohort + demographics]), then loops over
# meals x timepoints x offsets x models, calling run_mixed_effects_mediation.R
# with --weights-file. Each cell uses --model-n-phi PCs in the models (the
# manuscript's 3) and the meal-type-centered contrast. Finally collects every
# cell's .rds into one grid CSV.
#
# Run once per arm (demographics decoupled: weights vs models):
#   # no demographics anywhere (OhioT1DM-literal, full IOB sample):
#   Rscript run_all_timepoints.R --phi-file <demo CSV> --arm-tag base
#   # demographics balanced in the weights only, random-intercept model (primary):
#   Rscript run_all_timepoints.R --phi-file <demo CSV> --arm-tag demowt \
#       --demographics-weights TRUE --demographics-models FALSE
#   # demographics in both weights and models (old "demographics" arm):
#   Rscript run_all_timepoints.R --phi-file <demo CSV> --arm-tag demographics \
#       --demographics-weights TRUE --demographics-models TRUE

suppressPackageStartupMessages({
  script_dir <- tryCatch({
    normalizePath(dirname(sys.frame(1)$ofile))
  }, error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    if (length(file_arg) > 0) normalizePath(dirname(sub("^--file=", "", file_arg))) else getwd()
  })
  config_locations <- c(
    file.path(script_dir, "..", "config.R"),
    file.path(script_dir, "..", "..", "cma_cluster", "config.R"),
    file.path(getwd(), "cma_cluster", "config.R")
  )
  for (cfg_path in config_locations) {
    if (file.exists(cfg_path)) { source(cfg_path); break }
  }
  if (!exists("CONFIG")) stop("Could not find config.R.")
  library(optparse)
})

option_list <- list(
  make_option(c("--phi-file"), type = "character", default = NULL,
    help = "Embeddings CSV (demographics-merged if --demographics). Required."),
  make_option(c("--arm-tag"), type = "character", default = "base",
    help = paste(
      "Arm label for filenames. 'base' = the OhioT1DM-literal spec (no",
      "demographic covariates), directly comparable to the Ohio figures.",
      "[default: base]")),
  make_option(c("--demographics-weights"), type = "logical", default = FALSE,
    help = paste(
      "Add demographics (age/sex/HbA1c) to the npCBPS propensity model so the",
      "weights balance them. Requires a demographics-merged --phi-file; drops",
      "demographics-incomplete subjects. [default: FALSE]")),
  make_option(c("--demographics-models"), type = "logical", default = FALSE,
    help = paste(
      "Add demographics as fixed effects in the LMER mediator/outcome models.",
      "Inert with the subject random intercept (subject-constant covariates),",
      "so default FALSE. [default: FALSE]")),
  make_option(c("--cohort"), type = "character", default = "2",
    help = "Cohort filter. [default: 2]"),
  make_option(c("--balance-cohort"), type = "logical", default = TRUE,
    help = paste(
      "Add a cohort indicator to the npCBPS propensity model when >1 cohort.",
      "Set FALSE for the full-cohort demographics arm (cohort is collinear with",
      "age/HbA1c, which already capture it). [default: TRUE]")),
  make_option(c("--use-iob"), type = "logical", default = TRUE,
    help = paste(
      "Use iob_at_meal as a pre-treatment covariate (weights + models). When",
      "FALSE, the IOB-availability filter is also off, recovering IOB-missing",
      "episodes/subjects -- the no-IOB inclusivity arm. [default: TRUE]")),
  make_option(c("--split"), type = "character", default = "test",
    help = paste(
      "Within-subject temporal split to analyse: 'test' (OhioT1DM-matched",
      "held-out set; default), 'train', or 'all'. Applied to both the npCBPS",
      "weighting and every mediation cell. [default: test]"
    )),
  make_option(c("--balance-n-phi"), type = "integer", default = 6,
    help = "PCs balanced in npCBPS. [default: 6]"),
  make_option(c("--model-n-phi"), type = "integer", default = 3,
    help = "PCs used as covariates in the models. [default: 3]"),
  make_option(c("--covariate-prefix"), type = "character", default = "PC",
    help = "Covariate column prefix. [default: PC]"),
  make_option(c("--meals"), type = "character", default = "ALL,breakfast,lunch,dinner,snack",
    help = "Comma-separated meal subsets. [default: ALL,breakfast,lunch,dinner,snack]"),
  make_option(c("--timepoints"), type = "character",
    default = paste(seq(60, 210, by = 5), collapse = ","),
    help = paste(
      "Comma-separated outcome timepoints (min). Default is every 5 min from",
      "60 to 210 (31 points), matching the OhioT1DM sweep (seq(60,210,by=5)).",
      "The embeddings carry Y_60min..Y_210min at 5-min resolution.")),
  make_option(c("--offsets"), type = "character", default = "30",
    help = "Comma-separated carb offsets (g). [default: 30]"),
  make_option(c("--models"), type = "character", default = "lmer,qr50",
    help = "Comma-separated model tokens: lmer, qr25, qr50, qr75. [default: lmer,qr50]"),
  make_option(c("--contrast-anchor"), type = "character", default = "mealtype",
    help = "Contrast anchor passed through. [default: mealtype]"),
  make_option(c("--sims"), type = "integer", default = 1000,
    help = "Monte Carlo sims. [default: 1000]"),
  make_option(c("--output-dir"), type = "character", default = NULL,
    help = "Mediation output dir. Defaults to CONFIG$DIATREND_MEDIATION_RESULTS_DIR."),
  make_option(c("--weights-dir"), type = "character", default = NULL,
    help = "npCBPS weights dir. Defaults to CONFIG$DIATREND_WEIGHTS_DIR."),
  make_option(c("--run-id"), type = "character", default = NULL,
    help = "Batch id (shared by all cells). Defaults to a timestamp.")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`phi-file`)) stop("--phi-file is required.")
if (is.null(opt$`output-dir`)) opt$`output-dir` <- CONFIG$DIATREND_MEDIATION_RESULTS_DIR
if (is.null(opt$`weights-dir`)) opt$`weights-dir` <- CONFIG$DIATREND_WEIGHTS_DIR
if (is.null(opt$`run-id`)) opt$`run-id` <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)
dir.create(opt$`weights-dir`, recursive = TRUE, showWarnings = FALSE)

weights_script <- file.path(script_dir, "npcbps_weights.R")
mediation_script <- file.path(script_dir, "run_mixed_effects_mediation.R")
for (s in c(weights_script, mediation_script)) if (!file.exists(s)) stop("missing: ", s)

batch_id <- sprintf("%s_%s", opt$`run-id`, opt$`arm-tag`)
meals <- trimws(strsplit(opt$meals, ",")[[1]])
timepoints <- as.integer(trimws(strsplit(opt$timepoints, ",")[[1]]))
offsets <- as.integer(trimws(strsplit(opt$offsets, ",")[[1]]))
model_tokens <- trimws(strsplit(opt$models, ",")[[1]])

parse_model <- function(tok) {
  if (tok == "lmer") list(model = "lmer", tau = 0.5, tag = "lmer")
  else if (grepl("^qr[0-9]+$", tok)) {
    tau <- as.integer(sub("qr", "", tok)) / 100
    list(model = "qr", tau = tau, tag = sprintf("qr%02d", round(tau * 100)))
  } else stop("bad model token: ", tok)
}
models <- lapply(model_tokens, parse_model)

# ---- Step 1: npCBPS weights for this arm (once) ----
cat("\n", strrep("=", 64), "\n  npCBPS weights (arm: ", opt$`arm-tag`, ")\n",
    strrep("=", 64), "\n", sep = "")
w_args <- c(weights_script,
  "--phi-file", opt$`phi-file`, "--cohort", opt$cohort, "--use-iob", as.character(opt$`use-iob`),
  "--split", opt$split,
  "--demographics", as.character(opt$`demographics-weights`),
  "--balance-cohort", as.character(opt$`balance-cohort`),
  "--covariate-prefix", opt$`covariate-prefix`, "--n-phi", as.character(opt$`balance-n-phi`),
  "--output-dir", opt$`weights-dir`, "--run-id", batch_id)
cat("$ Rscript", paste(w_args, collapse = " "), "\n")
if (system2("Rscript", w_args) != 0) stop("npCBPS weighting failed.")
cohort_tag <- paste(as.integer(strsplit(opt$cohort, ",")[[1]]), collapse = "")
iob_tag <- if (isTRUE(opt$`use-iob`)) "iob" else "noiob"
weights_file <- file.path(opt$`weights-dir`,
  sprintf("npcbps_weights_c%s_%s_%s.csv", cohort_tag, iob_tag, batch_id))
if (!file.exists(weights_file)) stop("expected weights file not found: ", weights_file)

# ---- Step 2: sweep cells ----
n_cells <- length(meals) * length(timepoints) * length(offsets) * length(models)
cat(sprintf("\nSweeping %d cells: %d meals x %d timepoints x %d offsets x %d models\n",
            n_cells, length(meals), length(timepoints), length(offsets), length(models)))
cell <- 0
for (meal in meals) {
  for (mdl in models) {
    for (off in offsets) {
      for (tp in timepoints) {
        cell <- cell + 1
        cat(sprintf("\n[cell %d/%d] meal=%s model=%s offset=%d t=%d\n",
                    cell, n_cells, meal, mdl$tag, off, tp))
        m_args <- c(mediation_script,
          "--weights-file", weights_file, "--meal", meal, "--split", opt$split,
          "--covariate-prefix", opt$`covariate-prefix`, "--n-phi", as.character(opt$`model-n-phi`),
          "--use-iob", as.character(opt$`use-iob`), "--demographics", as.character(opt$`demographics-models`),
          "--model", mdl$model, "--quantile", as.character(mdl$tau),
          "--offset", as.character(off), "--timepoint", as.character(tp),
          "--contrast-anchor", opt$`contrast-anchor`, "--sims", as.character(opt$sims),
          "--output-dir", opt$`output-dir`, "--arm-tag", opt$`arm-tag`, "--run-id", batch_id)
        status <- system2("Rscript", m_args)
        if (status != 0) warning(sprintf("cell failed: meal=%s model=%s off=%d t=%d", meal, mdl$tag, off, tp))
      }
    }
  }
}

# ---- Step 3: collect all cell .rds into one grid CSV ----
cat("\n", strrep("=", 64), "\n  Collecting grid\n", strrep("=", 64), "\n", sep = "")
rds_files <- list.files(opt$`output-dir`, pattern = sprintf("_%s\\.rds$", batch_id), full.names = TRUE)
get4 <- function(v) c(est = v["est"], lo = v["lo"], hi = v["hi"], p = v["p"])
rows <- lapply(rds_files, function(f) {
  r <- readRDS(f)
  data.frame(
    arm = r$arm_tag, meal = r$meal, split = r$split, model = r$model,
    tau = r$quantile_tau, offset_g = r$offset_g, timepoint = r$timepoint_min,
    n_episodes = r$n_episodes, n_subjects = r$n_subjects,
    acme = r$acme["est"], acme_lo = r$acme["lo"], acme_hi = r$acme["hi"], acme_p = r$acme["p"],
    ade = r$ade["est"], ade_lo = r$ade["lo"], ade_hi = r$ade["hi"], ade_p = r$ade["p"],
    total = r$total["est"], total_lo = r$total["lo"], total_hi = r$total["hi"], total_p = r$total["p"],
    prop_mediated = r$prop_mediated["est"], prop_p = r$prop_mediated["p"],
    row.names = NULL
  )
})
grid <- do.call(rbind, rows)
grid <- grid[order(grid$meal, grid$model, grid$offset_g, grid$timepoint), ]
grid_file <- file.path(opt$`output-dir`, sprintf("grid_diatrend_%s.csv", batch_id))
write.csv(grid, grid_file, row.names = FALSE)

cat(sprintf("\nDone. %d cells -> grid:\n  %s\n", nrow(grid), grid_file))
cat("\nSignificant cells (any effect p < 0.05):\n")
sig <- grid[grid$acme_p < 0.05 | grid$ade_p < 0.05 | grid$total_p < 0.05, ]
if (nrow(sig) == 0) cat("  (none)\n") else {
  print(sig[, c("meal", "model", "offset_g", "timepoint",
                "acme", "acme_p", "ade", "ade_p", "total", "total_p")], row.names = FALSE)
}
