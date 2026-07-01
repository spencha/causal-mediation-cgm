#!/usr/bin/env Rscript
# install_r_packages.R
# ====================
# Install required R packages for this project
# Run with: Rscript install_r_packages.R

packages <- c(
  # Causal inference
  "mediation",
  "CBPS",

  # Regression models
  "quantreg",
  "mgcv",
  "lme4",      # lmer mediator/outcome models (run_mixed_effects_mediation.R)
  "survival",  # survreg Tobit mediator (lm/qr model branches)
  
  # Data manipulation
  "dplyr",
  "tidyr",
  "readr",
  "purrr",
  
  # Visualization
  "ggplot2",
  
  # Utilities
  "optparse"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat(sprintf("%s: already installed\n", pkg))
  }
}

cat("Installing required R packages...\n\n")
invisible(lapply(packages, install_if_missing))
cat("\nDone!\n")
