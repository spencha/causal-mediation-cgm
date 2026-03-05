# config.R
# ========
# Centralized configuration for R scripts in the causal mediation project.
#
# This module handles all path definitions. It supports three configuration
# methods (in order of precedence):
#
# 1. Environment variables (for HPC/cluster use)
# 2. Local config file (config_local.R in project root, git-ignored)
# 3. Default paths relative to this file
#
# Usage:
#   source("config.R")  # or source("cma_cluster/config.R") from project root
#   data_path <- file.path(CONFIG$ANALYSIS_DATA_DIR, "phi_embeddings.csv")
# ========

# Helper function to get path from environment or default
get_config_path <- function(env_var, default) {
  env_value <- Sys.getenv(env_var)
  if (env_value != "") {
    return(env_value)
  }
  return(default)
}

# Determine project root
get_project_root <- function() {
  # Check environment variable first
  env_root <- Sys.getenv("CAUSAL_AE_BASE_DIR")
  if (env_root != "") {
    return(normalizePath(env_root, mustWork = FALSE))
  }
  
  # Try to find based on this script's location
  # This file is in cma_cluster/, so parent is project root
  script_dir <- tryCatch({
    dirname(sys.frame(1)$ofile)
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(script_dir) && script_dir != "") {
    candidate <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
    if (dir.exists(file.path(candidate, "ae_python_code"))) {
      return(candidate)
    }
  }
  
  # Try working directory
  if (dir.exists(file.path(getwd(), "ae_python_code"))) {
    return(normalizePath(getwd()))
  }
  
  # Try parent of working directory
  parent <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
  if (dir.exists(file.path(parent, "ae_python_code"))) {
    return(parent)
  }
  
  # Fallback to working directory
  return(normalizePath(getwd()))
}

# Load local config if it exists
load_local_config <- function(project_root) {
  local_config_path <- file.path(project_root, "config_local.R")
  if (file.exists(local_config_path)) {
    source(local_config_path, local = TRUE)
    return(TRUE)
  }
  return(FALSE)
}

# Build configuration list
build_config <- function() {
  project_root <- get_project_root()

  # Check for local config
  has_local <- load_local_config(project_root)

  # Base directory - use LOCAL_BASE_DIR if defined, else project_root
  if (exists("LOCAL_BASE_DIR", inherits = FALSE)) {
    base_dir <- LOCAL_BASE_DIR
  } else {
    base_dir <- get_config_path("CAUSAL_AE_BASE_DIR", project_root)
  }

  # Detect flat cluster layout: when running from ~/cma_cluster directly,
  # analysis_data/ is a direct subdirectory (not under cma_cluster/cma_cluster/).
  # This avoids doubled paths like ~/cma_cluster/cma_cluster/analysis_data/.
  is_flat_layout <- dir.exists(file.path(base_dir, "analysis_data")) &&
                    !dir.exists(file.path(base_dir, "ae_python_code"))

  # Helper to resolve cma_cluster-relative paths
  cma_dir <- if (is_flat_layout) base_dir else file.path(base_dir, "cma_cluster")

  config <- list(
    # Project structure
    PROJECT_ROOT = project_root,
    BASE_DIR = base_dir,

    # Code directories
    AE_CODE_DIR = file.path(project_root, "ae_python_code"),
    CMA_CLUSTER_DIR = file.path(project_root, "cma_cluster"),
    DATA_PROCESSING_DIR = file.path(project_root, "data_processing"),

    # Data directories
    ANALYSIS_DATA_DIR = get_config_path(
      "CAUSAL_AE_DATA_DIR",
      file.path(cma_dir, "analysis_data")
    ),

    # Results directories
    MEDIATION_RESULTS_DIR = get_config_path(
      "CAUSAL_AE_MEDIATION_RESULTS_DIR",
      file.path(cma_dir, "mediation_results")
    ),

    # Figures
    FIGURES_DIR = get_config_path(
      "CAUSAL_AE_FIGURES_DIR",
      file.path(base_dir, "results_visualizations", "images")
    ),

    # Raw data (OhioT1DM)
    RAW_DATA_DIR = get_config_path(
      "CAUSAL_AE_RAW_DATA_DIR",
      file.path(base_dir, "OhioT1DM")
    ),

    # Meal windows for autoencoder
    MEAL_WINDOWS_DIR = file.path(base_dir, "ae_python_code", "meal_windows")
  )

  # Derived paths
  config$WEIGHTS_DIR <- file.path(config$ANALYSIS_DATA_DIR, "weights")

  # 2018 data paths
  config$OHIO_TRAIN_DIR <- file.path(config$RAW_DATA_DIR, "2018", "train")
  config$OHIO_TEST_DIR <- file.path(config$RAW_DATA_DIR, "2018", "test")

  # 2020 data paths
  config$OHIO_2020_TRAIN_DIR <- file.path(config$RAW_DATA_DIR, "2020", "train")
  config$OHIO_2020_TEST_DIR <- file.path(config$RAW_DATA_DIR, "2020", "test")

  # Meal windows directories for autoencoder
  config$MEAL_WINDOWS_COMBINED_DIR <- file.path(base_dir, "ae_python_code", "meal_windows_combined")

  # Horizon-specific embeddings directory
  config$HORIZON_EMBEDDINGS_DIR <- file.path(config$ANALYSIS_DATA_DIR, "horizon_embeddings")

  # Combined (2018 + 2020) train/test directories
  config$MEAL_WINDOWS_COMBINED_TRAIN_DIR <- file.path(base_dir, "ae_python_code", "meal_windows_combined", "train")
  config$MEAL_WINDOWS_COMBINED_TEST_DIR <- file.path(base_dir, "ae_python_code", "meal_windows_combined", "test")

  if (is_flat_layout) {
    cat("Note: Detected flat cluster layout (running from cma_cluster/ directly)\n")
  }

  return(config)
}

# Create output directories
ensure_dirs <- function(config) {
  # Note: HORIZON_EMBEDDINGS_DIR is created on-demand by scripts that use it
  dirs <- c(
    config$WEIGHTS_DIR,
    config$MEDIATION_RESULTS_DIR,
    config$FIGURES_DIR,
    config$MEAL_WINDOWS_COMBINED_TRAIN_DIR,
    config$MEAL_WINDOWS_COMBINED_TEST_DIR
  )

  for (d in dirs) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
  }
}

# Print configuration
print_config <- function(config) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat("PROJECT CONFIGURATION\n")
  cat(strrep("=", 60), "\n")
  cat(sprintf("  PROJECT_ROOT:          %s\n", config$PROJECT_ROOT))
  cat(sprintf("  BASE_DIR:              %s\n", config$BASE_DIR))
  cat(sprintf("  ANALYSIS_DATA_DIR:     %s\n", config$ANALYSIS_DATA_DIR))
  cat(sprintf("  MEDIATION_RESULTS_DIR: %s\n", config$MEDIATION_RESULTS_DIR))
  cat(sprintf("  FIGURES_DIR:           %s\n", config$FIGURES_DIR))
  cat(strrep("=", 60), "\n\n")
}

# Initialize global CONFIG object
CONFIG <- build_config()

# Print config when sourced interactively
if (interactive()) {
  print_config(CONFIG)
}
