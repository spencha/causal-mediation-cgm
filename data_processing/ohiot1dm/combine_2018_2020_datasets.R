#!/usr/bin/env Rscript
# =============================================================================
# combine_2018_2020_datasets.R
# Combines 2018 and 2020 OhioT1DM datasets for unified analysis
# =============================================================================

library(dplyr)

# --- START CONFIG BLOCK ---
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

# Find config.R (check multiple locations)
config_locations <- c(
  file.path(script_dir, "config.R"),
  file.path(script_dir, "..", "..", "cma_cluster", "config.R"),
  file.path(getwd(), "cma_cluster", "config.R")
)

config_loaded <- FALSE
for (cfg_path in config_locations) {
  if (file.exists(cfg_path)) {
    source(cfg_path)
    config_loaded <- TRUE
    break
  }
}

if (!config_loaded) {
  stop("Could not find config.R. Please run from project root or set CAUSAL_AE_BASE_DIR")
}
# --- END CONFIG BLOCK ---

cat("\n" , rep("=", 60), "\n")
cat("Combining 2018 and 2020 OhioT1DM Datasets\n")
cat(rep("=", 60), "\n\n")

# Load 2018 dataset
cat("Loading 2018 dataset...\n")
load(file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2018_5min.RData"))
data_2018 <- all_cleaned_data
aligned_2018 <- aligned_data_list
y_seq_2018 <- y_seq
y_seq_change_2018 <- y_seq_change

# Add cohort identifier if not present
if (!"cohort" %in% names(data_2018)) {
  data_2018$cohort <- "2018"
}

# Ensure subject_id is character
data_2018$subject_id <- as.character(data_2018$subject_id)

cat(sprintf("  2018 windows: %d\n", nrow(data_2018)))
cat(sprintf("  2018 subjects: %d\n", length(unique(data_2018$subject_id))))

# Load 2020 dataset
cat("\nLoading 2020 dataset...\n")
load(file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_2020_5min.RData"))
data_2020 <- all_cleaned_data
aligned_2020 <- aligned_data_list
y_seq_2020 <- y_seq
y_seq_change_2020 <- y_seq_change

# Add cohort identifier if not present
if (!"cohort" %in% names(data_2020)) {
  data_2020$cohort <- "2020"
}

# Ensure subject_id is character
data_2020$subject_id <- as.character(data_2020$subject_id)

cat(sprintf("  2020 windows: %d\n", nrow(data_2020)))
cat(sprintf("  2020 subjects: %d\n", length(unique(data_2020$subject_id))))

# Verify no window_id collisions
cat("\nVerifying unique window IDs...\n")
ids_2018 <- data_2018$window_id
ids_2020 <- data_2020$window_id
overlap <- intersect(ids_2018, ids_2020)

if (length(overlap) > 0) {
  stop(sprintf("Window ID collision detected! %d overlapping IDs.", length(overlap)))
}
cat("  No collisions detected.\n")

# Combine datasets
cat("\nCombining datasets...\n")
all_cleaned_data <- bind_rows(data_2018, data_2020)
aligned_data_list <- c(aligned_2018, aligned_2020)
y_seq <- rbind(y_seq_2018, y_seq_2020)
y_seq_change <- rbind(y_seq_change_2018, y_seq_change_2020)

# Verify unique window_ids
stopifnot(length(unique(all_cleaned_data$window_id)) == nrow(all_cleaned_data))

cat(sprintf("\n=== Combined Dataset Summary ===\n"))
cat(sprintf("Total windows: %d\n", nrow(all_cleaned_data)))
cat(sprintf("Total subjects: %d\n", length(unique(all_cleaned_data$subject_id))))
cat(sprintf("  2018: %d windows from %d subjects\n",
            sum(all_cleaned_data$cohort == "2018"),
            length(unique(all_cleaned_data$subject_id[all_cleaned_data$cohort == "2018"]))))
cat(sprintf("  2020: %d windows from %d subjects\n",
            sum(all_cleaned_data$cohort == "2020"),
            length(unique(all_cleaned_data$subject_id[all_cleaned_data$cohort == "2020"]))))

# Meal type distribution
cat("\nMeal type distribution:\n")
print(table(all_cleaned_data$meal_type, all_cleaned_data$cohort))

# Save combined dataset
output_path <- file.path(CONFIG$ANALYSIS_DATA_DIR, "z_meal_mediation_analysis_data_combined_5min.RData")
save(
  all_cleaned_data,
  aligned_data_list,
  y_seq,
  y_seq_change,
  pre_ints, post_X_ints, post_total_ints, interval_min,
  file = output_path
)

cat(sprintf("\nSaved combined dataset to: %s\n", output_path))
cat(rep("=", 60), "\n")
