#!/usr/bin/env Rscript
# ================================================================
# Bolus Timing Relative to Meal
# ================================================================
# Shows distribution of when the first bolus occurs relative to
# the meal event. Negative = bolus before meal, positive = after.
#
# Usage:
#   Rscript visualization_code/bolus_timing_distribution.R
# ================================================================

library(dplyr)
library(ggplot2)

# Load combined RData
rdata_path <- "cma_cluster/analysis_data/z_meal_mediation_analysis_data_combined_5min.RData"
if (!file.exists(rdata_path)) {
  # Try alternate locations
  rdata_path <- file.path(getwd(), rdata_path)
}
cat(sprintf("Loading: %s\n", rdata_path))
load(rdata_path)

cat(sprintf("Loaded %d meal windows\n", length(aligned_data_list)))
cat(sprintf("Interval: %d minutes\n", interval_min))

# pre_ints = number of pre-meal time steps (meal occurs at index pre_ints + 1 in 1-based R)
meal_index <- pre_ints  # 0-based index of meal in the window

results <- data.frame(
  window_id = integer(),
  cohort = character(),
  subject_id = character(),
  meal_type = character(),
  first_bolus_index = integer(),
  bolus_relative_to_meal = integer(),
  bolus_minutes_before_meal = numeric(),
  bolus_for_meal = numeric(),
  has_bolus = logical(),
  stringsAsFactors = FALSE
)

for (i in seq_along(aligned_data_list)) {
  w <- aligned_data_list[[i]]

  # Find first non-zero bolus in the pre-meal + early post-meal window
  # Window: full aligned data (pre_ints before meal through post)
  bolus_indices <- which(w$bolus > 0)

  # Get metadata from all_cleaned_data
  meta <- all_cleaned_data[i, ]

  if (length(bolus_indices) > 0) {
    first_bolus <- min(bolus_indices)
    # Relative to meal: negative = before meal, positive = after
    # meal is at row (pre_ints + 1) in 1-based indexing
    relative_index <- first_bolus - (meal_index + 1)
    minutes_before <- -relative_index * interval_min

    results <- rbind(results, data.frame(
      window_id = i,
      cohort = if ("cohort" %in% names(meta)) meta$cohort else "unknown",
      subject_id = if ("subject_id" %in% names(meta)) as.character(meta$subject_id) else "unknown",
      meal_type = if ("meal_type" %in% names(meta)) meta$meal_type else "unknown",
      first_bolus_index = first_bolus,
      bolus_relative_to_meal = relative_index,
      bolus_minutes_before_meal = minutes_before,
      bolus_for_meal = meta$bolus_for_meal,
      has_bolus = TRUE
    ))
  } else {
    results <- rbind(results, data.frame(
      window_id = i,
      cohort = if ("cohort" %in% names(meta)) meta$cohort else "unknown",
      subject_id = if ("subject_id" %in% names(meta)) as.character(meta$subject_id) else "unknown",
      meal_type = if ("meal_type" %in% names(meta)) meta$meal_type else "unknown",
      first_bolus_index = NA,
      bolus_relative_to_meal = NA,
      bolus_minutes_before_meal = NA,
      bolus_for_meal = meta$bolus_for_meal,
      has_bolus = FALSE
    ))
  }
}

# ================================================================
# SUMMARY
# ================================================================

cat("\n========================================\n")
cat("BOLUS TIMING SUMMARY\n")
cat("========================================\n\n")

cat(sprintf("Total windows: %d\n", nrow(results)))
cat(sprintf("Windows with bolus: %d (%.1f%%)\n",
            sum(results$has_bolus), 100 * mean(results$has_bolus)))
cat(sprintf("Windows without bolus: %d (%.1f%%)\n\n",
            sum(!results$has_bolus), 100 * mean(!results$has_bolus)))

bolus_windows <- results %>% filter(has_bolus)

cat("First bolus timing relative to meal (minutes):\n")
cat(sprintf("  Mean: %.1f min (negative = before meal)\n", mean(bolus_windows$bolus_minutes_before_meal)))
cat(sprintf("  Median: %.1f min\n", median(bolus_windows$bolus_minutes_before_meal)))
cat(sprintf("  Range: [%.0f, %.0f] min\n",
            min(bolus_windows$bolus_minutes_before_meal),
            max(bolus_windows$bolus_minutes_before_meal)))

cat(sprintf("\nBolus BEFORE meal: %d (%.1f%%)\n",
            sum(bolus_windows$bolus_minutes_before_meal > 0),
            100 * mean(bolus_windows$bolus_minutes_before_meal > 0)))
cat(sprintf("Bolus AT meal:     %d (%.1f%%)\n",
            sum(bolus_windows$bolus_minutes_before_meal == 0),
            100 * mean(bolus_windows$bolus_minutes_before_meal == 0)))
cat(sprintf("Bolus AFTER meal:  %d (%.1f%%)\n",
            sum(bolus_windows$bolus_minutes_before_meal < 0),
            100 * mean(bolus_windows$bolus_minutes_before_meal < 0)))

# By cohort
cat("\n--- By Cohort ---\n")
cohort_summary <- bolus_windows %>%
  group_by(cohort) %>%
  summarise(
    n = n(),
    pct_before = 100 * mean(bolus_minutes_before_meal > 0),
    pct_at = 100 * mean(bolus_minutes_before_meal == 0),
    pct_after = 100 * mean(bolus_minutes_before_meal < 0),
    mean_timing = mean(bolus_minutes_before_meal),
    median_timing = median(bolus_minutes_before_meal),
    .groups = "drop"
  )
print(cohort_summary)

# By cohort and meal type
cat("\n--- By Cohort and Meal Type ---\n")
cohort_meal_summary <- bolus_windows %>%
  group_by(cohort, meal_type) %>%
  summarise(
    n = n(),
    pct_before = 100 * mean(bolus_minutes_before_meal > 0),
    mean_timing = mean(bolus_minutes_before_meal),
    .groups = "drop"
  )
print(cohort_meal_summary, n = 20)

# ================================================================
# PLOT
# ================================================================

p <- ggplot(bolus_windows, aes(x = bolus_minutes_before_meal, fill = cohort)) +
  geom_histogram(binwidth = 5, position = "dodge", alpha = 0.7, color = "black", linewidth = 0.3) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
  labs(
    title = "Distribution of First Bolus Timing Relative to Meal",
    subtitle = "Positive = bolus before meal, Negative = bolus after meal",
    x = "Minutes Before Meal (negative = after)",
    y = "Count",
    fill = "Cohort"
  ) +
  theme_minimal() +
  theme(legend.position = "top")

# Output directories live under visualizations/ (separate from code)
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    normalizePath(dirname(sub("^--file=", "", file_arg)))
  } else {
    file.path(getwd(), "visualization_code", "data_distribution")
  }
})
project_root <- normalizePath(file.path(script_dir, "..", ".."))

figures_dir <- file.path(project_root, "visualizations", "data_distribution", "figures")
tables_dir <- file.path(project_root, "visualizations", "data_distribution", "tables")
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)

output_path <- file.path(figures_dir, "bolus_timing_distribution.pdf")
ggsave(output_path, p, width = 10, height = 6)
cat(sprintf("\nPlot saved to: %s\n", output_path))

# Also save results CSV
csv_path <- file.path(tables_dir, "bolus_timing_data.csv")
write.csv(results, csv_path, row.names = FALSE)
cat(sprintf("Data saved to: %s\n", csv_path))
