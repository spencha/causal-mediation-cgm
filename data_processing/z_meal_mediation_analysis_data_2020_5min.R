# =============================================================================
# z_meal_mediation_analysis_data_2020_5min.R
# Creates meal-centered windows for 2020 OhioT1DM cohort
#
# Uses data_train/data_test splits from the original OhioT1DM dataset
# (based on study days) to produce:
#   - z_meal_mediation_analysis_data_2020_5min.RData       (full data)
#   - z_meal_mediation_analysis_data_2020_TRAIN_5min.RData (train split)
#   - z_meal_mediation_analysis_data_2020_TEST_5min.RData  (test split)
# =============================================================================

# Load necessary libraries
library(dplyr)
library(tidyr)
library(zoo)
library(splines)
library(mgcv)
library(CBPS)
library(stochtree)
set.seed(123)

# --- START CONFIG BLOCK ---
script_dir <- tryCatch({
  normalizePath(dirname(sys.frame(1)$ofile))
}, error = function(e) {
  getwd()
})

config_locations <- c(
  file.path(script_dir, "config.R"),
  file.path(script_dir, "..", "cma_cluster", "config.R"),
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

DATA_PROCESSING_DIR <- CONFIG$DATA_PROCESSING_DIR

# force meal-centred
Z_var <- "meal"

# Source preprocessing functions
source(file.path(DATA_PROCESSING_DIR, "mediation_analysis_preprocessing_functions.R"))

# Load 5-minute preprocessed 2020 data (contains data_full, data_train, data_test)
load(file.path(DATA_PROCESSING_DIR, "data_preprocessed_2020-5min.RData"))

# Window ID offset to avoid collisions with 2018 data
WINDOW_ID_OFFSET_2020 <- 10000

# Helper function to compute max consecutive NAs in a vector
max_consecutive_na <- function(x) {
  if (!any(is.na(x))) return(0L)
  r <- rle(is.na(x))
  na_runs <- r$lengths[r$values]
  if (length(na_runs) == 0) return(0L)
  return(max(na_runs))
}

# =============================================================================
# PROCESSING FUNCTION
# =============================================================================
# Extracts meal-centered windows from a list of subject data frames.
# This function encapsulates all window extraction, filtering, and outcome
# computation logic so it can be called on data_full, data_train, or data_test.
# =============================================================================

process_subject_list <- function(subject_list, window_id_offset = WINDOW_ID_OFFSET_2020,
                                 cohort_label = "2020") {

  aligned_data_list <- list()
  interval_min <- 5

  for (sid in seq_along(subject_list)) {
    sd <- subject_list[[sid]]
    subject_id_str <- paste0(cohort_label, "_", sid)

    # make meal_type NA when meal==0, keep it when meal>0
    sd$meal_type[sd$meal == 0] <- NA

    # precompute sleep runs
    is_sleep     <- sd$sleep_fraction > 0
    r            <- rle(is_sleep)
    ends         <- cumsum(r$lengths)
    starts       <- ends - r$lengths + 1
    sleep_runs   <- which(r$values)
    sleep_starts <- starts[sleep_runs]
    sleep_ends   <- ends[sleep_runs]
    sleep_durs   <- (sleep_ends - sleep_starts + 1) * interval_min
    sleep_quals <- sapply(seq_along(sleep_runs), function(i) {
      start_i <- sleep_starts[i]
      end_i   <- sleep_ends[i]
      idxs    <- start_i:end_i
      mean(sd$sleep_quality[idxs], na.rm = TRUE)
    })

    # assign every non-NA meal_type its own meal_number
    sd$meal_number <- NA_integer_
    events <- which(!is.na(sd$meal_type))
    if (length(events)) {
      sd$meal_number[events] <- seq_along(events)
    }

    # one row per meal_number
    meal_groups <- tibble(
      idx         = events,
      day         = sd$day[events],
      meal_type   = sd$meal_type[events],
      meal_number = sd$meal_number[events]
    )

    # slide every meal window, centre = the meal itself
    pre_ints      <- 120L / interval_min   # 24
    post_ints     <- 240L / interval_min   # 48

    for (m in seq_len(nrow(meal_groups))) {
      mi   <- meal_groups$idx[m]
      mlab <- meal_groups$meal_type[m]

      centre    <- mi
      start_idx <- max(1L,        centre - pre_ints)
      end_idx   <- min(nrow(sd),  centre + post_ints)
      wd        <- sd[start_idx:end_idx, ]

      wd <- wd %>%
        mutate(
          time_relative       = seq_len(n()) - 1,
          time_since_bolus    = {
            bi <- which(bolus > 0)
            if (length(bi)) seq_len(n()) - 1 - (min(bi) - 1) else NA_real_
          },
          meal_at_time_0      = sd$meal[mi],
          last_sleep_duration = {
            v <- which(sleep_ends < centre)
            if (length(v)) sleep_durs[max(v)] else NA_real_
          },
          last_sleep_quality  = {
            v <- which(sleep_ends < centre)
            if (length(v)) sleep_quals[max(v)] else NA_real_
          },
          bolus_taken         = sum(bolus, na.rm = TRUE),
          subject_id          = subject_id_str,
          meal_day            = sd$day[mi],
          meal_number         = sd$meal_number[mi],
          meal_type           = mlab
        )

      aligned_data_list[[length(aligned_data_list) + 1]] <- wd
    }

    # Also add HyperCorrection windows (bolus without meal)
    bolus_idxs <- which(sd$bolus > 0)
    for (hi in bolus_idxs) {
      start_idx <- hi - pre_ints
      end_idx   <- hi + post_ints
      if (start_idx < 1 || end_idx > nrow(sd)) next
      if (sum(sd$meal[start_idx:end_idx], na.rm = TRUE) > 0) next

      wd <- sd[start_idx:end_idx, ]
      wd <- wd %>%
        mutate(
          time_relative       = seq_len(n()) - 1,
          time_since_bolus    = {
            bi <- which(bolus > 0)
            if (length(bi)) seq_len(n()) - 1 - (min(bi) - 1) else NA_real_
          },
          meal_at_time_0      = 0,
          last_sleep_duration = {
            v <- which(sleep_ends < hi)
            if (length(v)) sleep_durs[max(v)] else NA_real_
          },
          last_sleep_quality  = {
            v <- which(sleep_ends < hi)
            if (length(v)) sleep_quals[max(v)] else NA_real_
          },
          bolus_taken         = sum(bolus, na.rm = TRUE),
          subject_id          = subject_id_str,
          meal_day            = sd$day[hi],
          meal_number         = NA_integer_,
          meal_type           = "HyperCorrection"
        )
      aligned_data_list[[length(aligned_data_list) + 1]] <- wd
    }
  }

  # drop windows with no final glucose
  aligned_data_list <- Filter(
    function(df) !is.na(tail(df$glucose, 1)),
    aligned_data_list
  )

  # Drop windows with excessive consecutive missing glucose values
  MAX_CONSECUTIVE_NA <- 6
  n_before_na_filter <- length(aligned_data_list)
  aligned_data_list <- Filter(
    function(df) max_consecutive_na(df$glucose) <= MAX_CONSECUTIVE_NA,
    aligned_data_list
  )
  n_after_na_filter <- length(aligned_data_list)
  cat(sprintf("  Filtered out %d windows with >%d consecutive missing glucose values\n",
              n_before_na_filter - n_after_na_filter, MAX_CONSECUTIVE_NA))

  # ---- which windows cover a full [-2h,+4h] around the meal? ----
  pre_ints        <- 120 / 5    # 24
  post_total_ints <- 240 / 5    # 48
  post_X_ints     <- 60 / 5     # 12

  snack_threshold_g <- 5
  stopifnot(post_X_ints == 60 / interval_min)

  build_window_row <- function(df, idx) {
    centre <- pre_ints + 1L
    X_len  <- pre_ints + post_X_ints  # (-2h, +60m]

    g_filled <- df$glucose %>%
      na.approx(na.rm = FALSE) %>% na.locf(na.rm = FALSE) %>% na.locf(fromLast = TRUE)

    start_gluc      <- g_filled[1]
    glucose_at_meal <- g_filled[centre]
    final_glucose   <- tail(g_filled, 1)

    post_idx   <- seq(centre + post_X_ints + 1L, min(nrow(df), centre + post_total_ints))
    first_post <- suppressWarnings(min(post_idx[df$meal[post_idx] > snack_threshold_g]))
    if (!is.finite(first_post)) first_post <- NA_integer_

    data.frame(
      window_id             = idx + window_id_offset,
      subject_id            = df$subject_id[1],
      meal_type             = df$meal_type[1],
      start_carb            = df$meal[1],
      carb_for_meal         = sum(df$meal[1:X_len],  na.rm = TRUE),
      bolus_for_meal        = sum(df$bolus[1:X_len], na.rm = TRUE),
      total_carb            = sum(df$meal,  na.rm = TRUE),
      total_bolus           = sum(df$bolus, na.rm = TRUE),
      starting_glucose      = start_gluc,
      glucose_at_meal       = glucose_at_meal,
      final_glucose         = final_glucose,
      glucose_change        = final_glucose - start_gluc,
      glucose_centre_change = final_glucose - glucose_at_meal,
      any_post_carb_after60 = is.finite(first_post),
      first_post_idx        = first_post,
      first_post_carb_min   = if (is.finite(first_post)) (first_post - centre) * interval_min else NA_real_,
      cohort                = cohort_label,
      stringsAsFactors      = FALSE
    )
  }

  # keep only windows that fully cover [-2h, +4h]
  keep_idx <- sapply(aligned_data_list, function(df) {
    centre <- pre_ints + 1L
    start  <- centre - pre_ints
    end    <- centre + post_total_ints
    start >= 1 && end <= nrow(df)
  })
  complete_ids <- which(keep_idx)

  # Build all_cleaned_data + censor-aware y's
  rows <- Map(function(df, id) build_window_row(df, id), aligned_data_list[complete_ids], complete_ids)
  all_cleaned_data <- do.call(rbind, rows)

  # Store subject_id as character
  all_cleaned_data$subject_id <- as.character(all_cleaned_data$subject_id)

  # y_seq and y_seq_change with censoring
  make_y_rows <- function(i) {
    df     <- aligned_data_list[[i]]
    centre <- pre_ints + 1L
    y_idx  <- seq(centre + post_X_ints, centre + post_total_ints - 1L)

    g <- df$glucose %>%
      na.approx(na.rm = FALSE) %>% na.locf(na.rm = FALSE) %>% na.locf(fromLast = TRUE)

    first_post <- suppressWarnings(min(which(df$meal > snack_threshold_g & seq_along(df$meal) > (centre + post_X_ints))))
    censor_at  <- if (is.finite(first_post)) (first_post - 1L) else max(y_idx)

    keep <- y_idx <= censor_at
    y    <- rep(NA_real_, length(y_idx))
    yc   <- rep(NA_real_, length(y_idx))
    if (any(keep)) {
      y[keep]  <- g[y_idx[keep]]
      base_idx <- centre
      yc[keep] <- g[y_idx[keep]] - g[base_idx]
    }
    list(y = y, yc = yc, keep = as.integer(keep))
  }

  Yparts <- lapply(all_cleaned_data$window_id - window_id_offset, make_y_rows)
  y_seq         <- do.call(rbind, lapply(Yparts, `[[`, "y"))
  y_seq_change  <- do.call(rbind, lapply(Yparts, `[[`, "yc"))

  colnames(y_seq)        <- paste0("y", seq_len(ncol(y_seq)))
  colnames(y_seq_change) <- paste0("y", seq_len(ncol(y_seq_change)))

  # final mutate & filter
  all_cleaned_data <- all_cleaned_data %>%
    mutate(
      glucose_change = final_glucose - starting_glucose,
      glucose_centre_change = final_glucose - glucose_at_meal
    ) %>%
    filter(
      !(total_carb == 0 & total_bolus > 0 & glucose_change >= 0),
      !(total_carb  > 0 & total_bolus == 0 & glucose_change < 0)
    )

  # No HypoCorrection or HyperCorrection
  all_cleaned_data <- all_cleaned_data %>%
    dplyr::filter(meal_type != "HypoCorrection") %>%
    dplyr::filter(meal_type != "HyperCorrection")

  # Remove windows with post-carb after grace period
  all_cleaned_data <- all_cleaned_data %>%
    dplyr::filter(any_post_carb_after60 == FALSE)

  # Remove outliers with implausible carb:bolus ratios
  n_before_outlier <- nrow(all_cleaned_data)
  all_cleaned_data <- all_cleaned_data %>%
    dplyr::filter(!(carb_for_meal > 200 & bolus_for_meal < 15))
  n_after_outlier <- nrow(all_cleaned_data)
  if (n_before_outlier > n_after_outlier) {
    cat(sprintf("  Removed %d outlier(s) with implausible carb:bolus ratio (>200g carbs, <15U bolus)\n",
                n_before_outlier - n_after_outlier))
  }

  # drop unused factor levels
  if (is.factor(all_cleaned_data$meal_type)) {
    all_cleaned_data$meal_type <- droplevels(all_cleaned_data$meal_type)
  }

  # Sync aligned_data_list and y_seq matrices to match all_cleaned_data
  keep_idx <- unique(all_cleaned_data$window_id) - window_id_offset
  keep_idx <- as.integer(keep_idx[!is.na(keep_idx)])
  keep_idx <- keep_idx[keep_idx >= 1 & keep_idx <= length(aligned_data_list)]

  y_seq        <- y_seq[keep_idx, , drop = FALSE]
  y_seq_change <- y_seq_change[keep_idx, , drop = FALSE]
  aligned_data_list <- aligned_data_list[keep_idx]

  cat(sprintf("  Total windows: %d\n", nrow(all_cleaned_data)))
  cat(sprintf("  Unique subjects: %d\n", length(unique(all_cleaned_data$subject_id))))
  cat(sprintf("  Window ID range: %d - %d\n", min(all_cleaned_data$window_id), max(all_cleaned_data$window_id)))

  return(list(
    aligned_data_list = aligned_data_list,
    all_cleaned_data  = all_cleaned_data,
    y_seq             = y_seq,
    y_seq_change      = y_seq_change,
    pre_ints          = pre_ints,
    post_X_ints       = post_X_ints,
    post_total_ints   = post_total_ints,
    interval_min      = interval_min
  ))
}

# =============================================================================
# PROCESS ALL THREE SPLITS
# =============================================================================

cat("\n=== Processing 2020 OhioT1DM Cohort ===\n")
cat(sprintf("Number of subjects: %d\n", length(data_full)))

# --- Full data ---
cat("\n--- Processing FULL data ---\n")
full_result <- process_subject_list(data_full)

# --- Training data ---
cat("\n--- Processing TRAIN data ---\n")
train_result <- process_subject_list(data_train)

# --- Test data ---
cat("\n--- Processing TEST data ---\n")
test_result <- process_subject_list(data_test)

# =============================================================================
# SAVE RESULTS
# =============================================================================

save_result <- function(result, filename, extra_vars = list()) {
  aligned_data_list <- result$aligned_data_list
  all_cleaned_data  <- result$all_cleaned_data
  y_seq             <- result$y_seq
  y_seq_change      <- result$y_seq_change
  pre_ints          <- result$pre_ints
  post_X_ints       <- result$post_X_ints
  post_total_ints   <- result$post_total_ints
  interval_min      <- result$interval_min

  save_args <- list(
    aligned_data_list = aligned_data_list,
    all_cleaned_data  = all_cleaned_data,
    y_seq             = y_seq,
    y_seq_change      = y_seq_change,
    pre_ints          = pre_ints,
    post_X_ints       = post_X_ints,
    post_total_ints   = post_total_ints,
    interval_min      = interval_min,
    WINDOW_ID_OFFSET_2020 = WINDOW_ID_OFFSET_2020
  )
  for (nm in names(extra_vars)) {
    save_args[[nm]] <- extra_vars[[nm]]
  }

  filepath <- file.path(CONFIG$ANALYSIS_DATA_DIR, filename)
  env <- new.env(parent = emptyenv())
  for (nm in names(save_args)) {
    assign(nm, save_args[[nm]], envir = env)
  }
  save(list = names(save_args), file = filepath, envir = env)

  cat(sprintf("Saved to: %s (%d windows)\n", filepath, nrow(all_cleaned_data)))
}

# Save full data
cat("\n=== Saving Results ===\n")
save_result(full_result, "z_meal_mediation_analysis_data_2020_5min.RData")

# Save training data with split_info
train_split_info <- list(
  SPLIT_TYPE = "ohio_original",
  SPLIT_SOURCE = "data_train from data_preprocessed_2020-5min.RData",
  TRAIN_SUBJECTS = unique(train_result$all_cleaned_data$subject_id),
  TEST_SUBJECTS = unique(test_result$all_cleaned_data$subject_id),
  WINDOW_ID_OFFSET_2020 = WINDOW_ID_OFFSET_2020
)
save_result(train_result, "z_meal_mediation_analysis_data_2020_TRAIN_5min.RData",
            extra_vars = list(split_info = train_split_info))

# Save test data with split_info
test_split_info <- list(
  SPLIT_TYPE = "ohio_original",
  SPLIT_SOURCE = "data_test from data_preprocessed_2020-5min.RData",
  TRAIN_SUBJECTS = unique(train_result$all_cleaned_data$subject_id),
  TEST_SUBJECTS = unique(test_result$all_cleaned_data$subject_id),
  WINDOW_ID_OFFSET_2020 = WINDOW_ID_OFFSET_2020
)
save_result(test_result, "z_meal_mediation_analysis_data_2020_TEST_5min.RData",
            extra_vars = list(split_info = test_split_info))

# Print summary
cat("\n=== 2020 Meal Window Processing Complete ===\n")
cat(sprintf("Full:  %d windows from %d subjects\n",
            nrow(full_result$all_cleaned_data),
            length(unique(full_result$all_cleaned_data$subject_id))))
cat(sprintf("Train: %d windows from %d subjects\n",
            nrow(train_result$all_cleaned_data),
            length(unique(train_result$all_cleaned_data$subject_id))))
cat(sprintf("Test:  %d windows from %d subjects\n",
            nrow(test_result$all_cleaned_data),
            length(unique(test_result$all_cleaned_data$subject_id))))
