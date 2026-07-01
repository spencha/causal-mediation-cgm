# Function to create aligned data
create_aligned_data <- function(data_full, window_size = 10) {
  aligned_data_list <- list()
  for (subject_id in seq_along(data_full)) {
    subject_data <- data_full[[subject_id]]
    subject_data$meal_number <- NA
    unique_days <- unique(subject_data$day)
    
    for (day in unique_days) {
      day_indices <- which(subject_data$day == day)
      day_data <- subject_data[day_indices, ]
      meal_events <- which(day_data$meal > 0)
      if (length(meal_events) > 0) {
        meal_numbers <- seq_along(meal_events)
        subject_data$meal_number[day_indices[meal_events]] <- meal_numbers
      }
    }
    
    meal_indices <- which(subject_data$meal > 0)
    for (meal_idx in meal_indices) {
      meal_tt <- subject_data$tt[meal_idx]
      meal_day <- subject_data$day[meal_idx]
      meal_number <- subject_data$meal_number[meal_idx]
      meal_carbs <- subject_data$meal[meal_idx]
      
      time_of_day <- ifelse(
        meal_tt >= 8 & meal_tt <= 23, "morning",
        ifelse(meal_tt >= 24 & meal_tt <= 35, "afternoon", "evening")
      )
      
      start_idx <- meal_idx
      end_idx <- meal_idx + window_size
      if (end_idx > nrow(subject_data)) next
      
      window_data <- subject_data[start_idx:end_idx, ]
      window_data$time_relative <- 0:(nrow(window_data) - 1)
      window_data$meal_at_time_0 <- meal_carbs
      window_data$time_since_bolus <- NA
      
      bolus_indices <- which(window_data$bolus > 0)
      if (length(bolus_indices) > 0) {
        first_bolus_time <- min(bolus_indices)
        window_data$time_since_bolus <- window_data$time_relative - window_data$time_relative[first_bolus_time]
        window_data$time_since_bolus[window_data$time_relative < window_data$time_relative[first_bolus_time]] <- NA
      }
      
      window_data$bolus_taken <- sum(window_data$bolus, na.rm = TRUE)
      window_data$subject_id <- subject_id
      window_data$time_of_day <- time_of_day
      window_data$meal_day <- meal_day
      window_data$meal_number <- meal_number
      window_data$meal_carbs <- meal_carbs
      
      aligned_data_list[[length(aligned_data_list) + 1]] <- window_data
    }
  }
  do.call(rbind, aligned_data_list)
}

# Function to create nested data
create_nested_data_list <- function(aligned_data_list) {
  nested_data_list <- list()
  subject_ids <- unique(sapply(aligned_data_list, function(x) unique(x$subject_id)))
  time_of_day_categories <- c("morning", "afternoon", "evening")
  
  for (subject_id in subject_ids) {
    subject_aligned_data_list <- aligned_data_list[sapply(aligned_data_list, function(x) unique(x$subject_id) == subject_id)]
    subject_time_of_day_list <- list()
    for (tod in time_of_day_categories) {
      tod_aligned_data_list <- subject_aligned_data_list[sapply(subject_aligned_data_list, function(x) unique(x$time_of_day) == tod)]
      if (length(tod_aligned_data_list) > 0) {
        subject_time_of_day_list[[tod]] <- tod_aligned_data_list
      }
    }
    nested_data_list[[as.character(subject_id)]] <- subject_time_of_day_list
  }
  nested_data_list
}
