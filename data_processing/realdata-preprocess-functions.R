
#------------------------------------------------------------------------------
# 1. Functions for data preprocessing with Ohio Type 1 Diabetes dataset
#------------------------------------------------------------------------------


library(XML) #package for reading XML data
library(lubridate) #package for dealing with date and time

#Functions for measuring the effect of blood glucose control
ICG = function(x, a = 1.35, b = 2){
  return( - (x < 80) * ((abs(80 - x))^b) / 30 - (x > 140) * ((abs(x - 140))^a) / 30)
}

HyperIndex = function(x, a = 1.35){
  return( - (x > 140) * ((abs(x - 140))^a) / 30)
}

HypoIndex = function(x, b = 2){
  return( - (x < 80) * ((abs(80 - x))^b) / 30)
}
M100 = function(x){
  return( - 1000 * (abs(log10(x / 100)))^3)
}
InRange = function(x){
  return(as.numeric((x <= 140)&(x >= 70)))
}
#transform hour and minutes into minutes in a day
daytime = function(x){
  return(60 * hour(x) + minute(x))
}

#Put data into a list (2018 format)
preprocess0 = function(data0){
  data = list(glucose = data0[[1]],
            basal = data0[[3]],
            tempbasal = data0[[4]],
            bolus = data0[[5]][, c(1, 4)],
            meal = data0[[6]],
            sleep = data0[[7]],
            heart = data0[[13]],
            gsr = data0[[14]],
            skintemp = data0[[15]],
            airtemp = data0[[16]],
            steps = data0[[17]])
  return(data)
}

#------------------------------------------------------------------------------
# 2020 OhioT1DM Preprocessing Functions
#------------------------------------------------------------------------------
# The 2020 dataset has different XML variable names/positions:
# 1. glucose_level, 2. finger_stick, 3. basal, 4. temp_basal, 5. bolus,
# 6. meal, 7. sleep, 8. work, 9. stressors, 10. hypo_event, 11. illness,
# 12. exercise, 13. basis_heart_rate, 14. basis_gsr, 15. basis_skin_temperature,
# 16. basis_air_temperature, 17. basis_steps, 18. basis_sleep, 19. acceleration
#------------------------------------------------------------------------------

#' Safe column extraction with fallback
#' @param mat Matrix or data frame
#' @param cols Column indices to extract
#' @param fallback Value to return if extraction fails
safe_extract_cols <- function(mat, cols, fallback = NULL) {
  if (is.null(mat)) return(fallback)
  if (is.matrix(mat) || is.data.frame(mat)) {
    ncols <- ncol(mat)
    if (ncols == 0) return(fallback)
    valid_cols <- cols[cols <= ncols]
    if (length(valid_cols) == 0) return(fallback)
    if (length(valid_cols) == 1) {
      return(mat[, valid_cols, drop = FALSE])
    }
    return(mat[, valid_cols, drop = FALSE])
  }
  return(fallback)
}

#' Get variable by name from raw data with fallback
#' @param data0 Raw data list
#' @param varnames Vector of variable names from XML
#' @param target_name Name of variable to find
#' @param fallback_idx Fallback index if name not found
get_var_by_name <- function(data0, varnames, target_name, fallback_idx = NULL) {
  idx <- which(varnames == target_name)
  if (length(idx) > 0) {
    return(data0[[idx[1]]])
  }
  if (!is.null(fallback_idx) && fallback_idx <= length(data0)) {
    return(data0[[fallback_idx]])
  }
  return(NULL)
}

#' Put data into a list (2020 format - robust version)
#' Handles different column structures between 2018 and 2020
#' @param data0 Raw data list from XML parsing
#' @param varnames Optional vector of variable names to use for name-based lookup
preprocess0_2020 = function(data0, varnames = NULL){

  # If varnames provided, use name-based lookup
  if (!is.null(varnames)) {
    glucose <- get_var_by_name(data0, varnames, "glucose_level", 1)
    basal <- get_var_by_name(data0, varnames, "basal", 3)
    tempbasal <- get_var_by_name(data0, varnames, "temp_basal", 4)
    bolus_raw <- get_var_by_name(data0, varnames, "bolus", 5)
    meal <- get_var_by_name(data0, varnames, "meal", 6)
    sleep <- get_var_by_name(data0, varnames, "sleep", 7)
    heart <- get_var_by_name(data0, varnames, "basis_heart_rate", 13)
    gsr <- get_var_by_name(data0, varnames, "basis_gsr", 14)
    skintemp <- get_var_by_name(data0, varnames, "basis_skin_temperature", 15)
    airtemp <- get_var_by_name(data0, varnames, "basis_air_temperature", 16)
    steps <- get_var_by_name(data0, varnames, "basis_steps", 17)
  } else {
    # Use index-based lookup (same indices as 2018 for most variables)
    glucose <- data0[[1]]
    basal <- data0[[3]]
    tempbasal <- data0[[4]]
    bolus_raw <- data0[[5]]
    meal <- data0[[6]]
    sleep <- data0[[7]]
    heart <- data0[[13]]
    gsr <- data0[[14]]
    skintemp <- data0[[15]]
    airtemp <- data0[[16]]
    steps <- data0[[17]]
  }

  # Handle bolus - 2020 format may have different column structure
  # Try to extract timestamp (col 1) and dose (col 4 or col 2)
  bolus <- NULL
  if (!is.null(bolus_raw) && is.matrix(bolus_raw) && nrow(bolus_raw) > 0) {
    ncols <- ncol(bolus_raw)
    if (ncols >= 4) {
      # Standard 2018 format with columns 1 and 4
      bolus <- bolus_raw[, c(1, 4), drop = FALSE]
    } else if (ncols >= 2) {
      # 2020 might have timestamp and dose in columns 1 and 2
      bolus <- bolus_raw[, c(1, 2), drop = FALSE]
    } else if (ncols == 1) {
      # Only one column - create placeholder
      bolus <- cbind(bolus_raw[, 1], rep(0, nrow(bolus_raw)))
    }
  }
  if (is.null(bolus)) {
    # Create empty bolus matrix with correct structure
    bolus <- matrix(character(0), nrow = 0, ncol = 2)
  }

  # Handle sleep - 2020 format may have different structure
  # Expected columns: 1=ts_end, 2=ts_begin, 3=quality
  # But 2020 might have: 1=ts (single timestamp), or empty
  if (!is.null(sleep) && is.matrix(sleep) && nrow(sleep) > 0) {
    ncols <- ncol(sleep)
    if (ncols < 2) {
      # Only one column - sleep data is unusable, set to empty
      cat("Warning: Sleep data has only", ncols, "column(s), setting to empty\n")
      sleep <- matrix(character(0), nrow = 0, ncol = 3)
    } else if (ncols == 2) {
      # Two columns - assume ts_end, ts_begin, add placeholder quality
      cat("Warning: Sleep data has only 2 columns, adding default quality=1\n")
      sleep <- cbind(sleep, rep("1", nrow(sleep)))
    }
    # If ncols >= 3, keep as is
  } else {
    # No sleep data
    sleep <- matrix(character(0), nrow = 0, ncol = 3)
  }

  data = list(
    glucose = glucose,
    basal = basal,
    tempbasal = tempbasal,
    bolus = bolus,
    meal = meal,
    sleep = sleep,
    heart = heart,
    gsr = gsr,
    skintemp = skintemp,
    airtemp = airtemp,
    steps = steps
  )

  return(data)
}

#' Unified preprocessing function that auto-detects dataset year
#' @param data0 Raw data list from XML parsing
#' @param varnames Vector of variable names from XML
#' @param year Dataset year (2018 or 2020), or NULL for auto-detect
preprocess0_auto = function(data0, varnames = NULL, year = NULL) {

  # Auto-detect based on variable names if not specified
  if (is.null(year) && !is.null(varnames)) {
    if ("basis_heart_rate" %in% varnames || "glucose_level" %in% varnames) {
      year <- 2020
    } else {
      year <- 2018
    }
  }

  if (is.null(year)) {
    year <- 2018  # Default to 2018 format
  }

  if (year == 2020) {
    return(preprocess0_2020(data0, varnames))
  } else {
    return(preprocess0(data0))
  }
}

#' Helper function to safely create a sensor data frame
#' Returns empty data frame with correct columns if data is NULL or empty
#' @param sensor_data Matrix of sensor data (timestamp, value)
#' @param startdate Start date for day calculation
#' @param interval_min Interval in minutes
#' @param value_name Name of the value column (e.g., "heart", "gsr")
safe_sensor_df <- function(sensor_data, startdate, interval_min, value_name) {
  # Check if data is valid
  if (is.null(sensor_data) || !is.matrix(sensor_data) ||
      nrow(sensor_data) == 0 || ncol(sensor_data) < 2) {
    # Return empty data frame with correct structure
    df <- data.frame(day = integer(0), tt = integer(0))
    df[[value_name]] <- numeric(0)
    return(df)
  }

  # Create data frame from valid data
  tryCatch({
    df <- data.frame(
      day = as.numeric(date(dmy_hms(sensor_data[, 1])) - startdate),
      tt = floor(daytime(dmy_hms(sensor_data[, 1])) / interval_min)
    )
    df[[value_name]] <- as.numeric(sensor_data[, 2])
    return(df)
  }, error = function(e) {
    cat("Warning: Error processing", value_name, "data:", conditionMessage(e), "\n")
    df <- data.frame(day = integer(0), tt = integer(0))
    df[[value_name]] <- numeric(0)
    return(df)
  })
}

preprocess1 = function(data0, interval_min = 60,  startdate = NA){
  #start date is the first date of subject's participation in the experiment
  if (is.na(startdate)){
    startdate = date(dmy_hms(data0$glucose[1, 1]))
  }

  # Handle sleep data
  if (!is.null(data0$sleep) && is.matrix(data0$sleep) &&
      nrow(data0$sleep) > 0 && ncol(data0$sleep) >= 3) {
    sleep_df = data.frame(
      start_day      = as.numeric(lubridate::date(lubridate::dmy_hms(data0$sleep[, 2])) - startdate),
      start_daytime  = daytime(lubridate::dmy_hms(data0$sleep[, 2])),
      end_day        = as.numeric(lubridate::date(lubridate::dmy_hms(data0$sleep[, 1])) - startdate),
      end_daytime    = daytime(lubridate::dmy_hms(data0$sleep[, 1])),
      quality        = as.numeric(data0$sleep[, 3])
    )
    data0$sleep = sleep_df
  } else {
    data0$sleep = data.frame()
  }

  # Handle bolus - check for valid data
  if (!is.null(data0$bolus) && is.matrix(data0$bolus) &&
      nrow(data0$bolus) > 0 && ncol(data0$bolus) >= 2) {
    bolus_df <- data.frame(
      day = as.numeric(date(dmy_hms(data0$bolus[, 1])) - startdate),
      daytime = daytime(dmy_hms(data0$bolus[, 1])),
      tt = floor(daytime(dmy_hms(data0$bolus[, 1])) / interval_min),
      bolus = as.numeric(data0$bolus[, 2])
    )
  } else {
    bolus_df <- data.frame(day = integer(0), daytime = numeric(0),
                           tt = integer(0), bolus = numeric(0))
  }

  # Handle meal - check for valid data (needs at least 3 columns: timestamp, type, carbs)
  if (!is.null(data0$meal) && is.matrix(data0$meal) &&
      nrow(data0$meal) > 0 && ncol(data0$meal) >= 3) {
    meal_df <- data.frame(
      day = as.numeric(date(dmy_hms(data0$meal[, 1])) - startdate),
      daytime = daytime(dmy_hms(data0$meal[, 1])),
      tt = floor(daytime(dmy_hms(data0$meal[, 1])) / interval_min),
      meal = as.numeric(data0$meal[, 3])
    )
    meal_type_df <- data.frame(
      day = as.numeric(date(dmy_hms(data0$meal[, 1])) - startdate),
      daytime = daytime(dmy_hms(data0$meal[, 1])),
      tt = floor(daytime(dmy_hms(data0$meal[, 1])) / interval_min),
      meal_type = as.character(data0$meal[, 2])
    )
  } else if (!is.null(data0$meal) && is.matrix(data0$meal) &&
             nrow(data0$meal) > 0 && ncol(data0$meal) >= 2) {
    # Only 2 columns - assume timestamp and carbs, no type
    cat("Warning: Meal data has only 2 columns, assuming no meal type\n")
    meal_df <- data.frame(
      day = as.numeric(date(dmy_hms(data0$meal[, 1])) - startdate),
      daytime = daytime(dmy_hms(data0$meal[, 1])),
      tt = floor(daytime(dmy_hms(data0$meal[, 1])) / interval_min),
      meal = as.numeric(data0$meal[, 2])
    )
    meal_type_df <- data.frame(
      day = meal_df$day,
      daytime = meal_df$daytime,
      tt = meal_df$tt,
      meal_type = rep(NA_character_, nrow(meal_df))
    )
  } else {
    # No valid meal data
    cat("Warning: No valid meal data found for this subject\n")
    meal_df <- data.frame(day = integer(0), daytime = numeric(0),
                          tt = integer(0), meal = numeric(0))
    meal_type_df <- data.frame(day = integer(0), daytime = numeric(0),
                               tt = integer(0), meal_type = character(0))
  }

  #tt is the time index in the day
  #if interval_min=60 than all the time between 0:00-1:00 will have tt=0, all the time between 1:00-2:00 have tt=1, etc.
  data = list(glucose  = data.frame( day = as.numeric(date(dmy_hms(data0$glucose[, 1])) - startdate),
                                    tt = floor(daytime(dmy_hms(data0$glucose[, 1])) / interval_min),
                                    daytime = daytime(dmy_hms(data0$glucose[, 1])),
                                    glucose = as.numeric(data0$glucose[, 2])) ,
              basal    = data.frame(day = as.numeric(date(dmy_hms(data0$basal[, 1])) - startdate),
                                    daytime = daytime(dmy_hms(data0$basal[, 1])),
                                    rate = as.numeric(data0$basal[, 2])),
              bolus    = bolus_df,
              meal     = meal_df,
              meal_type = meal_type_df,

              sleep = data0$sleep,

              # Use safe_sensor_df for wearable sensor data
              heart    = safe_sensor_df(data0$heart, startdate, interval_min, "heart"),
              gsr      = safe_sensor_df(data0$gsr, startdate, interval_min, "gsr"),
              skintemp = safe_sensor_df(data0$skintemp, startdate, interval_min, "skintemp"),
              airtemp  = safe_sensor_df(data0$airtemp, startdate, interval_min, "airtemp"),
              steps    = safe_sensor_df(data0$steps, startdate, interval_min, "steps")
              )

  # Handle tempbasal
  if (!is.null(data0$tempbasal) && is.matrix(data0$tempbasal) &&
      nrow(data0$tempbasal) > 0 && ncol(data0$tempbasal) >= 3) {
    data[[12]] = data.frame(start_day = as.numeric(date(dmy_hms(data0$tempbasal[, 1])) - startdate),
                            start_daytime = daytime(dmy_hms(data0$tempbasal[, 1])),
                            end_day = as.numeric(date(dmy_hms(data0$tempbasal[, 2])) - startdate),
                            end_daytime = daytime(dmy_hms(data0$tempbasal[, 2])),
                            rate = as.numeric(data0$tempbasal[, 3])  )
    names(data)[12] = "tempbasal"
  }

  return(data)
}

#turn basal rate change information into basal rate for each interval
preprocess2_basal = function(data,  interval_min = 60, startbasal = NA){
  data1 = data
  firstday = min(data$glucose$day)
  lastday = max(data$glucose$day)
  nday = lastday - firstday + 1
  
  #initialize
  ntt = 1440 / interval_min
  datett = data.frame(day = rep(seq(firstday, lastday), each = ntt), 
                      tt  = rep(seq(0, ntt - 1), nday))
  tbasal = datett
  tbasal$rate = NA
  ob = 1
  trate = startbasal
  data1$basal = rbind(data1$basal,  c(lastday + 1, 0, 0)) #set an end point
  
  for (i in 1:(ntt * nday)){
    if ((data1$basal$day[ob] - firstday) * ntt + floor(data1$basal$daytime[ob] / interval_min) > i - 1) { 
      #basal rate not changing at this time interval
      tbasal$rate[i] = trate
    } else{
      #basal rate changing at this time interval
      begin_interval = 0
      tbasal$rate[i] = 0
      while ((data1$basal$day[ob] - firstday) * ntt + floor(data1$basal$daytime[ob] / interval_min) <= i - 1) {
        tbasal$rate[i] = tbasal$rate[i] +  trate *  (data1$basal$daytime[ob] %% interval_min - begin_interval) / interval_min
        begin_interval = data1$basal$daytime[ob] %% interval_min
        trate = data1$basal$rate[ob]
        ob = ob + 1
      }
      tbasal$rate[i] = tbasal$rate[i] +  trate *  (interval_min - begin_interval) / interval_min
    }
  }
  
  #adjust with temp basal information
  if (length(which(names(data1) == "tempbasal")) > 0){
    for (i in 1:nrow(data1$tempbasal)){
      begin_int_whole  = (data1$tempbasal$start_day[i] - firstday) * ntt + 
                         floor(data1$tempbasal$start_daytime[i] / interval_min) + 1
      end_int_whole    = (data1$tempbasal$end_day[i] - firstday) * ntt + 
                         floor(data1$tempbasal$end_daytime[i] / interval_min) + 1
      begin_int_rest   = data1$tempbasal$start_daytime[i] %% interval_min
      end_int_rest     = data1$tempbasal$end_daytime[i] %% interval_min
      trate = data1$tempbasal$rate[i]
      while (begin_int_whole  <  end_int_whole){
        tbasal[begin_int_whole, ]$rate = tbasal[begin_int_whole, ]$rate * (begin_int_rest) / interval_min + 
          trate * (interval_min - begin_int_rest) / interval_min
        begin_int_whole = begin_int_whole + 1
        begin_int_rest = 0
      }
      tbasal[ begin_int_whole, ]$rate = tbasal[ begin_int_whole, ]$rate * (begin_int_rest + interval_min - end_int_rest) / interval_min  + 
                                        trate * (end_int_rest - begin_int_rest) / interval_min
    }
  }
  names(tbasal)[which(names(tbasal) == "rate")] = "basal"
  return(tbasal)
}

#sleep into interval
preprocess2_sleep = function(data, interval_min = 60) {
  # If no sleep data, return NULL or an empty data frame
  if (!("sleep" %in% names(data))) {
    return(NULL)
  }
  if (nrow(data$sleep) == 0) {
    return(NULL)
  }
  
  # We’ll create something like: day, tt, fraction_sleep, avg_quality, etc.
  firstday = min(data$glucose$day, na.rm = TRUE)
  lastday  = max(data$glucose$day, na.rm = TRUE)
  nday     = lastday - firstday + 1
  ntt      = 1440 / interval_min
  datett   = data.frame(
    day = rep(seq(firstday, lastday), each = ntt),
    tt  = rep(seq(0, ntt - 1), nday)
  )
  datett$sleep_fraction = 0
  datett$sleep_quality  = NA
  
  # Loop over each sleep event
  for (i in seq_len(nrow(data$sleep))) {
    sd  = data$sleep$start_day[i]
    st  = data$sleep$start_daytime[i]
    ed  = data$sleep$end_day[i]
    et  = data$sleep$end_daytime[i]
    q   = data$sleep$quality[i]
    
    # Convert (day, daytime) to global index in [0, +∞)
    # e.g. “minutes from the start day”
    begin_global = (sd - firstday) * 1440 + st
    end_global   = (ed - firstday) * 1440 + et
    
    # For each (day, tt), compute that interval’s start and end in “global” minutes
    # If overlap with [begin_global, end_global] > 0, we fill fraction
    for (r in seq_len(nrow(datett))) {
      # This interval in minutes
      day_i    = datett$day[r]
      tt_i     = datett$tt[r]
      int_start = (day_i - firstday) * 1440 + (tt_i * interval_min)
      int_end   = int_start + interval_min
      
      # Overlap with the sleep window
      overlap = min(int_end, end_global) - max(int_start, begin_global)
      if (overlap > 0) {
        fraction = overlap / interval_min
        # Add fraction (there can be multiple sleeps per interval, so we might sum)
        datett$sleep_fraction[r] = datett$sleep_fraction[r] + fraction
        
        # You could store average or weighted quality. For simplicity, store the first quality or the max
        if (is.na(datett$sleep_quality[r])) {
          datett$sleep_quality[r] = q
        } else {
          # e.g. take an average if we have multiple sleep segments in same interval
          old_q = datett$sleep_quality[r]
          # Weighted average approach
          # datett$sleep_quality[r] = old_q*(1 - fraction) + q*fraction
          # Or just take the max
          datett$sleep_quality[r] = max(old_q, q)
        }
      }
    }
  }
  
  return(datett)
}

first_meal_type <- function(x, ...) {
  # Remove NAs
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  return(x[1])
}

#
#aggregated variables into time intervals
#Each variable has its own function for aggregating (sum or mean)
preprocess2_aggregate = function(data,
                                 variables = c("meal", "meal_type", "bolus",  "glucose", "heart", "gsr", "skintemp", "airtemp", "steps"),
                                 FUN = list(sum, first_meal_type, sum, mean, mean, mean, mean, mean, sum),
                                 glucose_eval = list(ICG, HyperIndex, HypoIndex, M100, InRange),
                                 eval_names = c("icg", "hyper", "hypo", "m100", "in_range"),
                                 missing_as_zero = c("meal", "bolus"),
                                 # missing_as_NA = c("meal_type"),
                                 interval_min = 60){
  firstday = min(data$glucose$day)
  lastday = max(data$glucose$day)
  nday = lastday - firstday + 1
  ntt = 1440 / interval_min
  datett = data.frame(day = rep(seq(firstday, lastday), each = ntt),
                    tt  = rep(seq(0, ntt - 1), nday))
  #find the index of the variables
  var_index = as.numeric(sapply(variables, function(x) which(names(data) == x)))
  data1 = datett
  for (i in 1:length(variables)){
    var_data <- data[[var_index[i]]]

    # Check if the data frame has rows to aggregate
    if (is.null(var_data) || !is.data.frame(var_data) || nrow(var_data) == 0) {
      # Create empty column with NA values for this variable
      data1[[variables[i]]] <- NA
      next
    }

    # Try to aggregate, with error handling
    tryCatch({
      temp = aggregate(as.formula(paste(variables[i], "~day + tt", sep = "")),
                     data = var_data,
                     FUN = FUN[[i]],
                     na.rm = TRUE)
      data1 = merge(data1, temp, by = c("day", "tt"), all = TRUE)
    }, error = function(e) {
      # If aggregation fails, add NA column
      cat("Warning: Could not aggregate", variables[i], "-", conditionMessage(e), "\n")
      data1[[variables[i]]] <<- NA
    })
  }
  for (i in 1:length(missing_as_zero)){
    m_index =  which(names(data1) == missing_as_zero[i])
    if (length(m_index) > 0) {
      data1[is.na(data1[, m_index]), m_index] = 0
    }
  }
  #transformations of glucose
  glucoseindex = which(names(data) == "glucose")
  gl = data[[glucoseindex]]$glucose
  for (i in 1:length(glucose_eval)){
    data[[glucoseindex]] = cbind(data[[glucoseindex]],  glucose_eval[[i]](gl))
    names(data[[glucoseindex]])[dim(data[[glucoseindex]])[2]] = eval_names[i]
    temp = aggregate(as.formula(paste(eval_names[i], "~day + tt", sep = "")),
                   data = data[[glucoseindex]],
                   FUN = mean,
                   na.rm = TRUE)
    data1 = merge(data1, temp, by = c("day", "tt"), all = TRUE)
  }
  return(data1)
}

#calculate the last observed glucose level within the time interval (vs average glucose level)
preprocess2_lastglucose = function(data, interval_min = 30){
  tglucose = data$glucose
  C = 100000
  tglucose$daytime_glucose = tglucose$daytime * C + tglucose$glucose
  tglucose2 = aggregate(daytime_glucose~tt + day, data = tglucose, FUN = max)
  tglucose2$lastglucose = tglucose2$daytime_glucose %% C
  tglucose2$daytime_glucose = NULL
  return(tglucose2)
}

preprocess2 = function(data, interval_min = 60, startbasal = NA){
  tbasal = preprocess2_basal(data, 
                           interval_min = interval_min, 
                           startbasal = startbasal)
  temp = preprocess2_aggregate(data, interval_min = interval_min)
  lastglucose = preprocess2_lastglucose(data, interval_min = interval_min)
  data1 = merge(tbasal, temp, by = c("day", "tt"), all = TRUE)
  data1 = merge(data1, lastglucose, by = c("day", "tt"), all = TRUE)
  
  # Now add the sleep intervals if needed:
  tsleep = preprocess2_sleep(data, interval_min = interval_min)
  if (!is.null(tsleep)) {
    data1 = merge(data1, tsleep, by = c("day", "tt"), all = TRUE)
  }
  
  data1 = data1[order(data1$day, data1$tt), ]
  return(data1)
}

#calculate past variable information
preprocess3_past = function(data,  variable = "basal",  interval_min = 60,  
                            lag_time_range = c(120, 240), 
                            new_var_name = "basal_2_4", 
                            FUN = mean){
  lag_time = floor(lag_time_range / interval_min)
  lagmin = lag_time[1]
  lagmax = lag_time[2] - 1
  data1 = data
  var_index = which(names(data) == variable)
  #data1$newvar = 0
  index = (1 + lagmax):(dim(data)[1])
  temp = matrix(NA, nrow = length(index), ncol = 0)
  for (i in lagmin:lagmax){
    temp = cbind(temp, data[index - i, var_index])
  }
  data1$newvar = NA
  data1$newvar[index] = apply(temp, 1, FUN = FUN)
  names(data1)[which(names(data1) == "newvar")] = new_var_name
  return(data1)
}

#calculate future variable information
preprocess3_future = function(data,  variable = "meal",  interval_min = 30,  
                          lag_time_range = c(0, 30), 
                          new_var_name = "meal_future_halfhour", 
                          FUN = sum){
  lag_time = floor(lag_time_range / interval_min)
  lagmin = lag_time[1] + 1
  lagmax = lag_time[2]
  data1 = data
  var_index = which(names(data) == variable)
  #data1$newvar = 0
  index = (1):(dim(data)[1] - lagmax)
  temp = matrix(NA, nrow = length(index), ncol = 0)
  for (i in lagmin:lagmax){
    temp = cbind(temp, data[index + i, var_index])
  }
  data1$newvar = NA
  data1$newvar[index] = apply(temp, 1, FUN = FUN)
  names(data1)[which(names(data1) == "newvar")] = new_var_name
  return(data1)
}





