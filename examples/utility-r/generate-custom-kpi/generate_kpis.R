#' @title 
#' Generate a full analysis table for a fixed set of KPIs, using a weekly and a 
#' daily Person Query from Viva Insights. 
#' 
#' @description
#' A full analysis table (data frame) is returned by this function. A weekly
#' Person Query (`weekly_pq`) and a daily Person Query (`daily_pq`) are required
#' as data inputs. The schema for the KPIs matrix can also be supplied, which
#' enables a more well-formatted output to be returned and provides
#' specifications such as `Order`, `Metric`, `Category`, `ToDisplay`,
#' `FullLabel`, `Unit`.
#' 
#' @details
#' There is a validation step to ensure that the weekly Person Query and the
#' daily Person Query have the same PersonId and Date range. The weekly Person
#' Query is used to define the population and the date range for the analysis.
#'
#' @param weekly_pq A data frame containing a weekly Person Query from Viva
#'   Insights.
#' @param daily_pq A data frame containing a daily Person Query from Viva
#'   Insights.
#' @param hrvar A character string specifying the HR variable to be used for
#'   grouping. If none is provided (`NULL`), a holder grouping variable
#'   `"Group"` with the values `"Total"` will be used in place.
#' @param schema A data frame containing the schema for the KPIs matrix. If no
#'   schema is supplied (`NULL`), then the unformatted calculations will be
#'   returned. The schema should contain
#' the following columns: 
#'  - `Order`: The order in which the KPIs should be displayed.
#'  - `Metric`: The name of the KPI.
#'  - `Category`: The category of the KPI.
#'  - `ToDisplay`: Whether the KPI should be displayed in the final output ("Yes")
#'  - `FullLabel`: The full label for the KPI.
#'  - `Unit`: The unit of the KPI.
#'
#'
generate_kpis <- function(
    weekly_pq,
    daily_pq,
    hrvar = NULL,
    schema = NULL){
  
  # Ensure that the weekly PQ and daily PQ have the same PersonId and Date range
  set_pid <- unique(weekly_pq$PersonId)
  set_min_date <- min(weekly_pq$MetricDate)
  set_max_date <- (max(weekly_pq$MetricDate)) + 6
  
  daily_pq <-
    daily_pq %>%
    filter(PersonId %in% set_pid) %>%
    filter(MetricDate >= set_min_date & MetricDate <= set_max_date)
  
  pq_date_range <- paste(set_min_date, "to", set_max_date)
  
  # Diagnostic messages
  message(paste("The weekly PQ has", n_distinct(weekly_pq$PersonId), "unique persons."))
  message(paste("The daily PQ has", n_distinct(daily_pq$PersonId), "unique persons."))
  message(
    paste(
      "The date range for the weekly PQ analysis is from",
      set_min_date, "to", set_max_date)
    )
  message(
    paste(
      "The date range for the daily PQ analysis is from",
      min(daily_pq$MetricDate), "to", max(daily_pq$MetricDate)
      )
    )
  
  # Handle NULL hrvar
  if(is.null(hrvar)){
    hrvar <- "Group"
    weekly_pq$Group <- "Total"
    daily_pq$Group <- "Total"
  } 
  
  # Only look at groups with at least n people
  groups_above_threshold <-
    weekly_pq %>%
    hrvar_count(hrvar = hrvar, return = "table") %>%
    filter(n >= 8)
  
  weekly_pq <- weekly_pq %>% filter(!!sym(hrvar) %in% groups_above_threshold[[hrvar]])
  daily_pq <- daily_pq %>% filter(!!sym(hrvar) %in% groups_above_threshold[[hrvar]])
  
  # This list excludes % of pop analysis metrics, which are calculated separately
  metric_list <- 
    c(
      "Collaboration_hours",
      "Meeting_hours",
      "Meeting_and_call_hours",
      "Chat_hours",
      "Email_hours",
      "Unscheduled_call_hours",
      "Meeting_hours_not_ended_on_time",
      "Meeting_hours_not_joined_on_time",
      "Large_and_long_meeting_hours",
      "Uninterrupted_hours",
      "Weekend_collaboration_hours",
      "Weekend_emails_sent",
      "Weekend_chats_sent"
    )
  
  # Output with meta data ---------------------------------------------------
  out_meta <-
    weekly_pq %>%
    group_by(!!sym(hrvar)) %>%
    summarise(n = n_distinct(PersonId)) %>%
    mutate(DateRange = pq_date_range) %>%
    select(!!sym(hrvar), n, DateRange)
  
  # Metrics requiring separate calculation ----------------------------------
  # Multitasking: % Spend > 15% of meeting time doing emails or chatting outside of the meeting 
  # Weekend: email, chat, call, two or more weekends per month
  sep_metric_list <- c(
    "Multitasking_hours",
    "Meeting_hours",
    "IsWeekendCollab" # Compute below
  )
  
  # Person level calculations for separate metrics
  sep_metric_tb_person <-
    weekly_pq %>%
    mutate(IsWeekendCollab =
             select(., 
                    Weekend_channel_message_posts,
                    Weekend_emails_sent,
                    Weekend_meetings,
                    Unscheduled_weekend_calls) %>%
             apply(1, function(x) any(x > 0))
    ) %>%
    group_by(PersonId, !!sym(hrvar)) %>%
    summarise(
      across(
        .cols = all_of(sep_metric_list),
        .fns = ~mean(.)
      ),
      .groups = "drop"
    ) %>%
    mutate(
      MultitaskingRate = Multitasking_hours / Meeting_hours,
    )
  
  # Table for Multitasking Rate
  tb_multitask <-
    sep_metric_tb_person %>%
    mutate(IsMultitasking15pct = ifelse(MultitaskingRate > 0.15, TRUE, FALSE)) %>%
    count(!!sym(hrvar), IsMultitasking15pct) %>%
    mutate(Pct_IsMultitasking15pct = n / sum(n)) %>%
    filter(IsMultitasking15pct) %>%
    mutate(Metric = "Multitasking_15pct") %>%
    select(Metric, !!sym(hrvar), Value = "Pct_IsMultitasking15pct")
  
  # Table for Weekend Collaboration
  tb_weekendcollab <-
    sep_metric_tb_person %>%
    mutate(WeekendCollabMoreThanTwiceMonthly = ifelse(IsWeekendCollab >= 0.5, TRUE, FALSE)) %>%
    group_by(!!sym(hrvar)) %>%
    summarise(Pct_WeekendCollabMoreThanTwiceMonthly = mean(WeekendCollabMoreThanTwiceMonthly)) %>%
    mutate(Metric = "WeekendCollabMoreThanTwiceMonthly") %>%
    select(Metric, !!sym(hrvar), Value = "Pct_WeekendCollabMoreThanTwiceMonthly")
  
  
  # Convert hourly columns into a long format -------------------------------
  # Uses DAILY level data
  # At the Person-Week level, With the following key columns: 
  # - Metric: 'Emails_sent_'
  # - Volume
  # - StartHour
  # - EndHour
  
  hour_str <-
    paste(
      str_pad(
        string = 0:23,
        width = 2,
        side = "left",
        pad = "0"
      ),
      str_pad(
        string = 1:24,
        width = 2,
        side = "left",
        pad = "0"
      ),
      sep = "_"
    )
  
  hour_df <-
    daily_pq %>%
    select(PersonId,
           MetricDate,
           !!sym(hrvar),
           starts_with("Emails_sent_"),
           starts_with("Chats_sent_"),
           paste0("Meetings_", hour_str),
           starts_with("Unscheduled_calls_")
    ) %>% 
    select(-ends_with("during_the_weekend")) %>%
    pivot_longer(
      cols = c(
        starts_with("Emails_sent_"),
        starts_with("Chats_sent_"),
        starts_with("Meetings_"),
        starts_with("Unscheduled_calls_")),
      names_to = "Metric",
      values_to = "Volume"
    ) %>%
    filter(Volume > 0) %>%
    mutate(Hours = str_extract(Metric, "\\d.+")) %>%
    mutate(Metric = str_remove(Metric, "\\d.+")) %>%
    separate_wider_delim(cols = Hours, delim = "_", names = c("StartHour", "EndHour")) %>%
    mutate(StartHour = as.numeric(StartHour),
           EndHour = as.numeric(EndHour)) 
  
  # Extract only first and last active hours
  hour_df_summary <-
    hour_df %>%
    arrange(StartHour) %>% 
    group_by(PersonId, !!sym(hrvar), MetricDate) %>% # Whichever signal
    summarise(FirstHourOfDay = first(StartHour),
              LastHourOfDay = last(EndHour),
              .groups = "drop")
  
  # Extract only first and last hours as a row
  out_first_last_hours <-
    hour_df_summary %>%
    group_by(!!sym(hrvar)) %>%
    summarise(
      FirstHourOfDay_mean = mean(FirstHourOfDay),
      LastHourOfDay_mean = mean(LastHourOfDay)
    ) %>%
    pivot_longer(cols = -!!sym(hrvar),
                 names_to = "Metric",
                 values_to = "Hour") %>%
    separate_wider_delim(cols = Metric, names = c("Metric", "Quantile"), delim = "_") %>%
    pivot_wider(names_from = Quantile, values_from = Hour) %>%
    select(Metric, !!sym(hrvar), Value = "mean")
  
  # Generate full matrix ----------------------------------------------------
  # Main matrix, i.e. contains intermediate metrics not required for the Excel
  main_mat <-
    weekly_pq %>%
    group_by(PersonId, !!sym(hrvar)) %>%
    summarise(
      across(
        .cols = all_of(metric_list),
        .fns = ~mean(.)
      ),
      .groups = "drop"
    ) %>%
    group_by(!!sym(hrvar)) %>%
    summarise(
      across(
        .cols = all_of(metric_list),
        .fns = ~mean(.)
      )
    ) %>%
    mutate(
      Percent_week_in_collaboration = Collaboration_hours / 40,
      Unscheduled_call_minutes = Unscheduled_call_hours * 60,
      Percent_meetings_joined_late = Meeting_hours_not_joined_on_time / Meeting_hours,
      Percent_meetings_ended_late = Meeting_hours_not_ended_on_time / Meeting_hours,
      Percent_meetings_large_and_long = Large_and_long_meeting_hours / Meeting_hours,
      Uninterrupted_hours_per_day = Uninterrupted_hours / 5,
      Uninterrupted_mins_per_day = Uninterrupted_hours_per_day * 60,
      Weekend_collab_minutes = Weekend_collaboration_hours * 60
    ) %>%
    pivot_longer(
      cols = -!!sym(hrvar),
      names_to = "Metric",
      values_to = "Value"
    ) %>%
    rbind(
      out_first_last_hours,# First and last hours of the day
      tb_multitask, # Multitasking
      tb_weekendcollab # Weekend collaboration
    )

  
  if(is.null(schema)){
    
    main_mat
    
  } else {
    
    main_mat_display <-
      main_mat %>%
      left_join(
        schema,
        by = "Metric"
      ) %>%
      arrange(Order) %>%
      select(Order, Category, everything()) %>%
      filter(ToDisplay == "Yes") %>%
      mutate(Order = 1:nrow(.)) %>%
      mutate(ValueRounded = round(Value, 1)) %>%
      mutate(ValueLabelled = case_when(
        Unit == "Percent" ~ paste0(round(Value * 100, 1), "%"),
        Unit == "Hours" ~ paste0(ValueRounded, " hours"),
        Unit == "Minutes" ~ paste0(ValueRounded, " minutes"),
        Unit == "Time" ~ convert_to_time(ValueRounded),
        TRUE ~ as.character(ValueRounded)
      )) %>%
      select(
        Order,
        Category,
        !!sym(hrvar),
        FullLabel,
        ValueLabelled,
        Value,
        Unit
      ) %>%
      # Get population n and DateRange
      left_join(
        out_meta,
        by = hrvar
      )
  }
  
  return(main_mat_display)
}

#' @title Convert decimal hours to time format
#' @description
#' Convert time values stored in decimals (e.g. 8.5 hours) to a time format (e.g. 8h 30m).
#' @param x A numeric vector of decimal hours.
#'
convert_to_time <- function(x) {
  hours <- floor(x)
  minutes <- round((x - hours) * 60)
  return(paste(hours, "h", minutes, "m"))
}