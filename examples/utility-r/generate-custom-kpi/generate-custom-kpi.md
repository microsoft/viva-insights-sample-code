# Generate custom KPIs from Viva Insights metrics in R

## Overview

This document walks through an example on how to set up a workflow to compute custom KPIs from Viva Insights metrics in R. 

Viva Insights provides over a hundred metrics that can be used to measure an organization's wellbeing, productivity, and collaborative culture. However, these metrics may not always come in the right unit to be communicated to a wider audience. For instance, _Collaboration hours_ may be more interpretable when expressed as a percentage of the expected weekly work hours, e.g. % of a 40-hour work week. Another example would be metrics like _% of population who are active on weekends_, which requires a different aggregation method that is less straightforward than a calculating a mean. 

In the example below, we walk through a script that computes the following KPIs: 

1. % of a 40h workweek spent in collaboration
2. Total hours per week in average spent in meetings
3. Total hours per week in average spent on chats
4. Total hours per week in average spent reading and writing emails
5. Total hours per week in average spent on unscheduled calls
6. % of meeting hours are ended late
7. % of meeting hours are joined late
8. % of meetings are long (>1 hour) and large meetings (9+ attendees)
9. % of spend >15% of meeting time doing emails or chatting outside of the meeting
10. Total hours per day of uninterrupted focus available for employees to do independent work
11. Total minutes per day of uninterrupted focus available for employees to do independent work
12. Average first active hours during the day
13. Average last active hours during the day
14. % who are active on weekends - two or more weekends per month
15. Average collaboration minutes per day during the weekend
16. Number of emails are sent over the weekend on average
17. Number of chats are sent over the weekend on average

Applying the same technique, it is possible to customize the above KPIs to your specific needs. We will also cover helpful techniques for Analysts, such as how to customize KPI labels so the outputs would work well to transfer directly to a presentation or a document. 

## Pre-requisites

To compute the above metrics, two Person Queries are required from Viva Insights. One needs to be grouped at the **week** level, and another needs to be grouped at the **day** level. The following metrics should be run for each: 

Person Query grouped by week: 

- Collaboration_hours
- Meeting_and_call_hours
- Chat_hours
- Email_hours
- Unscheduled_call_hours
- Meeting_hours_not_ended_on_time
- Meeting_hours_not_joined_on_time
- Large_and_long_meeting_hours
- Multitasking_hours
- Uninterrupted_hours
- Uninterrupted_hours_per_day
- Weekend_collaboration_hours

Person Query grouped by day: 
_For 'hh' below, select all hourly collaboration metrics, one for each hour of the day:_
- Chats_sent_hh_hh
- Emails_sent_hh_hh
- Meetings_hh_hh
- Unscheduled_calls_hh_hh

## Setting up parameters

First, we recommend setting up a file that specifies parameters like metric name (`Metric`), category (`Category`), full KPI label (`FullLabel`), unit (`Unit`), and so on. 

In this example, we will set up this file in JSON, which can be accessed in `generate-custom-kpi/kpi-parameters.json`. 

A preview of the JSON file looks like this: 

```json
[
    {
      "Order": 1,
      "Metric": "Collaboration_hours",
      "Category": "Collaboration KPIs",
      "ToDisplay": "",
      "FullLabel": "",
      "Unit": ""
    },
    {
      "Order": 2,
      "Metric": "Percent_week_in_collaboration",
      "Category": "Collaboration KPIs",
      "ToDisplay": "Yes",
      "FullLabel": "% of a 40h workweek spent in collaboration",
      "Unit": "Percent"
    },
    {
      "Order": 3,
      "Metric": "Meeting_hours",
      "Category": "Collaboration KPIs",
      "ToDisplay": "Yes",
      "FullLabel": "Total hours per week in average spent in meetings",
      "Unit": "Hours"
    },
...
]
```

To read this into R and examine the parameters as a data frame / tibble, you can run the following: 
```R
library(jsonlite)
library(here)
library(dplyr)

param_df <- readLines(
    here("examples", "utility-r", "generate-custom-kpi", "kpi-parameters.json")
    ) |>
    fromJSON() |>
    as_tibble()
```

## Setting up the function

The next step is to set up the function that computes the KPIs directly from the two queries that you have run. Here is the full worked out function for our example: 

```R
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
```

This function takes several inputs to generate a custom KPI matrix:

* The `weekly_pq` and `daily_pq` parameters are data frames containing weekly and daily Person Query data from Viva Insights, respectively. 
* The `hrvar` parameter is a character string that specifies the HR variable to be used for grouping. If no HR variable is provided, a default grouping variable named "Group" with the value "Total" will be used. 
* The `schema` parameter is a data frame that defines the structure of the KPIs matrix, which is the parameters data frame loaded in above. If no schema is provided, the function will return unformatted calculations. The schema should include columns for the order of KPIs, the name and category of each KPI, whether the KPI should be displayed, the full label for the KPI, and the unit of measurement.

## Usage

Here is an example of running the function: 
```R
library(vivainsights)

pq_week_df <- import_query(...) # replace with path
pq_day_df <- import_query(...) # replace with path

generate_kpis(
  weekly_pq = pq_week_df,
  daily_pq = pq_day_df,
  hrvar = 'Organization',
  schema = param_df # From above
)
```

The returned output will be a data frame that looks like this: 

| Order | Category           | Group    | FullLabel                                                                 | ValueLabelled | Value        | Unit    | n   | DateRange               |
|-------|--------------------|----------|--------------------------------------------------------------------------|---------------|--------------|---------|-----|-------------------------|
| 1     | Collaboration KPIs | Finance  | % of a 40h workweek spent in collaboration                               | 53.50%        | 0.53  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 2     | Collaboration KPIs | Finance  | Total hours per week in average spent in meetings                        | 35.9 hours    | 35.93  | Hours   | 100 | 2024-05-05 to 2024-05-29 |
| 3     | Collaboration KPIs | Finance  | Total hours per week in average spent on chats                           | 3.5 hours     | 3.55  | Hours   | 100 | 2024-05-05 to 2024-05-29 |
| 4     | Collaboration KPIs | Finance  | Total hours per week in average spent reading and writing emails         | 5.8 hours     | 5.83  | Hours   | 100 | 2024-05-05 to 2024-05-29 |
| 5     | Collaboration KPIs | Finance  | Total hours per week in average spent on unscheduled calls               | 54 minutes    | 54.05  | Minutes | 100 | 2024-05-05 to 2024-05-29 |
| 6     | Meetings behaviour | Finance  | % of meeting hours are ended late                                        | 38%           | 0.37  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 7     | Meetings behaviour | Finance  | % of meeting hours are joined late                                       | 35%           | 0.35  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 8     | Meetings behaviour | Finance  | % of meetings are long (>1 hour) and large meetings (9+ attendees)       | 35.50%        | 0.35  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 9     | Meetings behaviour | Finance  | % of spend >15% of meeting time doing emails or chatting outside of the meeting | 55.90%        | 0.559458591  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 10    | Meetings behaviour | Finance  | Total minutes per day of uninterrupted focus available for employees to do independent work | 309.8 minutes | 309.8387455  | Minutes | 100 | 2024-05-05 to 2024-05-29 |
| 11    | Work life balance  | Finance  | Average first active hours during the day                                | 8 h 5 m       | 8.33  | Time    | 100 | 2024-05-05 to 2024-05-29 |
| 12    | Work life balance  | Finance  | Average last active hours during the day                                 | 39 h 50 m     | 39.49  | Time    | 100 | 2024-05-05 to 2024-05-29 |
| 13    | Work life balance  | Finance  | % who are active on weekends - two or more weekends per month            | 55.50%        | 0.55  | Percent | 100 | 2024-05-05 to 2024-05-29 |
| 14    | Work life balance  | Finance  | Average collaboration minutes per day during the weekend                 | 34.5 minutes  | 34.35   | Minutes | 100 | 2024-05-05 to 2024-05-29 |
| 15    | Work life balance  | Finance  | Number of emails are sent over the weekend on average                    | 3.3           | 3.34  | Number  | 100 | 2024-05-05 to 2024-05-29 |
| 16    | Work life balance  | Finance  | Number of chats are sent over the weekend on average                     | 3.5           | 3.48  | Number  | 100 | 2024-05-05 to 2024-05-29 |
```

## Notes

Note that there is one further custom function that is used in the script called `convert_to_time()`, which converts time into nicely formatted hour and minutes. This function can be found at the bottom of `generate_kpis.R`.

At the beginning of the function, a check also ensures that the daily Person Query and the weekly Person Query are always aligned. A diagnostic message is printed out during the function run so as an Analyst, you can confirm whether you are running the data for the right date range. 

You can find the full parameters json (`kpi-parameters.json`) and the full function (`generate_kpis.R`) in the same directory as this Markdown article. 

