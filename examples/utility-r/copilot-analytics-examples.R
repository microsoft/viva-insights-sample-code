#' ==================================================================================================
#' Viva Insights - Copilot Analytics - Example R Script
#' This script provides a demo on how to generate example visuals using the 'vivainsights' R library,
#' using Copilot metrics from Viva Insights.
#' ==================================================================================================

# Load packages
library(tidyverse)
library(vivainsights)
library(here)

# Load the dataset
# Note: replace the path to your own dataset. The dataset should be a Person Query and contain the 
# required Copilot metrics. 
demo_pq <- import_query(here("data", "pq_data.csv"))

# In the code below, you can replace: 
# - the dataset name with your own dataset name
# - the metric names with your own metric names (specified in string)
# - the organizational attribute (`hrvar`) with your own organizational attribute

# Assign names of relevant metrics to variables -------------------------------------------------------
# Metrics: Copilot actions taken in
metrics_cop_actions_taken_in <-
  demo_pq %>%
  names() %>%
  str_subset("Copilot_actions_taken_in")

# Metrics: summarise
metrics_summarise_cop <-
  demo_pq %>%
  names() %>%
  str_subset("Summarise")

# Key metrics scan: Copilot actions taken in ----------------------------
# Create plot
temp_plot <- 
  keymetrics_scan(
    data = demo_pq,
    hrvar = "Organization",
    metrics = metrics_cop_actions_taken_in,
    return = "plot"
)

temp_plot %>%
  export(
    method = "png",
    path = here("output", "keymetrics_scan_cop_actions"),
    timestamp = FALSE,
    width = 9.25,
    height = 5
  )

# Key metrics scan: High vs Medium vs Low Copilot users ------------------
tb_cop_usage_segments <-
  demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)
  ) %>%
    rowSums()) %>%
  group_by(PersonId) %>%
  summarise(Total_Copilot_actions = mean(Total_Copilot_actions, na.rm = TRUE)) %>%
  mutate(CopilotUsageSegment = case_when(
    Total_Copilot_actions >= 10 ~ "Heavy\n(10+ actions)",
    Total_Copilot_actions >= 4 ~ "Medium\n(4-9 actions)",
    Total_Copilot_actions >= 1 ~ "Low\n(1-3 actions)",
    TRUE ~ "Non-user"
    )) %>%
  mutate(CopilotUsageSegment = factor(
    CopilotUsageSegment,
    levels = c(
      "Heavy\n(10+ actions)",
      "Medium\n(4-9 actions)",
      "Low\n(1-3 actions)",
      "Non-user"
  )))

# Join user segments with Heavy Copilot users
temp_plot <- 
  demo_pq %>%
  left_join(tb_cop_usage_segments, by = "PersonId") %>%
  keymetrics_scan(
    hrvar = "CopilotUsageSegment"
  )

temp_plot %>%
  export(
    method = "png",
    path = here("output", "keymetrics_scan_cop_segments"),
    timestamp = FALSE,
    width = 9.25,
    height = 5
  )

# Boxplot - Copilot Assisted Hours ---------------------------------------
# Create plot
temp_plot <-
  demo_pq %>%
  create_boxplot(
    hrvar = "Team",
    metric = "Copilot_assisted_hours",
    return = "plot"
  )

# Save plot to output folder
temp_plot %>%
  export(
    method = "png",
    path = here("output", "boxplot_copilot_assisted_hours"),
    timestamp = FALSE,
    width = 9.25,
    height = 5
  )

# Lorenz curve - Total Copilot actions --------------------------------
# Create plot
temp_plot <-
  demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)) %>%
           rowSums()) %>%
  create_lorenz(
    metric = "Total_Copilot_actions",
    return = "plot"
  )

# Save plot to output folder
temp_plot %>%
  export(
    method = "png",
    path = here("output", "lorenz_total_copilot_actions"),
    timestamp = FALSE,
    width = 6,
    height = 5
  )

# Cumulative share table
demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)) %>%
      rowSums()) %>%
  create_lorenz(
    metric = "Total_Copilot_actions",
    return = "table"
  ) %>%
  export()

# Ranked - Total Copilot Actions ----------------------------------------
# Top 10 - results copied to clipboard
demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)) %>%
      rowSums()) %>%
  mutate(Function = str_replace(Function, "Team", "Org")) %>%
  create_rank(
    metric = "Total_Copilot_actions",
    hrvar = c("Organization", "Function"),
    return = "table"
  ) %>%
  head(10) %>%
  export()

# Bottom 10 - results copied to clipboard
demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)) %>%
      rowSums()) %>%
  mutate(Function = str_replace(Function, "Team", "Org")) %>%
  create_rank(
    metric = "Total_Copilot_actions",
    hrvar = c("Organization", "Function"),
    return = "table"
  ) %>%
  tail(10) %>%
  export()

# Information value - Heavy Copilot Users ------------------------------
# Identify heavy Copilot users
tb_heavy_copilot_users <-
  demo_pq %>%
  mutate(Total_Copilot_actions = select(
    ., all_of(metrics_cop_actions_taken_in)
  ) %>%
    rowSums()) %>%
  group_by(PersonId) %>%
  summarise(Total_Copilot_actions = mean(Total_Copilot_actions, na.rm = TRUE)) %>%
  mutate(HeavyCopilotUsers = ifelse(Total_Copilot_actions >= 10, 1, 0))

# Join user segments with Heavy Copilot users
demo_pq %>%
  left_join(tb_heavy_copilot_users, by = "PersonId") %>%
  create_IV(
    outcome = "HeavyCopilotUsers",
    predictors = c(
      "Collaboration_hours",
      "Internal_network_size",
      "Influencer_score",
      "Emails_sent",
      "Active_connected_hours"
    ),
    return = "plot"
  ) 

# Internal and External network size -------------------------------------
# By Copilot usage segments

temp_plot <-
  demo_pq %>%
  left_join(tb_cop_usage_segments, by = "PersonId") %>%
  create_bubble(
    metric_x = "Internal_network_size",
    metric_y = "External_network_size",
    hrvar = "CopilotUsageSegment"
  )

temp_plot %>%
  export(
    method = "png",
    path = here("output", "network_sizes_by_cop_usage"),
    timestamp = FALSE,
    width = 9.25,
    height = 5
  )
