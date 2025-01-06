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
  demo_raw %>%
  names() %>%
  str_subset("Copilot_actions_taken_in")

# Metrics: summarise
metrics_summarise_cop <-
  demo_raw %>%
  names() %>%
  str_subset("Summarise")

# Key metrics scan: Copilot actions taken in ----------------------------
# Create plot
temp_plot <- 
  keymetrics_scan(
    data = demo_raw,
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

# Boxplot - Copilot Assisted Hours ---------------------------------------
# Create plot
temp_plot <-
  demo_raw %>%
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
  demo_raw %>%
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
demo_raw %>%
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
demo_raw %>%
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
demo_raw %>%
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