# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an accompaniment to the Extending Viva Insights with the {vivainsights} R
# library session. For more information about the package, please visit
# <https://microsoft.github.io/vivainsights/>.
#
# The main function demonstrateded in this example script is
# `create_rank()`.

## Load libraries ----------------------------------------------------------
library(vivainsights)
library(dplyr)

## Load documentation for function
?create_rank

## Automatic exploratory analysis ------------------------------------------

## Returns `simple` mode by default
## Uses `extract_hr()` to detect HR attributes in your data
pq_data %>%
  create_rank(
    metric = "Collaboration_hours"
  )

## Return pairwise combination table
## Will take longer due to number of combinations
pq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    mode = "combine"
  )

## Use specific HR attributes
pq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    hrvar = c("Organization", "LevelDesignation"),
    mode = "combine"
  )


## Return a plot - plot mode 1
pq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    return = "plot",
    plot_mode = 1 # Highlights top and bottom 5 overall
  )

## Return a plot - plot mode 2
pq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    return = "plot",
    plot_mode = 2 # Highlights top and bottom per HR attribute
  )

## Bar chart

pq_data %>%
  create_rank(
    metric = "Collaboration_hours"
  ) %>%
  head(5) %>%
  mutate(label = paste0(hrvar, " - ", group)) %>%
  create_bar_asis(
    group_var = "label",
    bar_var = "Collaboration_hours",
    title = "Top 5 Collaborator Groups",
    caption = "Source: Microsoft Viva Insights"
  )
