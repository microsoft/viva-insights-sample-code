# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an accompaniment to the Extending Workplace Analytics with the {wpa} R
# library session. For more information about the package, please visit
# <https://microsoft.github.io/wpa/>.
#
# The main function demonstrateded in this example script is
# `create_rank()`.

## Load libraries ----------------------------------------------------------
library(wpa)
library(dplyr)

## Load documentation for function
?create_rank

## Automatic exploratory analysis ------------------------------------------

## Returns `simple` mode by default
## Uses `extract_hr()` to detect HR attributes in your data
sq_data %>%
  create_rank(
    metric = "Collaboration_hours"
  )

## Return pairwise combination table
## Will take longer due to number of combinations
sq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    mode = "combine"
  )

## Use specific HR attributes
sq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    hrvar = c("Organization", "LevelDesignation"),
    mode = "combine"
  )


## Return a plot - plot mode 1
sq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    return = "plot",
    plot_mode = 1 # Highlights top and bottom 5 overall
  )

## Return a plot - plot mode 2
sq_data %>%
  create_rank(
    metric = "Collaboration_hours",
    return = "plot",
    plot_mode = 2 # Highlights top and bottom per HR attribute
  )

## Bar chart

sq_data %>%
  create_rank(
    metric = "Collaboration_hours"
  ) %>%
  head(5) %>%
  create_bar_asis(
    group_var = "group",
    bar_var = "Collaboration_hours"
  )
