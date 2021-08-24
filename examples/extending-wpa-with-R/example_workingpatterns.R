# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an accompaniment to the Extending Workplace Analytics with the {wpa} R
# library session. For more information about the package, please visit
# <https://microsoft.github.io/wpa/>.
#
# The main function demonstrateded in this example script is
# `workpatterns_classify()` and `flex_index()`.

## Load libraries ----------------------------------------------------------
library(wpa)
library(dplyr)

## Load documentation for function
?workpatterns_classify
?flex_index

## Working Patterns --------------------------------------------------------

## Explore structure of Hourly Collaboration data
em_data %>% glimpse()

## Plot a breakdown of working patterns for entire org
## using default settings
em_data %>%
  workpatterns_classify()

## Plot split by HR attribute
em_data %>%
  workpatterns_classify(
    return = "plot-hrvar"
  )

## Plot area chart
em_data %>%
  workpatterns_classify(
    return = "plot-area"
  )

## Plot dist chart
## Distribution of signals by hour
em_data %>%
  workpatterns_classify(
    return = "plot-dist"
  )

## Return a list of outputs from working patterns
## More performance than running each output individually
wp_list <-
  em_data %>%
  workpatterns_classify(
      return = "list"
  )

## Get summary table
wp_list$table

## Get flat data output
wp_list$data

## Flexibility Index -------------------------------------------------------

## Return to top 10 most common working patterns
em_data %>%
  flex_index(
    plot_method = "common"
  )

## Plot a sample of 10 working patterns
em_data %>%
  flex_index(
    plot_method = "sample"
  )

## Plot a time series view of flexibility index
em_data %>%
  flex_index(
    plot_method = "time"
  )

## Return underlying data
em_data %>%
  flex_index(
    return = "data"
  )

## Return summary table
em_data %>%
  flex_index(
    return = "table"
  )

## Return summary table with HR attribute splits
em_data %>%
  flex_index(
    hrvar = "Organization",
    return = "table"
  )
