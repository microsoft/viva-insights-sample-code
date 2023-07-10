# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an additional ONA example script to accompany `example_ONA.R`. For
# more information about the package, please visit
# <https://microsoft.github.io/wpa/>.
#
# The main function demonstrateded in this example script is
# `network_g2g()`.

## Load libraries ----------------------------------------------------------
library(wpa)
library(igraph)
library(tidyverse)

## Basic outputs -----------------------------------------------------------

## Examine build-in group-to-group dataset
g2g_data %>% glimpse()

## Visualize network
g2g_data %>% network_g2g()

## Return an interaction matrix
g2g_data %>% network_g2g(return = "table")

## Return an igraph object
g2g_data %>% network_g2g(return = "network")

## Return edge list, with weight re-calculated as proportion of total
g2g_data %>% network_g2g(return = "data")

## Return a sankey visualization
g2g_data %>%
  network_g2g(return = "data") %>%
  create_sankey(
    var1 = "TimeInvestorOrg",
    var2 = "CollaboratorOrg",
    count = "metric_prop"
  )

# Customize visualization -------------------------------------------------

## Visualize network - customize metric and threshold
## `Meeting_hours` and 5% threshold
g2g_data %>%
  network_g2g(time_investor = "TimeInvestors_Organization",
              collaborator = "Collaborators_Organization",
              metric = "Meeting_hours",
              exc_threshold = 0.05)

## Change layout
g2g_data %>%
  network_g2g(time_investor = "TimeInvestors_Organization",
              collaborator = "Collaborators_Organization",
              metric = "Meeting_hours",
              exc_threshold = 0.05,
              algorithm = "mds") # multi-dimensionality scaling


## Overlay org colours
# Return a network plot - custom-specific colours
# Get labels of orgs and assign random colours
org_str <- unique(g2g_data$TimeInvestors_Organization)

col_str <-
  sample(
    x = c("red", "green", "blue"),
    size = length(org_str),
    replace = TRUE
  )

# Create and supply a named vector to `node_colour`
names(col_str) <- org_str

g2g_data %>%
  network_g2g(node_colour = col_str)


## Overlay org sizes
# Return a network plot with circle layout
# Vary node colours and add org sizes
org_tb <- hrvar_count(
  sq_data,
  hrvar = "Organization",
  return = "table"
)

# `org_tb` can be replaced with another data frame to control for the size of
# the bubbles

g2g_data %>%
  network_g2g(algorithm = "circle",
              node_colour = "vary",
              org_count = org_tb)

# Return an interaction matrix
# Minimum arguments specified
g2g_data %>%
  network_g2g(return = "table")
