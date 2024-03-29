# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an accompaniment to the Extending Workplace Analytics with the {wpa} R
# library session. For more information about the package, please visit
# <https://microsoft.github.io/wpa/>.
#
# The main function demonstrateded in this example script is
# `network_p2p()`.

## Load libraries ----------------------------------------------------------
library(wpa)
library(igraph)
library(dplyr)

## ONA and Community Detection ---------------------------------------------

## Simulate a p2p data set with 800 edges
p2p_data <- p2p_data_sim(size = 200, nei = 4)

## Explore structure
p2p_data %>% glimpse()

## Create a basic visualization
p2p_data %>%
  network_p2p(
    hrvar = "Organization",
    display = "hrvar",
    return = "plot",
    path = NULL # Render in browser instead of saving to PDF
  )

## Return underlying igraph object
g <-
  p2p_data %>%
  network_p2p(
    hrvar = "Organization",
    display = "hrvar",
    return = "network"
  )

## Compute closeness
# Calculate the shortest paths between all nodes, then
# assigns each node a score based on its sum of shortest paths.
igraph::closeness(g) %>%
  tibble::enframe()

## Compute degree
# Number of adjacent edges
igraph::degree(g) %>%
  tibble::enframe() %>%
  summary()

## Compute betweeness
# Number of shortest paths going through a vertex
igraph::betweenness(g) %>%
  tibble::enframe()

## Use Louvain community detection
p2p_data %>%
  network_p2p(
    display = "louvain",
    return = "plot",
    path = NULL # Render in browser instead of saving to PDF
  )

## Return table
p2p_data %>%
  network_p2p(
    display = "louvain",
    return = "table",
    path = NULL # Render in browser instead of saving to PDF
  )

## Create sankey visualization
p2p_data %>%
  network_p2p(
    display = "louvain",
    return = "sankey",
    path = NULL # Render in browser instead of saving to PDF
  )

## Describe communities
## Returns a list of data frames
p2p_data %>%
  network_p2p(
    display = "louvain",
    return = "describe",
    desc_hrvar = c("Organization", "LevelDesignation", "City")
  )

## For large graphs, use fast plotting method
p2p_data %>%
  network_p2p(
    display = "louvain",
    return = "plot",
    size_threshold = 0, # Coerce to 0
    path = NULL # Render in browser instead of saving to PDF
  )

## Return a data frame matching HR variable and communities to nodes
## Using Louvain communities
p2p_data %>%
  network_p2p(display = "louvain",
              return = "data")
