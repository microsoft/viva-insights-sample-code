# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# This is an accompaniment to the Extending Viva Insights with the {vivainsights} R
# library session. For more information about the package, please visit
# <https://microsoft.github.io/vivainsights/>.
#
# The main function demonstrateded in this example script is
# `network_p2p()`.

## Load libraries ----------------------------------------------------------
library(vivainsights)
library(igraph)
library(dplyr)

## ONA and Community Detection ---------------------------------------------

## Simulate a p2p data set with 800 edges
p2p_data <- p2p_data_sim(size = 200, nei = 4)

## Explore structure
p2p_data %>% glimpse()

## Create a basic visualization
# Renders in browser
p2p_data %>%
  network_p2p(
    hrvar = "Organization",
    return = "plot"
  )

## Return underlying igraph object
g <-
  p2p_data %>%
  network_p2p(
    hrvar = "Organization",
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

## Summary of all the centrality metrics by `node_id`
network_summary(g)

## Use Louvain community detection
# Custom resolution argument to control number of clusters returned
# See <https://igraph.org/r/doc/cluster_louvain.html> for more information
p2p_data %>%
  network_p2p(
    community = "louvain",
    comm_args = list(resolution = 0.5),
    return = "plot"
  )

## Use Leiden community detection
# See <https://igraph.org/r/doc/cluster_leiden.html> for more information
# Change the `community` and `comm_args` to use different algorithms
p2p_data %>%
  network_p2p(
    community = "leiden",
    comm_args = list(resolution_parameter = 0.5),
    return = "plot"
  )

## Return table
p2p_data %>%
  network_p2p(
    community = "louvain",
    comm_args = list(resolution = 0.5),
    return = "table"
  )

## Create sankey visualization
p2p_data %>%
  network_p2p(
    community = "louvain",
    comm_args = list(resolution = 0.5),
    return = "sankey"
  )

## For large graphs, use fast plotting method
# 'igraph' style
p2p_data %>%
  network_p2p(
    community = "louvain",
    return = "plot",
    style = "igraph"
  )

## Return a data frame matching HR variable and communities to nodes
## Using Louvain communities
p2p_data %>%
  network_p2p(
    community = "louvain",
    return = "data"
    )
