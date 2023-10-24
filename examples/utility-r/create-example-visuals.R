#' This script provides a demo on how to generate example visuals using the 'vivainsights' R library

# Set up -------------------------------------------------------------------------------------------
# Load packages
library(tidyverse)
library(vivainsights)

# Set output visual path 
out_vis <- here::here("examples", "utility-r", "example-visuals")

# SVG output: `create_rank()` -----------------------------------------------------------------------

pq_data %>%
    create_rank(
        metric = "Collaboration_hours",
        hrvar = c("Organization", "LevelDesignation", "FunctionType", "SupervisorIndicator"),
        return = "plot") %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_rank.svg", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `create_trend()` ----------------------------------------------------------------------

pq_data %>%
    create_trend(
        metric = "Email_hours",
        hrvar = "LevelDesignation"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_trend", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `create_bar()` ------------------------------------------------------------------------

pq_data %>%
    create_bar(
        metric = "Internal_network_size",
        hrvar = "Organization"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_bar", sep = "/"),
        timestamp = FALSE
        )      

# SVG output: `create_boxplot()` --------------------------------------------------------------------

pq_data %>%
    create_boxplot(
        metric = "Collaboration_hours",
        hrvar = "Organization"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_boxplot", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `create_scatter()` --------------------------------------------------------------------

pq_data %>%
    create_scatter(
        metric_x = "Email_hours",
        metric_y = "Meeting_hours",
        hrvar = "FunctionType"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_scatter", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `create_bubble` --------------------------------------------------------------------        

pq_data %>%
    create_bubble(
        metric_x = "Email_hours",
        metric_y = "Meeting_hours",
        hrvar = "FunctionType"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "create_bubble", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `keymetrics_scan()` -------------------------------------------------------------------

pq_data %>%
    keymetrics_scan() %>%
    export(
        method = "svg",
        path = paste(out_vis, "keymetrics_scan", sep = "/"),
        timestamp = FALSE
        )

# SVG output: `network_g2g()` -----------------------------------------------------------------------

g2g_data %>%
    network_g2g(
        metric = "Meeting_Count",
        primary = "PrimaryCollaborator_Organization",
        secondary = "SecondaryCollaborator_Organization",
        exc_threshold = 0.05
        ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "network_g2g", sep = "/"),
        timestamp = FALSE
        )        

# SVG output: `network_p2p()` -----------------------------------------------------------------------

set.seed(100)
p2p_data <- p2p_data_sim()

p2p_data %>%
    network_p2p(
        style = "ggraph",
        return = "plot"
    ) %>%
    export(
        method = "svg",
        path = paste(out_vis, "network_p2p", sep = "/"),
        timestamp = FALSE
        )      