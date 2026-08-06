# Example 5 - Person-to-person network analysis.
#
# A P2P export is an edge list: one row per collaboration tie between two people,
# with tie-strength scores. network_p2p builds the graph and can detect
# communities. network_summary computes node centrality.
#
# The built-in R p2p_data ships with HR attributes already, so network_p2p runs
# directly. network_p2p requires a single-date snapshot.

suppressMessages(library(vivainsights))

cat("Edge-list columns:", paste(names(p2p_data), collapse = ", "), "\n")
cat("Ties:", nrow(p2p_data), "\n")

# Filter to a single-date snapshot (required by network_p2p).
one_date <- sort(unique(p2p_data$MetricDate))[1]
one <- subset(p2p_data, MetricDate == one_date)
cat(sprintf("Snapshot %s: %d ties\n", one_date, nrow(one)))

# Build the graph object.
graph <- network_p2p(data = one, return = "network")
cat("Graph object class:", class(graph)[1], "\n")

# Summarise node centrality.
summary_tbl <- network_summary(graph, return = "table")
cat(sprintf("\nNode centrality table: %d nodes x %d metrics\n",
            nrow(summary_tbl), ncol(summary_tbl)))
print(head(summary_tbl))
