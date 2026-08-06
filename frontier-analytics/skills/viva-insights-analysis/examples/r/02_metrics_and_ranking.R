# Example 2 - Compare a metric across groups and scan several metrics at once.
#
# Demonstrates: create_rank (ranked group table + plot) and keymetrics_scan
# (multi-metric heatmap across groups). The plot is written to the system temp
# directory so nothing is left in the skill folder.

suppressMessages({ library(vivainsights); library(ggplot2) })

pq <- pq_data
outdir <- tempdir()

# 1. Rank Organizations by mean weekly collaboration hours (table).
rank_tbl <- create_rank(pq, metric = "Collaboration_hours",
                        hrvar = "Organization", mingroup = 5, return = "table")
cat("Ranked groups by Collaboration_hours:\n")
print(head(rank_tbl, 10))

# 2. Same, as a plot saved to disk.
rank_plot <- create_rank(pq, metric = "Collaboration_hours",
                         hrvar = "Organization", mingroup = 5, return = "plot")
rank_png <- file.path(outdir, "rank_collaboration.png")
suppressMessages(ggsave(rank_png, rank_plot, width = 8, height = 5, dpi = 150))
cat(sprintf("Saved rank plot: %s\n", rank_png))

# 3. Scan several metrics across groups at once (explicit metric list for
#    reproducibility, since defaults differ between R and Python).
scan <- keymetrics_scan(
  pq, hrvar = "Organization", mingroup = 5,
  metrics = c("Collaboration_hours", "Email_hours", "Meetings",
              "After_hours_collaboration_hours", "Internal_network_size"),
  return = "table"
)
cat("\nKey-metrics scan (head):\n")
print(head(scan))
