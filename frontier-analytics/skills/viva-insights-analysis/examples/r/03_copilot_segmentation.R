# Example 3 - Copilot usage segmentation.
#
# Builds a total Copilot actions metric from the per-app action columns, then
# classifies the population into usage segments (Power / Habitual / Novice /
# Low / Non-user) with identify_usage_segments.
#
# Convention reminders (see reference/analysis-conventions.md):
# - Fill the metric NAs with 0 before segmenting, or the rolling window leaves
#   people unclassified.
# - Define the licensed population by enabled days for a snapshot rather than by
#   action presence.

suppressMessages(library(vivainsights))

pq <- pq_data

# 1. Build a total Copilot actions metric from the per-app action columns.
action_cols <- grep("^Copilot_actions_taken_in_", names(pq), value = TRUE)
cat("Per-app action columns:", paste(action_cols, collapse = ", "), "\n")
pq$Total_Copilot_actions <- rowSums(pq[, action_cols], na.rm = TRUE)  # NA -> 0

# 2. Classify into 12-week usage segments.
seg <- identify_usage_segments(pq, metric = "Total_Copilot_actions",
                               version = "12w", return = "data")

# 3. Segment distribution across the whole panel of person-weeks.
#    For a point-in-time adoption report, filter to a single MetricDate first
#    (for example seg[seg$MetricDate == max(seg$MetricDate), ]).
cat("\nUsage segment distribution (all person-weeks):\n")
print(table(seg$UsageSegments_12w, useNA = "ifany"))
