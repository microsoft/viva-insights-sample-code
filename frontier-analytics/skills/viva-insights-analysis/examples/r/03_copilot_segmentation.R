# Example 3 - Copilot usage segmentation.
#
# Builds a total Copilot actions metric from the per-app action columns, then
# classifies the population into usage segments (Power / Habitual / Novice /
# Low / Non-user) with identify_usage_segments.
#
# Convention reminders (see reference/analysis-conventions.md):
# - Preserve observed nulls and the full weekly calendar.
# - Require known eligible activity history throughout the rolling window,
#   rather than inferring eligibility from action presence.

suppressMessages(library(vivainsights))

pq <- pq_data

# 1. Build a total Copilot actions metric from the per-app action columns.
action_cols <- grep("^Copilot_actions_taken_in_", names(pq), value = TRUE)
cat("Per-app action columns:", paste(action_cols, collapse = ", "), "\n")
stopifnot(length(action_cols) > 0)
if (anyNA(pq[, action_cols, drop = FALSE])) {
  stop("Unknown activity makes segments unavailable in this minimal sample recipe")
}
pq$Total_Copilot_actions <- rowSums(pq[, action_cols, drop = FALSE], na.rm = FALSE)
pq$MetricDate <- as.Date(pq$MetricDate)
stopifnot(!anyNA(pq$MetricDate),
          !anyDuplicated(pq[, c("PersonId", "MetricDate")]),
          length(unique(as.POSIXlt(pq$MetricDate)$wday)) == 1L)
pq <- pq[order(pq$PersonId, pq$MetricDate), ]

# 2. Keep the calendar intact for rolling classification, then gate each window.
seg <- identify_usage_segments(pq, metric = "Total_Copilot_actions",
                               version = "12w", return = "data")
for (width in c(4L, 12L)) {
  seg <- seg |>
    dplyr::group_by(PersonId) |>
    dplyr::mutate(.window_ok = slider::slide_lgl(
      !is.na(Total_Copilot_enabled_days) & Total_Copilot_enabled_days > 0 &
        !is.na(Total_Copilot_actions),
      ~ length(.x) == width && all(.x), .before = width - 1L
    ) & as.integer(MetricDate - dplyr::lag(MetricDate, width - 1L)) == (width - 1L) * 7L) |>
    dplyr::ungroup()
  columns <- c(paste0("UsageSegments_", width, "w"), paste0("IsHabit", width, "w"),
               paste0("target_metric_l", width, "w"))
  seg[!seg$.window_ok, columns] <- NA
}
seg$.window_ok <- NULL

# 3. Segment distribution across the whole panel of person-weeks.
#    For a point-in-time adoption report, filter to a single MetricDate first
#    (for example seg[seg$MetricDate == max(seg$MetricDate), ]).
cat("\nUsage segment distribution (person-weeks; NA = unavailable history):\n")
print(table(seg$UsageSegments_12w, useNA = "ifany"))
