# Example 4 - Meeting Query quality filter.
#
# Roughly half of raw Meeting Query rows are not real collaborative meetings
# (personal holds, all-day markers, cancelled items, huge broadcasts). This shows
# the generalised quality filter and a per-rule attrition table.
#
# Note: the built-in mt_data sample is small and synthetic, so very few rows
# survive the strict filter. The point here is the filter mechanics and the
# attrition breakdown, not the surviving counts. Adapt thresholds to your data.
#
# Reminder: Duration and *_hours columns are in HOURS, not minutes.

suppressMessages(library(vivainsights))

mt <- mt_data
n0 <- nrow(mt)

# Per-rule attrition: how many rows each rule would remove on its own.
rules <- list(
  "attendees < 2"              = mt$Number_of_attendees < 2,
  "intended <= 1"              = mt$Intended_participant_count <= 1,
  "intended > 150 (broadcast)" = mt$Intended_participant_count > 150,
  "cancelled"                  = as.logical(mt$Cancelled),
  "all-day"                    = as.logical(mt$All_Day_Meeting)
)
attrition <- data.frame(
  rule = names(rules),
  rows_flagged = vapply(rules, function(x) sum(x, na.rm = TRUE), integer(1)),
  row.names = NULL
)
cat(sprintf("Raw rows: %d\n", n0))
print(attrition)

# Combined quality filter.
mq <- subset(
  mt,
  Number_of_attendees >= 2 &
    Intended_participant_count > 1 &
    Intended_participant_count <= 150 &
    !as.logical(Cancelled) &
    !as.logical(All_Day_Meeting)
)
cat(sprintf("\nAfter quality filter: %d -> %d rows (%.1f%% retained)\n",
            n0, nrow(mq), 100 * nrow(mq) / n0))
