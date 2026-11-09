# Example 4 - Meeting Query quality filter.
#
# Raw Meeting Query rows can include personal holds, all-day markers, cancelled
# items and broadcasts. This shows a generalised quality filter and a per-rule
# attrition table. Exclusion rates depend on the export and analytical question.
#
# Note: the built-in mt_data sample is small and synthetic, so very few rows
# survive the strict filter. The point here is the filter mechanics and the
# attrition breakdown rather than the surviving counts. Adapt thresholds to your data.
#
# Reminder: Duration and *_hours columns are measured in hours rather than minutes.

suppressMessages(library(vivainsights))
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
if (length(script) != 1L) stop("Run this example with Rscript r/04_meeting_quality.R")
source(file.path(dirname(script), "..", "meeting_quality.R"))

mt <- mt_data
n0 <- nrow(mt)

# Per-rule attrition: how many rows each rule would remove on its own.
rules <- list(
  "attendees < 2"              = mt$Number_of_attendees < 2,
  "intended <= 1"              = mt$Intended_participant_count <= 1,
  "intended > 150 (broadcast)" = mt$Intended_participant_count > 150,
  "cancelled"                  = normalise_flag(mt$Cancelled),
  "all-day"                    = normalise_flag(mt$All_Day_Meeting),
  "duration < 1 minute"        = mt$Duration < 1 / 60,
  "duration > 8 hours"         = mt$Duration > 8
)
attrition <- data.frame(
  rule = names(rules),
  rows_flagged = vapply(rules, function(x) sum(x, na.rm = TRUE), integer(1)),
  row.names = NULL
)
cat(sprintf("Raw rows: %d\n", n0))
print(attrition)

# Combined quality filter.
mq <- mt[quality_mask(mt), , drop = FALSE]
missing <- !complete.cases(mt[c("Number_of_attendees", "Intended_participant_count", "Duration")]) |
  is.na(normalise_flag(mt$Cancelled)) | is.na(normalise_flag(mt$All_Day_Meeting))
cat("Missing quality inputs:", sum(missing), "\n")
cat(sprintf("\nAfter quality filter: %d -> %d rows (%.1f%% retained)\n",
            n0, nrow(mq), 100 * nrow(mq) / n0))
