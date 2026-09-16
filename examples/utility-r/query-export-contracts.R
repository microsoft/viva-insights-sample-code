# Confirmed ordered export headers; HR selections and custom Person Query metrics
# below specify this fixture recipe, not a universal tenant configuration.
fixture_hr_columns <- c("Organization", "FunctionType", "LevelDesignation", "IsManager",
                        "Team", "Role", "Seniority", "Tenure", "IsDeveloper")
fixture_person_metrics <- c(
  "Collaboration_hours", "Meeting_hours", "Email_hours", "Chat_hours",
  "Scheduled_call_hours", "Unscheduled_call_hours", "Channel_message_hours",
  "Active_connected_hours", "External_collaboration_hours", "Available_to_focus_hours",
  "Uninterrupted_hours", "Interrupted_hours", "Open_1_hour_block",
  "Recurring_meeting_hours", "Conflicting_meeting_hours",
  "Meeting_hours_with_six_or_fewer_hours_of_advanced_notice",
  "Emails_sent", "Chats_sent", "Meetings", "Calls", "After_hours_collaboration_hours",
  "After_hours_meeting_hours", "After_hours_email_hours", "After_hours_chat_hours",
  "Weekend_collaboration_hours", "Collaboration_span", "Internal_network_size",
  "External_network_size", "Strong_ties", "Diverse_ties", "Network_outside_organization",
  "Total_Copilot_enabled_days")
query_export_headers <- list(
  "person-query/PersonQuery.csv" = c("PersonId", "MetricDate", fixture_person_metrics,
                                    fixture_hr_columns),
  "consumption-query/PeopleMetaData.csv" = c("PeopleHistoricalId", "IsCopilotLicensed",
                                            fixture_hr_columns),
  "consumption-query/PersonM365CreditsMetrics.csv" = c(
    "PersonId", "ServiceId", "ServiceName", "SpendingPolicyId", "MetricDate",
    "Session count", "Spending policy limit", "Total Copilot Credits used",
    "User limit", "PeopleHistoricalId"),
  "consumption-query/PersonGitHubCreditsMetrics.csv" = c(
    "PersonId", "MetricDate", "Total GitHub AI Credits used", "PeopleHistoricalId"),
  "github-query/PersonGitHubActivityMetrics.csv" = c(
    "PersonId", "MetricDate", "Agent adoption", "Code completions accepted",
    "Code completions suggested", "User-initiated chat requests"),
  "github-query/GitHubActivityBreakdownByFeatureMetrics.csv" = c(
    "PersonId", "MetricDate", "Feature", "Feature Usage Count"),
  "github-query/GitHubActivityBreakdownByLanguageFeatureMetrics.csv" = c(
    "PersonId", "MetricDate", "Language", "Feature", "Usage count by language and feature"),
  "github-query/GitHubActivityBreakdownByLanguageModelMetrics.csv" = c(
    "PersonId", "MetricDate", "Language", "Model", "Usage count by language and model"),
  "github-query/GitHubActivityBreakdownByModelFeatureMetrics.csv" = c(
    "PersonId", "MetricDate", "Model", "Feature", "Usage count by model and feature")
)

validate_query_bundle <- function(bundle) {
  if (!identical(names(bundle), names(query_export_headers)))
    stop("Query bundle must contain the exact ordered fixture exports.")
  for (path in names(query_export_headers)) {
    data <- bundle[[path]]
    if (!identical(names(data), query_export_headers[[path]]))
      stop("Ordered header contract failed: ", path)
    # Validate every output before opening even the first destination.
    if (any(vapply(data, function(x)
      is.character(x) && any(grepl('[,"\r\n]', x)), logical(1))))
      stop("Unquoted export contains a delimiter: ", path)
  }
  invisible(TRUE)
}

write_query_bundle <- function(bundle, data_dir) {
  validate_query_bundle(bundle)
  for (path in names(bundle)) {
    dest <- file.path(data_dir, path)
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    write.table(bundle[[path]], dest, sep = ",", quote = FALSE,
                row.names = FALSE, na = "", fileEncoding = "UTF-8")
    message(sprintf("  %-70s %8d rows", path, nrow(bundle[[path]])))
  }
}
