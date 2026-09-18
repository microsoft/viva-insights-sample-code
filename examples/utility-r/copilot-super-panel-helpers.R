# Helpers for joining the synthetic Person, Consumption and GitHub Copilot
# exports into one person-week analytical panel.

suppressPackageStartupMessages({
  library(dplyr)
})

read_demo_export <- function(data_dir, ...) {
  read.csv(file.path(data_dir, ...),
           check.names = FALSE, stringsAsFactors = FALSE,
           na.strings = c("", "NA"))
}

week_start_sunday <- function(x) {
  x <- as.Date(x)
  x - as.integer(format(x, "%w"))
}

assert_unique_key <- function(data, keys, label) {
  missing <- setdiff(keys, names(data))
  if (length(missing)) {
    stop(label, " missing key columns: ", paste(missing, collapse = ", "))
  }
  if (anyNA(data[keys])) {
    stop(label, " contains missing join keys.")
  }
  if (anyDuplicated(data[keys])) {
    stop(label, " is not unique at grain: ", paste(keys, collapse = " + "))
  }
  invisible(TRUE)
}

clean_label <- function(x, fallback = "Not disclosed") {
  x <- as.character(x)
  if_else(is.na(x) | !nzchar(trimws(x)), fallback, x)
}

safe_divide <- function(num, den) {
  den <- rep_len(den, length(num))
  out <- rep(NA_real_, length(num))
  valid <- !is.na(den) & den > 0
  out[valid] <- num[valid] / den[valid]
  out
}

build_copilot_super_panel <- function(data_dir = "_data") {
  pq <- read_demo_export(data_dir, "person-query", "PersonQuery.csv") |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate))
  people_meta <- read_demo_export(data_dir, "consumption-query",
                                  "PeopleMetaData.csv")
  m365_daily_raw <- read_demo_export(data_dir, "consumption-query",
                                     "PersonM365CreditsMetrics.csv") |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           ServiceName = clean_label(ServiceName))
  github_credits_daily <- read_demo_export(data_dir, "consumption-query",
                                           "PersonGitHubCreditsMetrics.csv") |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate))
  github_activity_daily <- read_demo_export(data_dir, "github-query",
                                            "PersonGitHubActivityMetrics.csv") |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           `Agent adoption` = tolower(as.character(`Agent adoption`)) %in%
             c("true", "yes", "1"))

  feature_daily <- read_demo_export(data_dir, "github-query",
                                    "GitHubActivityBreakdownByFeatureMetrics.csv") |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           Feature = clean_label(Feature))
  language_feature_daily <- read_demo_export(
    data_dir, "github-query", "GitHubActivityBreakdownByLanguageFeatureMetrics.csv"
  ) |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           Language = clean_label(Language),
           Feature = clean_label(Feature))
  language_model_daily <- read_demo_export(
    data_dir, "github-query", "GitHubActivityBreakdownByLanguageModelMetrics.csv"
  ) |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           Language = clean_label(Language),
           Model = clean_label(Model))
  model_feature_daily <- read_demo_export(
    data_dir, "github-query", "GitHubActivityBreakdownByModelFeatureMetrics.csv"
  ) |>
    mutate(MetricDate = as.Date(MetricDate),
           WeekStart = week_start_sunday(MetricDate),
           Model = clean_label(Model),
           Feature = clean_label(Feature))

  assert_unique_key(pq, c("PersonId", "MetricDate"), "Person query")
  assert_unique_key(people_meta, "PeopleHistoricalId", "People metadata")
  assert_unique_key(github_credits_daily, c("PersonId", "MetricDate"),
                    "GitHub credit consumption")
  assert_unique_key(github_activity_daily, c("PersonId", "MetricDate"),
                    "GitHub activity")
  assert_unique_key(feature_daily, c("PersonId", "MetricDate", "Feature"),
                    "GitHub feature breakdown")
  assert_unique_key(language_feature_daily,
                    c("PersonId", "MetricDate", "Language", "Feature"),
                    "GitHub language-feature breakdown")
  assert_unique_key(language_model_daily,
                    c("PersonId", "MetricDate", "Language", "Model"),
                    "GitHub language-model breakdown")
  assert_unique_key(model_feature_daily,
                    c("PersonId", "MetricDate", "Model", "Feature"),
                    "GitHub model-feature breakdown")

  person_keys <- bind_rows(
    select(m365_daily_raw, PersonId, PeopleHistoricalId),
    select(github_credits_daily, PersonId, PeopleHistoricalId)
  ) |>
    filter(!is.na(PersonId), nzchar(PersonId),
           !is.na(PeopleHistoricalId), nzchar(PeopleHistoricalId)) |>
    distinct()
  if (any(count(person_keys, PersonId)$n > 1L)) {
    stop("Crosswalk invariant failed: a PersonId carries more than one PeopleHistoricalId.")
  }
  if (any(count(person_keys, PeopleHistoricalId)$n > 1L)) {
    stop("Crosswalk invariant failed: a PeopleHistoricalId maps to more than one PersonId.")
  }
  if (!all(person_keys$PeopleHistoricalId %in% people_meta$PeopleHistoricalId)) {
    stop("Crosswalk invariant failed: a crosswalk PeopleHistoricalId is absent from PeopleMetaData.")
  }

  person_attributes <- pq |>
    arrange(PersonId, MetricDate) |>
    group_by(PersonId) |>
    summarise(
      Organization = first(clean_label(Organization)),
      FunctionType = first(clean_label(FunctionType)),
      LevelDesignation = first(clean_label(LevelDesignation)),
      IsManager = first(clean_label(IsManager)),
      Team = first(clean_label(Team)),
      Role = first(clean_label(Role)),
      Seniority = first(clean_label(Seniority)),
      Tenure = first(clean_label(Tenure)),
      IsDeveloper = first(clean_label(IsDeveloper)),
      M365EnabledUser = any(Total_Copilot_enabled_days > 0, na.rm = TRUE),
      .groups = "drop"
    )

  m365_weekly <- m365_daily_raw |>
    mutate(Day = MetricDate) |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      m365_observed_days = n_distinct(Day),
      m365_active_days = n_distinct(Day[`Session count` > 0 |
                                           `Total Copilot Credits used` > 0]),
      m365_sessions = sum(`Session count`, na.rm = TRUE),
      m365_credits = sum(`Total Copilot Credits used`, na.rm = TRUE),
      m365_service_count = n_distinct(ServiceName),
      .groups = "drop"
    )
  assert_unique_key(m365_weekly, c("PersonId", "MetricDate"),
                    "M365 consumption aggregated to person-week")

  m365_service_weekly <- m365_daily_raw |>
    group_by(PersonId, MetricDate = WeekStart, ServiceName) |>
    summarise(
      m365_sessions = sum(`Session count`, na.rm = TRUE),
      m365_credits = sum(`Total Copilot Credits used`, na.rm = TRUE),
      .groups = "drop"
    )
  assert_unique_key(m365_service_weekly,
                    c("PersonId", "MetricDate", "ServiceName"),
                    "M365 service consumption aggregated to person-week-service")

  github_credits_weekly <- github_credits_daily |>
    mutate(Day = MetricDate) |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_credit_days = n_distinct(Day),
      github_credits = sum(`Total GitHub AI Credits used`, na.rm = TRUE),
      .groups = "drop"
    )
  assert_unique_key(github_credits_weekly, c("PersonId", "MetricDate"),
                    "GitHub credits aggregated to person-week")

  github_activity_weekly <- github_activity_daily |>
    mutate(Day = MetricDate) |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_observed_days = n_distinct(Day),
      github_active_days = n_distinct(Day[
        `Code completions suggested` > 0 |
          `Code completions accepted` > 0 |
          `User-initiated chat requests` > 0 |
          `Agent adoption`
      ]),
      github_code_suggestions = sum(`Code completions suggested`, na.rm = TRUE),
      github_code_acceptances = sum(`Code completions accepted`, na.rm = TRUE),
      github_chat_requests = sum(`User-initiated chat requests`, na.rm = TRUE),
      github_agent_adoption_days = sum(`Agent adoption`, na.rm = TRUE),
      .groups = "drop"
    )
  assert_unique_key(github_activity_weekly, c("PersonId", "MetricDate"),
                    "GitHub activity aggregated to person-week")

  github_feature_weekly <- feature_daily |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_feature_usage_count = sum(`Feature Usage Count`, na.rm = TRUE),
      github_distinct_features = n_distinct(Feature),
      .groups = "drop"
    )
  github_language_feature_weekly <- language_feature_daily |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_language_feature_usage_count =
        sum(`Usage count by language and feature`, na.rm = TRUE),
      github_distinct_languages_feature = n_distinct(Language),
      .groups = "drop"
    )
  github_language_model_weekly <- language_model_daily |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_language_model_usage_count =
        sum(`Usage count by language and model`, na.rm = TRUE),
      github_distinct_languages_model = n_distinct(Language),
      github_distinct_models_language = n_distinct(Model),
      .groups = "drop"
    )
  github_model_feature_weekly <- model_feature_daily |>
    group_by(PersonId, MetricDate = WeekStart) |>
    summarise(
      github_model_feature_usage_count =
        sum(`Usage count by model and feature`, na.rm = TRUE),
      github_distinct_models_feature = n_distinct(Model),
      .groups = "drop"
    )

  panel <- pq |>
    select(-WeekStart) |>
    left_join(m365_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_credits_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_activity_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_feature_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_language_feature_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_language_model_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    left_join(github_model_feature_weekly, by = c("PersonId", "MetricDate"),
              relationship = "one-to-one") |>
    mutate(
      m365_observed_use = coalesce(m365_credits > 0 | m365_sessions > 0, FALSE),
      github_observed_use = case_when(
        !is.na(github_active_days) ~ github_active_days > 0,
        !is.na(github_credits) ~ github_credits > 0,
        TRUE ~ FALSE
      ),
      any_observed_product_use = m365_observed_use | github_observed_use,
      github_activity_observed = !is.na(github_observed_days),
      m365_credit_row_observed = !is.na(m365_observed_days),
      credits_per_m365_session = safe_divide(m365_credits, m365_sessions)
    )
  assert_unique_key(panel, c("PersonId", "MetricDate"),
                    "Copilot super panel")
  if (nrow(panel) != nrow(pq)) {
    stop("Copilot super panel changed the Person Query row count.")
  }

  list(
    panel = panel,
    person_attributes = person_attributes,
    person_keys = person_keys,
    people_meta = people_meta,
    m365_daily_raw = m365_daily_raw,
    m365_weekly = m365_weekly,
    m365_service_weekly = m365_service_weekly,
    github_credits_daily = github_credits_daily,
    github_credits_weekly = github_credits_weekly,
    github_activity_daily = github_activity_daily,
    github_activity_weekly = github_activity_weekly,
    github_feature_weekly = github_feature_weekly,
    github_language_feature_weekly = github_language_feature_weekly,
    github_language_model_weekly = github_language_model_weekly,
    github_model_feature_weekly = github_model_feature_weekly
  )
}
