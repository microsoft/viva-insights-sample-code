# Developer Experience and Copilot: read real-schema synthetic query exports.
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

MIN_GROUP_N <- 10L
BASELINE_WEEKS <- 8L
AFTER_HOURS_CONVENTION <- 3
PERSISTENCE_WEEKS <- 4L
DATA_DIR <- '_data'
stopifnot(AFTER_HOURS_CONVENTION >= 0, PERSISTENCE_WEEKS >= 2,
          PERSISTENCE_WEEKS <= BASELINE_WEEKS)
options(OutDec = '.')

read_export <- function(...) {
  read.csv(file.path(DATA_DIR, ...), check.names = FALSE,
           stringsAsFactors = FALSE, na.strings = c('', 'NA'))
}
as_bool <- function(x) {
  if (is.logical(x)) return(x)
  value <- tolower(trimws(as.character(x)))
  case_when(value %in% c('true', 'yes', '1') ~ TRUE,
            value %in% c('false', 'no', '0') ~ FALSE,
            TRUE ~ NA)
}
week_start <- function(x) {
  d <- as.Date(x)
  d - as.POSIXlt(d)$wday
}
assert_key <- function(data, keys) {
  stopifnot(all(keys %in% names(data)), !anyDuplicated(data[keys]), !anyNA(data[keys]))
}
assert_columns <- function(data, cols, label) {
  missing <- setdiff(cols, names(data))
  if (length(missing)) stop(label, ' missing columns: ', paste(missing, collapse = ', '))
}
safe_div <- function(num, den) ifelse(is.na(den) | den == 0, NA_real_, num / den)
missing_label <- function(x) {
  value <- trimws(as.character(x))
  if_else(is.na(value) | value %in% c('', 'NA', '#N/A', 'N/A', 'NULL'),
          'Unknown / not supplied', value)
}

# This is an analyst-side evidence object, NOT an additional query-export schema.
# A caller must obtain completeness independently of the activity rows, licence
# metadata and calendar. No evidence is inferred from filenames or fixture shape.
derive_m365_coverage <- function(person_weeks, evidence = NULL) {
  assert_columns(person_weeks, c('PersonId', 'Week', 'Total_Copilot_enabled_days'),
                 'Person Query eligibility')
  assert_key(person_weeks, c('PersonId', 'Week'))
  enabled <- person_weeks$Total_Copilot_enabled_days
  stopifnot(is.numeric(enabled), all(is.na(enabled) |
              (is.finite(enabled) & enabled >= 0 & enabled <= 7)))
  result <- person_weeks |>
    transmute(PersonId, Week,
              M365_eligible = Total_Copilot_enabled_days > 0)
  if (is.null(evidence)) {
    return(result |> mutate(M365_complete = NA,
                            M365_covered_days = NA_integer_))
  }
  assert_columns(evidence, c('PersonId', 'Week', 'Complete', 'Covered_days',
                            'Evidence_source'), 'Independent coverage evidence')
  assert_key(evidence, c('PersonId', 'Week'))
  stopifnot(inherits(evidence$Week, 'Date'), is.logical(evidence$Complete),
            is.numeric(evidence$Covered_days),
            all(!is.na(evidence$Evidence_source) &
                  nzchar(trimws(evidence$Evidence_source))),
            all(is.na(evidence$Covered_days) |
                  (is.finite(evidence$Covered_days) & evidence$Covered_days >= 0 &
                     evidence$Covered_days <= 7 &
                     evidence$Covered_days == floor(evidence$Covered_days))),
            all(!coalesce(evidence$Complete, FALSE) |
                  (!is.na(evidence$Covered_days) & evidence$Covered_days > 0)))
  result |>
    left_join(evidence |> transmute(PersonId, Week, M365_complete = Complete,
                                    M365_covered_days = Covered_days),
              by = c('PersonId', 'Week'), relationship = 'one-to-one')
}

observed_or_missing <- function(value, eligible, complete) {
  if_else(coalesce(eligible & complete, FALSE), coalesce(value, 0), NA_real_)
}

# Suppress the whole partition, not just its small cell: published totals and
# other margins must not make the withheld value recoverable by subtraction.
disclose_partition <- function(data, count = 'People', minimum = MIN_GROUP_N) {
  values <- data[[count]]
  if (anyNA(values) || any(values > 0 & values < minimum)) return(data[0, ])
  data
}
disclosable_split <- function(total, part, minimum = MIN_GROUP_N) {
  !is.na(total) & !is.na(part) & total >= minimum &
    (part == 0 | part >= minimum) &
    (total - part == 0 | total - part >= minimum)
}
# An empty population has nobody to protect, and this report already publishes
# zero cells elsewhere (see disclose_partition). This decides WHERE the split
# gate is required; it does not change the gate. Suppressing "0 developers were
# validly observed" taught nothing and hid the coverage story it exists to tell.
split_or_empty <- function(total, part, minimum = MIN_GROUP_N) {
  (!is.na(total) & total == 0 & !is.na(part) & part == 0) |
    disclosable_split(total, part, minimum)
}

# Protect differences between overlapping service/category supports as well as
# each positive-contributor count. Every membership pattern includes the full
# parent population, so absent/noncontributing complements cannot be inferred.
support_release_safe <- function(data, groups, metrics, parent_ids) {
  parent_ids <- unique(parent_ids)
  if (length(parent_ids) < MIN_GROUP_N || anyNA(parent_ids)) return(FALSE)
  if (!all(data$PersonId %in% parent_ids)) return(FALSE)
  memberships <- setNames(rep('', length(parent_ids)), parent_ids)
  for (metric in metrics) {
    if (anyNA(data[[metric]]) || any(!is.finite(data[[metric]]))) return(FALSE)
    cells <- interaction(data[groups], drop = TRUE, lex.order = TRUE)
    for (cell in levels(cells)) {
      rows <- cells == cell
      observed <- unique(data$PersonId[rows])
      positive <- unique(data$PersonId[rows & data[[metric]] > 0])
      if (!disclosable_split(length(parent_ids), length(observed)) ||
          !disclosable_split(length(parent_ids), length(positive))) return(FALSE)
      memberships[observed] <- paste0(memberships[observed], '|', metric, ':', cell, ':row')
      memberships[positive] <- paste0(memberships[positive], '|', metric, ':', cell, ':positive')
    }
  }
  all(table(memberships) >= MIN_GROUP_N)
}

pq <- read_export('person-query', 'PersonQuery.csv') |>
  mutate(MetricDate = as.Date(MetricDate),
         Week = MetricDate,
         IsDeveloper = as_bool(IsDeveloper),
         IsManager = as_bool(IsManager),
         across(any_of(c('Team', 'Role', 'Seniority', 'Tenure')), missing_label))
people_meta <- read_export('consumption-query', 'PeopleMetaData.csv') |>
  mutate(IsCopilotLicensed = as_bool(IsCopilotLicensed),
         IsDeveloper = as_bool(IsDeveloper),
         IsManager = as_bool(IsManager))
gh_daily <- read_export('github-query', 'PersonGitHubActivityMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate),
         Week = week_start(MetricDate),
         `Agent adoption` = as_bool(`Agent adoption`))
m365_raw <- read_export('consumption-query', 'PersonM365CreditsMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate),
         Week = week_start(MetricDate))
github_credits_daily <- read_export('consumption-query', 'PersonGitHubCreditsMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate),
         Week = week_start(MetricDate))
feature_daily <- read_export('github-query', 'GitHubActivityBreakdownByFeatureMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate),
         Feature = missing_label(Feature))
language_feature_daily <- read_export('github-query', 'GitHubActivityBreakdownByLanguageFeatureMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate),
         across(c(Language, Feature), missing_label))
language_model_daily <- read_export('github-query', 'GitHubActivityBreakdownByLanguageModelMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate),
         across(c(Language, Model), missing_label))
model_feature_daily <- read_export('github-query', 'GitHubActivityBreakdownByModelFeatureMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate),
         across(c(Model, Feature), missing_label))

assert_columns(pq, c('PersonId', 'MetricDate', 'Collaboration_hours', 'Meeting_hours',
  'Scheduled_call_hours', 'Available_to_focus_hours', 'Uninterrupted_hours',
  'Interrupted_hours', 'Open_1_hour_block', 'Recurring_meeting_hours', 'Conflicting_meeting_hours',
  'Meeting_hours_with_six_or_fewer_hours_of_advanced_notice',
  'After_hours_collaboration_hours', 'Total_Copilot_enabled_days', 'Team', 'Role',
  'Seniority', 'Tenure', 'IsDeveloper'), 'PersonQuery.csv')
assert_columns(people_meta, c('PeopleHistoricalId', 'IsCopilotLicensed', 'Team',
  'Role', 'Seniority', 'Tenure', 'IsDeveloper'), 'PeopleMetaData.csv')
assert_columns(gh_daily, c('PersonId', 'MetricDate', 'Agent adoption',
  'Code completions accepted', 'Code completions suggested',
  'User-initiated chat requests'), 'PersonGitHubActivityMetrics.csv')
assert_columns(m365_raw, c('PersonId', 'ServiceName', 'MetricDate', 'Session count',
  'Total Copilot Credits used', 'PeopleHistoricalId'), 'PersonM365CreditsMetrics.csv')

assert_key(pq, c('PersonId', 'MetricDate'))
assert_key(people_meta, 'PeopleHistoricalId')
assert_key(gh_daily, c('PersonId', 'MetricDate'))
assert_key(github_credits_daily, c('PersonId', 'MetricDate'))
assert_key(feature_daily, c('PersonId', 'MetricDate', 'Feature'))
assert_key(language_feature_daily, c('PersonId', 'MetricDate', 'Language', 'Feature'))
assert_key(language_model_daily, c('PersonId', 'MetricDate', 'Language', 'Model'))
assert_key(model_feature_daily, c('PersonId', 'MetricDate', 'Model', 'Feature'))

weeks <- sort(unique(pq$Week))
baseline_start <- tail(weeks, BASELINE_WEEKS)[1]
period_end <- max(weeks) + 6
period_label <- paste(format(baseline_start, '%d %b %Y'), 'to', format(period_end, '%d %b %Y'))
history_label <- paste(format(min(weeks), '%d %b %Y'), 'to', format(period_end, '%d %b %Y'))
product_start <- min(c(gh_daily$MetricDate, m365_raw$MetricDate, github_credits_daily$MetricDate))
product_end <- max(c(gh_daily$MetricDate, m365_raw$MetricDate, github_credits_daily$MetricDate))
product_label <- paste(format(product_start, '%d %b %Y'), 'to', format(product_end, '%d %b %Y'))
baseline_days <- seq(baseline_start, period_end, by = 'day')
baseline_weekdays <- baseline_days[as.POSIXlt(baseline_days)$wday %in% 1:5]
product_weeks <- sort(unique(week_start(seq(product_start, product_end, by = 'day'))))
teams <- sort(unique(pq$Team[pq$IsDeveloper]))

people <- pq |>
  distinct(PersonId, Organization, FunctionType, LevelDesignation, IsManager,
           Team, Role, Seniority, Tenure, IsDeveloper)
assert_key(people, 'PersonId')
# Population and window facts are derived from the data, never hard-coded, so
# regenerating the fixtures with different parameters cannot leave a stale
# literal behind. These assert internal consistency, not a specific size.
N_ROSTER <- n_distinct(people$PersonId)
N_DEVELOPERS <- sum(people$IsDeveloper)
developer_team_sizes <- as.integer(table(people$Team[people$IsDeveloper]))
stopifnot(N_ROSTER == n_distinct(pq$PersonId),
          N_DEVELOPERS >= MIN_GROUP_N, N_DEVELOPERS <= N_ROSTER,
          length(developer_team_sizes) > 1L,
          length(unique(developer_team_sizes)) == 1L,
          length(weeks) > BASELINE_WEEKS,
          all(diff(as.numeric(weeks)) == 7),
          all(format(weeks, '%w') == '0'),
          baseline_start >= min(product_weeks))

# The scheduled working-hours basis is configurable per tenant through metric
# rules, so read it from the data rather than assuming 40.
working_hours_basis <- unique(round(pq$Meeting_hours + pq$Scheduled_call_hours +
                                      pq$Available_to_focus_hours, 2))
stopifnot(length(working_hours_basis) == 1L)

pq_eligibility <- pq |>
  group_by(PersonId) |>
  summarise(PQ_eligible = TRUE,
            PQ_complete = n_distinct(Week) == length(weeks),
            .groups = 'drop')
person_history <- m365_raw |>
  distinct(PersonId, PeopleHistoricalId) |>
  bind_rows(github_credits_daily |> distinct(PersonId, PeopleHistoricalId)) |>
  distinct(PersonId, PeopleHistoricalId)
person_history_counts <- person_history |> count(PersonId)
stopifnot(all(person_history_counts$n == 1L))
m365_product_weeks <- derive_m365_coverage(pq)
# Historical internal name: GH_eligible denotes observed row presence only,
# never an independently established GitHub licence or provisioning status.
gh_eligibility <- gh_daily |>
  distinct(PersonId, Week) |>
  mutate(GH_eligible = TRUE)

expected_weekdays <- tibble(MetricDate = seq(product_start, product_end, by = 'day')) |>
  mutate(Week = week_start(MetricDate),
         weekday = as.POSIXlt(MetricDate)$wday) |>
  filter(weekday %in% 1:5) |>
  count(Week, name = 'Expected_weekdays')
gh_coverage <- gh_daily |>
  filter(as.POSIXlt(MetricDate)$wday %in% 1:5) |>
  count(PersonId, Week, name = 'GH_observed_days') |>
  left_join(expected_weekdays, by = 'Week', relationship = 'many-to-one') |>
  mutate(GH_complete = GH_observed_days == 5L & Expected_weekdays == 5L,
         GH_covered_days = GH_observed_days) |>
  select(PersonId, Week, GH_complete, GH_covered_days)
m365_daily <- m365_raw |>
  group_by(PersonId, PeopleHistoricalId, MetricDate, Week) |>
  summarise(M365_sessions = sum(`Session count`),
            M365_credits = sum(`Total Copilot Credits used`),
            M365_services = n_distinct(ServiceName),
            .groups = 'drop')
assert_key(m365_daily, c('PersonId', 'MetricDate'))
m365_weekly <- m365_daily |>
  group_by(PersonId, Week) |>
  summarise(M365_active_days = n_distinct(MetricDate[M365_sessions > 0 | M365_credits > 0]),
            M365_sessions = sum(M365_sessions),
            M365_credits = sum(M365_credits),
            .groups = 'drop')
developer_ids <- people$PersonId[coalesce(people$IsDeveloper, FALSE)]
m365_service_source <- m365_raw |>
  filter(MetricDate %in% baseline_weekdays, PersonId %in% developer_ids) |>
  mutate(ServiceName = missing_label(ServiceName))
m365_service_safe <- support_release_safe(
  m365_service_source, 'ServiceName', c('Session count', 'Total Copilot Credits used'),
  developer_ids)
m365_service_mix <- m365_service_source |>
  group_by(ServiceName) |>
  summarise(People = n_distinct(PersonId),
            Sessions = sum(`Session count`),
            Credits = sum(`Total Copilot Credits used`),
            .groups = 'drop') |>
  disclose_partition() |>
  arrange(desc(Credits)) |>
  mutate(Share = Credits / sum(Credits))
if (!m365_service_safe) m365_service_mix <- m365_service_mix[0, ]

github_credits_weekly <- github_credits_daily |>
  filter(as.POSIXlt(MetricDate)$wday %in% 1:5) |>
  group_by(PersonId, Week) |>
  summarise(GH_credit_days = n_distinct(MetricDate[as.POSIXlt(MetricDate)$wday %in% 1:5]),
            GH_credits = sum(`Total GitHub AI Credits used`), .groups = 'drop')
gh_weekly <- gh_daily |>
  filter(as.POSIXlt(MetricDate)$wday %in% 1:5) |>
  group_by(PersonId, Week) |>
  summarise(GH_observed_days = n_distinct(MetricDate),
            GH_active_days = n_distinct(MetricDate[`Code completions suggested` > 0 |
              `Code completions accepted` > 0 | `User-initiated chat requests` > 0 |
              `Agent adoption`]),
            # Days the sparse credit export is expected to carry: the credit
            # feed only records billable usage, so a day with suggestions but
            # no accepted completion or chat request legitimately has no row.
            GH_billable_days = n_distinct(MetricDate[`Code completions accepted` > 0 |
              `User-initiated chat requests` > 0]),
            GH_suggestions = sum(`Code completions suggested`),
            GH_accepted = sum(`Code completions accepted`),
            GH_chats = sum(`User-initiated chat requests`),
            GH_agent_days = sum(`Agent adoption`),
            .groups = 'drop') |>
  left_join(github_credits_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one')

coverage <- expand_grid(PersonId = people$PersonId, Week = weeks) |>
  left_join(pq_eligibility |> select(PersonId, PQ_eligible, PQ_complete),
            by = 'PersonId', relationship = 'many-to-one') |>
  left_join(gh_eligibility, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(gh_coverage, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(m365_product_weeks,
            by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  mutate(GH_covered_days = if_else(coalesce(GH_eligible, FALSE),
                                  GH_covered_days, NA_integer_))
assert_key(coverage, c('PersonId', 'Week'))

panel <- coverage |>
  left_join(people, by = 'PersonId', relationship = 'many-to-one') |>
  left_join(pq |> select(-Organization, -FunctionType, -LevelDesignation, -IsManager,
                         -Team, -Role, -Seniority, -Tenure, -IsDeveloper),
            by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(gh_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(m365_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  mutate(GH_activity_valid = coalesce(GH_eligible & GH_complete, FALSE),
         GH_billable_days = if_else(GH_activity_valid,
                                    coalesce(GH_billable_days, 0L), NA_integer_),
         GH_credit_days = if_else(GH_activity_valid,
                                  coalesce(GH_credit_days, 0L), NA_integer_),
         # Two different coverage questions, deliberately separated. The
         # activity export carries explicit weekday zero rows, so its five rows
         # establish that the week was observed whether or not the person used
         # anything. The credit export is SPARSE: it carries a row only where
         # billable usage occurred, so requiring five credit rows would discard
         # fully observed but partly inactive weeks. Credit coverage is instead
         # resolved when every observed billable day carries a credit row, and a
         # week with no billable day resolves to a measured zero. This is
         # coverage logic, not a privacy control: each flag gates only the
         # measures it actually governs.
         GH_credit_valid = coalesce(GH_activity_valid &
                                      GH_credit_days == GH_billable_days, FALSE),
         M365_valid = coalesce(M365_eligible & M365_complete, FALSE),
         across(c(GH_active_days, GH_suggestions, GH_accepted, GH_chats,
                  GH_agent_days),
                ~if_else(GH_activity_valid, .x, NA_real_)),
         GH_credits = if_else(GH_credit_valid, coalesce(GH_credits, 0), NA_real_),
         across(c(M365_active_days, M365_sessions, M365_credits),
                ~observed_or_missing(.x, M365_eligible, M365_complete)))
assert_key(panel, c('PersonId', 'Week'))

metric_labels <- c(Meeting_hours = 'Meetings',
                   Available_to_focus_hours = 'Available-to-focus hours',
                   Uninterrupted_hours = 'Uninterrupted time',
                   Interrupted_hours = 'Interrupted time',
                   Collaboration_hours = 'Collaboration hours',
                   After_hours_collaboration_hours = 'After-hours collaboration',
                   Recurring_meeting_hours = 'Recurring meetings',
                   Conflicting_meeting_hours = 'Conflicting meetings',
                   Meeting_hours_with_six_or_fewer_hours_of_advanced_notice = 'Short-notice meetings')
work_metrics <- names(metric_labels)
baseline_panel <- panel |>
  filter(IsDeveloper, PQ_eligible, PQ_complete, Week >= baseline_start)
baseline <- baseline_panel |>
  group_by(PersonId, Team, Role, Seniority, Tenure) |>
  summarise(Weeks_observed = n(),
            Weeks_above = sum(After_hours_collaboration_hours >= AFTER_HOURS_CONVENTION),
            across(all_of(work_metrics), mean),
            GH_all_valid = all(GH_activity_valid),
            GH_credits_all_valid = all(GH_credit_valid),
            M365_all_valid = all(M365_valid),
            GH_active_days = if (all(GH_activity_valid)) sum(GH_active_days) else NA_real_,
            M365_active_days = if (all(M365_valid)) sum(M365_active_days) else NA_real_,
            GH_suggestions = if (all(GH_activity_valid)) sum(GH_suggestions) else NA_real_,
            GH_accepted = if (all(GH_activity_valid)) sum(GH_accepted) else NA_real_,
            GH_chats = if (all(GH_activity_valid)) sum(GH_chats) else NA_real_,
            GH_agent_days = if (all(GH_activity_valid)) sum(GH_agent_days) else NA_real_,
            GH_credits = if (all(GH_credit_valid)) sum(GH_credits) else NA_real_,
            M365_sessions = if (all(M365_valid)) sum(M365_sessions) else NA_real_,
            M365_credits = if (all(M365_valid)) sum(M365_credits) else NA_real_,
            GH_eligible_all = all(GH_eligible),
            M365_eligible_all = all(M365_eligible),
            .groups = 'drop') |>
  mutate(Joint = case_when(
    is.na(GH_eligible_all) | is.na(M365_eligible_all) ~ 'Observation unresolved',
    !GH_eligible_all & !M365_eligible_all ~ 'Observation unresolved',
    !GH_eligible_all ~ 'Observation unresolved',
    !M365_eligible_all ~ 'GitHub observed; M365 not enabled all weeks',
    !GH_all_valid | !M365_all_valid ~ 'Observation unresolved',
    GH_active_days > 0 & M365_active_days > 0 ~ 'Both recorded',
    GH_active_days > 0 ~ 'GitHub only recorded',
    M365_active_days > 0 ~ 'M365 only recorded',
    TRUE ~ 'Neither recorded'))
joint_levels <- c('Both recorded', 'GitHub only recorded', 'M365 only recorded',
                  'Neither recorded', 'GitHub observed; M365 not enabled all weeks',
                  'Observation unresolved')
baseline$Joint <- factor(baseline$Joint, levels = joint_levels)
# Developers valid for BOTH product feeds. Empty by default: Microsoft 365
# completeness is not established by these exports, which is why the two-product
# comparison stays unavailable and the page-6 comparison uses gh_observed below.
matched <- baseline |> filter(GH_all_valid, M365_all_valid)

# The GitHub feed resolves its own observation from explicit weekday zero rows,
# so a recorded-use comparison is available on that population without inventing
# Microsoft 365 completeness evidence. The two-product comparison above stays
# unavailable by default; this one is the part the exports can actually support.
gh_use_levels <- c('GitHub use recorded', 'No GitHub use recorded',
                   'GitHub observation unresolved')
baseline <- baseline |>
  mutate(GH_use = factor(case_when(
    !GH_all_valid ~ 'GitHub observation unresolved',
    GH_active_days > 0 ~ 'GitHub use recorded',
    TRUE ~ 'No GitHub use recorded'), levels = gh_use_levels))
gh_observed <- baseline |> filter(GH_all_valid)

measure_values <- c(
  unlist(pq |> select(where(is.numeric), -Total_Copilot_enabled_days), use.names = FALSE),
  gh_daily$`Code completions accepted`,
  gh_daily$`Code completions suggested`,
  gh_daily$`User-initiated chat requests`,
  m365_raw$`Session count`,
  m365_raw$`Total Copilot Credits used`,
  github_credits_daily$`Total GitHub AI Credits used`,
  feature_daily$`Feature Usage Count`,
  language_feature_daily$`Usage count by language and feature`,
  language_model_daily$`Usage count by language and model`,
  model_feature_daily$`Usage count by model and feature`
)
stopifnot(nrow(coverage) == N_ROSTER * length(weeks),
          nrow(panel) == nrow(pq),
          nrow(baseline) == N_DEVELOPERS,
          nrow(baseline_panel) == N_DEVELOPERS * BASELINE_WEEKS,
          all(baseline$Weeks_observed == BASELINE_WEEKS),
          all(is.finite(measure_values)),
          all(measure_values >= 0),
          all(gh_daily$`Code completions accepted` <= gh_daily$`Code completions suggested`),
          all(panel$GH_active_days[panel$GH_activity_valid] <=
                panel$GH_covered_days[panel$GH_activity_valid]),
          all(panel$M365_active_days[panel$M365_valid] <= panel$M365_covered_days[panel$M365_valid]),
          all(is.na(panel$GH_active_days[!panel$GH_activity_valid])),
          all(is.na(panel$GH_credits[!panel$GH_credit_valid])),
          all(!panel$GH_credit_valid | !is.na(panel$GH_credits)),
          all(!panel$GH_credit_valid | panel$GH_activity_valid),
          all(is.na(panel$M365_active_days[!panel$M365_valid])),
          max(abs(pq$Meeting_hours + pq$Scheduled_call_hours +
                    pq$Available_to_focus_hours - working_hours_basis)) <= .0011,
          max(abs(pq$Uninterrupted_hours + pq$Interrupted_hours - pq$Available_to_focus_hours)) <= .0011,
          all(pq$Uninterrupted_hours <= pq$Available_to_focus_hours + .0011),
          all(pq$Open_1_hour_block <= pq$Available_to_focus_hours + .0011),
          all(pq$Collaboration_hours + .0011 >= pq$Meeting_hours + pq$Scheduled_call_hours),
          all(pq$Recurring_meeting_hours <= pq$Meeting_hours + .0011),
          all(pq$Conflicting_meeting_hours <= pq$Meeting_hours + .0011),
          all(pq$Meeting_hours_with_six_or_fewer_hours_of_advanced_notice <= pq$Meeting_hours + .0011),
          all(pq$Open_1_hour_block == floor(pq$Open_1_hour_block)))

# The generator checks its synthetic allocation invariant. Do not enforce that
# invariant here: matching headers do not establish real-export metric equality
# or model attribution for completion counts.

baseline_feature_daily <- feature_daily |> filter(MetricDate %in% baseline_weekdays, PersonId %in% developer_ids)
baseline_language_feature_daily <- language_feature_daily |> filter(MetricDate %in% baseline_weekdays, PersonId %in% developer_ids)
baseline_language_model_daily <- language_model_daily |> filter(MetricDate %in% baseline_weekdays, PersonId %in% developer_ids)
baseline_model_feature_daily <- model_feature_daily |> filter(MetricDate %in% baseline_weekdays, PersonId %in% developer_ids)
agent_features <- c('chat_panel_agent_mode', 'agent_edit', 'copilot_cli', 'copilot_app')
feature_mix <- baseline_feature_daily |>
  group_by(Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Feature Usage Count`), .groups = 'drop') |>
  filter(Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count),
         Feature_type = if_else(Feature %in% agent_features, 'Agent-type feature', 'Other feature'))
model_mix <- baseline_model_feature_daily |>
  group_by(Model) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by model and feature`), .groups = 'drop') |>
  filter(Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count))
language_mix <- baseline_language_model_daily |>
  group_by(Language) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and model`), .groups = 'drop') |>
  filter(Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count))
model_feature_mix <- baseline_model_feature_daily |>
  group_by(Model, Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by model and feature`), .groups = 'drop') |>
  filter(Count > 0) |>
  group_by(Model) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
language_model_mix <- baseline_language_model_daily |>
  group_by(Language, Model) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and model`), .groups = 'drop') |>
  filter(Count > 0) |>
  group_by(Language) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
language_feature_mix <- baseline_language_feature_daily |>
  group_by(Language, Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and feature`), .groups = 'drop') |>
  filter(Count > 0) |>
  group_by(Language) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
breakdown_safe <- all(vapply(list(feature_mix, model_mix, language_mix,
  model_feature_mix, language_model_mix, language_feature_mix),
  function(x) all(x$People >= MIN_GROUP_N), logical(1))) &&
  support_release_safe(baseline_feature_daily, 'Feature', 'Feature Usage Count', developer_ids) &&
  support_release_safe(baseline_model_feature_daily, c('Model', 'Feature'),
                       'Usage count by model and feature', developer_ids) &&
  support_release_safe(baseline_language_model_daily, c('Language', 'Model'),
                       'Usage count by language and model', developer_ids) &&
  support_release_safe(baseline_language_feature_daily, c('Language', 'Feature'),
                       'Usage count by language and feature', developer_ids)
if (!breakdown_safe) {
  feature_mix <- feature_mix[0, ]
  model_mix <- model_mix[0, ]
  language_mix <- language_mix[0, ]
  model_feature_mix <- model_feature_mix[0, ]
  language_model_mix <- language_model_mix[0, ]
  language_feature_mix <- language_feature_mix[0, ]
}

# Presentation helpers use group-level aggregates only. No point per person.
wrap <- function(x, width = 100) stringr::str_wrap(x, width = width)
f1 <- function(x) ifelse(!is.finite(x), 'N/A — unavailable or withheld',
                         formatC(x, format = 'f', digits = 1, big.mark = ',', decimal.mark = '.'))
pct <- function(x) ifelse(!is.finite(x), 'N/A — unavailable or withheld',
                          scales::percent(x, accuracy = .1, decimal.mark = '.', big.mark = ','))
num <- function(x) ifelse(is.na(x), 'N/A — unavailable or withheld',
                         format(x, big.mark = ',', scientific = FALSE, trim = TRUE))
# A rate over an empty valid population is undefined, not withheld. Saying so
# is more honest than one suppression marker doing both jobs.
pct_or_reason <- function(rate, denominator) ifelse(
  !is.na(denominator) & denominator == 0, 'Undefined — no valid denominator',
  pct(rate))
leader_label <- function(labels, values) {
  keep <- is.finite(values)
  if (!any(keep)) return('Unavailable')
  as.character(labels[keep][which.max(values[keep])])
}
PAL <- c('#2563eb', '#0f766e', '#7c3aed', '#db2777', '#f97316', '#64748b', '#0891b2', '#ca8a04')
joint_pal <- setNames(PAL, joint_levels)
theme_report <- function() theme_minimal(base_size = 12) + theme(
  panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
  plot.title.position = 'plot', plot.title = element_text(face = 'bold', colour = '#20243b', size = 15),
  plot.subtitle = element_text(colour = '#475569', size = 11, margin = margin(b = 12)),
  plot.caption = element_text(hjust = 0, colour = '#475569', size = 9, margin = margin(t = 14)),
  plot.margin = margin(14, 24, 14, 16), legend.position = 'bottom', legend.title = element_blank(),
  strip.text = element_text(face = 'bold', size = 11), axis.title = element_text(size = 11),
  legend.text = element_text(size = 10))
chart_labs <- function(title, subtitle, caption, x = NULL, y = NULL) labs(
  title = wrap(title, 88), subtitle = wrap(subtitle, 110), caption = wrap(caption, 115), x = x, y = y)
base_caption <- function(n = nrow(baseline), unit = 'hours/person/week') paste0(
  period_label, ' | ', num(n), ' distinct developers | ', BASELINE_WEEKS,
  ' complete Person Query weeks | ', unit, '. Dates use UTC convention.')

interval_data <- function(data, metrics, group = 'Team') {
  sizes <- data |> count(.data[[group]], name = 'People', .drop = TRUE)
  if (!nrow(data) || !nrow(disclose_partition(sizes))) {
    return(tibble(Group = character(), Metric = character(),
                  p25 = double(), p50 = double(), p75 = double()))
  }
  bind_rows(lapply(metrics, function(m) {
    source <- data |> mutate(MetricDate = baseline_start)
    result <- vivainsights::create_boxplot(source, metric = m, hrvar = group,
                                           mingroup = MIN_GROUP_N, return = 'table')
    result |> transmute(Group = as.character(group), Metric = unname(metric_labels[m]),
                        p25, p50, p75)
  }))
}
interval_plot <- function(data, metrics, group = 'Team', title, subtitle, caption = base_caption(nrow(data))) {
  d <- interval_data(data, metrics, group)
  if (!nrow(d)) return(unavailable_plot(title))
  first_metric <- unname(metric_labels[metrics[1]])
  group_order <- if (group == 'Joint') joint_levels[1:4]
    else if (group == 'GH_use') gh_use_levels[1:2]
    else d |>
    filter(Metric == first_metric) |> arrange(desc(p50)) |> pull(Group)
  d$Group <- factor(d$Group, levels = rev(group_order))
  d$Metric <- factor(d$Metric, levels = unname(metric_labels[metrics]))
  ggplot(d, aes(x = p50, y = Group)) +
    geom_linerange(aes(xmin = p25, xmax = p75), linewidth = 2.4, colour = '#bfdbfe') +
    geom_point(size = 2.6, colour = '#2563eb') +
    facet_wrap(~Metric, nrow = 1, scales = 'free_x', labeller = label_wrap_gen(22)) +
    scale_x_continuous(expand = expansion(mult = c(.12, .14))) + theme_report() +
    chart_labs(title, subtitle, caption, x = 'Hours per person per week')
}
composition <- function(data, group, category) {
  counts <- data |> count(Group = missing_label(.data[[group]]),
                         Category = missing_label(.data[[category]]), name = 'People', .drop = TRUE)
  counts <- disclose_partition(counts)
  counts |> group_by(Group) |> mutate(Share = People / sum(People)) |> ungroup()
}
unavailable_plot <- function(title = NULL) {
  ggplot() + annotate('text', x = 0, y = 0,
                      label = 'Unavailable: independent coverage missing or privacy threshold not met.') +
    theme_void() + labs(title = title)
}
stack_plot <- function(d, title, subtitle, caption, palette = PAL, sort_category = NULL, group_order = NULL) {
  if (!nrow(d)) return(unavailable_plot(title))
  d <- d |> mutate(Group = as.character(Group), Category = as.character(Category))
  d$Category <- factor(d$Category, levels = unique(d$Category))
  target <- if (is.null(sort_category)) as.character(d$Category)[1] else sort_category
  if (is.null(group_order)) {
    group_order <- d |>
      filter(as.character(Category) == target) |>
      arrange(desc(Share)) |>
      pull(Group) |> as.character()
    group_order <- c(group_order, setdiff(unique(d$Group), group_order))
    subtitle <- paste0(subtitle, ' Groups are ranked by the share of ', target, '.')
  }
  d$Group <- factor(d$Group, levels = rev(group_order))
  colours <- setNames(rep(palette, length.out = nlevels(d$Category)), levels(d$Category))
  dark <- c('#2563eb', '#0f766e', '#7c3aed', '#db2777', '#64748b', '#0891b2')
  d$TextColour <- ifelse(colours[as.character(d$Category)] %in% dark, 'white', '#182033')
  ggplot(d, aes(x = Share, y = Group, fill = Category, group = Category)) + geom_col(width = .6) +
    geom_text(aes(label = ifelse(Share >= .06, pct(Share), ''), colour = TextColour),
              position = position_stack(vjust = .5), size = 3.2) +
    scale_colour_identity() + scale_fill_manual(values = colours, labels = function(x) wrap(x, 25)) +
    scale_x_continuous(labels = label_percent(), expand = expansion(mult = c(0, .01))) +
    guides(fill = guide_legend(nrow = 2, byrow = TRUE)) + theme_report() +
    chart_labs(title, subtitle, caption, x = 'Share of developers')
}
html_table <- function(data, caption = NULL, escape = TRUE) {
  if (is.null(data) || !nrow(data)) {
    cat('<p>Unavailable: independent coverage missing or privacy threshold not met.</p>')
    return(invisible(NULL))
  }
  cat('<div class="table-scroll">')
  print(knitr::kable(data, format = 'html', row.names = FALSE, escape = escape, caption = caption))
  cat('</div>')
}
composition_table <- function(d) d |> transmute(Group, Category, `Developers (count)` = People, `Share (%)` = pct(Share))
metric_count <- function(x, unit) {
  display <- if (x >= 1e6) paste0(f1(x / 1e6), 'M') else if (x >= 1e3) paste0(f1(x / 1e3), 'K') else num(x)
  sprintf('<span title="%s %s">%s %s</span>', num(x), unit, display, unit)
}
card <- function(label, value, detail = '') cat(sprintf(
  '<div class="kpi"><span>%s</span><strong>%s</strong><small>%s</small></div>', label, value, detail))

joint_counts <- baseline |> count(Joint, name = 'People', .drop = TRUE) |>
  mutate(Share = People / nrow(baseline))
joint_counts_public <- joint_counts |>
  disclose_partition() |>
  mutate(Display = Joint) |>
  arrange(Display)
team_context <- composition(baseline, 'Team', 'Role')
role_context <- composition(gh_observed, 'GH_use', 'Role')
team_joint_context <- composition(gh_observed, 'GH_use', 'Team')
composition_counts <- bind_rows(lapply(c('Role','Seniority','Tenure'), function(a)
  baseline |> count(Category=.data[[a]], name='People') |> mutate(Attribute=a)))
# One shared gate for every linked HR margin and cross-tab in this report. A
# family is published only when every cell of every published margin meets the
# floor, so no published margin can be differenced against another to expose a
# sub-floor group. Gating instead on the full Team x Role x Seniority x Tenure
# intersection withheld the whole family unconditionally: at this population
# size that intersection can never reach ten people, and it is never published.
hr_published <- list(
  baseline |> count(Team, Role, name = 'People'),
  baseline |> count(Role, name = 'People'),
  baseline |> count(Seniority, name = 'People'),
  baseline |> count(Tenure, name = 'People'),
  gh_observed |> count(GH_use, Role, name = 'People', .drop = TRUE),
  gh_observed |> count(GH_use, Team, name = 'People', .drop = TRUE))
hr_release_safe <- all(vapply(hr_published,
  function(x) nrow(x) > 0L && nrow(disclose_partition(x)) > 0L, logical(1)))
if (!hr_release_safe) {
  team_context <- team_context[0, ]
  role_context <- role_context[0, ]
  team_joint_context <- team_joint_context[0, ]
  composition_counts <- composition_counts[0, ]
}
work_summary <- bind_rows(lapply(c('Collaboration_hours', 'Meeting_hours', 'Available_to_focus_hours',
                                   'Uninterrupted_hours', 'Interrupted_hours',
                                   'After_hours_collaboration_hours'), function(m) tibble(
  Metric = metric_labels[[m]], `25th percentile` = f1(quantile(baseline[[m]], .25)),
  Median = f1(median(baseline[[m]])), `75th percentile` = f1(quantile(baseline[[m]], .75)))))

credit_release_safe <- setNames(vapply(c('GH', 'M365'), function(product) {
  flag <- if (product == 'GH') 'GH_credits_all_valid' else 'M365_all_valid'
  valid <- baseline |> filter(.data[[flag]])
  support_release_safe(valid, 'Team', paste0(product, '_credits'), baseline$PersonId)
}, logical(1)), c('GH', 'M365'))
product_summary <- bind_rows(lapply(c('GH', 'M365'), function(product) {
  gh <- product == 'GH'
  valid <- baseline |> filter(if (gh) GH_all_valid else M365_all_valid)
  active <- valid |> filter(if (gh) GH_active_days > 0 else M365_active_days > 0)
  volume <- if (gh) 'GH_credits' else 'M365_credits'
  # Credit coverage is resolved separately from activity coverage, so the
  # credit population is its own denominator and gets its own release check.
  credited <- baseline |> filter(if (gh) GH_credits_all_valid else M365_all_valid)
  credit_pop_safe <- disclosable_split(nrow(baseline), nrow(credited))
  split_safe <- disclosable_split(nrow(baseline), nrow(valid)) &&
    split_or_empty(nrow(valid), nrow(active))
  credit_active <- credited |> filter(.data[[volume]] > 0)
  top_n <- ceiling(.1 * nrow(credit_active))
  top_ids <- order(credit_active[[volume]], decreasing = TRUE)[seq_len(top_n)]
  top_safe <- credit_pop_safe && credit_release_safe[[product]] &&
    top_n >= MIN_GROUP_N && nrow(credit_active) - top_n >= MIN_GROUP_N &&
    sum(credit_active[[volume]][top_ids] > 0) >= MIN_GROUP_N &&
    sum(credit_active[[volume]][-top_ids] > 0) >= MIN_GROUP_N
  share <- if (top_safe) safe_div(
    sum(sort(credit_active[[volume]], decreasing = TRUE)[seq_len(top_n)]),
    sum(credit_active[[volume]])) else NA_real_
  credits_mean <- if (credit_pop_safe && credit_release_safe[[product]] && nrow(credited))
    mean(credited[[volume]]) / BASELINE_WEEKS else NA_real_
  tibble(Product = if (gh) 'GitHub Copilot' else 'Microsoft 365 Copilot',
         `Valid developers (count)` = num(if (split_safe) nrow(valid) else NA_integer_),
         `Recorded active (count)` = num(if (split_safe) nrow(active) else NA_integer_),
         `Recorded active (%)` = if (split_safe) pct_or_reason(safe_div(nrow(active), nrow(valid)), nrow(valid)) else pct(NA_real_),
         `Active days / developer / week (mean)` = f1(if (split_safe) mean(if (gh) valid$GH_active_days else valid$M365_active_days) / BASELINE_WEEKS else NA_real_),
         `Credit-resolved developers (count)` = num(if (credit_pop_safe) nrow(credited) else NA_integer_),
         `Credits / developer / week (mean)` = f1(credits_mean),
         `Session or request unit` = if (gh) 'GitHub credits' else 'M365 credits',
         `Top 10% active credit share (%)` = pct(share),
         `Top group (count)` = num(if (top_safe) top_n else NA_integer_))
}))
team_league <- baseline |>
  group_by(Team) |>
  summarise(Developers = n(),
            Collaboration_hours = median(Collaboration_hours),
            Meeting_hours = median(Meeting_hours),
            Available_to_focus_hours = median(Available_to_focus_hours),
            Uninterrupted_hours = median(Uninterrupted_hours),
            Interrupted_hours = median(Interrupted_hours),
            After_hours_collaboration_hours = median(After_hours_collaboration_hours),
            GH_valid_n = sum(GH_all_valid),
            GH_active_n = sum(GH_all_valid & coalesce(GH_active_days, 0) > 0),
            GH_active_share = safe_div(GH_active_n, GH_valid_n),
            GH_credit_n = sum(GH_credits_all_valid),
            GH_intensity = mean(GH_credits[GH_credits_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
            M365_valid_n = sum(M365_all_valid),
            M365_active_n = sum(M365_all_valid & coalesce(M365_active_days, 0) > 0),
            M365_active_share = safe_div(M365_active_n, M365_valid_n),
            M365_credit_n = sum(M365_all_valid),
            M365_intensity = mean(M365_credits[M365_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
            .groups = 'drop') |>
  filter(Developers >= MIN_GROUP_N) |>
  mutate(GH_safe = disclosable_split(Developers, GH_valid_n) &
                    split_or_empty(GH_valid_n, GH_active_n),
         GH_credit_safe = GH_safe & disclosable_split(Developers, GH_credit_n),
         M365_safe = disclosable_split(Developers, M365_valid_n) &
                      split_or_empty(M365_valid_n, M365_active_n),
         M365_credit_safe = M365_safe & disclosable_split(Developers, M365_credit_n),
         across(c(GH_valid_n, GH_active_n, GH_active_share),
                ~if_else(GH_safe, .x, NA_real_)),
         GH_intensity = if_else(GH_credit_safe, GH_intensity, NA_real_),
         across(c(M365_valid_n, M365_active_n, M365_active_share),
                ~if_else(M365_safe, .x, NA_real_)),
         M365_intensity = if_else(M365_credit_safe, M365_intensity, NA_real_)) |>
  arrange(desc(Collaboration_hours)) |>
  mutate(Rank = row_number(), .before = Team)
for (product in c('GH', 'M365')) {
  if (!credit_release_safe[[product]] ||
      !all(team_league[[paste0(product, '_credit_safe')]]))
    team_league[[paste0(product, '_intensity')]] <- NA_real_
  if (!all(team_league[[paste0(product, '_safe')]])) {
    for (measure in c('_valid_n', '_active_n', '_active_share', '_intensity')) {
      team_league[[paste0(product, measure)]] <- NA_real_
    }
  }
}

league_metric_levels <- c('Collaboration hours', 'Meeting hours', 'Available-to-focus hours',
  'Uninterrupted hours', 'Interrupted time', 'After-hours collaboration',
  'GitHub recorded-active share (%)', 'GitHub credits',
  'Microsoft 365 recorded-active share (%)', 'Microsoft 365 credits')
team_league_plot <- bind_rows(
  team_league |> transmute(Team, Metric = 'Collaboration hours', Value = Collaboration_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'Meeting hours', Value = Meeting_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'Available-to-focus hours', Value = Available_to_focus_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'Uninterrupted hours', Value = Uninterrupted_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'Interrupted time', Value = Interrupted_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'After-hours collaboration', Value = After_hours_collaboration_hours, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'GitHub recorded-active share (%)', Value = GH_active_share * 100, Label = pct(GH_active_share)),
  team_league |> transmute(Team, Metric = 'GitHub credits', Value = GH_intensity, Label = f1(Value)),
  team_league |> transmute(Team, Metric = 'Microsoft 365 recorded-active share (%)', Value = M365_active_share * 100, Label = pct(M365_active_share)),
  team_league |> transmute(Team, Metric = 'Microsoft 365 credits', Value = M365_intensity, Label = f1(Value))) |>
  mutate(Metric = factor(Metric, levels = league_metric_levels))

trend <- panel |> filter(IsDeveloper, PQ_eligible, PQ_complete) |>
  group_by(Week) |> summarise(PQ_n = n_distinct(PersonId),
    across(all_of(c('Collaboration_hours', 'Meeting_hours', 'Available_to_focus_hours',
                    'Uninterrupted_hours', 'Interrupted_hours',
                    'After_hours_collaboration_hours')), median),
    GH_n = sum(GH_activity_valid), GH_active = sum(GH_active_days > 0, na.rm = TRUE),
    M365_n = sum(M365_valid), M365_active = sum(M365_active_days > 0, na.rm = TRUE),
    Joint_n = sum(GH_activity_valid & M365_valid), .groups = 'drop') |>
  mutate(GH_safe = disclosable_split(PQ_n, GH_n) & split_or_empty(GH_n, GH_active),
         M365_safe = disclosable_split(PQ_n, M365_n) & split_or_empty(M365_n, M365_active),
         GH_rate = if_else(GH_safe, safe_div(GH_active, GH_n), NA_real_),
         M365_rate = if_else(M365_safe, safe_div(M365_active, M365_n), NA_real_),
         across(c(GH_n, GH_active), ~if_else(GH_safe, .x, NA_real_)),
         across(c(M365_n, M365_active), ~if_else(M365_safe, .x, NA_real_)),
         Joint_n = if_else(GH_safe & M365_safe & disclosable_split(PQ_n, Joint_n),
                           Joint_n, NA_real_))
stopifnot(all(trend$PQ_n == N_DEVELOPERS), all(trend$GH_rate >= 0 & trend$GH_rate <= 1, na.rm = TRUE),
          all(trend$M365_rate >= 0 & trend$M365_rate <= 1, na.rm = TRUE))

source_contracts <- tibble(
  `Relative file under _data` = c(
    'person-query/PersonQuery.csv',
    'consumption-query/PeopleMetaData.csv',
    'consumption-query/PersonM365CreditsMetrics.csv',
    'consumption-query/PersonGitHubCreditsMetrics.csv',
    'github-query/PersonGitHubActivityMetrics.csv',
    'github-query/GitHubActivityBreakdownByFeatureMetrics.csv',
    'github-query/GitHubActivityBreakdownByLanguageFeatureMetrics.csv',
    'github-query/GitHubActivityBreakdownByLanguageModelMetrics.csv',
    'github-query/GitHubActivityBreakdownByModelFeatureMetrics.csv'),
  Grain = c('Person x week', 'Person', 'Person x service x day', 'Person x day',
            'Person x day', 'Person x day x feature',
            'Person x day x language x feature',
            'Person x day x language x model',
            'Person x day x model x feature'),
  Contract = c(
    paste('Weekly panel spanning', length(weeks), 'weeks. Person Query anchors the roster.'),
    'Consumption metadata keyed by PeopleHistoricalId; static licence context only, not period eligibility.',
    'Microsoft 365 Copilot credits and session counts; service rows are aggregated before daily or weekly use.',
    'GitHub AI credits by person and day.',
    'Synthetic GitHub activity includes explicit inactive weekday zeros; absent rows leave provisioning unknown.',
    'Feature usage count. Synthetic fixture allocation is not a verified real metric equality.',
    'Language x feature usage; shared allocation is synthetic only.',
    'Language x model usage; model attribution, including completions, is illustrative.',
    'Model x feature usage; model attribution, including completions, is illustrative.'))
coverage_derivation <- tibble(
  Question = c('Person Query roster and weekly completeness',
               'Microsoft 365 Copilot eligibility',
               'GitHub Copilot observation',
               'GitHub weekly activity coverage',
               'GitHub weekly credit coverage',
               'Microsoft 365 weekly observation',
               'Recorded product use'),
  `Signal used` = c('PersonQuery.csv has one row per PersonId x week in this sample.',
                    'Person Query Total_Copilot_enabled_days in the relevant week; static metadata never overrides zero enabled days.',
                    'An activity row establishes observation for that week, not licence or provisioning status. Absence is unknown.',
                    'Five explicit weekday activity rows, including measured zeros; a weekday-only reporting convention, not an ingestion certificate. No absent GitHub values are zero-filled.',
                    'The credit export is sparse and carries a row only for billable usage, so coverage is resolved when every observed billable day has a credit row. A week with no billable day resolves to a measured zero. Credit coverage never invalidates observed activity.',
                    'Unknown by default. Independent completeness evidence is required; licence metadata and a calendar are insufficient.',
                    'Positive observations are not proof of complete collection. M365 zero filling requires independently evidenced completeness and positive weekly enabled days.'))
validation_summary <- tibble(
  Check = c('Population', 'Teams', 'Person Query developer-weeks', 'Baseline developer-weeks',
            'Focus-time identities', 'Source keys and joins', 'Non-negative metrics and acceptance bounds',
            'GitHub synthetic allocation invariant', 'Eligibility and observation derivation',
            'Activity and credit coverage separated', 'Joint reconciliation', 'Privacy floor'),
  Result = c(paste(num(nrow(people)), 'people;', num(sum(people$IsDeveloper)), 'developers'),
             'Team metrics require the privacy floor; complementary product cells are withheld together.',
             num(nrow(pq |> filter(IsDeveloper))), num(nrow(baseline_panel)),
             'Available-to-focus hours = working-hours basis minus meetings and scheduled calls; uninterrupted + interrupted = available-to-focus',
             'Unique keys; no join amplification. PeopleHistoricalId is read from the activity crosswalk, never reconstructed from PersonId.',
             'Passed',
             'Checked only by the synthetic generator; not enforced as a real-export contract. Model attribution is illustrative, including completions.',
             'Weekly enabled days govern M365 eligibility; independent M365 completeness unavailable by default. No coverage file invented.',
             paste(num(sum(baseline$GH_all_valid)), 'developers have complete GitHub activity weeks;',
                   num(sum(baseline$GH_credits_all_valid)),
                   'also have resolved credit coverage. Sparse credit rows never void observed activity.'),
             'Internal joint counts reconcile to the roster; public partitions are withheld if any positive cell is below the privacy floor.',
             paste('At least', MIN_GROUP_N, 'distinct people in every published group')))
