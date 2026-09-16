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
  tolower(as.character(x)) %in% c('true', 'yes', '1')
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

pq <- read_export('person-query', 'PersonQuery.csv') |>
  mutate(MetricDate = as.Date(MetricDate),
         Week = MetricDate,
         IsDeveloper = as_bool(IsDeveloper),
         IsManager = as_bool(IsManager))
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
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate))
language_feature_daily <- read_export('github-query', 'GitHubActivityBreakdownByLanguageFeatureMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate))
language_model_daily <- read_export('github-query', 'GitHubActivityBreakdownByLanguageModelMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate))
model_feature_daily <- read_export('github-query', 'GitHubActivityBreakdownByModelFeatureMetrics.csv') |>
  mutate(MetricDate = as.Date(MetricDate), Week = week_start(MetricDate))

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
          N_DEVELOPERS > 0, N_DEVELOPERS <= N_ROSTER,
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
            M365_eligible_pq = any(Total_Copilot_enabled_days > 0, na.rm = TRUE),
            .groups = 'drop')
meta_eligibility <- people_meta |>
  transmute(PeopleHistoricalId,
            M365_eligible_meta = IsCopilotLicensed)
person_history <- m365_raw |>
  distinct(PersonId, PeopleHistoricalId) |>
  bind_rows(github_credits_daily |> distinct(PersonId, PeopleHistoricalId)) |>
  distinct(PersonId, PeopleHistoricalId)
person_history_counts <- person_history |> count(PersonId)
stopifnot(all(person_history_counts$n == 1L))
m365_eligibility <- people |>
  left_join(person_history, by = 'PersonId', relationship = 'one-to-one') |>
  left_join(meta_eligibility, by = 'PeopleHistoricalId', relationship = 'many-to-one') |>
  left_join(pq_eligibility, by = 'PersonId', relationship = 'one-to-one') |>
  transmute(PersonId,
            M365_eligible = coalesce(M365_eligible_meta, M365_eligible_pq, FALSE))
gh_eligibility <- gh_daily |>
  distinct(PersonId) |>
  mutate(GH_eligible = TRUE)

expected_weekdays <- tibble(MetricDate = seq(product_start, product_end, by = 'day')) |>
  mutate(Week = week_start(MetricDate),
         weekday = as.POSIXlt(MetricDate)$wday) |>
  filter(weekday %in% 1:5) |>
  count(Week, name = 'Expected_weekdays')
gh_coverage <- gh_daily |>
  count(PersonId, Week, name = 'GH_observed_days') |>
  left_join(expected_weekdays, by = 'Week', relationship = 'many-to-one') |>
  mutate(GH_complete = GH_observed_days == Expected_weekdays,
         GH_covered_days = GH_observed_days) |>
  select(PersonId, Week, GH_complete, GH_covered_days)
m365_product_weeks <- expand_grid(PersonId = m365_eligibility$PersonId, Week = product_weeks) |>
  left_join(m365_eligibility, by = 'PersonId', relationship = 'many-to-one') |>
  left_join(expected_weekdays, by = 'Week', relationship = 'many-to-one') |>
  mutate(M365_complete = M365_eligible & !is.na(Expected_weekdays),
         M365_covered_days = if_else(M365_complete, Expected_weekdays, NA_integer_))

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
m365_service_mix <- m365_raw |>
  filter(MetricDate %in% baseline_weekdays) |>
  group_by(ServiceName) |>
  summarise(People = n_distinct(PersonId),
            Sessions = sum(`Session count`),
            Credits = sum(`Total Copilot Credits used`),
            .groups = 'drop') |>
  filter(People >= MIN_GROUP_N) |>
  arrange(desc(Credits)) |>
  mutate(Share = Credits / sum(Credits))

github_credits_weekly <- github_credits_daily |>
  group_by(PersonId, Week) |>
  summarise(GH_credits = sum(`Total GitHub AI Credits used`), .groups = 'drop')
gh_weekly <- gh_daily |>
  group_by(PersonId, Week) |>
  summarise(GH_observed_days = n_distinct(MetricDate),
            GH_active_days = n_distinct(MetricDate[`Code completions suggested` > 0 |
              `Code completions accepted` > 0 | `User-initiated chat requests` > 0 |
              `Agent adoption`]),
            GH_suggestions = sum(`Code completions suggested`),
            GH_accepted = sum(`Code completions accepted`),
            GH_chats = sum(`User-initiated chat requests`),
            GH_agent_days = sum(`Agent adoption`),
            .groups = 'drop') |>
  left_join(github_credits_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one')

coverage <- expand_grid(PersonId = people$PersonId, Week = weeks) |>
  left_join(pq_eligibility |> select(PersonId, PQ_eligible, PQ_complete),
            by = 'PersonId', relationship = 'many-to-one') |>
  left_join(gh_eligibility, by = 'PersonId', relationship = 'many-to-one') |>
  left_join(m365_eligibility, by = 'PersonId', relationship = 'many-to-one') |>
  left_join(gh_coverage, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(m365_product_weeks |> select(PersonId, Week, M365_complete, M365_covered_days),
            by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  mutate(GH_eligible = coalesce(GH_eligible, FALSE),
         M365_eligible = coalesce(M365_eligible, FALSE),
         GH_complete = coalesce(GH_complete, FALSE),
         GH_covered_days = if_else(GH_eligible & !is.na(GH_covered_days), GH_covered_days, NA_integer_),
         M365_complete = coalesce(M365_complete, FALSE))
assert_key(coverage, c('PersonId', 'Week'))

panel <- coverage |>
  left_join(people, by = 'PersonId', relationship = 'many-to-one') |>
  left_join(pq |> select(-Organization, -FunctionType, -LevelDesignation, -IsManager,
                         -Team, -Role, -Seniority, -Tenure, -IsDeveloper),
            by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(gh_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  left_join(m365_weekly, by = c('PersonId', 'Week'), relationship = 'one-to-one') |>
  mutate(GH_valid = GH_eligible & GH_complete,
         M365_valid = M365_eligible & M365_complete,
         across(c(GH_active_days, GH_suggestions, GH_accepted, GH_chats,
                  GH_agent_days, GH_credits),
                ~if_else(GH_valid, coalesce(.x, 0), NA_real_)),
         across(c(M365_active_days, M365_sessions, M365_credits),
                ~if_else(M365_valid, coalesce(.x, 0), NA_real_)))
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
            GH_all_valid = all(GH_valid),
            M365_all_valid = all(M365_valid),
            GH_active_days = if (all(GH_valid)) sum(GH_active_days) else NA_real_,
            M365_active_days = if (all(M365_valid)) sum(M365_active_days) else NA_real_,
            GH_suggestions = if (all(GH_valid)) sum(GH_suggestions) else NA_real_,
            GH_accepted = if (all(GH_valid)) sum(GH_accepted) else NA_real_,
            GH_chats = if (all(GH_valid)) sum(GH_chats) else NA_real_,
            GH_agent_days = if (all(GH_valid)) sum(GH_agent_days) else NA_real_,
            GH_credits = if (all(GH_valid)) sum(GH_credits) else NA_real_,
            M365_sessions = if (all(M365_valid)) sum(M365_sessions) else NA_real_,
            M365_credits = if (all(M365_valid)) sum(M365_credits) else NA_real_,
            GH_eligible_all = all(GH_eligible),
            M365_eligible_all = all(M365_eligible),
            .groups = 'drop') |>
  mutate(Joint = case_when(
    !GH_eligible_all & !M365_eligible_all ~ 'Neither provisioned/licensed',
    !GH_eligible_all ~ 'M365 only eligible',
    !M365_eligible_all ~ 'GitHub only eligible',
    !GH_all_valid | !M365_all_valid ~ 'Observation unresolved',
    GH_active_days > 0 & M365_active_days > 0 ~ 'Both recorded',
    GH_active_days > 0 ~ 'GitHub only recorded',
    M365_active_days > 0 ~ 'M365 only recorded',
    TRUE ~ 'Neither recorded'))
joint_levels <- c('Both recorded', 'GitHub only recorded', 'M365 only recorded',
                  'Neither recorded', 'GitHub only eligible', 'M365 only eligible',
                  'Neither provisioned/licensed', 'Observation unresolved')
baseline$Joint <- factor(baseline$Joint, levels = joint_levels)
matched <- baseline |> filter(GH_all_valid, M365_all_valid)

measure_values <- c(
  unlist(pq |> select(where(is.numeric)), use.names = FALSE),
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
          all(panel$GH_active_days[panel$GH_valid] <= panel$GH_covered_days[panel$GH_valid]),
          all(panel$M365_active_days[panel$M365_valid] <= panel$M365_covered_days[panel$M365_valid]),
          all(is.na(panel$GH_active_days[!panel$GH_valid])),
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

activity_totals <- gh_daily |>
  transmute(PersonId, MetricDate,
            Expected = `Code completions accepted` + `User-initiated chat requests`)
feature_totals <- feature_daily |>
  group_by(PersonId, MetricDate) |>
  summarise(FeatureTotal = sum(`Feature Usage Count`), .groups = 'drop')
language_feature_totals <- language_feature_daily |>
  group_by(PersonId, MetricDate) |>
  summarise(LanguageFeatureTotal = sum(`Usage count by language and feature`), .groups = 'drop')
language_model_totals <- language_model_daily |>
  group_by(PersonId, MetricDate) |>
  summarise(LanguageModelTotal = sum(`Usage count by language and model`), .groups = 'drop')
model_feature_totals <- model_feature_daily |>
  group_by(PersonId, MetricDate) |>
  summarise(ModelFeatureTotal = sum(`Usage count by model and feature`), .groups = 'drop')
breakdown_check <- activity_totals |>
  left_join(feature_totals, by = c('PersonId', 'MetricDate'), relationship = 'one-to-one') |>
  left_join(language_feature_totals, by = c('PersonId', 'MetricDate'), relationship = 'one-to-one') |>
  left_join(language_model_totals, by = c('PersonId', 'MetricDate'), relationship = 'one-to-one') |>
  left_join(model_feature_totals, by = c('PersonId', 'MetricDate'), relationship = 'one-to-one') |>
  mutate(across(ends_with('Total'), ~coalesce(.x, 0)))
stopifnot(all(breakdown_check$Expected == breakdown_check$FeatureTotal),
          all(breakdown_check$Expected == breakdown_check$LanguageFeatureTotal),
          all(breakdown_check$Expected == breakdown_check$LanguageModelTotal),
          all(breakdown_check$Expected == breakdown_check$ModelFeatureTotal))

baseline_feature_daily <- feature_daily |> filter(MetricDate %in% baseline_weekdays)
baseline_language_feature_daily <- language_feature_daily |> filter(MetricDate %in% baseline_weekdays)
baseline_language_model_daily <- language_model_daily |> filter(MetricDate %in% baseline_weekdays)
baseline_model_feature_daily <- model_feature_daily |> filter(MetricDate %in% baseline_weekdays)
agent_features <- c('chat_panel_agent_mode', 'agent_edit', 'copilot_cli', 'copilot_app')
feature_mix <- baseline_feature_daily |>
  group_by(Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Feature Usage Count`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count),
         Feature_type = if_else(Feature %in% agent_features, 'Agent-type feature', 'Other feature'))
model_mix <- baseline_model_feature_daily |>
  group_by(Model) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by model and feature`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count))
language_mix <- baseline_language_model_daily |>
  group_by(Language) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and model`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  arrange(desc(Count)) |>
  mutate(Share = Count / sum(Count))
model_feature_mix <- baseline_model_feature_daily |>
  group_by(Model, Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by model and feature`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  group_by(Model) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
language_model_mix <- baseline_language_model_daily |>
  group_by(Language, Model) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and model`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  group_by(Language) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
language_feature_mix <- baseline_language_feature_daily |>
  group_by(Language, Feature) |>
  summarise(People = n_distinct(PersonId), Count = sum(`Usage count by language and feature`), .groups = 'drop') |>
  filter(People >= MIN_GROUP_N, Count > 0) |>
  group_by(Language) |>
  mutate(Share = Count / sum(Count)) |>
  ungroup()
stopifnot(nrow(feature_mix) > 0, nrow(model_mix) > 0, nrow(language_mix) > 0,
          nrow(model_feature_mix) > 0, nrow(language_model_mix) > 0,
          nrow(language_feature_mix) > 0)

# Presentation helpers use group-level aggregates only. No point per person.
wrap <- function(x, width = 100) stringr::str_wrap(x, width = width)
f1 <- function(x) ifelse(is.na(x), 'N/A',
                         formatC(x, format = 'f', digits = 1, big.mark = ',', decimal.mark = '.'))
pct <- function(x) ifelse(is.na(x), 'N/A',
                          scales::percent(x, accuracy = .1, decimal.mark = '.', big.mark = ','))
num <- function(x) format(x, big.mark = ',', scientific = FALSE, trim = TRUE)
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
  first_metric <- unname(metric_labels[metrics[1]])
  group_order <- if (group == 'Joint') joint_levels[1:4] else d |>
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
  counts <- data |> count(Group = .data[[group]], Category = .data[[category]], name = 'People', .drop = TRUE)
  counts <- counts |> group_by(Group) |> filter(all(People >= MIN_GROUP_N)) |> ungroup()
  stopifnot(nrow(counts) > 0, all(counts$People >= MIN_GROUP_N))
  counts |> group_by(Group) |> mutate(Share = People / sum(People)) |> ungroup()
}
stack_plot <- function(d, title, subtitle, caption, palette = PAL, sort_category = NULL, group_order = NULL) {
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
  stopifnot(!is.null(data), nrow(data) > 0)
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
  mutate(Display = if_else(People < MIN_GROUP_N,
                           'Small or exception groups withheld',
                           as.character(Joint))) |>
  group_by(Display) |>
  summarise(People = sum(People), .groups = 'drop') |>
  mutate(Share = People / nrow(baseline),
         Display = factor(Display, levels = c(joint_levels,
           'Small or exception groups withheld'))) |>
  arrange(Display)
team_context <- composition(baseline, 'Team', 'Role')
role_context <- composition(matched, 'Joint', 'Role')
team_joint_context <- composition(matched, 'Joint', 'Team')
work_summary <- bind_rows(lapply(c('Collaboration_hours', 'Meeting_hours', 'Available_to_focus_hours',
                                   'Uninterrupted_hours', 'Interrupted_hours',
                                   'After_hours_collaboration_hours'), function(m) tibble(
  Metric = metric_labels[[m]], `25th percentile` = f1(quantile(baseline[[m]], .25)),
  Median = f1(median(baseline[[m]])), `75th percentile` = f1(quantile(baseline[[m]], .75)))))

product_summary <- bind_rows(lapply(c('GH', 'M365'), function(product) {
  gh <- product == 'GH'
  valid <- baseline |> filter(if (gh) GH_all_valid else M365_all_valid)
  active <- valid |> filter(if (gh) GH_active_days > 0 else M365_active_days > 0)
  volume <- if (gh) 'GH_credits' else 'M365_credits'
  stopifnot(nrow(valid) >= MIN_GROUP_N, nrow(active) >= MIN_GROUP_N,
            ceiling(.1 * nrow(active)) >= MIN_GROUP_N)
  top_n <- ceiling(.1 * nrow(active))
  share <- sum(sort(active[[volume]], decreasing = TRUE)[seq_len(top_n)]) / sum(active[[volume]])
  tibble(Product = if (gh) 'GitHub Copilot' else 'Microsoft 365 Copilot',
         `Valid developers (count)` = nrow(valid),
         `Recorded active (count)` = nrow(active),
         `Recorded active (%)` = pct(nrow(active) / nrow(valid)),
         `Active days / developer / week (mean)` = f1(mean(if (gh) valid$GH_active_days else valid$M365_active_days) / BASELINE_WEEKS),
         `Credits / developer / week (mean)` = f1(mean(valid[[volume]]) / BASELINE_WEEKS),
         `Session or request unit` = if (gh) 'GitHub credits' else 'M365 credits',
         `Top 10% active credit share (%)` = pct(share),
         `Top group (count)` = top_n)
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
            GH_intensity = mean(GH_credits[GH_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
            M365_valid_n = sum(M365_all_valid),
            M365_active_n = sum(M365_all_valid & coalesce(M365_active_days, 0) > 0),
            M365_active_share = safe_div(M365_active_n, M365_valid_n),
            M365_intensity = mean(M365_credits[M365_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
            .groups = 'drop') |>
  mutate(across(c(GH_active_share, GH_intensity, M365_active_share, M365_intensity),
                ~if_else(Developers >= MIN_GROUP_N & GH_valid_n >= MIN_GROUP_N &
                           M365_valid_n >= MIN_GROUP_N, .x, NA_real_))) |>
  arrange(desc(Collaboration_hours)) |>
  mutate(Rank = row_number(), .before = Team)
stopifnot(all(team_league$Developers >= MIN_GROUP_N), all(team_league$GH_valid_n >= MIN_GROUP_N),
          all(team_league$M365_valid_n >= MIN_GROUP_N))

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
    GH_n = sum(GH_valid), GH_active = sum(GH_active_days > 0, na.rm = TRUE),
    M365_n = sum(M365_valid), M365_active = sum(M365_active_days > 0, na.rm = TRUE),
    Joint_n = sum(GH_valid & M365_valid), .groups = 'drop') |>
  mutate(GH_rate = safe_div(GH_active, GH_n),
         M365_rate = safe_div(M365_active, M365_n))
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
    paste('Weekly complete panel for', num(nrow(people)), 'people and', length(weeks),
          'weeks;', num(sum(people$IsDeveloper)), 'developers in',
          n_distinct(people$Team[people$IsDeveloper]), 'teams of',
          min(as.integer(table(people$Team[people$IsDeveloper]))), '.'),
    'Consumption metadata keyed by PeopleHistoricalId; used for Copilot licence status.',
    'Microsoft 365 Copilot credits and session counts; service rows are aggregated before daily or weekly use.',
    'GitHub AI credits by person and day.',
    'GitHub activity with explicit zero rows for provisioned inactive weekdays.',
    'Feature margin for accepted completions plus user-initiated chat requests.',
    'Language x feature margin of the same GitHub allocation.',
    'Language x model margin of the same GitHub allocation.',
    'Model x feature margin of the same GitHub allocation.'))
coverage_derivation <- tibble(
  Question = c('Person Query roster and weekly completeness',
               'Microsoft 365 Copilot eligibility',
               'GitHub Copilot provisioning',
               'GitHub weekly coverage',
               'Microsoft 365 weekly observation',
               'Recorded product use'),
  `Signal used` = c('PersonQuery.csv has one row per PersonId x week in this sample.',
                    'PeopleMetaData.csv IsCopilotLicensed, with Person Query enabled days as a cross-check.',
                    'Presence of at least one row in PersonGitHubActivityMetrics.csv.',
                    'Count observed weekday rows per person-week, with a maximum of five.',
                    'The product feed window is treated as the observation window; ingestion completeness is not separately observable.',
                    'Positive observed activity or credits/sessions. Missing is filled as zero only after eligibility and observation are established.'))
validation_summary <- tibble(
  Check = c('Population', 'Teams', 'Person Query developer-weeks', 'Baseline developer-weeks',
            'Focus-time identities', 'Source keys and joins', 'Non-negative metrics and acceptance bounds',
            'GitHub breakdown reconciliation', 'Eligibility and observation derivation',
            'Joint reconciliation', 'Privacy floor'),
  Result = c(paste(num(nrow(people)), 'people;', num(sum(people$IsDeveloper)), 'developers'),
             paste(n_distinct(people$Team[people$IsDeveloper]), 'teams of',
                   min(as.integer(table(people$Team[people$IsDeveloper]))), 'developers'),
             num(nrow(pq |> filter(IsDeveloper))), num(nrow(baseline_panel)),
             'Available-to-focus hours = working-hours basis minus meetings and scheduled calls; uninterrupted + interrupted = available-to-focus',
             'Unique keys; no join amplification', 'Passed',
             'Feature, language-feature, language-model and model-feature margins reconcile to accepted completions plus chat requests',
             'Derived from real query fields; no coverage file used',
             paste(sum(joint_counts$People), '=', nrow(matched), '+', nrow(baseline) - nrow(matched)),
             paste('At least', MIN_GROUP_N, 'distinct people in every published group')))
