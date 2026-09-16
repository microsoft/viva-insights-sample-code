# ---------------------------------------------------------------------------
# Simulate Viva Insights query exports with real export schemas.
#
# This script generates fully synthetic CSV files that match, column for column,
# the output of three real Viva Insights queries:
#
#   1. Person query          -> _data/person-query/
#   2. Consumption query     -> _data/consumption-query/
#   3. GitHub Copilot query  -> _data/github-query/
#
# The VALUES are simulated. The SCHEMA is not. Every column name below appears
# in a real export or in the official metric reference:
#   https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query
#   https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics
#
# No column is invented. Organisational attributes (Organization, FunctionType,
# Team, and so on) are customer-defined in a real tenant, so the specific
# attribute names here are illustrative while the mechanism is real.
#
# Run from examples/utility-r:  Rscript simulate-query-exports.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

SEED <- 20260916L
set.seed(SEED)
options(OutDec = ".", stringsAsFactors = FALSE)

DATA_DIR <- "_data"
N_PEOPLE <- 480L
N_DEVELOPERS <- 360L

# Scheduled working hours per week. In a real tenant this basis is configurable
# through metric rules, so it should not be treated as universal.
WORKING_HOURS_PER_WEEK <- 40

# The Person query covers 26 weeks. Product feeds cover a shorter, more recent
# window, which is the common real-world case: Copilot and GitHub telemetry is
# typically available for a shorter history than collaboration metrics.
weeks <- seq(as.Date("2026-01-04"), by = "week", length.out = 26L)
product_weeks <- tail(weeks, 13L)
period_end <- max(weeks) + 6L
product_start <- min(product_weeks)

# ---------------------------------------------------------------------------
# Identifiers
#
# Real PersonId values are GUIDs. PeopleHistoricalId, the join key between the
# Consumption activity files and PeopleMetaData, is the PersonId with a numeric
# suffix appended. Both shapes are reproduced here.
# ---------------------------------------------------------------------------
hex <- function(n) paste0(sample(c(0:9, letters[1:6]), n, replace = TRUE), collapse = "")
make_guid <- function() {
  paste(hex(8), hex(4), paste0("3", hex(3)), paste0(sample(c("8", "9", "a", "b"), 1), hex(3)),
        hex(12), sep = "-")
}
HISTORICAL_SUFFIX <- "1784160000"

person_ids <- vapply(seq_len(N_PEOPLE), function(i) make_guid(), character(1))
stopifnot(!anyDuplicated(person_ids))

# ---------------------------------------------------------------------------
# Roster and organisational attributes
# ---------------------------------------------------------------------------
teams <- c("Data Engineering", "Developer Experience", "Identity",
           "Mobile", "Payments", "Platform")
organizations <- c("Business Unit A", "Business Unit B", "Business Unit C",
                   "Business Unit D", "Business Unit E")
functions <- c("Engineering", "Product", "Finance", "Operations",
               "Marketing", "Sales", "Legal", "Human Resources")
levels_designation <- c("Individual Contributor", "Manager", "Senior Manager", "Director")
roles <- c("Application engineering", "Infrastructure engineering", "Engineering management")
seniority <- c("Early career", "Experienced", "Senior / lead")
tenures <- c("Under 2 years", "2 to 5 years", "Over 5 years")

team_effects <- tibble(
  Team = teams,
  meeting_delta        = c(0.6, -0.8, 0.0, -3.0, 1.6, 5.0),
  focus_delta          = c(0.3, 2.1, 0.0, 3.7, -1.0, -4.1),
  after_delta          = c(0.0, -0.4, 0.0, -0.6, 2.4, 0.4),
  scheduled_call_delta = c(0.0, -0.1, 0.0, -0.4, 0.2, 0.6),
  email_delta          = c(0.2, -0.1, 0.0, -0.5, 0.8, 1.5),
  chat_delta           = c(0.1, 0.0, 0.0, -0.3, 0.6, 1.2),
  recurring_delta      = c(0.02, -0.02, 0.00, -0.05, 0.03, 0.10),
  conflict_delta       = c(0.00, -0.01, 0.00, -0.02, 0.11, 0.03),
  short_notice_delta   = c(0.00, -0.01, 0.00, -0.02, 0.12, 0.02),
  gh_prop_delta        = c(0.06, 0.24, 0.00, -0.18, 0.02, 0.04),
  m365_prop_delta      = c(0.24, 0.05, 0.00, -0.17, 0.02, 0.05),
  gh_intensity_mult    = c(1.10, 1.85, 1.00, 0.72, 1.05, 1.08),
  m365_intensity_mult  = c(1.85, 1.05, 1.00, 0.72, 1.05, 1.08)
)

function_effects <- tibble(
  FunctionType = functions,
  fn_credit_mult = c(1.35, 1.20, 0.80, 0.90, 1.05, 1.15, 0.65, 0.75),
  fn_collab_delta = c(-1.2, 0.8, -0.4, 0.2, 1.1, 2.4, -0.9, 0.5)
)

is_developer <- seq_len(N_PEOPLE) <= N_DEVELOPERS
people <- tibble(
  PersonId = person_ids,
  IsDeveloper = is_developer,
  Team = c(sample(rep(teams, each = N_DEVELOPERS / length(teams))),
           sample(teams, N_PEOPLE - N_DEVELOPERS, replace = TRUE)),
  Role = c(sample(roles, N_DEVELOPERS, TRUE, c(.48, .32, .20)),
           rep("Other job families", N_PEOPLE - N_DEVELOPERS)),
  Seniority = sample(seniority, N_PEOPLE, TRUE, c(.26, .43, .31)),
  Tenure = sample(tenures, N_PEOPLE, TRUE, c(.34, .42, .24)),
  Organization = sample(organizations, N_PEOPLE, TRUE),
  FunctionType = c(sample(functions, N_DEVELOPERS, TRUE, c(.52, .16, .05, .09, .05, .06, .03, .04)),
                   sample(functions, N_PEOPLE - N_DEVELOPERS, TRUE,
                          c(.05, .18, .14, .16, .14, .16, .08, .09))),
  LevelDesignation = sample(levels_designation, N_PEOPLE, TRUE, c(.62, .21, .12, .05))
) |>
  mutate(
    IsManager = if_else(LevelDesignation == "Individual Contributor", "No", "Yes"),
    IsDeveloperFlag = if_else(IsDeveloper, "Yes", "No"),
    PeopleHistoricalId = paste0(PersonId, HISTORICAL_SUFFIX)
  )

stopifnot(all(table(people$Team[people$IsDeveloper]) == N_DEVELOPERS / length(teams)))

# ---------------------------------------------------------------------------
# Latent per-person traits driving both collaboration and AI usage.
# Composition confounding is deliberate so that raw associations are not causal.
# ---------------------------------------------------------------------------
latent <- people |>
  left_join(team_effects, by = "Team", relationship = "many-to-one") |>
  left_join(function_effects, by = "FunctionType", relationship = "many-to-one") |>
  mutate(
    meeting_base = 7 + 3 * (Role == "Engineering management") +
      3 * (LevelDesignation %in% c("Senior Manager", "Director")) +
      .9 * (Seniority == "Senior / lead") + meeting_delta + fn_collab_delta +
      rnorm(n(), 0, 1.8),
    focus_base = focus_delta + rnorm(n(), 0, 1.7),
    after_base = pmax(.05, rlnorm(n(), .35, .70) + after_delta),
    network_base = pmax(8, round(rlnorm(n(), 3.7, .45) + 6 * (IsManager == "Yes"))),

    # Copilot licensing. Unlicensed people never appear in the credit feeds.
    IsCopilotLicensed = runif(n()) < .82,

    # Separate product propensities. Team signatures are cross-sectional only.
    gh_base_prop = if_else(runif(n()) < .25, 0, runif(n(), .2, .85)),
    m365_base_prop = if_else(runif(n()) < .22, 0, runif(n(), .18, .8)),
    gh_prop = if_else(IsDeveloper, pmin(.96, pmax(0, gh_base_prop + gh_prop_delta)), 0),
    m365_prop = pmin(.96, pmax(0, m365_base_prop + m365_prop_delta)),
    gh_intensity = rlnorm(n(), 2.4, .65) * gh_intensity_mult,
    m365_intensity = rlnorm(n(), 1.8, .60) * m365_intensity_mult * fn_credit_mult,

    # Not every licensed developer has GitHub Copilot provisioned.
    gh_provisioned = IsDeveloper & runif(n()) < .92,

    # Credit cost per session varies by how much agent and premium-model work
    # a person does. This is the honest replacement for the fabricated token
    # fields that used to sit in this simulation.
    premium_affinity = pmin(1, pmax(0, rbeta(n(), 2, 4) + .15 * (Seniority == "Senior / lead"))),
    credits_per_session = pmax(8, rlnorm(n(), 3.6, .55) * (1 + .9 * premium_affinity)),
    gh_credits_per_action = pmax(.2, rlnorm(n(), .35, .5) * (1 + 1.2 * premium_affinity))
  )

# ---------------------------------------------------------------------------
# Person query, weekly, 26 weeks.
#
# Every metric below is a genuine Viva Insights person-query metric. Metrics
# that did not survive verification against the metric reference (for example
# an "available to focus hours" field) have been removed: the real pair is
# Uninterrupted_hours and Interrupted_hours.
# ---------------------------------------------------------------------------
pq <- expand_grid(PersonId = people$PersonId, MetricDate = weeks) |>
  left_join(latent, by = "PersonId", relationship = "many-to-one") |>
  mutate(
    seasonal = .6 * sin(2 * pi * as.numeric(MetricDate - min(weeks)) / 182),
    Meeting_hours = pmin(26, pmax(.5, meeting_base + seasonal + rnorm(n(), 0, 1.5))),
    Scheduled_call_hours = pmin(3.5, pmax(.25, runif(n(), .5, 2) + scheduled_call_delta)),
    Unscheduled_call_hours = pmax(0, rlnorm(n(), -.2, .6) + .02 * Meeting_hours),
    Email_hours = pmax(.2, rlnorm(n(), .55, .35) + .03 * Meeting_hours + email_delta),
    Chat_hours = pmax(.1, rlnorm(n(), .20, .40) + .02 * Meeting_hours + chat_delta),
    Channel_message_hours = pmax(0, rlnorm(n(), -.9, .6)),
    Collaboration_hours = pmin(45, Meeting_hours + Scheduled_call_hours +
      Unscheduled_call_hours + Email_hours + Chat_hours + Channel_message_hours),
    Active_connected_hours = pmin(55, Collaboration_hours + pmax(0, rnorm(n(), 9, 3))),
    External_collaboration_hours = pmax(0, Collaboration_hours *
      pmin(.45, pmax(0, rbeta(n(), 2, 12)))),

    # "Available-to-focus hours" is the hours remaining during working hours
    # after excluding meetings and scheduled calls. Uninterrupted and
    # interrupted hours partition it.
    Available_to_focus_hours = pmax(0, WORKING_HOURS_PER_WEEK - Meeting_hours -
      Scheduled_call_hours),
    Uninterrupted_hours = pmin(Available_to_focus_hours,
      pmax(0, 16 - .55 * Meeting_hours + focus_base + rnorm(n(), 0, 1.8))),
    Interrupted_hours = Available_to_focus_hours - Uninterrupted_hours,
    Open_1_hour_block = pmin(floor(Available_to_focus_hours),
      pmax(0, round(22 - .6 * Meeting_hours + rnorm(n(), 0, 2)))),

    Recurring_meeting_hours = Meeting_hours *
      pmin(.90, pmax(.05, runif(n(), .35, .75) + recurring_delta)),
    Conflicting_meeting_hours = Meeting_hours *
      pmin(.45, pmax(.00, runif(n(), .01, .18) + conflict_delta)),
    Meeting_hours_with_six_or_fewer_hours_of_advanced_notice = Meeting_hours *
      pmin(.50, pmax(.00, runif(n(), .03, .23) + short_notice_delta)),

    Emails_sent = pmax(0, round(rnorm(n(), 24 + 6 * Email_hours, 7))),
    Chats_sent = pmax(0, round(rnorm(n(), 40 + 22 * Chat_hours, 14))),
    Meetings = pmax(0, round(Meeting_hours / pmax(.4, rnorm(n(), .85, .12)))),
    Calls = pmax(0, round((Scheduled_call_hours + Unscheduled_call_hours) /
      pmax(.25, rnorm(n(), .6, .1)))),

    After_hours_collaboration_hours = pmin(16, pmax(0, after_base + rnorm(n(), 0, .65))),
    After_hours_meeting_hours = After_hours_collaboration_hours * runif(n(), .1, .45),
    After_hours_email_hours = After_hours_collaboration_hours * runif(n(), .15, .4),
    After_hours_chat_hours = After_hours_collaboration_hours * runif(n(), .05, .3),
    Weekend_collaboration_hours = pmax(0, rlnorm(n(), -.5, 1) * (1 + .2 * (IsManager == "Yes"))),
    Collaboration_span = pmin(16, pmax(4, 8 + .12 * Meeting_hours +
      .8 * After_hours_collaboration_hours + rnorm(n(), 0, 1.1))),

    Internal_network_size = pmax(1, round(network_base + rnorm(n(), 0, 4))),
    External_network_size = pmax(0, round(rlnorm(n(), 2.1, .7))),
    Strong_ties = pmax(0, round(Internal_network_size * runif(n(), .15, .45))),
    Diverse_ties = pmax(0, round(Internal_network_size * runif(n(), .1, .35))),
    Network_outside_organization = pmax(0, round(Internal_network_size * runif(n(), .05, .25))),

    # Licensing is observable in the Person query and is the eligibility signal
    # that replaces the previously fabricated coverage file.
    Total_Copilot_enabled_days = if_else(IsCopilotLicensed, 7L, 0L)
  ) |>
  select(
    PersonId, MetricDate,
    Collaboration_hours, Meeting_hours, Email_hours, Chat_hours,
    Scheduled_call_hours, Unscheduled_call_hours, Channel_message_hours,
    Active_connected_hours, External_collaboration_hours,
    Available_to_focus_hours, Uninterrupted_hours, Interrupted_hours,
    Open_1_hour_block,
    Recurring_meeting_hours, Conflicting_meeting_hours,
    Meeting_hours_with_six_or_fewer_hours_of_advanced_notice,
    Emails_sent, Chats_sent, Meetings, Calls,
    After_hours_collaboration_hours, After_hours_meeting_hours,
    After_hours_email_hours, After_hours_chat_hours,
    Weekend_collaboration_hours, Collaboration_span,
    Internal_network_size, External_network_size, Strong_ties, Diverse_ties,
    Network_outside_organization, Total_Copilot_enabled_days,
    Organization, FunctionType, LevelDesignation, IsManager,
    Team, Role, Seniority, Tenure, IsDeveloper = IsDeveloperFlag
  ) |>
  mutate(across(where(is.numeric), ~round(.x, 3)))

# ---------------------------------------------------------------------------
# Daily weekday grid for the product feeds
# ---------------------------------------------------------------------------
product_days <- seq(product_start, period_end, by = "day")
product_days <- product_days[as.POSIXlt(product_days)$wday %in% 1:5]

daily_grid <- expand_grid(PersonId = people$PersonId, MetricDate = product_days) |>
  left_join(
    latent |> select(PersonId, IsCopilotLicensed, gh_provisioned, gh_prop, m365_prop,
                     gh_intensity, m365_intensity, credits_per_session,
                     gh_credits_per_action, premium_affinity),
    by = "PersonId", relationship = "many-to-one"
  )

# ---------------------------------------------------------------------------
# GitHub Copilot query: PersonGitHubActivityMetrics
#
# Real exports include explicit zero rows for provisioned users on days with no
# activity, so those are reproduced rather than dropped.
# ---------------------------------------------------------------------------
gh_raw <- daily_grid |>
  filter(gh_provisioned) |>
  mutate(
    active = rbinom(n(), 1, gh_prop),
    suggested = active * rpois(n(), gh_intensity),
    accepted = rbinom(n(), suggested, .53),
    chat = active * rpois(n(), gh_intensity * .28)
  ) |>
  select(PersonId, MetricDate, suggested, accepted, chat, premium_affinity,
         gh_credits_per_action)

# Feature, language and model vocabularies observed in real GitHub query exports.
CHAT_FEATURES <- c("chat_panel_ask_mode", "chat_panel_agent_mode", "chat_panel_edit_mode",
                   "chat_panel_plan_mode", "chat_panel_custom_mode",
                   "chat_panel_unknown_mode", "chat_inline", "agent_edit",
                   "copilot_cli", "copilot_app")
AGENT_FEATURES <- c("chat_panel_agent_mode", "agent_edit", "copilot_cli", "copilot_app")
COMPLETION_FEATURE <- "code_completion"
LANGUAGES <- c("typescript", "python", "javascript", "c#", "java", "go",
               "sql", "json", "markdown", "yaml", "unknown")
MODELS <- c("gpt-5.4", "gpt-5.4-mini", "gpt-5.6-terra", "gpt-5.3-codex",
            "claude-sonnet-5", "claude-haiku-4.5")
PREMIUM_MODELS <- c("gpt-5.6-terra", "gpt-5.3-codex", "claude-sonnet-5")

# A single three-way (Feature x Language x Model) table per person-day. Every
# published breakdown file is an exact margin of it, so all four GitHub
# breakdown files reconcile with each other and with the activity file.
chat_weights <- c(.26, .18, .12, .05, .04, .03, .14, .08, .06, .04)
language_weights <- c(.16, .15, .13, .09, .08, .07, .07, .09, .08, .05, .03)

build_breakdown <- function(df) {
  rows <- which(df$accepted + df$chat > 0)
  if (!length(rows)) return(tibble())
  out <- vector("list", length(rows))
  for (k in seq_along(rows)) {
    i <- rows[k]
    feat_names <- character(0)
    feat_counts <- integer(0)
    if (df$accepted[i] > 0) {
      feat_names <- COMPLETION_FEATURE
      feat_counts <- as.integer(df$accepted[i])
    }
    if (df$chat[i] > 0) {
      n_feat <- min(length(CHAT_FEATURES), 1L + rbinom(1, 3, .45))
      picked <- sample(CHAT_FEATURES, n_feat, prob = chat_weights)
      counts <- as.vector(rmultinom(1, df$chat[i], rep(1, n_feat)))
      keep <- counts > 0
      feat_names <- c(feat_names, picked[keep])
      feat_counts <- c(feat_counts, counts[keep])
    }
    if (!length(feat_names)) next

    n_lang <- min(length(LANGUAGES), 1L + rbinom(1, 2, .5))
    langs <- sample(LANGUAGES, n_lang, prob = language_weights)
    n_model <- min(length(MODELS), 1L + rbinom(1, 2, .4))
    prem <- df$premium_affinity[i]
    model_prob <- ifelse(MODELS %in% PREMIUM_MODELS, .2 + .8 * prem, .8 - .5 * prem)
    models <- sample(MODELS, n_model, prob = model_prob)

    cells <- expand_grid(Feature = feat_names, Language = langs, Model = models)
    alloc <- lapply(seq_along(feat_names), function(f) {
      cell_n <- n_lang * n_model
      as.vector(rmultinom(1, feat_counts[f], rep(1, cell_n)))
    })
    cells$Count <- as.integer(unlist(alloc))
    cells <- cells[cells$Count > 0, , drop = FALSE]
    cells$PersonId <- df$PersonId[i]
    cells$MetricDate <- df$MetricDate[i]
    out[[k]] <- cells
  }
  bind_rows(out)
}

message("Building GitHub breakdown allocation ...")
breakdown <- build_breakdown(gh_raw)

agent_days <- breakdown |>
  filter(Feature %in% AGENT_FEATURES) |>
  distinct(PersonId, MetricDate) |>
  mutate(agent_flag = TRUE)

gh_activity <- gh_raw |>
  left_join(agent_days, by = c("PersonId", "MetricDate"), relationship = "one-to-one") |>
  transmute(
    PersonId, MetricDate,
    `Agent adoption` = tolower(as.character(coalesce(agent_flag, FALSE))),
    `Code completions accepted` = as.integer(accepted),
    `Code completions suggested` = as.integer(suggested),
    `User-initiated chat requests` = as.integer(chat)
  ) |>
  arrange(MetricDate, PersonId)

feature_metrics <- breakdown |>
  group_by(PersonId, MetricDate, Feature) |>
  summarise(`Feature Usage Count` = sum(Count), .groups = "drop") |>
  arrange(MetricDate, PersonId, Feature)

language_feature_metrics <- breakdown |>
  group_by(PersonId, MetricDate, Language, Feature) |>
  summarise(`Usage count by language and feature` = sum(Count), .groups = "drop") |>
  arrange(MetricDate, PersonId, Language, Feature)

language_model_metrics <- breakdown |>
  group_by(PersonId, MetricDate, Language, Model) |>
  summarise(`Usage count by language and model` = sum(Count), .groups = "drop") |>
  arrange(MetricDate, PersonId, Language, Model)

model_feature_metrics <- breakdown |>
  group_by(PersonId, MetricDate, Model, Feature) |>
  summarise(`Usage count by model and feature` = sum(Count), .groups = "drop") |>
  arrange(MetricDate, PersonId, Model, Feature)

# ---------------------------------------------------------------------------
# Consumption query: PersonGitHubCreditsMetrics
# ---------------------------------------------------------------------------
gh_credits <- gh_raw |>
  filter(accepted + chat > 0) |>
  left_join(select(people, PersonId, PeopleHistoricalId), by = "PersonId",
            relationship = "many-to-one") |>
  transmute(
    PersonId, MetricDate,
    `Total GitHub AI Credits used` = round((accepted + chat) * gh_credits_per_action *
      runif(n(), .7, 1.4), 6),
    PeopleHistoricalId
  ) |>
  arrange(MetricDate, PersonId)

# ---------------------------------------------------------------------------
# Consumption query: PersonM365CreditsMetrics
#
# One row per PersonId, ServiceId and MetricDate. Service identifiers are fixed
# per service, matching the real one-GUID-per-service pattern.
# ---------------------------------------------------------------------------
services <- tibble(
  ServiceName = c("WorkIQ", "Cowork", "Researcher", "Analyst"),
  ServiceId = vapply(seq_len(4), function(i) make_guid(), character(1)),
  weight = c(.46, .28, .16, .10)
)

NO_POLICY <- "00000000-0000-0000-0000-000000000000"
managed_policy <- make_guid()

# A minority of people sit under a spending policy. For everyone else the
# policy and limit columns are empty in the real export, which is reproduced.
policy_people <- people |>
  transmute(
    PersonId,
    SpendingPolicyId = if_else(runif(n()) < .22, managed_policy, NO_POLICY),
    `Spending policy limit` = NA_real_,
    `User limit` = NA_real_
  ) |>
  mutate(
    `Spending policy limit` = if_else(SpendingPolicyId == NO_POLICY, NA_real_, 25000),
    `User limit` = if_else(SpendingPolicyId == NO_POLICY, NA_real_,
                           round(runif(n(), 400, 1800)))
  )

m365_credits <- daily_grid |>
  filter(IsCopilotLicensed) |>
  mutate(active = rbinom(n(), 1, m365_prop)) |>
  filter(active == 1) |>
  select(PersonId, MetricDate, m365_intensity, credits_per_session)

m365_credits <- m365_credits |>
  mutate(n_services = 1L + rbinom(n(), 2, .35)) |>
  uncount(n_services, .id = "service_slot") |>
  group_by(PersonId, MetricDate) |>
  mutate(ServiceName = sample(services$ServiceName, n(), replace = FALSE,
                              prob = services$weight)) |>
  ungroup() |>
  left_join(select(services, ServiceName, ServiceId), by = "ServiceName",
            relationship = "many-to-one") |>
  left_join(policy_people, by = "PersonId", relationship = "many-to-one") |>
  left_join(select(people, PersonId, PeopleHistoricalId), by = "PersonId",
            relationship = "many-to-one") |>
  mutate(
    `Session count` = pmax(1L, rpois(n(), m365_intensity / 2)),
    `Total Copilot Credits used` = round(`Session count` * credits_per_session *
      runif(n(), .6, 1.5), 7)
  ) |>
  select(PersonId, ServiceId, ServiceName, SpendingPolicyId, MetricDate,
         `Session count`, `Spending policy limit`, `Total Copilot Credits used`,
         `User limit`, PeopleHistoricalId) |>
  arrange(MetricDate, PersonId, ServiceName)

# ---------------------------------------------------------------------------
# Consumption query: PeopleMetaData
#
# PeopleHistoricalId and IsCopilotLicensed are always present and cannot be
# removed. The remaining columns are the HR attributes selected at query setup.
# ---------------------------------------------------------------------------
people_metadata <- latent |>
  transmute(
    PeopleHistoricalId,
    IsCopilotLicensed = tolower(as.character(IsCopilotLicensed)),
    Organization, FunctionType, LevelDesignation, IsManager,
    Team, Role, Seniority, Tenure,
    IsDeveloper = IsDeveloperFlag
  ) |>
  arrange(PeopleHistoricalId)

# ---------------------------------------------------------------------------
# Contract assertions. The build fails rather than exporting a broken schema.
# ---------------------------------------------------------------------------
assert_key <- function(data, keys, label) {
  stopifnot(!anyNA(data[keys]))
  if (anyDuplicated(data[keys])) stop("Duplicate key in ", label)
  invisible(TRUE)
}

assert_key(people_metadata, "PeopleHistoricalId", "PeopleMetaData")
assert_key(pq, c("PersonId", "MetricDate"), "PersonQuery")
assert_key(gh_activity, c("PersonId", "MetricDate"), "PersonGitHubActivityMetrics")
assert_key(gh_credits, c("PersonId", "MetricDate"), "PersonGitHubCreditsMetrics")
assert_key(m365_credits, c("PersonId", "ServiceId", "MetricDate"), "PersonM365CreditsMetrics")
assert_key(feature_metrics, c("PersonId", "MetricDate", "Feature"), "ByFeature")
assert_key(language_feature_metrics, c("PersonId", "MetricDate", "Language", "Feature"),
           "ByLanguageFeature")
assert_key(language_model_metrics, c("PersonId", "MetricDate", "Language", "Model"),
           "ByLanguageModel")
assert_key(model_feature_metrics, c("PersonId", "MetricDate", "Model", "Feature"),
           "ByModelFeature")

stopifnot(
  nrow(people_metadata) == N_PEOPLE,
  nrow(pq) == N_PEOPLE * length(weeks),
  all(gh_activity$`Code completions accepted` <= gh_activity$`Code completions suggested`),
  all(pq$Recurring_meeting_hours <= pq$Meeting_hours + 1e-9),
  all(pq$Conflicting_meeting_hours <= pq$Meeting_hours + 1e-9),
  all(pq$Meeting_hours_with_six_or_fewer_hours_of_advanced_notice <= pq$Meeting_hours + 1e-9),
  all(pq$Collaboration_hours + 1e-9 >= pq$Meeting_hours),
  all(pq$Uninterrupted_hours <= pq$Available_to_focus_hours + 1e-9),
  max(abs(pq$Available_to_focus_hours - pq$Uninterrupted_hours -
            pq$Interrupted_hours)) <= .0011,
  max(abs(pq$Meeting_hours + pq$Scheduled_call_hours +
            pq$Available_to_focus_hours - WORKING_HOURS_PER_WEEK)) <= .0011,
  all(pq$Open_1_hour_block <= pq$Available_to_focus_hours + 1e-9),
  all(pq$Total_Copilot_enabled_days %in% c(0L, 7L)),
  all(m365_credits$`Session count` >= 1),
  all(m365_credits$`Total Copilot Credits used` >= 0),
  all(gh_credits$`Total GitHub AI Credits used` >= 0)
)

# Every breakdown file must be an exact margin of the same allocation, and the
# feature margin must reconcile to the activity file.
recon <- feature_metrics |>
  group_by(PersonId, MetricDate) |>
  summarise(feature_total = sum(`Feature Usage Count`), .groups = "drop") |>
  left_join(
    gh_activity |>
      transmute(PersonId, MetricDate,
                activity_total = `Code completions accepted` + `User-initiated chat requests`),
    by = c("PersonId", "MetricDate"), relationship = "one-to-one"
  )
stopifnot(!anyNA(recon$activity_total), all(recon$feature_total == recon$activity_total))

margin_total <- function(df, col) {
  df |> group_by(PersonId, MetricDate) |>
    summarise(total = sum(.data[[col]]), .groups = "drop") |>
    arrange(PersonId, MetricDate)
}
m_feat <- margin_total(feature_metrics, "Feature Usage Count")
stopifnot(
  identical(m_feat, margin_total(language_feature_metrics, "Usage count by language and feature")),
  identical(m_feat, margin_total(language_model_metrics, "Usage count by language and model")),
  identical(m_feat, margin_total(model_feature_metrics, "Usage count by model and feature"))
)

# Credit feeds must only contain licensed or provisioned people.
licensed_ids <- latent$PersonId[latent$IsCopilotLicensed]
provisioned_ids <- latent$PersonId[latent$gh_provisioned]
stopifnot(
  all(m365_credits$PersonId %in% licensed_ids),
  all(gh_activity$PersonId %in% provisioned_ids),
  all(gh_credits$PersonId %in% provisioned_ids),
  all(m365_credits$PeopleHistoricalId %in% people_metadata$PeopleHistoricalId),
  all(gh_credits$PeopleHistoricalId %in% people_metadata$PeopleHistoricalId)
)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
export_csv <- function(data, folder, filename) {
  dest <- file.path(DATA_DIR, folder)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  chr <- data |> select(where(is.character))
  if (ncol(chr) && any(vapply(chr, function(x) any(grepl('[,"]', x)), logical(1)))) {
    stop("Unquoted export would be corrupted by a comma or quote in ", filename)
  }
  if (any(grepl('[,"]', names(data)))) stop("Header contains a delimiter in ", filename)
  write.table(data, file.path(dest, filename), sep = ",", quote = FALSE,
              row.names = FALSE, na = "", fileEncoding = "UTF-8")
  message(sprintf("  %-52s %8d rows", file.path(folder, filename), nrow(data)))
}

message("Writing exports ...")
export_csv(pq, "person-query", "PersonQuery.csv")

export_csv(people_metadata, "consumption-query", "PeopleMetaData.csv")
export_csv(m365_credits, "consumption-query", "PersonM365CreditsMetrics.csv")
export_csv(gh_credits, "consumption-query", "PersonGitHubCreditsMetrics.csv")

export_csv(gh_activity, "github-query", "PersonGitHubActivityMetrics.csv")
export_csv(feature_metrics, "github-query", "GitHubActivityBreakdownByFeatureMetrics.csv")
export_csv(language_feature_metrics, "github-query",
           "GitHubActivityBreakdownByLanguageFeatureMetrics.csv")
export_csv(language_model_metrics, "github-query",
           "GitHubActivityBreakdownByLanguageModelMetrics.csv")
export_csv(model_feature_metrics, "github-query",
           "GitHubActivityBreakdownByModelFeatureMetrics.csv")

message("Done.")
