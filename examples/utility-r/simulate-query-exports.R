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
source("query-export-contracts.R")
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
# Real PersonId values are GUIDs. PeopleHistoricalId is the join key between the
# Consumption activity files and PeopleMetaData. It is an OPAQUE key: analyst
# code must read the PersonId -> PeopleHistoricalId crosswalk out of the
# activity files, never reconstruct it from PersonId. A real export may happen
# to show a constant numeric suffix across one download, but that coincidence
# is not a contract and silently breaks on real tenants.
#
# These fixtures therefore mint the historical key INDEPENDENTLY: it keeps the
# same opaque GUID-plus-digits shape, but no string operation on PersonId
# reproduces it. Code that reconstructs the key fails here instead of passing
# the demo and failing in production.
# ---------------------------------------------------------------------------
hex <- function(n) paste0(sample(c(0:9, letters[1:6]), n, replace = TRUE), collapse = "")
make_guid <- function() {
  paste(hex(8), hex(4), paste0("3", hex(3)), paste0(sample(c("8", "9", "a", "b"), 1), hex(3)),
        hex(12), sep = "-")
}
make_history_id <- function() {
  paste0(make_guid(), paste0(sample(0:9, 10L, replace = TRUE), collapse = ""))
}

person_ids <- vapply(seq_len(N_PEOPLE), function(i) make_guid(), character(1))
historical_ids <- vapply(seq_len(N_PEOPLE), function(i) make_history_id(), character(1))
stopifnot(!anyDuplicated(person_ids), !anyDuplicated(historical_ids),
          !any(person_ids %in% historical_ids),
          !any(startsWith(historical_ids, person_ids)),
          !any(endsWith(historical_ids, person_ids)),
          !any(mapply(grepl, person_ids, historical_ids, MoreArgs = list(fixed = TRUE))))

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
  Role = "Other job families",
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
    PeopleHistoricalId = historical_ids
  )

# Role composition is set per team rather than sampled person by person. A
# published Team x Role cell has to clear the ten-person floor, so an
# unconstrained draw at 60 developers per team routinely produces an
# eight-person cell and the whole linked HR family has to be withheld. The
# mixes below still differ team by team, so the composition chart stays
# informative.
team_role_mix <- list(
  "Data Engineering"     = c(30L, 18L, 12L),
  "Developer Experience" = c(26L, 22L, 12L),
  "Identity"             = c(28L, 19L, 13L),
  "Mobile"               = c(32L, 16L, 12L),
  "Payments"             = c(27L, 20L, 13L),
  "Platform"             = c(24L, 24L, 12L))
stopifnot(setequal(names(team_role_mix), teams),
          all(vapply(team_role_mix, sum, integer(1)) == N_DEVELOPERS / length(teams)),
          all(unlist(team_role_mix) >= 10L))
for (tm in teams) {
  slot <- which(people$IsDeveloper & people$Team == tm)
  people$Role[slot] <- sample(rep(roles, team_role_mix[[tm]]))
}

# GitHub Copilot provisioning and adoption are managed per team in this
# fixture, with fixed counts rather than independent coin flips. A published
# team rate must leave a complement of at least ten people, so provisioning 92%
# of every 60-developer team would be unreportable by construction: the five
# unprovisioned developers could be recovered by subtraction. The rollout is
# deliberately uneven so team differences remain visible.
team_rollout <- tibble(
  Team = teams,
  provisioned_n = c(44L, 48L, 42L, 46L, 45L, 43L),
  adopted_n     = c(30L, 34L, 28L, 32L, 31L, 29L))
stopifnot(setequal(team_rollout$Team, teams),
          all(N_DEVELOPERS / length(teams) - team_rollout$provisioned_n >= 10L),
          all(team_rollout$provisioned_n - team_rollout$adopted_n >= 10L),
          all(team_rollout$adopted_n >= 10L))

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

    # Not every licensed developer has GitHub Copilot provisioned, and
    # provisioned is not adopted. Both are resolved from the per-team rollout
    # below, so the counts are fixed while who is included is random.
    gh_rollout_draw = runif(n()),

    # Credit cost per session varies by how much agent and premium-model work
    # a person does. This is the honest replacement for the fabricated token
    # fields that used to sit in this simulation.
    premium_affinity = pmin(1, pmax(0, rbeta(n(), 2, 4) + .15 * (Seniority == "Senior / lead"))),
    credits_per_session = pmax(8, rlnorm(n(), 3.6, .55) * (1 + .9 * premium_affinity)),
    gh_credits_per_action = pmax(.2, rlnorm(n(), .35, .5) * (1 + 1.2 * premium_affinity))
  ) |>
  left_join(team_rollout, by = "Team", relationship = "many-to-one") |>
  group_by(Team) |>
  mutate(rollout_rank = as.integer(rank(
    if_else(IsDeveloper, gh_rollout_draw, 2 + gh_rollout_draw), ties.method = "first"))) |>
  ungroup() |>
  mutate(
    gh_provisioned = IsDeveloper & rollout_rank <= provisioned_n,
    # Provisioned non-adopters stay in the activity export as explicit weekday
    # zero rows, which is what lets the reports distinguish an observed zero
    # from an unknown.
    gh_adopted = gh_provisioned & rollout_rank <= adopted_n,
    m365_adopted = IsCopilotLicensed & m365_base_prop > 0
  )

# ---------------------------------------------------------------------------
# Person query, weekly, 26 weeks.
#
# Every selected metric below is a genuine Viva Insights person-query metric.
# Available-to-focus hours is partitioned into uninterrupted and interrupted
# hours; the hyphens in the published metric name matter when locating it.
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
    latent |> select(PersonId, Team, IsCopilotLicensed, gh_provisioned, gh_adopted,
                     m365_adopted, gh_prop, m365_prop, gh_intensity, m365_intensity,
                     credits_per_session, gh_credits_per_action, premium_affinity),
    by = "PersonId", relationship = "many-to-one"
  )

# ---------------------------------------------------------------------------
# GitHub Copilot query: persistent per-person Copilot profiles
#
# Real developers do not re-roll their languages, models and Copilot surfaces
# every day. They work in the languages their team owns, on the models the
# tenant has enabled, through a stable set of surfaces. An earlier revision of
# this generator drew those categories afresh on every person-day, which gave
# almost every developer a unique category signature. That is both unrealistic
# and unpublishable: linked breakdown cross-tabs then have to be withheld in
# full even though every displayed cell covers dozens of people.
#
# Categorical membership is therefore COHORT-LEVEL and persistent, while the
# volumes stay individual, seasonal and team-weighted:
#   * one core surface set shared by every adopting developer;
#   * languages follow the team, so a language cohort is a whole team;
#   * enabled models are a tenant-level setting, so they are org-wide.
# ---------------------------------------------------------------------------

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

# Surfaces every adopting developer uses, plus two that differ by team.
CORE_FEATURES <- c(COMPLETION_FEATURE, "chat_panel_ask_mode", "chat_inline")
TEAM_FEATURES <- list(
  "Data Engineering"     = c("chat_panel_edit_mode", "chat_panel_agent_mode"),
  "Developer Experience" = c("agent_edit", "copilot_cli"),
  "Identity"             = c("chat_panel_custom_mode", "chat_panel_agent_mode"),
  "Mobile"               = c("chat_panel_plan_mode", "chat_panel_agent_mode"),
  "Payments"             = c("chat_panel_unknown_mode", "copilot_app"),
  "Platform"             = c("copilot_cli", "agent_edit"))
# Languages follow team ownership. Unclassified files turn up everywhere.
TEAM_LANGUAGES <- list(
  "Data Engineering"     = c("python", "sql", "yaml"),
  "Developer Experience" = c("typescript", "javascript", "markdown"),
  "Identity"             = c("c#", "go", "json"),
  "Mobile"               = c("typescript", "java", "json"),
  "Payments"             = c("java", "sql", "go"),
  "Platform"             = c("go", "python", "yaml"))
SHARED_LANGUAGE <- "unknown"
# Model availability is a tenant-level setting, so the enabled list is org-wide.
ENABLED_MODELS <- c("gpt-5.4", "gpt-5.4-mini", "claude-sonnet-5")
stopifnot(
  all(CORE_FEATURES %in% c(COMPLETION_FEATURE, CHAT_FEATURES)),
  all(unlist(TEAM_FEATURES) %in% CHAT_FEATURES),
  !any(unlist(TEAM_FEATURES) %in% CORE_FEATURES),
  setequal(c(CORE_FEATURES, unlist(TEAM_FEATURES)), c(COMPLETION_FEATURE, CHAT_FEATURES)),
  all(unlist(TEAM_LANGUAGES) %in% LANGUAGES), SHARED_LANGUAGE %in% LANGUAGES,
  all(ENABLED_MODELS %in% MODELS),
  setequal(names(TEAM_FEATURES), teams), setequal(names(TEAM_LANGUAGES), teams),
  all(lengths(TEAM_FEATURES) == 2L), all(lengths(TEAM_LANGUAGES) == 3L))

profile_cells <- lapply(teams, function(tm)
  expand_grid(Feature = c(CORE_FEATURES, TEAM_FEATURES[[tm]]),
              Language = c(TEAM_LANGUAGES[[tm]], SHARED_LANGUAGE),
              Model = ENABLED_MODELS) |>
    mutate(Team = tm, cell_id = row_number()))
names(profile_cells) <- teams
PROFILE_CELLS <- unique(vapply(profile_cells, nrow, integer(1)))
# A developer revisits the whole profile at least once every four weeks, so any
# four consecutive weeks - and therefore the eight-week baseline window - always
# contains the complete cohort profile.
COVERAGE_WEEKS <- 4L
WEEKLY_CELLS <- as.integer(ceiling(PROFILE_CELLS / COVERAGE_WEEKS))
stopifnot(length(PROFILE_CELLS) == 1L, COVERAGE_WEEKS <= 8L,
          length(product_weeks) > COVERAGE_WEEKS)

week_start <- function(d) d - as.POSIXlt(d)$wday
adopters <- latent |> filter(gh_adopted) |>
  select(PersonId, Team, gh_prop, gh_intensity, premium_affinity)
stopifnot(nrow(adopters) >= 10L, all(table(adopters$Team) >= 10L),
          N_DEVELOPERS - nrow(adopters) >= 10L)

message("Building GitHub breakdown allocation ...")
adopter_weeks <- expand_grid(PersonId = adopters$PersonId, Week = product_weeks) |>
  left_join(adopters, by = "PersonId", relationship = "many-to-one") |>
  mutate(
    week_no = match(Week, product_weeks) - 1L,
    # An adopting developer uses Copilot on at least one weekday each week;
    # how many days, and how much, still varies by person and season.
    active_days = pmax(1L, rbinom(n(), 5L, gh_prop)),
    seasonal = 1 + .18 * sin(2 * pi * as.numeric(Week - min(weeks)) / 182),
    week_volume = pmax(WEEKLY_CELLS,
      round(.8 * gh_intensity * active_days * seasonal * rlnorm(n(), 0, .3))))

# Persistent per-person cell order, and a per-week rotation through it.
cell_order <- t(apply(matrix(runif(nrow(adopters) * PROFILE_CELLS),
                             nrow(adopters), PROFILE_CELLS), 1, order))
day_order <- t(apply(matrix(runif(nrow(adopter_weeks) * 5L),
                            nrow(adopter_weeks), 5L), 1, order))
pw_row <- rep(seq_len(nrow(adopter_weeks)), each = WEEKLY_CELLS)
slot <- rep(seq_len(WEEKLY_CELLS) - 1L, times = nrow(adopter_weeks))

feature_weight <- c(code_completion = 3.4, chat_panel_ask_mode = 1.5, chat_inline = 1.0,
                    chat_panel_agent_mode = 1.2, chat_panel_edit_mode = .9,
                    chat_panel_plan_mode = .6, chat_panel_custom_mode = .5,
                    chat_panel_unknown_mode = .4, agent_edit = 1.0,
                    copilot_cli = .7, copilot_app = .5)
language_weight <- setNames(c(.16, .15, .13, .09, .08, .07, .07, .09, .08, .05, .03),
                            LANGUAGES)

cell_rows <- adopter_weeks[pw_row, ] |>
  mutate(
    cell_pos = (week_no * WEEKLY_CELLS + slot) %% PROFILE_CELLS + 1L,
    cell_id = cell_order[cbind(match(PersonId, adopters$PersonId), cell_pos)],
    weekday = day_order[cbind(pw_row, 1L + floor(runif(n()) * active_days))],
    MetricDate = Week + weekday) |>
  left_join(bind_rows(profile_cells), by = c("Team", "cell_id"),
            relationship = "many-to-one") |>
  mutate(weight = as.numeric(feature_weight[Feature]) *
           as.numeric(language_weight[Language]) *
           if_else(Model %in% PREMIUM_MODELS, .3 + 1.6 * premium_affinity,
                   1 - .55 * premium_affinity)) |>
  group_by(PersonId, Week) |>
  mutate(Count = 1L + rpois(n(), (week_volume - WEEKLY_CELLS) * weight / sum(weight))) |>
  ungroup()

breakdown <- cell_rows |>
  group_by(PersonId, MetricDate, Feature, Language, Model) |>
  summarise(Count = as.integer(sum(Count)), .groups = "drop")

# ---------------------------------------------------------------------------
# GitHub Copilot query: PersonGitHubActivityMetrics
#
# The activity file is derived from the allocation above, so the breakdown
# margins reconcile with it by construction. Real exports include explicit zero
# rows for provisioned users on days with no activity, so those are reproduced
# rather than dropped.
# ---------------------------------------------------------------------------
day_totals <- breakdown |>
  group_by(PersonId, MetricDate) |>
  summarise(accepted = as.integer(sum(Count[Feature == COMPLETION_FEATURE])),
            chat = as.integer(sum(Count[Feature != COMPLETION_FEATURE])),
            agent_flag = any(Feature %in% AGENT_FEATURES), .groups = "drop")

gh_raw <- daily_grid |>
  filter(gh_provisioned) |>
  left_join(day_totals, by = c("PersonId", "MetricDate"), relationship = "one-to-one") |>
  mutate(
    accepted = coalesce(accepted, 0L),
    chat = coalesce(chat, 0L),
    agent_flag = coalesce(agent_flag, FALSE),
    # Suggestions are shown far more often than they are accepted, and an
    # adopting developer can be shown suggestions on a day they accept nothing.
    # Those are observed activity days that carry no billable credit row, which
    # is exactly why credit coverage must be resolved separately from activity
    # coverage.
    suggested = accepted + rpois(n(), if_else(gh_adopted, .89 * accepted + .5, 0)))

gh_activity <- gh_raw |>
  transmute(
    PersonId, MetricDate,
    `Agent adoption` = tolower(as.character(agent_flag)),
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
#
# Service adoption, like language and model choice on the GitHub side, is a
# persistent property of a person rather than a fresh daily draw: teams
# standardise on a small set of Copilot surfaces. Re-rolling the service set
# every day gave each person a near-unique service signature, which no
# group-disclosure rule can publish even when every service covers hundreds of
# people.
# ---------------------------------------------------------------------------
services <- tibble(
  ServiceName = c("WorkIQ", "Cowork", "Researcher", "Analyst"),
  ServiceId = vapply(seq_len(4), function(i) make_guid(), character(1)),
  weight = c(.46, .28, .16, .10)
)
CORE_SERVICES <- c("WorkIQ", "Cowork")
TEAM_SERVICES <- list(
  "Data Engineering"     = "Analyst",
  "Developer Experience" = "Researcher",
  "Identity"             = "Researcher",
  "Mobile"               = "Analyst",
  "Payments"             = c("Researcher", "Analyst"),
  "Platform"             = character(0))
stopifnot(setequal(names(TEAM_SERVICES), teams),
          all(CORE_SERVICES %in% services$ServiceName),
          all(unlist(TEAM_SERVICES) %in% services$ServiceName),
          setequal(c(CORE_SERVICES, unlist(TEAM_SERVICES)), services$ServiceName))
service_profiles <- bind_rows(lapply(teams, function(tm)
  tibble(Team = tm, ServiceName = c(CORE_SERVICES, TEAM_SERVICES[[tm]]))))

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

m365_people <- latent |> filter(m365_adopted) |>
  select(PersonId, Team, m365_prop, m365_intensity, credits_per_session)
m365_weeks <- expand_grid(PersonId = m365_people$PersonId, Week = product_weeks) |>
  left_join(m365_people, by = "PersonId", relationship = "many-to-one") |>
  mutate(active_days = pmax(1L, rbinom(n(), 5L, m365_prop)),
         week_row = row_number())
m365_day_order <- t(apply(matrix(runif(nrow(m365_weeks) * 5L),
                                 nrow(m365_weeks), 5L), 1, order))

m365_credits <- m365_weeks |>
  left_join(service_profiles, by = "Team", relationship = "many-to-many") |>
  mutate(service_days = pmax(1L, rbinom(n(), active_days, .55))) |>
  uncount(service_days) |>
  group_by(week_row, ServiceName) |>
  mutate(day_slot = row_number()) |>
  ungroup() |>
  mutate(MetricDate = Week + m365_day_order[cbind(week_row, day_slot)]) |>
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

# Synthetic allocation invariant only, not a verified real-export definition.
# Model assignment to completions is illustrative. Never impose these equalities
# on replacement data without authoritative metric-definition evidence.
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

# ---------------------------------------------------------------------------
# Publication self-check.
#
# A report cannot publish linked breakdown cross-tabs unless every cell, its
# complement inside the developer roster, and every per-person membership
# signature clear the ten-person floor. That is a property of the DATA, so the
# generator asserts it here. It must never be satisfied by relaxing a reader's
# disclosure rule.
# ---------------------------------------------------------------------------
developer_ids <- sort(people$PersonId[people$IsDeveloper])
baseline_breakdown <- breakdown |> filter(MetricDate >= tail(weeks, 8L)[1])
stopifnot(all(baseline_breakdown$PersonId %in% developer_ids),
          all(baseline_breakdown$Count > 0))
assert_releasable <- function(data, groups) {
  cells <- interaction(data[groups], drop = TRUE, lex.order = TRUE)
  memberships <- setNames(rep("", length(developer_ids)), developer_ids)
  for (cell in levels(cells)) {
    observed <- unique(data$PersonId[cells == cell])
    complement <- length(developer_ids) - length(observed)
    if (length(observed) < 10L || (complement > 0L && complement < 10L))
      stop("Breakdown cell below the publication floor: ",
           paste(groups, collapse = " x "), " / ", cell)
    memberships[observed] <- paste0(memberships[observed], "|", cell)
  }
  if (any(table(memberships) < 10L))
    stop("Breakdown membership signature below the publication floor: ",
         paste(groups, collapse = " x "))
  invisible(TRUE)
}
assert_releasable(baseline_breakdown, "Feature")
assert_releasable(baseline_breakdown, c("Model", "Feature"))
assert_releasable(baseline_breakdown, c("Language", "Model"))
assert_releasable(baseline_breakdown, c("Language", "Feature"))
assert_releasable(m365_credits |>
                    filter(MetricDate >= tail(weeks, 8L)[1], PersonId %in% developer_ids),
                  "ServiceName")

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

# The historical key must stay opaque. Every published crosswalk row is checked
# against naive reconstruction from PersonId, including the constant-suffix
# pattern a real download can coincidentally display.
crosswalk <- bind_rows(distinct(m365_credits, PersonId, PeopleHistoricalId),
                       distinct(gh_credits, PersonId, PeopleHistoricalId)) |>
  distinct(PersonId, PeopleHistoricalId)
observed_suffixes <- unique(substr(crosswalk$PeopleHistoricalId,
                                   nchar(crosswalk$PersonId) + 1L,
                                   nchar(crosswalk$PeopleHistoricalId)))
stopifnot(
  nrow(count(crosswalk, PersonId) |> filter(n > 1L)) == 0L,
  !any(startsWith(crosswalk$PeopleHistoricalId, crosswalk$PersonId)),
  !any(mapply(grepl, crosswalk$PersonId, crosswalk$PeopleHistoricalId,
              MoreArgs = list(fixed = TRUE))),
  !any(outer(crosswalk$PersonId, observed_suffixes, paste0) ==
         crosswalk$PeopleHistoricalId)
)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
bundle <- list(
  "person-query/PersonQuery.csv" = pq,
  "consumption-query/PeopleMetaData.csv" = people_metadata,
  "consumption-query/PersonM365CreditsMetrics.csv" = m365_credits,
  "consumption-query/PersonGitHubCreditsMetrics.csv" = gh_credits,
  "github-query/PersonGitHubActivityMetrics.csv" = gh_activity,
  "github-query/GitHubActivityBreakdownByFeatureMetrics.csv" = feature_metrics,
  "github-query/GitHubActivityBreakdownByLanguageFeatureMetrics.csv" = language_feature_metrics,
  "github-query/GitHubActivityBreakdownByLanguageModelMetrics.csv" = language_model_metrics,
  "github-query/GitHubActivityBreakdownByModelFeatureMetrics.csv" = model_feature_metrics
)
message("Validating entire bundle before writing exports ...")
write_query_bundle(bundle, DATA_DIR)

message("Done.")
