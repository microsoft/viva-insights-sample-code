# Synthetic dashboard calculations. Source copilot-super-panel-helpers.R first.
suppressPackageStartupMessages(library(dplyr))

business_active_union <- function(...) Reduce(`|`, list(...))

business_public_count <- function(n, population, minimum = 10L) {
  if (population < minimum || is.na(n) ||
      (n > 0 && n < minimum) ||
      (population - n > 0 && population - n < minimum)) return(NA_real_)
  as.numeric(n)
}

business_recorded_total <- function(data, metric, population, minimum = 10L) {
  values <- data[[metric]]
  contributors <- n_distinct(data$PersonId[!is.na(values) & values > 0])
  if (all(is.na(values)) ||
      is.na(business_public_count(contributors, population, minimum))) return(NA_real_)
  sum(values, na.rm = TRUE)
}

business_check_numbers <- function(data, columns, label, allow_missing = FALSE) {
  if (!all(columns %in% names(data))) stop(label, ": required metric columns missing.")
  for (column in columns) {
    x <- data[[column]]
    if (!is.numeric(x) || any(!is.finite(x) & !is.na(x)) ||
        any(x < 0, na.rm = TRUE) || (!allow_missing && anyNA(x)))
      stop(label, ": invalid or unavailable values in ", column)
  }
}

business_collaboration_fields <- c("Emails_sent", "Chats_sent", "Meetings",
                                   "Collaboration_hours")
business_collaboration_credits <- c(M365 = "m365_credits", GitHub = "github_credits")
business_collaboration_measures <- c(business_collaboration_fields,
                                     unname(business_collaboration_credits))
business_credit_bands <- c("none", "low", "medium", "high")

read_business_leaders_data <- function(data_dir = "_data", scenario = FALSE) {
  stopifnot(length(scenario) == 1L, is.logical(scenario), !is.na(scenario))
  # This is the generator's declared product window, not a cadence inferred from rows.
  scope <- as.Date(c("2026-04-05", "2026-07-04"))
  weeks <- seq(scope[1], scope[2], by = "week")
  paths <- c("person-query/PersonQuery.csv", "consumption-query/PeopleMetaData.csv",
    "consumption-query/PersonM365CreditsMetrics.csv",
    "consumption-query/PersonGitHubCreditsMetrics.csv",
    "github-query/PersonGitHubActivityMetrics.csv",
    "github-query/GitHubActivityBreakdownByFeatureMetrics.csv",
    "agent-query/PersonAgentActivityMetrics.csv", "agent-query/AgentMetadata.csv")
  exports <- setNames(lapply(paths, function(path) read_demo_export(data_dir, path)), paths)
  if (scenario) exports <- simulate_business_leaders_ai(exports)
  super <- build_copilot_super_panel(data_dir, include_github_breakdowns = FALSE,
                                      exports = exports)
  person_metrics <- c("Total_Copilot_enabled_days", "Total_Copilot_actions_taken",
                      "Total_Copilot_active_days", "Cowork_tasks_submitted",
                      "Days_of_active_Cowork_usage")
  business_check_numbers(super$panel, person_metrics, "Person query", TRUE)
  business_check_numbers(super$panel, business_collaboration_fields,
                         "Person query collaboration", TRUE)
  if (any(vapply(super$panel[business_collaboration_fields],
                 function(x) any(is.nan(x)), logical(1))))
    stop("Person query collaboration: NaN is not a recorded activity value.")
  if (any(super$panel$Total_Copilot_enabled_days > 7, na.rm = TRUE) ||
      any(super$panel$Total_Copilot_active_days > 7, na.rm = TRUE) ||
      any(super$panel$Days_of_active_Cowork_usage > 7, na.rm = TRUE))
    stop("Person query: weekly day counts exceed seven.")
  panel <- super$panel |> filter(MetricDate %in% weeks)
  scoped_dates <- super$panel$MetricDate[super$panel$MetricDate >= scope[1] &
                                         super$panel$MetricDate <= scope[2]]
  if (any(!scoped_dates %in% weeks))
    stop("Person query does not use the declared Sunday-start weekly calendar.")
  if (nrow(panel) != n_distinct(panel$PersonId) * length(weeks) ||
      anyNA(panel[c("Organization", "FunctionType", "Team")]))
    stop("The synthetic demo requires a full weekly roster and HR attributes.")
  if (any(coalesce(panel$Total_Copilot_enabled_days, 0) == 0 &
          (coalesce(panel$Total_Copilot_actions_taken, 0) > 0 |
           coalesce(panel$Total_Copilot_active_days, 0) > 0 |
           coalesce(panel$Cowork_tasks_submitted, 0) > 0 |
           coalesce(panel$Days_of_active_Cowork_usage, 0) > 0)))
    stop("Conflicting licence and licensed-only activity evidence.")

  m365 <- super$m365_daily_raw
  github_credits <- super$github_credits_daily
  github <- exports[["github-query/PersonGitHubActivityMetrics.csv"]]
  business_check_numbers(m365, c("Session count", "Total Copilot Credits used"),
                         "M365 consumption")
  business_check_numbers(github_credits, "Total GitHub AI Credits used", "GitHub credits")
  business_check_numbers(github, c("Code completions accepted", "Code completions suggested",
                                   "User-initiated chat requests"), "GitHub activity")
  adoption <- tolower(as.character(github$`Agent adoption`))
  if (anyNA(adoption) || any(!adoption %in% c("true", "false", "0", "1")))
    stop("GitHub activity: invalid Agent adoption indicator.")
  if (any(github$`Code completions accepted` > github$`Code completions suggested`))
    stop("GitHub activity: accepted completions exceed suggested completions.")
  assert_unique_key(m365, c("PersonId", "ServiceId", "SpendingPolicyId", "MetricDate"),
                    "M365 consumption")
  policies <- m365 |>
    group_by(PersonId, ServiceId, MetricDate) |>
    summarise(positive = sum(`Total Copilot Credits used` > 0 | `Session count` > 0),
              .groups = "drop")
  if (any(policies$positive > 1))
    stop("Multiple positive spending policies require verified additivity.")

  agents <- exports[["agent-query/PersonAgentActivityMetrics.csv"]] |>
    mutate(MetricDate = as.Date(MetricDate))
  agent_meta <- exports[["agent-query/AgentMetadata.csv"]]
  assert_unique_key(agents, c("PersonId", "AgentId", "MetricDate"), "Agent activity")
  assert_unique_key(agent_meta, "AgentId", "Agent metadata")
  if (!all(agents$AgentId %in% agent_meta$AgentId))
    stop("Agent activity has unmatched agent metadata.")
  returning <- c("Returning Agent 365 user (most recent and prior 7 days)",
                  "Returning Agent 365 user (most recent and prior 28 days)")
  business_check_numbers(agents, c("Conversational agent sessions", returning),
                         "Agent activity")
  if (any(!unlist(agents[returning]) %in% c(0, 1)))
    stop("Agent returning-user fields must be indicators.")
  consistency <- agents |>
    group_by(PersonId, MetricDate) |>
    summarise(across(all_of(returning), n_distinct), .groups = "drop")
  if (any(unlist(consistency[returning]) != 1L))
    stop("Repeated person-level returning indicators disagree across agents.")
  for (feed in list(m365, github_credits, super$github_activity_daily, agents)) {
    if (anyNA(feed$MetricDate) ||
        any(as.Date(feed$MetricDate) < scope[1] | as.Date(feed$MetricDate) > scope[2]))
      stop("Product activity falls outside the declared synthetic window.")
  }
  if (any(!agents$MetricDate %in% weeks) ||
      any(!as.integer(format(as.Date(github$MetricDate), "%w")) %in% 1:5))
    stop("Unexpected Agent cadence or GitHub collection calendar.")
  roster <- unique(panel$PersonId)
  outside <- any(!agents$PersonId %in% roster)
  agents <- agents |>
    filter(PersonId %in% roster) |>
    left_join(agent_meta, by = "AgentId", relationship = "many-to-one")
  agent_weekly <- agents |>
    group_by(PersonId, MetricDate) |>
    summarise(agent_sessions = sum(`Conversational agent sessions`),
              returning_7 = max(.data[[returning[1]]]),
              returning_28 = max(.data[[returning[2]]]), .groups = "drop")
  panel <- panel |>
    left_join(agent_weekly, by = c("PersonId", "MetricDate"), relationship = "one-to-one") |>
    mutate(
      # Only sparse, in-scope credit/session omissions are measured zeros.
      across(c(m365_credits, m365_sessions, github_credits), ~ coalesce(.x, 0)),
      licensed = coalesce(Total_Copilot_enabled_days, 0) > 0,
      copilot_active = licensed &
        (Total_Copilot_actions_taken > 0 | Total_Copilot_active_days > 0),
      cowork_active = licensed &
        (Cowork_tasks_submitted > 0 | Days_of_active_Cowork_usage > 0),
      agent_sessions = coalesce(agent_sessions, 0),
      agents_active = agent_sessions > 0,
      metered_active = coalesce(m365_sessions, 0) > 0 | coalesce(m365_credits, 0) > 0,
      github_active = case_when(
        coalesce(github_active_days, 0) > 0 | coalesce(github_credits, 0) > 0 ~ TRUE,
        github_observed_days == 5 ~ FALSE,
        TRUE ~ NA),
      any_active = business_active_union(copilot_active, cowork_active, agents_active,
                                         metered_active, github_active))
  feature <- exports[["github-query/GitHubActivityBreakdownByFeatureMetrics.csv"]] |>
    mutate(MetricDate = as.Date(MetricDate))
  assert_unique_key(feature, c("PersonId", "MetricDate", "Feature"), "GitHub features")
  business_check_numbers(feature, "Feature Usage Count", "GitHub features")
  if (anyNA(feature$MetricDate) || any(!feature$PersonId %in% roster) ||
      any(feature$MetricDate < scope[1] | feature$MetricDate > scope[2]))
    stop("GitHub features fall outside the roster or declared window.")
  if (!all(m365$PersonId %in% roster) || !all(github_credits$PersonId %in% roster) ||
      !all(github$PersonId %in% roster))
    stop("Consumption or GitHub activity is outside the Person query roster.")
  list(panel = panel, agents = agents, m365 = m365, feature = feature,
       scope = scope, weeks = weeks, outside_agents = outside,
       exports = exports, scenario = scenario)
}

business_people <- function(panel) {
  panel |>
    arrange(MetricDate) |>
    group_by(PersonId) |>
    summarise(across(c(Organization, FunctionType, Team), last),
              across(c(any_active, copilot_active, cowork_active, agents_active,
                        github_active, metered_active), any), .groups = "drop")
}

business_breakdown <- function(data, group, metric, population, minimum, weeks) {
  rows <- data |>
    group_by(Category = .data[[group]]) |>
    summarise(Users = n_distinct(PersonId[.data[[metric]] > 0]),
              Volume = sum(.data[[metric]]), .groups = "drop") |>
    filter(Users > 0)
  safe <- vapply(rows$Users, function(n)
    !is.na(business_public_count(n, population, minimum)), logical(1))
  withheld <- any(!safe)
  # Add a second withheld category so additive totals cannot reveal a small cell.
  if (withheld && any(safe)) {
    candidates <- which(safe)
    safe[candidates[which.min(rows$Users[candidates])]] <- FALSE
  }
  list(rows = rows[safe, ] |>
         mutate(PerPersonWeek = Volume / (population * weeks)) |>
         arrange(desc(Volume)), withheld = withheld)
}

business_group_summary <- function(panel, attribute, minimum) {
  people <- business_people(panel)
  groups <- people |> count(Group = .data[[attribute]], name = "People")
  result <- lapply(groups$Group, function(group) {
    members <- people[people[[attribute]] == group, ]
    part <- panel[panel$PersonId %in% members$PersonId, ]
    n <- nrow(members)
    active <- business_public_count(sum(members$any_active, na.rm = TRUE), n, minimum)
    actions <- business_recorded_total(part, "Total_Copilot_actions_taken", n, minimum)
    data.frame(Group = group, People = business_public_count(n, nrow(people), minimum),
               Users = active, Reach = safe_divide(active, n),
               Actions = safe_divide(actions, nrow(part)),
               LowerBound = anyNA(members$any_active),
               ActionsLowerBound = anyNA(part$Total_Copilot_actions_taken[part$licensed]))
  })
  rows <- bind_rows(result)
  suppress_complement <- function(rows, column, dependent) {
    missing <- which(is.na(rows[[column]]))
    available <- which(!is.na(rows[[column]]))
    if (length(missing) == 1L && length(available)) {
      other <- available[which.min(groups$People[available])]
      rows[other, dependent] <- NA
    }
    rows
  }
  rows[is.na(rows$People), c("Users", "Reach", "Actions")] <- NA
  rows <- suppress_complement(rows, "People", c("People", "Users", "Reach", "Actions"))
  rows <- suppress_complement(rows, "Users", c("Users", "Reach"))
  rows <- suppress_complement(rows, "Actions", "Actions")
  rows
}

business_credit_band_people <- function(panel, minimum) {
  people <- panel |>
    arrange(MetricDate) |>
    group_by(PersonId) |>
    summarise(across(c(Organization, FunctionType, Team), last),
              across(all_of(unname(business_collaboration_credits)),
                     ~ if (anyNA(.x)) NA_real_ else sum(.x) / n()),
              .groups = "drop")
  status <- list()
  for (product in names(business_collaboration_credits)) {
    credit <- business_collaboration_credits[[product]]
    values <- people[[credit]]
    if (anyNA(values)) {
      status[[product]] <- "incomplete"
      people[[paste0("band_", product)]] <- NA_character_
      next
    }
    positive <- values[values > 0]
    cuts <- if (length(positive))
      as.numeric(quantile(positive, c(1 / 3, 2 / 3), type = 1, names = FALSE)) else
      c(NA_real_, NA_real_)
    band <- ifelse(values == 0, "none",
                   ifelse(values <= cuts[1], "low",
                          ifelse(values <= cuts[2], "medium", "high")))
    counts <- table(factor(band, levels = business_credit_bands))
    safe <- vapply(as.numeric(counts), function(n)
      !is.na(business_public_count(n, nrow(people), minimum)), logical(1))
    status[[product]] <- if (all(safe)) "ready" else "privacy"
    people[[paste0("band_", product)]] <- band
  }
  list(people = people, status = status)
}

# Keep at least two suppressed cells in any released row or column margin.
business_complementary_mask <- function(counts, primary, protect_columns = rep(TRUE, ncol(counts))) {
  mask <- primary
  repeat {
    changed <- FALSE
    for (i in seq_len(nrow(counts))) {
      if (sum(mask[i, ]) != 1L) next
      candidates <- which(counts[i, ] > 0 & !mask[i, ])
      if (!length(candidates)) return(NULL)
      j <- candidates[which.min(counts[i, candidates])]
      mask[i, j] <- TRUE
      changed <- TRUE
    }
    for (j in which(protect_columns)) {
      if (sum(mask[, j]) != 1L) next
      candidates <- which(counts[, j] > 0 & !mask[, j])
      if (!length(candidates)) return(NULL)
      i <- candidates[which.min(counts[candidates, j])]
      mask[i, j] <- TRUE
      changed <- TRUE
    }
    if (!changed) return(mask)
  }
}

business_collaboration_gap <- function(part) {
  # Bit positions follow business_collaboration_measures.
  flags <- vapply(part[business_collaboration_measures], anyNA, logical(1))
  as.integer(sum(2L ^ (which(flags) - 1L)))
}

business_collaboration_rates <- function(part, population, minimum) {
  if (!nrow(part)) return(setNames(rep(NA_real_, length(business_collaboration_measures)),
                                   business_collaboration_measures))
  vapply(business_collaboration_measures, function(metric)
    safe_divide(business_recorded_total(part, metric, population, minimum), nrow(part)),
    numeric(1))
}

business_collaboration_window <- function(panel, minimum) {
  assert_unique_key(panel, c("PersonId", "MetricDate"), "Collaboration person-week")
  business_check_numbers(panel, business_collaboration_measures, "Collaboration panel", TRUE)
  if (any(vapply(panel[business_collaboration_measures],
                 function(x) any(is.nan(x)), logical(1))))
    stop("Collaboration panel: NaN is not a recorded value.")
  weeks <- sort(unique(panel$MetricDate))
  if (nrow(panel) != n_distinct(panel$PersonId) * length(weeks))
    stop("Collaboration comparisons require a complete weekly roster.")
  people_result <- business_credit_band_people(panel, minimum)
  people <- people_result$people
  status <- people_result$status
  attributes <- c("Organization", "FunctionType", "Team")

  make_scope <- function(members) {
    part <- panel[panel$PersonId %in% members$PersonId, ]
    n <- nrow(members)
    trend <- bind_rows(lapply(weeks, function(week) {
      week_part <- part[part$MetricDate == week, ]
      rates <- business_collaboration_rates(week_part, n, minimum)
      as.data.frame(c(list(MetricDate = as.Date(week, origin = "1970-01-01")),
                      as.list(rates), list(Gap = business_collaboration_gap(week_part))))
    }))
    scope <- list(People = n, Trend = trend)
    for (product in names(business_collaboration_credits)) {
      if (status[[product]] != "ready") {
        scope[[product]] <- NULL
        next
      }
      scope[[product]] <- bind_rows(lapply(business_credit_bands, function(band) {
        ids <- members$PersonId[members[[paste0("band_", product)]] == band]
        band_part <- part[part$PersonId %in% ids, ]
        rates <- business_collaboration_rates(band_part, length(ids), minimum)
        as.data.frame(c(list(Band = band, People = length(ids),
                             Credit = rates[[business_collaboration_credits[[product]]]]),
                        as.list(rates[business_collaboration_fields]),
                        list(Gap = business_collaboration_gap(band_part))))
      }))
    }
    scope
  }

  scopes <- list(All = list("All people" = make_scope(people)))
  for (attribute in attributes) {
    groups <- sort(unique(people[[attribute]]))
    sizes <- vapply(groups, function(group) sum(people[[attribute]] == group), integer(1))
    if (any(vapply(sizes, function(n)
      is.na(business_public_count(n, nrow(people), minimum)), logical(1)))) {
      scopes[[attribute]] <- list()
    } else {
      scopes[[attribute]] <- setNames(lapply(groups, function(group)
        make_scope(people[people[[attribute]] == group, ])), groups)
    }
  }

  for (product in names(business_collaboration_credits)) {
    if (status[[product]] != "ready") next
    fields <- c("Credit", business_collaboration_fields)
    all_bands <- scopes$All[[1]][[product]]
    all_counts <- matrix(all_bands$People, nrow = 1)
    for (field in fields) {
      primary <- matrix(is.na(all_bands[[field]]) & all_bands$People > 0, nrow = 1)
      mask <- business_complementary_mask(all_counts, primary,
                                           rep(FALSE, length(business_credit_bands)))
      if (is.null(mask)) mask <- all_counts > 0
      all_bands[[field]][mask[1, ]] <- NA_real_
    }
    scopes$All[[1]][[product]] <- all_bands

    for (attribute in attributes) {
      parts <- scopes[[attribute]]
      if (!length(parts)) next
      counts <- do.call(rbind, lapply(parts, function(x) x[[product]]$People))
      primary <- matrix(FALSE, nrow(counts), ncol(counts))
      for (i in seq_len(nrow(counts))) for (j in seq_len(ncol(counts)))
        primary[i, j] <- is.na(business_public_count(counts[i, j], parts[[i]]$People,
                                                       minimum))
      base_mask <- business_complementary_mask(counts, primary)
      if (is.null(base_mask)) base_mask <- counts > 0
      for (i in seq_along(parts)) {
        hidden <- base_mask[i, ]
        parts[[i]][[product]]$People[hidden] <- NA_real_
        parts[[i]][[product]][hidden, c(fields, "Gap")] <- NA
      }
      for (field in fields) {
        values <- do.call(rbind, lapply(parts, function(x) x[[product]][[field]]))
        primary <- base_mask | (is.na(values) & counts > 0)
        globally_hidden <- is.na(all_bands[[field]]) & all_counts[1, ] > 0
        if (any(globally_hidden))
          primary[, globally_hidden] <- counts[, globally_hidden, drop = FALSE] > 0
        mask <- business_complementary_mask(counts, primary, !globally_hidden)
        if (is.null(mask)) mask <- counts > 0
        for (i in seq_along(parts)) parts[[i]][[product]][[field]][mask[i, ]] <- NA_real_
      }
      scopes[[attribute]] <- parts
    }
  }

  # A single hidden group cannot be recovered from the whole-roster weekly total.
  for (attribute in attributes) {
    parts <- scopes[[attribute]]
    if (length(parts) < 2L) next
    for (i in seq_along(weeks)) for (metric in business_collaboration_measures) {
      values <- vapply(parts, function(x) x$Trend[[metric]][i], numeric(1))
      if (sum(is.na(values)) != 1L) next
      candidates <- which(!is.na(values))
      if (length(candidates)) {
        population <- vapply(parts[candidates], `[[`, numeric(1), "People")
        other <- candidates[which.min(population)]
        parts[[other]]$Trend[[metric]][i] <- NA_real_
      }
    }
    scopes[[attribute]] <- parts
  }
  list(status = status, scopes = scopes)
}

business_window_summary <- function(data, lookback, minimum = 10L) {
  weeks <- tail(data$weeks, lookback)
  panel <- data$panel |> filter(MetricDate %in% weeks)
  people <- business_people(panel)
  population <- nrow(people)
  public <- function(x) business_public_count(sum(x, na.rm = TRUE), population, minimum)
  total <- function(metric, part = panel)
    business_recorded_total(part, metric, population, minimum)
  trend <- panel |>
    group_by(MetricDate) |>
    summarise(across(c(any_active, copilot_active, cowork_active, agents_active,
                        github_active, metered_active), public), .groups = "drop")
  latest <- panel |> filter(MetricDate == max(weeks))
  latest_agents <- latest |> filter(agents_active)
  returning_population <- nrow(latest_agents)
  returning_n <- business_public_count(sum(latest_agents$returning_7),
                                       returning_population, minimum)
  licensed <- business_public_count(sum(latest$licensed), population, minimum)
  eligible <- latest |> filter(licensed)
  adoption <- if (!anyNA(eligible$copilot_active) && nrow(eligible) > 0)
    safe_divide(business_public_count(sum(eligible$copilot_active), nrow(eligible),
                                     minimum), nrow(eligible)) else NA_real_
  m365 <- data$m365 |> filter(MetricDate >= min(weeks), MetricDate <= max(weeks) + 6L)
  agents <- data$agents |> filter(MetricDate %in% weeks)
  feature <- data$feature |> filter(MetricDate >= min(weeks), MetricDate <= max(weeks) + 6L)
  agent_breakdown <- business_breakdown(agents, "AgentName", "Conversational agent sessions",
                                         population, minimum, length(weeks))
  credit_breakdown <- business_breakdown(m365, "ServiceName", "Total Copilot Credits used",
                                          population, minimum, length(weeks))
  feature_breakdown <- business_breakdown(feature, "Feature", "Feature Usage Count",
                                           population, minimum, length(weeks))
  github_incomplete <- any(is.na(panel$github_observed_days) | panel$github_observed_days < 5)
  feature_breakdown$lower_bound <- github_incomplete
  accepted <- total("github_code_acceptances")
  suggested <- total("github_code_suggestions")
  products <- data.frame(
    Product = c("M365 Copilot", "Cowork", "Agents", "GitHub Copilot", "Metered M365 services"),
    Users = c(public(people$copilot_active), public(people$cowork_active),
              public(people$agents_active), public(people$github_active),
              public(people$metered_active)),
    Volume = c(total("Total_Copilot_actions_taken"), total("Cowork_tasks_submitted"),
               total("agent_sessions"), total("github_chat_requests"), total("m365_sessions")),
    Unit = c("actions", "tasks submitted", "sessions", "chat requests", "sessions"),
    LowerBound = c(anyNA(people$copilot_active), anyNA(people$cowork_active),
                   FALSE, anyNA(people$github_active), FALSE),
    VolumeLowerBound = c(anyNA(panel$Total_Copilot_actions_taken[panel$licensed]),
                         anyNA(panel$Cowork_tasks_submitted[panel$licensed]),
                         FALSE, github_incomplete, FALSE))
  products$PerPersonWeek <- products$Volume / nrow(panel)
  list(
    lookback = lookback, start = as.character(min(weeks)),
    end = as.character(max(weeks) + 6L), population = population, person_weeks = nrow(panel),
    users = public(people$any_active), lower_bound = anyNA(people$any_active),
    licensed = licensed, adoption = adoption,
    m365_credits = total("m365_credits"), github_credits = total("github_credits"),
    m365_credits_per_person_week = safe_divide(total("m365_credits"), nrow(panel)),
    github_credits_per_person_week = safe_divide(total("github_credits"), nrow(panel)),
    accepted = accepted, suggested = suggested,
    accepted_per_person_week = safe_divide(accepted, nrow(panel)),
    acceptance = safe_divide(accepted, suggested),
    agent_returning = safe_divide(returning_n, returning_population),
    returning_week = as.character(max(weeks)),
    products = products, trend = trend,
    credit_trend = panel |>
      group_by(MetricDate) |>
      summarise(M365 = total("m365_credits", pick(everything())),
                GitHub = total("github_credits", pick(everything())),
                M365PerPerson = M365 / n(), GitHubPerPerson = GitHub / n(),
                .groups = "drop"),
    groups = setNames(lapply(c("Organization", "FunctionType", "Team"), function(attribute)
      business_group_summary(panel, attribute, minimum)),
      c("Organization", "FunctionType", "Team")),
    agents = agent_breakdown, services = credit_breakdown, features = feature_breakdown,
    collaboration = business_collaboration_window(panel, minimum))
}

business_dashboard_payload <- function(data, minimum = 10L) {
  if (length(minimum) != 1 || is.na(minimum) || minimum < 10 || minimum != floor(minimum))
    stop("The dashboard privacy minimum must be an integer of at least ten.")
  if (n_distinct(data$panel$PersonId) < minimum)
    stop("The roster is below the dashboard privacy minimum.")
  list(synthetic = TRUE, scenario = data$scenario,
       minimum = minimum, outside_agents = data$outside_agents,
       generated = format(Sys.time(), "%Y-%m-%d %H:%M UTC", tz = "UTC"),
       windows = setNames(lapply(c(4L, 8L, 13L), function(n)
         business_window_summary(data, n, minimum)), c("4", "8", "13")))
}

business_payload_json <- function(payload) {
  # Compact only the collaboration tables; their row and column order is fixed.
  for (window in names(payload$windows)) {
    item <- payload$windows[[window]]
    for (attribute in names(item$collaboration$scopes)) {
      scopes <- item$collaboration$scopes[[attribute]]
      for (group in names(scopes)) {
        scope <- scopes[[group]]
        if (!identical(as.character(scope$Trend$MetricDate),
                       as.character(item$trend$MetricDate)))
          stop("Collaboration and dashboard weekly dates do not align.")
        scope$Trend <- lapply(seq_len(nrow(scope$Trend)), function(i)
          unname(as.list(scope$Trend[i, c(business_collaboration_measures, "Gap")])))
        for (product in names(business_collaboration_credits)) {
          rows <- scope[[product]]
          if (is.null(rows)) next
          if (!identical(rows$Band, business_credit_bands))
            stop("Unexpected collaboration credit-band order.")
          scope[[product]] <- lapply(seq_len(nrow(rows)), function(i)
            unname(as.list(rows[i, c("People", "Credit", business_collaboration_fields, "Gap")])))
        }
        scopes[[group]] <- scope
      }
      item$collaboration$scopes[[attribute]] <- scopes
    }
    payload$windows[[window]] <- item
  }
  json <- jsonlite::toJSON(payload, auto_unbox = TRUE, dataframe = "rows",
                            na = "null", digits = NA)
  # JSON is embedded in an inert script element, including analyst-provided labels.
  gsub("<", "\\u003c", json, fixed = TRUE)
}
