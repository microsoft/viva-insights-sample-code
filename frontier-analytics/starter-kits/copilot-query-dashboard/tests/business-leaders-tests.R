# Invoked by run-tests.R. All inputs are committed synthetic fixtures.
business <- new.env()
sys.source(file.path(utility, "copilot-super-panel-helpers.R"), business)
sys.source(file.path(utility, "business-leaders-ai-simulation.R"), business)
sys.source(file.path(utility, "business-leaders-ai-helpers.R"), business)
business_data <- business$read_business_leaders_data(file.path(utility, "_data"))
business_payload <- business$business_dashboard_payload(business_data)
test("AI union preserves positives across sources and unresolved non-use", {
  result <- business$business_active_union(c(TRUE, FALSE, NA, FALSE),
                                            c(TRUE, TRUE, FALSE, FALSE))
  stopifnot(identical(result, c(TRUE, TRUE, NA, FALSE)))
})
test("business dashboard joins preserve the full scoped weekly roster", {
  panel <- business_data$panel
  stopifnot(nrow(panel) == 480L * 13L,
            !anyDuplicated(panel[c("PersonId", "MetricDate")]),
            length(unique(panel$MetricDate)) == 13L,
            min(panel$MetricDate) == as.Date("2026-04-05"),
            max(panel$MetricDate) + 6L == as.Date("2026-07-04"),
            all(business_data$agents$PersonId %in% panel$PersonId),
            business_data$outside_agents,
            anyNA(panel$Total_Copilot_actions_taken[panel$licensed]),
            all(panel$copilot_active[!panel$licensed] == FALSE),
            !anyNA(panel[c("m365_credits", "m365_sessions", "github_credits")]),
            anyNA(panel$github_active))
})
test("sparse in-scope credit omissions are zero without erasing activity gaps", {
  panel <- business_data$panel
  credits <- business_data$m365
  missing_credit <- !paste(panel$PersonId, panel$MetricDate) %in%
    paste(credits$PersonId, credits$WeekStart)
  stopifnot(any(missing_credit), all(panel$m365_credits[missing_credit] == 0),
            all(panel$m365_sessions[missing_credit] == 0),
            anyNA(panel$Total_Copilot_actions_taken[panel$licensed]),
            anyNA(panel$github_active))
})
test("window users are distinct people rather than person-weeks or service sums", {
  for (n in c(4, 8, 13)) {
    summary <- business_payload$windows[[as.character(n)]]
    panel <- business_data$panel |>
      dplyr::filter(MetricDate %in% tail(business_data$weeks, n))
    expected <- dplyr::n_distinct(panel$PersonId[panel$any_active %in% TRUE])
    stopifnot(summary$population == 480, summary$users == expected,
              nrow(summary$trend) == n,
              summary$users < sum(summary$products$Users, na.rm = TRUE))
    for (week in unique(panel$MetricDate)) {
      expected_week <- sum(panel$any_active[panel$MetricDate == week], na.rm = TRUE)
      actual <- summary$trend$any_active[summary$trend$MetricDate == week]
      stopifnot(actual == business$business_public_count(expected_week, 480))
    }
  }
})
test("weekly licence denominator includes licensed zeros and withholds genuine unknowns", {
  data <- business_data
  latest <- data$panel$MetricDate == max(data$weeks)
  data$panel$licensed[latest] <- TRUE
  data$panel$copilot_active[latest] <- c(rep(TRUE, 470), rep(FALSE, 10))
  known <- business$business_window_summary(data, 4)
  stopifnot(known$licensed == 480, known$adoption == 470 / 480)
  data$panel$copilot_active[which(latest)[1]] <- NA
  unresolved <- business$business_window_summary(data, 4)
  stopifnot(is.na(unresolved$adoption))
})
test("recorded action intensity retains inactive and unresolved people in the roster", {
  panel <- business_data$panel
  small <- head(panel[panel$MetricDate == max(panel$MetricDate), ], 20)
  small$Team <- "Synthetic test group"
  small$agents_active <- small$any_active <- TRUE
  small$copilot_active <- c(rep(TRUE, 10), rep(FALSE, 9), TRUE)
  small$Total_Copilot_actions_taken <- c(rep(2, 10), rep(0, 9), NA)
  small$licensed <- TRUE
  summary <- business$business_group_summary(small, "Team", 10)
  stopifnot(summary$Actions == 1, summary$People == 20, summary$ActionsLowerBound)
})
test("credit units reconcile independently to their raw records", {
  for (n in c(4, 8, 13)) {
    window <- business_payload$windows[[as.character(n)]]
    start <- as.Date(window$start)
    raw_m365 <- business_data$m365
    raw_github <- business$read_demo_export(file.path(utility, "_data"),
                      "consumption-query", "PersonGitHubCreditsMetrics.csv")
    stopifnot(abs(window$m365_credits -
                    sum(raw_m365$`Total Copilot Credits used`[
                      raw_m365$MetricDate >= start])) < 1e-6,
              abs(window$github_credits -
                    sum(raw_github$`Total GitHub AI Credits used`[
                      as.Date(raw_github$MetricDate) >= start])) < 1e-6,
              abs(sum(window$credit_trend$M365) - window$m365_credits) < 1e-6,
              abs(sum(window$credit_trend$GitHub) - window$github_credits) < 1e-6,
              window$m365_credits_per_person_week == window$m365_credits / (480 * n),
              window$github_credits_per_person_week == window$github_credits / (480 * n))
  }
})
test("GitHub acceptance uses pooled received counts rather than mean percentages", {
  summary <- business_payload$windows[["4"]]
  github <- business$read_demo_export(file.path(utility, "_data"),
                                      "github-query", "PersonGitHubActivityMetrics.csv")
  github <- github[as.Date(github$MetricDate) >= as.Date(summary$start), ]
  expected <- sum(github$`Code completions accepted`) /
    sum(github$`Code completions suggested`)
  stopifnot(abs(summary$acceptance - expected) < 1e-6,
            is.na(business$safe_divide(0, 0)))
})
test("returning agent users deduplicate repeated indicators at the latest week", {
  agents <- business_data$agents
  latest <- agents[agents$MetricDate == max(business_data$weeks), ]
  distinct <- latest[!duplicated(latest$PersonId), ]
  indicator <- "Returning Agent 365 user (most recent and prior 7 days)"
  expected <- sum(distinct[[indicator]]) / nrow(distinct)
  stopifnot(anyDuplicated(latest$PersonId) > 0L,
            abs(business_payload$windows[["13"]]$agent_returning - expected) < 1e-6,
            identical(business_payload$windows[["4"]]$agent_returning,
                      business_payload$windows[["13"]]$agent_returning))
})
test("privacy counts distinct contributors and protects small complements", {
  repeated <- data.frame(PersonId = rep(sprintf("synthetic-%02d", 1:9), 20),
                          Credits = 1)
  stopifnot(is.na(business$business_recorded_total(repeated, "Credits", 30)),
            is.na(business$business_public_count(10, 19)),
            business$business_public_count(10, 20) == 10,
            is.na(business$business_public_count(0, 9)))
  expect_error(business$business_dashboard_payload(business_data, 9), "at least ten")
})
test("small Agent categories and a complementary category are absent from payload", {
  for (window in business_payload$windows) {
    stopifnot(window$agents$withheld,
              !"Field service agent" %in% window$agents$rows$Category,
              all(window$agents$rows$Users >= 10),
              sum(window$agents$rows$Volume) <
                sum(business_data$agents$`Conversational agent sessions`[
                  business_data$agents$MetricDate >= as.Date(window$start)]))
  }
})
test("aggregate JSON embeds no person agent or historical identifiers", {
  json <- business$business_payload_json(business_payload)
  ids <- unique(c(business_data$panel$PersonId, business_data$agents$AgentId))
  stopifnot(!any(vapply(ids, function(id) grepl(id, json, fixed = TRUE), logical(1))),
            !grepl("PersonId|AgentId|PeopleHistoricalId", json))
  injected <- business_payload
  injected$windows[["4"]]$groups$Team$Group[1] <- "</script><script>alert(1)</script>"
  encoded <- business$business_payload_json(injected)
  stopifnot(!grepl("</script>", encoded, fixed = TRUE),
            identical(jsonlite::fromJSON(encoded)$windows[["4"]]$groups$Team$Group[1],
                      "</script><script>alert(1)</script>"))
})
test("in-memory exports preserve the shared loader's file-based results", {
  disk <- business$build_copilot_super_panel(file.path(utility, "_data"), FALSE)
  memory <- business$build_copilot_super_panel(exports = business_data$exports,
                                               include_github_breakdowns = FALSE)
  stopifnot(identical(disk, memory))
  expect_error(business$build_copilot_super_panel(exports = list()), "Missing supplied export")
})

fixture_paths <- file.path(utility, "_data", names(business_data$exports))
fixture_hashes <- tools::md5sum(fixture_paths)
rng_before_scenario <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
business_scenario <- business$read_business_leaders_data(file.path(utility, "_data"),
                                                         scenario = TRUE)
scenario_payload <- business$business_dashboard_payload(business_scenario)
test("rollout simulation is reproducible without changing fixtures or caller RNG", {
  stopifnot(identical(rng_before_scenario,
                      get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)),
            identical(business_scenario$exports,
                      business$simulate_business_leaders_ai(business_data$exports)),
            identical(fixture_hashes, tools::md5sum(fixture_paths)))
  old <- business_data$exports[["person-query/PersonQuery.csv"]]
  new <- business_scenario$exports[["person-query/PersonQuery.csv"]]
  metrics <- c("Total_Copilot_actions_taken", "Total_Copilot_active_days",
                "Cowork_tasks_submitted", "Days_of_active_Cowork_usage")
  stopifnot(identical(is.na(old[metrics]), is.na(new[metrics])),
            identical(old[setdiff(names(old), metrics)], new[setdiff(names(new), metrics)]))
})
test("rollout source facts produce growth, a temporary dip and recovery", {
  trend <- scenario_payload$windows[["13"]]$trend
  for (metric in c("copilot_active", "agents_active", "github_active", "any_active")) {
    values <- trend[[metric]]
    stopifnot(max(values, na.rm = TRUE) - min(values, na.rm = TRUE) > 20,
              tail(values[!is.na(values)], 1) > head(values[!is.na(values)], 1),
              values[8] < values[7], values[8] < values[9])
  }
})
test("scenario GitHub features and credits reconcile to daily activity", {
  raw <- business_scenario$exports
  github <- raw[["github-query/PersonGitHubActivityMetrics.csv"]]
  features <- raw[["github-query/GitHubActivityBreakdownByFeatureMetrics.csv"]]
  credits <- raw[["consumption-query/PersonGitHubCreditsMetrics.csv"]]
  allocated <- features |>
    dplyr::group_by(PersonId, MetricDate) |>
    dplyr::summarise(completions = sum(`Feature Usage Count`[Feature == "code_completion"]),
      chats = sum(`Feature Usage Count`[Feature != "code_completion"]),
      agent = any(Feature %in% c("chat_panel_agent_mode", "agent_edit", "copilot_cli",
                                 "copilot_app")), .groups = "drop")
  check <- dplyr::left_join(github, allocated, by = c("PersonId", "MetricDate"),
                             relationship = "one-to-one")
  stopifnot(all(github$`Code completions accepted` <= github$`Code completions suggested`),
    all(check$`Code completions accepted` == dplyr::coalesce(check$completions, 0)),
    all(check$`User-initiated chat requests` == dplyr::coalesce(check$chats, 0)),
    all(tolower(check$`Agent adoption`) == ifelse(dplyr::coalesce(check$agent, FALSE),
                                                 "true", "false")),
    all(features$`Feature Usage Count` > 0),
    all(features$`Feature Usage Count` == floor(features$`Feature Usage Count`)))
  index <- match(paste(credits$PersonId, credits$MetricDate),
                   paste(github$PersonId, github$MetricDate))
  stopifnot(!anyNA(index), all(credits$`Total GitHub AI Credits used` > 0),
    all((github$`Code completions accepted` + github$`User-initiated chat requests`)[index] > 0))
})
test("scenario Person and metered-day facts preserve licence and day constraints", {
  panel <- business_scenario$panel
  daily <- business_scenario$m365 |>
    dplyr::group_by(PersonId, WeekStart) |>
    dplyr::summarise(days = dplyr::n_distinct(MetricDate), .groups = "drop")
  check <- dplyr::left_join(panel, daily, by = c("PersonId", "MetricDate" = "WeekStart"),
                             relationship = "one-to-one")
  stopifnot(all(panel$Total_Copilot_active_days <= 5, na.rm = TRUE),
    all(panel$Days_of_active_Cowork_usage <= panel$Total_Copilot_active_days, na.rm = TRUE),
    all(panel$Total_Copilot_active_days <= panel$Total_Copilot_enabled_days, na.rm = TRUE),
    all((panel$Total_Copilot_actions_taken > 0) == (panel$Total_Copilot_active_days > 0),
        na.rm = TRUE),
    all((panel$Cowork_tasks_submitted > 0) == (panel$Days_of_active_Cowork_usage > 0),
        na.rm = TRUE),
    all(check$Total_Copilot_active_days >= check$days, na.rm = TRUE),
    all(check$licensed[!is.na(check$days)]))
})
test("scenario returning-agent flags are rebuilt from actual weekly participation", {
  agents <- business_scenario$agents
  active <- unique(agents[c("PersonId", "MetricDate")])
  active$week <- match(active$MetricDate, business_scenario$weeks)
  for (weeks in c(1L, 4L)) {
    field <- if (weeks == 1L) "Returning Agent 365 user (most recent and prior 7 days)" else
      "Returning Agent 365 user (most recent and prior 28 days)"
    previous <- lapply(seq_len(nrow(active)), function(i) {
      earlier <- active$week[active$PersonId == active$PersonId[i]]
      as.integer(any(earlier < active$week[i] & earlier >= active$week[i] - weeks))
    })
    index <- match(paste(agents$PersonId, agents$MetricDate),
                     paste(active$PersonId, active$MetricDate))
    stopifnot(all(agents[[field]] == unlist(previous)[index]))
  }
})
test("all windows and groupings use full-roster recorded intensity", {
  for (n in c(4, 8, 13)) {
    window <- scenario_payload$windows[[as.character(n)]]
    panel <- business_scenario$panel |>
      dplyr::filter(MetricDate %in% tail(business_scenario$weeks, n))
    metrics <- c("Total_Copilot_actions_taken", "Cowork_tasks_submitted",
                   "agent_sessions", "github_chat_requests", "m365_sessions")
    expected <- vapply(metrics, function(metric) sum(panel[[metric]], na.rm = TRUE),
                         numeric(1)) / (480 * n)
    stopifnot(window$person_weeks == 480 * n,
      isTRUE(all.equal(unname(expected), window$products$PerPersonWeek)),
      window$m365_credits_per_person_week == sum(panel$m365_credits) / (480 * n),
      window$github_credits_per_person_week == sum(panel$github_credits) / (480 * n),
      window$accepted_per_person_week ==
        sum(panel$github_code_acceptances, na.rm = TRUE) / (480 * n),
      all(window$credit_trend$M365PerPerson == window$credit_trend$M365 / 480),
      all(window$credit_trend$GitHubPerPerson == window$credit_trend$GitHub / 480))
    for (attribute in names(window$groups)) {
      groups <- window$groups[[attribute]]
      for (i in seq_len(nrow(groups))) {
        part <- panel[panel[[attribute]] == groups$Group[i], ]
        if (!is.na(groups$Actions[i]))
          stopifnot(groups$Actions[i] ==
            sum(part$Total_Copilot_actions_taken, na.rm = TRUE) /
              (dplyr::n_distinct(part$PersonId) * n))
      }
    }
    for (name in c("agents", "services", "features")) {
      rows <- window[[name]]$rows
      stopifnot(all(rows$PerPersonWeek == rows$Volume / (480 * n)))
    }
  }
})

test("activity-rate coverage is independent of observed user status", {
  data <- business_scenario
  selected <- data$panel$MetricDate %in% tail(data$weeks, 4)
  data$panel$copilot_active[selected] <- TRUE
  data$panel$cowork_active[selected] <- TRUE
  data$panel$github_active[selected] <- TRUE
  data$panel$github_observed_days[selected] <- 5
  data$panel$github_observed_days[which(selected)[1]] <- 4
  data$panel$github_chat_requests[selected] <- 1
  data$panel$github_code_acceptances[selected] <- 1
  data$panel$github_code_suggestions[selected] <- 2
  licensed <- which(selected & data$panel$licensed)
  data$panel$Total_Copilot_actions_taken[licensed] <- 1
  data$panel$Cowork_tasks_submitted[licensed] <- 1
  data$panel$Total_Copilot_actions_taken[licensed[1]] <- NA
  data$panel$Cowork_tasks_submitted[licensed[1]] <- NA
  window <- business$business_window_summary(data, 4)
  stopifnot(!any(window$products$LowerBound[c(1, 2, 4)]),
            all(window$products$VolumeLowerBound[c(1, 2, 4)]),
            !any(window$products$VolumeLowerBound[c(3, 5)]),
            window$features$lower_bound)
  data$panel$Total_Copilot_actions_taken[selected & data$panel$licensed] <- 0
  data$panel$Cowork_tasks_submitted[selected & data$panel$licensed] <- 0
  data$panel$github_observed_days[selected] <- 5
  complete <- business$business_window_summary(data, 4)
  stopifnot(!any(complete$products$VolumeLowerBound), !complete$features$lower_bound)
})
test("aggregate JSON preserves lower-bound precision before display", {
  payload <- scenario_payload
  payload$windows[["4"]]$products$PerPersonWeek[1] <- 1.9999999
  payload$windows[["4"]]$products$VolumeLowerBound[1] <- TRUE
  encoded <- jsonlite::fromJSON(business$business_payload_json(payload))
  stopifnot(encoded$windows[["4"]]$products$PerPersonWeek[1] <= 1.9999999,
            encoded$windows[["4"]]$products$PerPersonWeek[1] > 1.99)
})

collaboration_fixture <- function() {
  ids <- sprintf("fixture-person-%03d", 1:80)
  panel <- expand.grid(PersonId = ids,
                       MetricDate = seq(as.Date("2026-01-04"), by = "week", length.out = 4),
                       KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  index <- match(panel$PersonId, ids)
  alpha <- c(1L, 21:39, 41:50, 61:70)
  panel$Organization <- ifelse(index %in% alpha, "Alpha", "Beta")
  panel$FunctionType <- panel$Team <- "All"
  panel$m365_credits <- rep(c(rep(0, 20), rep(1, 20), rep(2, 20), rep(3, 20)), 4)
  panel$github_credits <- rep(c(rep(0, 20), rep(3, 20), rep(2, 20), rep(1, 20)), 4)
  panel$Emails_sent <- ifelse(index <= 10, 0, 4)
  panel$Chats_sent <- 3
  panel$Meetings <- ifelse(index <= 10, 0, 2)
  panel$Collaboration_hours <- 1.5
  panel
}

test("group headcounts, users and action rates protect complementary totals", {
  panel <- collaboration_fixture()
  index <- as.integer(sub(".*-", "", panel$PersonId))
  panel$licensed <- TRUE
  panel$any_active <- panel$copilot_active <- panel$cowork_active <- TRUE
  panel$agents_active <- panel$github_active <- panel$metered_active <- TRUE
  panel$Total_Copilot_actions_taken <- 1
  panel$Team <- ifelse(index == 1, "Small", ifelse(index <= 40, "Large A", "Large B"))
  groups <- business$business_group_summary(panel, "Team", 10)
  small <- groups$Group == "Small"
  hidden <- is.na(groups$People)
  stopifnot(is.na(groups$People[small]), sum(hidden) >= 2L,
            all(is.na(groups$Users[hidden])), all(is.na(groups$Reach[hidden])),
            all(is.na(groups$Actions[hidden])))
  panel$Team <- ifelse(index <= 40, "A", "B")
  panel$any_active <- panel$copilot_active <- index <= 5 | index > 40
  panel$cowork_active <- panel$agents_active <- FALSE
  panel$github_active <- panel$metered_active <- FALSE
  groups <- business$business_group_summary(panel, "Team", 10)
  stopifnot(all(is.na(groups$Users)), all(is.na(groups$Reach)))
  panel$any_active <- panel$copilot_active <- TRUE
  panel$Total_Copilot_actions_taken <- as.numeric(index <= 5 | index > 40)
  groups <- business$business_group_summary(panel, "Team", 10)
  stopifnot(all(groups$People == 40), all(groups$Users == 40),
            all(is.na(groups$Actions)),
            business$business_recorded_total(panel, "Total_Copilot_actions_taken", 80) == 180)
})
test("collaboration bands include zero activity and use full person-week denominators", {
  panel <- collaboration_fixture()
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(identical(unlist(result$status), c(M365 = "ready", GitHub = "ready")))
  for (product in c("M365", "GitHub")) {
    bands <- result$scopes$All[[1]][[product]]
    stopifnot(identical(bands$Band, c("none", "low", "medium", "high")),
              identical(bands$People, rep(20L, 4)),
              isTRUE(all.equal(bands$Credit, c(0, 1, 2, 3))),
              isTRUE(all.equal(bands$Emails_sent, c(2, 4, 4, 4))),
              all(bands$Chats_sent == 3), all(bands$Collaboration_hours == 1.5))
  }
  stopifnot(all(result$scopes$All[[1]]$Trend$Emails_sent == 3.5))
})
test("positive credit ties stay together and empty bands remain distinct from zeros", {
  panel <- collaboration_fixture()
  index <- as.integer(sub(".*-", "", panel$PersonId))
  panel$m365_credits <- ifelse(index <= 20, 0, ifelse(index <= 60, 1, 2))
  bands <- business$business_collaboration_window(panel, 10)$scopes$All[[1]]$M365
  stopifnot(identical(bands$People, c(20L, 40L, 0L, 20L)),
            is.na(bands$Credit[3]), bands$Credit[1] == 0,
            bands$Credit[2] == 1, bands$Credit[4] == 2)
  panel$m365_credits <- 0
  bands <- business$business_collaboration_window(panel, 10)$scopes$All[[1]]$M365
  stopifnot(identical(bands$People, c(80L, 0L, 0L, 0L)),
            bands$Credit[1] == 0, all(is.na(bands$Credit[-1])))
  panel$m365_credits <- ifelse(index == 1, 1, 0)
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(result$status$M365 == "privacy", is.null(result$scopes$All[[1]]$M365),
            result$status$GitHub == "ready")
})
test("collaboration missingness stays distinct from zero and protects linked totals", {
  panel <- collaboration_fixture()
  first <- panel$PersonId == "fixture-person-011" & panel$MetricDate == min(panel$MetricDate)
  panel$Emails_sent[first] <- NA_real_
  result <- business$business_collaboration_window(panel, 10)
  bands <- result$scopes$All[[1]]$M365
  stopifnot(abs(bands$Emails_sent[1] - 1.95) < 1e-10,
            bitwAnd(bands$Gap[1], 1L) == 1L,
            bitwAnd(result$scopes$All[[1]]$Trend$Gap[1], 1L) == 1L)
  panel$Chats_sent[as.integer(sub(".*-", "", panel$PersonId)) <= 20] <- NA_real_
  result <- business$business_collaboration_window(panel, 10)
  bands <- result$scopes$All[[1]]$M365
  stopifnot(is.na(bands$Chats_sent[1]), sum(is.na(bands$Chats_sent)) >= 2L)
  panel$m365_credits[first] <- NA_real_
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(result$status$M365 == "incomplete", result$status$GitHub == "ready",
            is.null(result$scopes$All[[1]]$M365),
            !is.null(result$scopes$All[[1]]$GitHub),
            bitwAnd(result$scopes$All[[1]]$Trend$Gap[1], 16L) == 16L)
})
test("collaboration filters protect small cells, margins and weekly complements", {
  panel <- collaboration_fixture()
  result <- business$business_collaboration_window(panel, 10)
  alpha <- result$scopes$Organization$Alpha$M365
  beta <- result$scopes$Organization$Beta$M365
  mask <- rbind(is.na(alpha$People), is.na(beta$People))
  stopifnot(mask[1, 1], mask[2, 2], all(rowSums(mask) != 1L),
            all(colSums(mask) != 1L), any(!mask),
            all(is.na(alpha$Credit[mask[1, ]])),
            all(is.na(beta$Emails_sent[mask[2, ]])))
  index <- as.integer(sub(".*-", "", panel$PersonId))
  panel$Emails_sent <- ifelse(panel$Organization == "Alpha" & index %in% c(1, 21:24), 1,
                              ifelse(panel$Organization == "Beta", 1, 0))
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(all(is.na(result$scopes$Organization$Alpha$Trend$Emails_sent)),
            all(is.na(result$scopes$Organization$Beta$Trend$Emails_sent)),
            all(!is.na(result$scopes$All[[1]]$Trend$Emails_sent)))
  panel$FunctionType <- ifelse(index <= 5, "Rare", "Common")
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(length(result$scopes$FunctionType) == 0L,
            length(result$scopes$Organization) == 2L)
})
test("collaboration filters use latest group membership for the whole window", {
  panel <- collaboration_fixture()
  index <- as.integer(sub(".*-", "", panel$PersonId))
  panel$Team <- ifelse(panel$MetricDate == max(panel$MetricDate),
                       ifelse(index <= 40, "Latest A", "Latest B"), "Old group")
  panel$Emails_sent <- ifelse(index <= 40, 10, 30)
  result <- business$business_collaboration_window(panel, 10)
  stopifnot(setequal(names(result$scopes$Team), c("Latest A", "Latest B")),
            result$scopes$Team[["Latest A"]]$People == 40,
            all(result$scopes$Team[["Latest A"]]$Trend$Emails_sent == 10),
            all(result$scopes$Team[["Latest B"]]$Trend$Emails_sent == 30))
})
test("collaboration rejects invalid values and incomplete person-week rosters", {
  panel <- collaboration_fixture()
  for (value in list(-1, Inf, NaN, "invalid")) {
    bad <- panel
    bad$Emails_sent[1] <- value
    expect_error(business$business_collaboration_window(bad, 10),
                 if (is.numeric(value) && is.nan(value)) "NaN" else "invalid")
  }
  expect_error(business$business_collaboration_window(panel[-1, ], 10), "complete weekly roster")
  expect_error(business$business_collaboration_window(rbind(panel, panel[1, ]), 10),
               "not unique")
})
test("collaboration results reconcile to raw person-weeks in every window and filter", {
  measures <- business$business_collaboration_fields
  for (n in c(4, 8, 13)) {
    panel <- business_scenario$panel |>
      dplyr::filter(MetricDate %in% tail(business_scenario$weeks, n))
    latest <- panel[panel$MetricDate == max(panel$MetricDate), ]
    result <- scenario_payload$windows[[as.character(n)]]$collaboration
    for (product in c("M365", "GitHub")) {
      credit <- business$business_collaboration_credits[[product]]
      totals <- tapply(panel[[credit]], panel$PersonId, sum) / n
      cuts <- quantile(totals[totals > 0], c(1 / 3, 2 / 3), type = 1, names = FALSE)
      band <- ifelse(totals == 0, "none", ifelse(totals <= cuts[1], "low",
                     ifelse(totals <= cuts[2], "medium", "high")))
      check_scope <- function(scope, member_ids) {
        for (i in seq_along(business$business_credit_bands)) {
          ids <- intersect(member_ids, names(band)[band == business$business_credit_bands[i]])
          row <- scope[[product]][i, ]
          if (!is.na(row$People)) stopifnot(row$People == length(ids))
          part <- panel[panel$PersonId %in% ids, ]
          for (metric in c("Credit", measures)) {
            if (!is.na(row[[metric]]) && length(ids)) {
              field <- if (metric == "Credit") credit else metric
              stopifnot(isTRUE(all.equal(row[[metric]], sum(part[[field]]) / (length(ids) * n))))
            }
          }
        }
      }
      check_scope(result$scopes$All[[1]], unique(panel$PersonId))
      for (attribute in c("Organization", "FunctionType", "Team")) {
        for (group in names(result$scopes[[attribute]])) {
          ids <- latest$PersonId[latest[[attribute]] == group]
          scope <- result$scopes[[attribute]][[group]]
          stopifnot(scope$People == length(ids))
          check_scope(scope, ids)
          for (i in seq_len(nrow(scope$Trend))) for (metric in c(measures, credit)) {
            actual <- scope$Trend[[metric]][i]
            if (!is.na(actual)) {
              part <- panel[panel$PersonId %in% ids &
                              panel$MetricDate == scope$Trend$MetricDate[i], ]
              stopifnot(isTRUE(all.equal(actual, sum(part[[metric]]) / length(ids))))
            }
          }
        }
      }
    }
  }
})
test("business renderer resolves relative output from an unrelated working directory", {
  destination <- "business-leaders-relative.html"
  command <- file.path(utility, "render-business-leaders-ai.R")
  result <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(command), shQuote(destination)), stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(result, "status")), file.exists(destination),
            file.info(destination)$size < 200 * 1024)
  html <- paste(readLines(destination, warn = FALSE), collapse = "\n")
  stopifnot(grepl("SYNTHETIC DEMO", html, fixed = TRUE),
            grepl('id="dashboard-data"', html, fixed = TRUE),
            !grepl('src="https?://|href="https?://[^"]+\\.css', html),
            !any(vapply(unique(business_data$panel$PersonId),
                        function(id) grepl(id, html, fixed = TRUE), logical(1))))
})
