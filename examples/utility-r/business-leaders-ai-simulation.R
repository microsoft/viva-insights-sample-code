# A seeded rollout scenario on raw synthetic records, before any aggregation.
# The committed CSVs and the other dashboards' datasets are not changed.
simulate_business_leaders_ai <- function(exports, seed = 20261105L) {
  previous_seed <- get0(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    if (is.null(previous_seed)) rm(".Random.seed", envir = .GlobalEnv) else
      assign(".Random.seed", previous_seed, envir = .GlobalEnv)
  })
  set.seed(seed)
  person_path <- "person-query/PersonQuery.csv"
  github_path <- "github-query/PersonGitHubActivityMetrics.csv"
  feature_path <- "github-query/GitHubActivityBreakdownByFeatureMetrics.csv"
  github_credit_path <- "consumption-query/PersonGitHubCreditsMetrics.csv"
  m365_path <- "consumption-query/PersonM365CreditsMetrics.csv"
  agent_path <- "agent-query/PersonAgentActivityMetrics.csv"
  weeks <- seq(as.Date("2026-04-05"), by = "week", length.out = 13L)
  people <- exports[[person_path]] |>
    distinct(PersonId, Team)
  if (anyDuplicated(people$PersonId)) stop("The scenario requires stable synthetic teams.")
  outsiders <- setdiff(exports[[agent_path]]$PersonId, people$PersonId) |> unique()
  people <- bind_rows(people, data.frame(PersonId = outsiders, Team = "Outside roster"))
  team_shift <- c("Data Engineering" = -2, "Developer Experience" = -2,
                   Identity = 0, Mobile = 1, Payments = 2, Platform = 3,
                   "Outside roster" = 0)
  if (!all(people$Team %in% names(team_shift)))
    stop("Unknown synthetic team in rollout scenario.")
  launch <- function() pmin(13L, pmax(1L,
    sample(1:13, nrow(people), replace = TRUE) + team_shift[people$Team]))
  starts <- data.frame(PersonId = people$PersonId, copilot_start = launch(),
                        github_start = launch(), agent_start = launch())
  starts$cowork_start <- pmin(13L, starts$copilot_start +
                               sample(0:3, nrow(people), replace = TRUE))
  plan <- tidyr::expand_grid(PersonId = people$PersonId, Week = weeks) |>
    left_join(starts, by = "PersonId", relationship = "many-to-one") |>
    mutate(week_no = match(Week, weeks))
  # A brief calendar dip in week 8 and a busier final week affect all products.
  calendar <- c(1, .95, 1.06, 1.02, .94, 1.08, 1, .64, .98, 1.08, 1, .85, 1.13)
  participation <- pmin(.97, .80 * calendar[plan$week_no])
  plan$copilot_on <- plan$week_no >= plan$copilot_start &
    runif(nrow(plan)) < participation
  plan$github_on <- plan$week_no >= plan$github_start &
    runif(nrow(plan)) < participation
  plan$agent_on <- plan$week_no >= plan$agent_start &
    runif(nrow(plan)) < participation
  plan$cowork_on <- plan$copilot_on & plan$week_no >= plan$cowork_start &
    runif(nrow(plan)) < .88
  demand <- (.65 + .055 * plan$week_no) * calendar[plan$week_no]
  plan$demand <- demand * rlnorm(nrow(plan), -.5 * .18^2, .18)
  plan_keys <- paste(plan$PersonId, plan$Week)
  lookup <- function(data, column) {
    i <- match(paste(data$PersonId, week_start_sunday(data$MetricDate)), plan_keys)
    if (anyNA(i)) stop("Scenario record outside the declared population or window.")
    plan[[column]][i]
  }
  scale_count <- function(x, factor) as.integer(pmax(0, round(x * factor)))

  person <- exports[[person_path]]
  in_window <- as.Date(person$MetricDate) %in% weeks
  current <- person[in_window, ]
  on <- lookup(current, "copilot_on")
  factor <- lookup(current, "demand")
  metrics <- c("Total_Copilot_actions_taken", "Total_Copilot_active_days",
                "Cowork_tasks_submitted", "Days_of_active_Cowork_usage")
  missing <- lapply(current[metrics], is.na)
  days <- ifelse(current$Total_Copilot_active_days > 0,
    pmin(5L, pmax(1L, scale_count(current$Total_Copilot_active_days, pmin(1.2, factor)))), 0L)
  current$Total_Copilot_active_days <- ifelse(on, days, 0L)
  current$Total_Copilot_actions_taken <- ifelse(
    on & days > 0, pmax(1L, scale_count(current$Total_Copilot_actions_taken, factor)), 0L)
  cowork_on <- lookup(current, "cowork_on")
  cowork_days <- pmin(current$Total_Copilot_active_days,
                       scale_count(current$Days_of_active_Cowork_usage, factor))
  current$Days_of_active_Cowork_usage <- ifelse(cowork_on, cowork_days, 0L)
  current$Cowork_tasks_submitted <- ifelse(cowork_on & cowork_days > 0,
    pmax(1L, scale_count(current$Cowork_tasks_submitted, factor)), 0L)
  for (metric in metrics) current[[metric]][missing[[metric]]] <- NA
  person[in_window, metrics] <- current[metrics]
  m365 <- exports[[m365_path]]
  m365 <- m365[lookup(m365, "copilot_on"), ]
  factor <- lookup(m365, "demand")
  old_sessions <- m365$`Session count`
  m365$`Session count` <- pmax(1L, scale_count(old_sessions, factor))
  m365$`Total Copilot Credits used` <- round(m365$`Total Copilot Credits used` *
    m365$`Session count` / old_sessions, 7)
  exports[[m365_path]] <- m365
  daily_support <- m365 |>
    mutate(Week = as.character(week_start_sunday(MetricDate))) |>
    group_by(PersonId, Week) |>
    summarise(days = n_distinct(MetricDate), .groups = "drop")
  i <- match(paste(person$PersonId, person$MetricDate),
             paste(daily_support$PersonId, daily_support$Week))
  observed <- !is.na(i) & !is.na(person$Total_Copilot_active_days)
  person$Total_Copilot_active_days[observed] <- pmax(
    person$Total_Copilot_active_days[observed], daily_support$days[i[observed]])
  exports[[person_path]] <- person

  github <- exports[[github_path]]
  factor <- lookup(github, "demand") * lookup(github, "github_on")
  old_volume <- github$`Code completions accepted` + github$`User-initiated chat requests`
  for (metric in c("Code completions accepted", "Code completions suggested",
                    "User-initiated chat requests"))
    github[[metric]] <- scale_count(github[[metric]], factor)
  github_keys <- paste(github$PersonId, github$MetricDate)
  credits <- exports[[github_credit_path]]
  i <- match(paste(credits$PersonId, credits$MetricDate), github_keys)
  if (anyNA(i) || any(old_volume[i] <= 0))
    stop("Synthetic credit records require matching billable GitHub activity.")
  new_volume <- github$`Code completions accepted` + github$`User-initiated chat requests`
  credits$`Total GitHub AI Credits used` <- round(
    credits$`Total GitHub AI Credits used` * new_volume[i] / old_volume[i], 6)
  exports[[github_credit_path]] <- credits[new_volume[i] > 0, ]

  feature_columns <- names(exports[[feature_path]])
  # Largest-remainder allocation keeps integer feature counts equal to daily totals.
  features <- exports[[feature_path]] |>
    left_join(github |> select(PersonId, MetricDate, `Code completions accepted`,
                                `User-initiated chat requests`),
              by = c("PersonId", "MetricDate"), relationship = "many-to-one") |>
    group_by(PersonId, MetricDate, completion = Feature == "code_completion") |>
    mutate(target = if_else(completion, `Code completions accepted`,
                              `User-initiated chat requests`),
           share = target * `Feature Usage Count` / sum(`Feature Usage Count`),
           allocated = floor(share),
           `Feature Usage Count` = as.integer(allocated +
             (rank(-(share - allocated), ties.method = "first") <=
                first(target) - sum(allocated)))) |>
    ungroup() |>
    select(all_of(feature_columns)) |>
    filter(`Feature Usage Count` > 0)
  exports[[feature_path]] <- features
  agent_features <- c("chat_panel_agent_mode", "agent_edit", "copilot_cli", "copilot_app")
  feature_agent_days <- features[features$Feature %in% agent_features, ]
  github$`Agent adoption` <- as.character(github_keys %in%
    paste(feature_agent_days$PersonId, feature_agent_days$MetricDate))
  exports[[github_path]] <- github

  agents <- exports[[agent_path]]
  agents <- agents[lookup(agents, "agent_on"), ]
  agents$`Conversational agent sessions` <- pmax(1L,
    scale_count(agents$`Conversational agent sessions`, lookup(agents, "demand")))
  active <- agents |>
    distinct(PersonId, MetricDate) |>
    mutate(week_no = match(as.Date(MetricDate), weeks))
  histories <- split(active$week_no, active$PersonId)
  # This new-program scenario has no agent use before the first exported week.
  returning <- function(window) vapply(seq_len(nrow(active)), function(i) {
    prior <- histories[[active$PersonId[i]]]
    as.integer(any(prior < active$week_no[i] & prior >= active$week_no[i] - window))
  }, integer(1))
  active$`Returning Agent 365 user (most recent and prior 7 days)` <- returning(1L)
  active$`Returning Agent 365 user (most recent and prior 28 days)` <- returning(4L)
  columns <- names(agents)
  agents <- agents |> select(-starts_with("Returning Agent 365 user")) |>
    left_join(active |> select(-week_no), by = c("PersonId", "MetricDate"),
              relationship = "many-to-one")
  exports[[agent_path]] <- agents[columns]
  exports
}
