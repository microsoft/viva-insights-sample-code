# Runs inside the existing base-R runner; inputs here are committed synthetic fixtures.
utility <- file.path(repo, "examples", "utility-r")
source(file.path(utility, "query-export-contracts.R"))
expect_error <- function(code, pattern) {
  error <- tryCatch({ force(code); NULL }, error = conditionMessage)
  stopifnot(!is.null(error), grepl(pattern, error))
}
bundle <- lapply(names(query_export_headers), function(path)
  read.csv(file.path(utility, "_data", path), check.names = FALSE,
           stringsAsFactors = FALSE))
names(bundle) <- names(query_export_headers)
super_helpers <- new.env()
sys.source(file.path(utility, "copilot-super-panel-helpers.R"), super_helpers)
test("all nine synthetic exports have independently specified ordered headers", {
  stopifnot(length(query_export_headers) == 9L)
  validate_query_bundle(bundle)
})
test("super panel joins all product files without multiplying person-weeks", {
  joined <- super_helpers$build_copilot_super_panel(file.path(utility, "_data"))
  panel <- joined$panel
  person_query <- bundle[["person-query/PersonQuery.csv"]]
  m365 <- bundle[["consumption-query/PersonM365CreditsMetrics.csv"]]
  m365$MetricDate <- as.Date(m365$MetricDate)
  m365$WeekStart <- super_helpers$week_start_sunday(m365$MetricDate)
  expected_services <- aggregate(ServiceName ~ PersonId + WeekStart, m365,
                                 function(x) length(unique(x)))
  service_match <- match(paste(joined$m365_weekly$PersonId,
                               joined$m365_weekly$MetricDate),
                         paste(expected_services$PersonId,
                               expected_services$WeekStart))
  stopifnot(nrow(panel) == nrow(person_query),
            !anyDuplicated(panel[c("PersonId", "MetricDate")]),
            anyDuplicated(m365[c("PersonId", "MetricDate")]) > 0L,
            !anyDuplicated(joined$m365_weekly[c("PersonId", "MetricDate")]),
            !anyDuplicated(joined$github_activity_weekly[c("PersonId", "MetricDate")]),
            !anyDuplicated(joined$github_feature_weekly[c("PersonId", "MetricDate")]),
            max(joined$m365_weekly$m365_observed_days) == 5L,
            max(joined$github_activity_weekly$github_observed_days) == 5L,
            max(joined$github_credits_weekly$github_credit_days) > 1L,
            max(joined$m365_weekly$m365_active_days) > 1L,
            max(joined$github_activity_weekly$github_active_days) > 1L,
            all(!is.na(service_match)),
            all(joined$m365_weekly$m365_service_count ==
                  expected_services$ServiceName[service_match]),
            all(c("m365_credits", "m365_sessions", "github_credits",
                  "github_code_acceptances", "github_chat_requests",
                  "github_feature_usage_count", "github_language_model_usage_count",
                  "github_model_feature_usage_count") %in% names(panel)),
            all(joined$m365_weekly$PersonId %in% person_query$PersonId),
            all(joined$github_activity_weekly$PersonId %in% person_query$PersonId))
})
test("consumption setup derives product mix and enabled population from people", {
  rmd_lines <- readLines(file.path(utility,
                                   "copilot-consumption-ways-of-working-simulation.Rmd"))
  setup_start <- grep("^```\\{r setup", rmd_lines)[1] + 1L
  setup_end <- which(seq_along(rmd_lines) > setup_start &
                       rmd_lines == "```")[1] - 1L
  env <- new.env()
  old <- getwd()
  tryCatch({
    setwd(utility)
    eval(parse(text = rmd_lines[setup_start:setup_end]), env)
  }, finally = setwd(old))
  stopifnot("ProductUsageMix" %in% names(env$person_base),
            setequal(levels(env$person_base$ProductUsageMix),
                     c("Observed both", "Observed M365 only",
                       "Observed GitHub only", "No observed product use")),
            all(table(env$person_base$ProductUsageMix) >= 10L),
            !any(grepl("non.?users?", levels(env$person_base$ProductUsageMix),
                       ignore.case = TRUE)),
            all(env$person_base$weekly_m365_credits[
              env$person_base$m365_observed_weeks > 0] ==
                env$person_base$total_m365_credits[
                  env$person_base$m365_observed_weeks > 0] /
                env$person_base$m365_observed_weeks[
                  env$person_base$m365_observed_weeks > 0]),
            all(env$person_base$active_weeks ==
                  vapply(env$person_base$PersonId, function(id) {
                    sum(with(env$person_week[env$person_week$PersonId == id, ],
                             m365_credits > 0 | m365_sessions > 0),
                        na.rm = TRUE)
                  }, numeric(1))),
            env$overall$Licensed ==
              env$publication_count(sum(env$person_base$M365EnabledUser),
                                    nrow(env$person_base)))
})
test("rename drop extra and order mutations fail before any output is written", {
  for (path in names(bundle)) for (mutation in c("rename", "drop", "extra", "order")) {
    bad <- bundle
    if (mutation == "rename") names(bad[[path]])[3] <- "Agent adoption TYPO"
    if (mutation == "drop") bad[[path]] <- bad[[path]][-3]
    if (mutation == "extra") bad[[path]]$Fabricated <- 0
    if (mutation == "order") bad[[path]] <- bad[[path]][c(2, 1, 3:ncol(bad[[path]]))]
    dest <- file.path(root, "rejected-schema")
    expect_error(write_query_bundle(bad, dest), "header contract")
    stopifnot(!file.exists(dest))
  }
})
test("bundle write validation includes late-file delimiter failures", {
  bad <- bundle
  bad[[9]]$Feature[1] <- "synthetic,invalid"
  dest <- file.path(root, "rejected-delimiter")
  expect_error(write_query_bundle(bad, dest), "delimiter")
  stopifnot(!file.exists(dest))
})
consumption <- new.env()
lines <- readLines(file.path(utility, "copilot-consumption-ways-of-working-simulation.Rmd"))
start <- grep("^```\\{r setup", lines)[1] + 1L
end <- grep("^super <-", lines)[1] - 1L
eval(parse(text = lines[start:end]), consumption)
test("publication floors count distinct people and suppress complementary cells", {
  d <- data.frame(PersonId = sprintf("synthetic-%02d", 1:20),
                  Group = rep(c("A", "B"), each = 10), Credits = 1)
  stopifnot(nrow(consumption$publication_partition(d, "Group", "Credits")) == 20L,
            nrow(consumption$publication_partition(d[-1, ], "Group", "Credits")) == 0L,
            nrow(consumption$publication_partition(rbind(d[1:9, ], d[1:9, ]),
                                                   "Group", "Credits")) == 0L)
  d$Credits[1] <- 0
  stopifnot(nrow(consumption$publication_partition(d, "Group", "Credits")) == 0L)
  stopifnot(is.na(consumption$publication_count(10, 19)),
            consumption$publication_count(10, 20) == 10,
            is.na(consumption$publication_count(9, 20)),
            is.na(consumption$publication_count(0, 9)))
  d$Group[1] <- NA
  stopifnot(nrow(consumption$publication_partition(d, "Group")) == 0L)
  d$PersonId[1] <- NA
  stopifnot(nrow(consumption$publication_partition(d, "Group")) == 0L)
})
test("zero through nine sessions remain limited rather than standard cost", {
  profiles <- consumption$consumption_profile(rep("Lower credit", 12),
                                              c(0:10, NA), rep("Standard cost per session", 12))
  stopifnot(all(profiles[c(1:10, 12)] == "Limited M365 sessions"),
            profiles[11] == "Lower credit, standard cost/session")
})
github <- new.env()
helper <- readLines(file.path(utility, "github-developer-experience-helpers.R"))
eval(parse(text = helper[1:(grep("^super_panel_inputs <-", helper)[1] - 1L)]), github)
test("weekly enabled days override metadata and completeness defaults unknown", {
  p <- data.frame(PersonId = c("synthetic-a", "synthetic-b", "synthetic-c"),
                  Week = as.Date("2026-06-07"), Total_Copilot_enabled_days = c(7, 0, NA),
                  IsCopilotLicensed = TRUE)
  cov <- github$derive_m365_coverage(p)
  stopifnot(identical(cov$M365_eligible, c(TRUE, FALSE, NA)),
            all(is.na(cov$M365_complete)),
            all(is.na(github$observed_or_missing(c(NA, 10, 0),
                                                cov$M365_eligible, cov$M365_complete))))
  evidence <- data.frame(PersonId = p$PersonId, Week = p$Week, Complete = TRUE,
                         Covered_days = 5, Evidence_source = "Independent synthetic test ledger")
  cov <- github$derive_m365_coverage(p, evidence)
  values <- github$observed_or_missing(c(NA, 10, 0), cov$M365_eligible, cov$M365_complete)
  stopifnot(values[1] == 0, all(is.na(values[2:3])))
  evidence$Evidence_source[1] <- ""
  expect_error(github$derive_m365_coverage(p, evidence), "not TRUE")
})
test("GitHub disclosure withholds entire partitions and unsafe complements", {
  stopifnot(nrow(github$disclose_partition(data.frame(People = c(10, 9)))) == 0,
            nrow(github$disclose_partition(data.frame(People = c(10, 10)))) == 2,
            !github$disclosable_split(19, 10), github$disclosable_split(20, 10),
            github$missing_label(NA_character_) == "Unknown / not supplied")
})
test("removed M365 person-week stays unknown and unequal breakdowns remain loadable", {
  copy <- file.path(root, "mutated-fixtures")
  for (path in names(bundle)) {
    target <- file.path(copy, "_data", path)
    dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
    file.copy(file.path(utility, "_data", path), target)
  }
  mpath <- file.path(copy, "_data", "consumption-query", "PersonM365CreditsMetrics.csv")
  m <- read.csv(mpath, check.names = FALSE)
  victim <- m$PersonId[nrow(m)]
  week <- github$week_start(m$MetricDate[nrow(m)])
  m <- m[!(m$PersonId == victim & github$week_start(m$MetricDate) == week), ]
  write.csv(m, mpath, row.names = FALSE, na = "")
  fpath <- file.path(copy, "_data", "github-query", "GitHubActivityBreakdownByFeatureMetrics.csv")
  f <- read.csv(fpath, check.names = FALSE)
  f[["Feature Usage Count"]][1] <- f[["Feature Usage Count"]][1] + 1
  write.csv(f, fpath, row.names = FALSE)
  old <- getwd()
  result <- new.env()
  tryCatch({
    setwd(copy)
    sys.source(file.path(utility, "github-developer-experience-helpers.R"), result)
  }, finally = setwd(old))
  row <- result$m365_product_weeks
  row <- row[row$PersonId == victim & row$Week == week, ]
  stopifnot(nrow(row) == 1L, is.na(row$M365_complete),
            is.na(github$observed_or_missing(NA_real_, row$M365_eligible, row$M365_complete)))
})
test("Consumption keeps missing weeks unknown and GitHub units cannot affect M365 bands", {
  setup_end <- which(seq_along(lines) > start & lines == "```")[1] - 1L
  run_setup <- function(directory) {
    env <- new.env()
    old <- getwd()
    tryCatch({
      setwd(directory)
      eval(parse(text = lines[start:setup_end]), env)
    }, finally = setwd(old))
    env
  }
  baseline <- run_setup(copy)
  removed <- baseline$person_week
  removed <- removed[removed$PersonId == victim & removed$MetricDate == week, ]
  stopifnot(nrow(removed) == 1, is.na(removed$m365_credits), is.na(removed$m365_sessions))
  gpath <- file.path(copy, "_data", "consumption-query", "PersonGitHubCreditsMetrics.csv")
  g <- read.csv(gpath, check.names = FALSE)
  g[["Total GitHub AI Credits used"]] <- g[["Total GitHub AI Credits used"]] * 1000
  write.csv(g, gpath, row.names = FALSE)
  scaled <- run_setup(copy)
  cols <- c("PersonId", "weekly_m365_credits", "CreditQuartile", "CreditBand",
            "ConsumptionProfile", "credits_per_m365_session")
  stopifnot(identical(baseline$person_base[cols], scaled$person_base[cols]),
            all(baseline$person_base$ConsumptionProfile[
              baseline$person_base$total_m365_sessions < 10] == "Limited M365 sessions"))
})
test("overlapping supports reject one-person contributors and membership complements", {
  d <- data.frame(PersonId = sprintf("synthetic-%02d", 1:20),
                  Service = "A", Credits = 1)
  stopifnot(github$support_release_safe(d, "Service", "Credits", d$PersonId))
  overlapping <- rbind(d, transform(d[1:19, ], Service = "B"))
  stopifnot(!github$support_release_safe(overlapping, "Service", "Credits", d$PersonId))
  d$Credits <- c(77, rep(0, 19))
  stopifnot(!github$support_release_safe(d, "Service", "Credits", d$PersonId))
})
load_github <- function(directory) {
  env <- new.env()
  old <- getwd()
  tryCatch({
    setwd(directory)
    sys.source(file.path(utility, "github-developer-experience-helpers.R"), env)
  }, finally = setwd(old))
  env
}
test("developer service totals are invariant to non-developer volume changes", {
  m <- bundle[["consumption-query/PersonM365CreditsMetrics.csv"]]
  m$ServiceName <- "SyntheticService"
  write.csv(m, mpath, row.names = FALSE, na = "")
  before <- load_github(copy)
  stopifnot(nrow(before$m365_service_mix) > 0,
            all(before$m365_service_source$PersonId %in% before$developer_ids))
  other <- !m$PersonId %in% before$developer_ids
  stopifnot(any(other))
  m[["Total Copilot Credits used"]][other] <- m[["Total Copilot Credits used"]][other] * 1000
  m[["Session count"]][other] <- m[["Session count"]][other] * 1000
  write.csv(m, mpath, row.names = FALSE, na = "")
  after <- load_github(copy)
  stopifnot(identical(before$m365_service_mix, after$m365_service_mix))
})
test("all linked HR margins and one-contributor service totals are withheld", {
  ppath <- file.path(copy, "_data", "person-query", "PersonQuery.csv")
  p <- bundle[["person-query/PersonQuery.csv"]]
  ids <- sort(unique(p$PersonId[p$IsDeveloper == "Yes"]))
  stopifnot(length(ids) == 360L)
  index <- match(p$PersonId, ids)
  dev <- !is.na(index)
  p$Team[dev] <- ifelse(index[dev] <= 180, "SyntheticTeamA", "SyntheticTeamB")
  p$Role[dev] <- ifelse(index[dev] <= 90 | (index[dev] > 180 & index[dev] <= 351),
                        "PrivacyRoleA", "PrivacyRoleB")
  write.csv(p, ppath, row.names = FALSE)
  m <- bundle[["consumption-query/PersonM365CreditsMetrics.csv"]]
  m[["Session count"]] <- 0
  m[["Total Copilot Credits used"]] <- 0
  one <- which(m$PersonId %in% ids & as.Date(m$MetricDate) >= as.Date("2026-05-10"))[1]
  m[["Session count"]][one] <- 1
  m[["Total Copilot Credits used"]][one] <- 9876543
  write.csv(m, mpath, row.names = FALSE, na = "")
  g <- bundle[["consumption-query/PersonGitHubCreditsMetrics.csv"]]
  g[["Total GitHub AI Credits used"]] <- 0
  g[["Total GitHub AI Credits used"]][which(g$PersonId %in% ids &
     as.Date(g$MetricDate) >= as.Date("2026-05-10"))[1]] <- 8765432
  write.csv(g, gpath, row.names = FALSE, na = "")
  adversarial <- load_github(copy)
  stopifnot(!adversarial$hr_release_safe, !adversarial$m365_service_safe,
            nrow(adversarial$team_context) == 0,
            nrow(adversarial$composition_counts) == 0,
            nrow(adversarial$role_context) == 0,
            nrow(adversarial$m365_service_mix) == 0,
            !adversarial$credit_release_safe[["GH"]],
            all(is.na(adversarial$team_league$GH_intensity)),
            all(grepl("unavailable", adversarial$product_summary[["Credits / developer / week (mean)"]])))
  if (exists("cli") && "--render-demos" %in% cli) {
    for (name in c("github-copilot-developer-productivity-simulation.Rmd",
                   "github-developer-experience-helpers.R"))
      file.copy(file.path(utility, name), file.path(copy, name), overwrite = TRUE)
    html <- rmarkdown::render(file.path(copy, "github-copilot-developer-productivity-simulation.Rmd"),
                             output_file = "adversarial-publication.html", quiet = TRUE)
    text <- readLines(html, warn = FALSE)
    stopifnot(!any(grepl("PrivacyRoleA|PrivacyRoleB|9,876,543|9876543|8765432|8,765,432", text)),
              any(grepl("Unavailable: independent coverage", text, fixed = TRUE)))
  }
})
test("reuse guidance rejects old coverage equality and real-adapter promises", {
  guide <- readLines(file.path(utility, "copilot-consumption-github-demo-reports.md"))
  stopifnot(!any(grepl("missing row is no provisioning|so they reconcile with", guide)),
            any(grepl("independent|independently", guide)),
            any(grepl("synthetic", guide)),
            any(grepl("credits-only", guide)))
  for (path in c("frontier-analytics/prompts/copilot-dashboards/consumption-dashboard.md",
                 "_pages/frontier-analytics-prompt-consumption-dashboard.md")) {
    prose <- readLines(file.path(repo, path))
    stopifnot(!any(grepl("Build from credits and sessions only", prose)),
              any(grepl("supported M365 credits panels", prose)))
  }
})
stage_fixtures <- function(name) {
  dest <- file.path(root, name)
  for (path in names(bundle)) {
    target <- file.path(dest, "_data", path)
    dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
    stopifnot(file.copy(file.path(utility, "_data", path), target))
  }
  dest
}
# Smallest number of developers sharing an identical pattern of cell membership.
# support_release_safe requires this to reach the privacy floor, which is exactly
# what a fresh per-person-day category draw destroys.
signature_floor <- function(data, groups, parent_ids) {
  cells <- interaction(data[groups], drop = TRUE, lex.order = TRUE)
  memberships <- setNames(rep("", length(parent_ids)), parent_ids)
  for (cell in levels(cells)) {
    observed <- unique(data$PersonId[cells == cell])
    memberships[observed] <- paste0(memberships[observed], "|", cell)
  }
  min(table(memberships))
}
breakdown_groups <- list(
  list(file = "GitHubActivityBreakdownByFeatureMetrics.csv", source = "baseline_feature_daily",
       groups = "Feature", measure = "Feature Usage Count"),
  list(file = "GitHubActivityBreakdownByModelFeatureMetrics.csv", source = "baseline_model_feature_daily",
       groups = c("Model", "Feature"), measure = "Usage count by model and feature"),
  list(file = "GitHubActivityBreakdownByLanguageModelMetrics.csv", source = "baseline_language_model_daily",
       groups = c("Language", "Model"), measure = "Usage count by language and model"),
  list(file = "GitHubActivityBreakdownByLanguageFeatureMetrics.csv", source = "baseline_language_feature_daily",
       groups = c("Language", "Feature"), measure = "Usage count by language and feature"))
test("breakdown margins and cross-tabs are released for the committed fixtures", {
  released <- load_github(stage_fixtures("breakdown-released"))
  mixes <- list(Feature = released$feature_mix, Model = released$model_mix,
                Language = released$language_mix, ModelFeature = released$model_feature_mix,
                LanguageModel = released$language_model_mix,
                LanguageFeature = released$language_feature_mix)
  stopifnot(released$breakdown_safe,
            all(vapply(mixes, nrow, integer(1)) > 0L),
            all(vapply(mixes, function(x) all(x$People >= 10L), logical(1))),
            nrow(released$feature_mix) >= 5L, nrow(released$model_mix) >= 2L,
            nrow(released$language_mix) >= 5L, nrow(released$model_feature_mix) >= 10L,
            nrow(released$language_model_mix) >= 10L,
            nrow(released$language_feature_mix) >= 10L)
  for (spec in breakdown_groups) {
    source <- released[[spec$source]]
    stopifnot(released$support_release_safe(source, spec$groups, spec$measure,
                                            released$developer_ids),
              signature_floor(source, spec$groups, released$developer_ids) >= 10L)
  }
  page <- paste(readLines(file.path(utility,
    "github-copilot-developer-productivity-simulation.html"), warn = FALSE), collapse = "")
  # Only the ranked cross-tab cells are published as HTML text; the margins are
  # published as chart images, so assert against what the tables actually print.
  top_rows <- function(x, n = 15L) x[order(-x$Count), ][seq_len(min(n, nrow(x))), ]
  shown <- c(top_rows(released$model_feature_mix)$Model,
             top_rows(released$model_feature_mix)$Feature,
             top_rows(released$language_model_mix)$Language,
             top_rows(released$language_model_mix)$Model,
             top_rows(released$language_feature_mix, 20L)$Language,
             top_rows(released$language_feature_mix, 20L)$Feature)
  stopifnot(!grepl("GitHub breakdowns unavailable or withheld", page, fixed = TRUE),
            grepl("Largest reportable Model x Feature cells", page, fixed = TRUE),
            grepl("Largest reportable Language x Model cells", page, fixed = TRUE),
            grepl("Largest reportable Language x Feature cells", page, fixed = TRUE),
            length(unique(shown)) >= 6L,
            all(vapply(unique(shown), function(x) grepl(x, page, fixed = TRUE), logical(1))))
})
test("per-person-day category redraws lose the shared support and stay withheld", {
  redrawn <- stage_fixtures("breakdown-redrawn")
  set.seed(20260917L)
  for (spec in breakdown_groups) {
    path <- file.path(redrawn, "_data", "github-query", spec$file)
    d <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
    for (column in spec$groups) {
      vocabulary <- sort(unique(d[[column]]))
      # Each person draws a private subset, then each row draws inside it: the
      # per-person-day allocation the generator used to produce.
      subsets <- lapply(unique(d$PersonId), function(p)
        sample(vocabulary, sample(2:max(2L, length(vocabulary) - 2L), 1L)))
      names(subsets) <- unique(d$PersonId)
      d[[column]] <- vapply(seq_len(nrow(d)),
        function(i) sample(subsets[[d$PersonId[i]]], 1L), character(1))
    }
    keys <- c("PersonId", "MetricDate", spec$groups)
    d <- aggregate(d[spec$measure], d[keys], sum)[, c(keys, spec$measure)]
    write.csv(d, path, row.names = FALSE)
  }
  scrambled <- load_github(redrawn)
  stopifnot(!scrambled$breakdown_safe,
            nrow(scrambled$feature_mix) == 0L, nrow(scrambled$model_mix) == 0L,
            nrow(scrambled$language_mix) == 0L, nrow(scrambled$model_feature_mix) == 0L,
            nrow(scrambled$language_model_mix) == 0L,
            nrow(scrambled$language_feature_mix) == 0L)
  floors <- vapply(breakdown_groups, function(spec)
    signature_floor(scrambled[[spec$source]], spec$groups, scrambled$developer_ids),
    integer(1))
  stopifnot(any(floors < 10L))
})
test("PeopleHistoricalId is opaque and joins only through the activity crosswalk", {
  meta <- bundle[["consumption-query/PeopleMetaData.csv"]]
  columns <- c("PersonId", "PeopleHistoricalId")
  crosswalk <- unique(rbind(
    bundle[["consumption-query/PersonM365CreditsMetrics.csv"]][, columns],
    bundle[["consumption-query/PersonGitHubCreditsMetrics.csv"]][, columns]))
  stopifnot(nrow(crosswalk) >= 10L, !anyDuplicated(crosswalk$PersonId),
            !anyDuplicated(crosswalk$PeopleHistoricalId))
  # Every constant suffix a naive reader could infer from this crosswalk, plus
  # the literal suffix an earlier revision of the generator used.
  suffixes <- c(unique(substr(crosswalk$PeopleHistoricalId,
                             nchar(crosswalk$PersonId) + 1L,
                             nchar(crosswalk$PeopleHistoricalId))), "1784160000")
  stopifnot(length(suffixes) > 1L,
            !any(outer(crosswalk$PersonId, suffixes, paste0) ==
                   crosswalk$PeopleHistoricalId),
            !any(startsWith(crosswalk$PeopleHistoricalId, crosswalk$PersonId)),
            !any(endsWith(crosswalk$PeopleHistoricalId, crosswalk$PersonId)),
            !any(mapply(grepl, crosswalk$PersonId, crosswalk$PeopleHistoricalId,
                        MoreArgs = list(fixed = TRUE))),
            !any(substr(crosswalk$PeopleHistoricalId, 1L,
                        nchar(crosswalk$PersonId)) == crosswalk$PersonId))
  # Reading the crosswalk still resolves every credit row to its metadata row.
  stopifnot(all(crosswalk$PeopleHistoricalId %in% meta$PeopleHistoricalId))
  env <- load_github(stage_fixtures("history-crosswalk"))
  stopifnot(nrow(env$person_history) == nrow(crosswalk),
            all(env$person_history$PeopleHistoricalId %in% meta$PeopleHistoricalId),
            setequal(env$person_history$PersonId, crosswalk$PersonId))
})
test("sparse GitHub credit rows never void observed activity", {
  directory <- stage_fixtures("credit-sparsity")
  env <- load_github(directory)
  weeks <- env$panel
  sparse <- weeks[weeks$GH_activity_valid & !is.na(weeks$GH_credit_days) &
                    weeks$GH_credit_days < 5L, ]
  stopifnot(nrow(sparse) >= 10L, all(sparse$GH_credit_valid),
            all(!is.na(sparse$GH_active_days)), all(!is.na(sparse$GH_suggestions)),
            all(!is.na(sparse$GH_credits)))
  quiet <- weeks[weeks$GH_activity_valid & weeks$GH_billable_days == 0L, ]
  stopifnot(nrow(quiet) >= 10L, all(quiet$GH_credit_valid), all(quiet$GH_credits == 0))
  # Losing the credit rows for an observed billable week unresolves the credit
  # measures only; the observed activity week stays valid.
  billable <- weeks[weeks$GH_activity_valid & !is.na(weeks$GH_credit_days) &
                      weeks$GH_credit_days > 0L & weeks$Week >= env$baseline_start, ]
  victim <- billable$PersonId[1]
  week <- billable$Week[1]
  path <- file.path(directory, "_data", "consumption-query",
                    "PersonGitHubCreditsMetrics.csv")
  credits <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  keep <- !(credits$PersonId == victim & env$week_start(credits$MetricDate) == week)
  stopifnot(any(!keep))
  write.csv(credits[keep, ], path, row.names = FALSE)
  after <- load_github(directory)
  row <- after$panel[after$panel$PersonId == victim & after$panel$Week == week, ]
  developer <- after$baseline[after$baseline$PersonId == victim, ]
  stopifnot(nrow(row) == 1L, row$GH_activity_valid, !row$GH_credit_valid,
            !is.na(row$GH_active_days), !is.na(row$GH_suggestions),
            !is.na(row$GH_accepted), is.na(row$GH_credits),
            nrow(developer) == 1L, developer$GH_all_valid,
            !developer$GH_credits_all_valid, !is.na(developer$GH_active_days),
            is.na(developer$GH_credits))
})

test("no published label asserts non-use from absent M365 data", {
  env <- run_setup(stage_fixtures("consumption-honest-label"))
  segments <- levels(env$person_base$CreditActivitySegment)
  absent <- "No observed M365 credits"
  # The segment and the cost profile share the condition !positive_consumer, so
  # they must share the honest label. "Non-user" would assert non-use from data
  # this report states is not established as complete.
  stopifnot(absent %in% segments,
            !any(grepl("non.?users?", segments, ignore.case = TRUE)),
            sum(env$person_base$CreditActivitySegment == absent) > 0L,
            all(env$person_base$CostPerSessionProfile[
              env$person_base$CreditActivitySegment == absent] %in%
                c(absent, "Limited M365 sessions")),
            !any(grepl("non.?users?", levels(env$person_base$CreditQuartile),
                       ignore.case = TRUE)))
  page <- paste(readLines(file.path(utility,
    "copilot-consumption-ways-of-working-simulation.html"), warn = FALSE),
    collapse = "")
  stopifnot(!grepl("non.?users?", page, ignore.case = TRUE),
            grepl(absent, page, fixed = TRUE))
})
test("credit concentration withholds unless the top group and its complement clear the floor", {
  cs <- consumption$concentration_share
  # A top decile of 4 people is below the floor even though the population is not.
  stopifnot(is.na(cs(rep(1, 40), 0.10)))
  # 100 equal contributors: the top decile holds exactly a tenth of the total.
  stopifnot(abs(cs(rep(1, 100), 0.10) - 0.10) < 1e-9)
  # The complement must clear the floor too, or the remainder is identifiable.
  stopifnot(is.na(cs(rep(1, 100), 0.95)))
  # Zero and non-finite values leave both the ranking and the denominator.
  stopifnot(abs(cs(c(rep(0, 50), rep(1, 100)), 0.10) - 0.10) < 1e-9,
            abs(cs(c(NA, NaN, rep(1, 100)), 0.10) - 0.10) < 1e-9)
  # Concentration tracks skew rather than population size.
  stopifnot(abs(cs(c(rep(100, 10), rep(1, 90)), 0.10) - 1000 / 1090) < 1e-9)
  stopifnot(is.na(cs(rep(1, 9), 0.10)), is.na(cs(numeric(0), 0.10)),
            is.na(cs(rep(0, 100), 0.10)))
})
test("heavy-user pattern reconciles to the population and never publishes a small band", {
  env <- run_setup(stage_fixtures("consumption-intensity"))
  summary <- env$intensity_summary
  base <- env$person_base
  stopifnot(nrow(summary) > 0,
            sum(summary$People) == nrow(base),
            all(summary$People >= env$MIN_GROUP_N),
            all(c("Heavy, sustained", "Heavy, intermittent") %in%
                  as.character(summary$IntensityPattern)))
  heavy <- base[grepl("^Heavy", base$IntensityPattern), ]
  sustained <- base[base$IntensityPattern == "Heavy, sustained", ]
  intermittent <- base[base$IntensityPattern == "Heavy, intermittent", ]
  absent <- base[base$IntensityPattern == "No observed M365 credits", ]
  stopifnot(all(heavy$weekly_m365_credits >= env$heavy_user_cut),
            all(sustained$m365_high_week_share >= env$SUSTAINED_WEEK_SHARE),
            all(intermittent$m365_high_week_share < env$SUSTAINED_WEEK_SHARE),
            all(absent$total_m365_credits == 0),
            # An unobserved week is never counted as a high-credit week.
            all(base$m365_high_weeks <= base$m365_observed_weeks))
  # The published concentration equals an independent recomputation.
  positive <- sort(base$total_m365_credits[base$total_m365_credits > 0],
                   decreasing = TRUE)
  expected <- sum(positive[seq_len(ceiling(0.10 * length(positive)))]) /
    sum(positive)
  reported <- env$concentration_summary$`Top decile`[
    env$concentration_summary$Product == "Microsoft 365 Copilot credits"]
  stopifnot(abs(reported - expected) < 1e-9,
            env$concentration_summary$People[
              env$concentration_summary$Product == "GitHub AI credits"] ==
              sum(base$total_github_credits > 0))
  # A narrower heavy-user cut that isolates a handful of people is withheld in
  # full, including the complementary bands, rather than published in part.
  narrowed <- base
  narrowed$IntensityPattern <- as.character(narrowed$IntensityPattern)
  narrowed$IntensityPattern[which(narrowed$IntensityPattern ==
                                    "Heavy, sustained")[1:3]] <- "Heavy, rare"
  stopifnot(nrow(env$publication_partition(narrowed, "IntensityPattern",
    c("total_m365_credits", "total_m365_sessions"))) == 0)
})
test("crosswalk rejects a shared historical key and an unbacked historical key", {
  mpath_for <- function(directory) file.path(directory, "_data", "consumption-query",
                                             "PersonM365CreditsMetrics.csv")
  shared <- stage_fixtures("crosswalk-shared-key")
  m <- read.csv(mpath_for(shared), check.names = FALSE, stringsAsFactors = FALSE)
  keys <- unique(m[, c("PersonId", "PeopleHistoricalId")])
  stopifnot(nrow(keys) > 1L)
  # Two different people now carry one PeopleHistoricalId. Counting only
  # PeopleHistoricalId per PersonId still sees exactly one each, so this passes
  # a one-directional check while the metadata join is no longer 1:1.
  m$PeopleHistoricalId[m$PersonId == keys$PersonId[2]] <- keys$PeopleHistoricalId[1]
  write.csv(m, mpath_for(shared), row.names = FALSE, na = "")
  expect_error(load_github(shared), "maps to more than one PersonId")
  unbacked <- stage_fixtures("crosswalk-unbacked-key")
  m <- read.csv(mpath_for(unbacked), check.names = FALSE, stringsAsFactors = FALSE)
  meta <- bundle[["consumption-query/PeopleMetaData.csv"]]
  orphan <- "synthetic-history-absent-from-metadata"
  stopifnot(!orphan %in% meta$PeopleHistoricalId)
  # Both cardinalities still hold; the key simply has no metadata row.
  m$PeopleHistoricalId[m$PersonId == keys$PersonId[1]] <- orphan
  write.csv(m, mpath_for(unbacked), row.names = FALSE, na = "")
  expect_error(load_github(unbacked), "absent from PeopleMetaData")
})
test("credit validity compares credit dates, not matching day counts", {
  directory <- stage_fixtures("credit-date-mismatch")
  before <- load_github(directory)
  candidates <- before$panel[before$panel$GH_activity_valid &
                               !is.na(before$panel$GH_billable_days) &
                               before$panel$GH_billable_days > 0L &
                               before$panel$GH_billable_days < 5L, ]
  stopifnot(nrow(candidates) > 0L)
  victim <- candidates$PersonId[1]
  week <- candidates$Week[1]
  activity <- before$gh_daily[before$gh_daily$PersonId == victim &
                                before$gh_daily$Week == week, ]
  billable <- as.character(activity$MetricDate[
    activity[["Code completions accepted"]] > 0 |
      activity[["User-initiated chat requests"]] > 0])
  quiet <- setdiff(as.character(activity$MetricDate), billable)
  stopifnot(length(billable) > 0L, length(quiet) > 0L)
  path <- file.path(directory, "_data", "consumption-query",
                    "PersonGitHubCreditsMetrics.csv")
  credits <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  moved <- which(credits$PersonId == victim & credits$MetricDate %in% billable)[1]
  stopifnot(!is.na(moved))
  # Move one credit row off a billable day onto an observed non-billable day in
  # the same week. The credit-day count and the billable-day count both stay
  # put, so a count comparison still reads 1 == 1 while a billable day has lost
  # its credit row and another date has gained one it should not have.
  credits$MetricDate[moved] <- quiet[1]
  write.csv(credits, path, row.names = FALSE, na = "")
  after <- load_github(directory)
  row_before <- before$panel[before$panel$PersonId == victim &
                               before$panel$Week == week, ]
  row_after <- after$panel[after$panel$PersonId == victim &
                             after$panel$Week == week, ]
  stopifnot(nrow(row_before) == 1L, nrow(row_after) == 1L,
            row_before$GH_credit_valid,
            identical(row_before$GH_credit_days, row_after$GH_credit_days),
            identical(row_before$GH_billable_days, row_after$GH_billable_days),
            !row_after$GH_credit_valid,
            row_after$GH_missing_credit_days == 1L,
            row_after$GH_unexpected_credit_days == 1L,
            is.na(row_after$GH_credits),
            row_after$GH_activity_valid, !is.na(row_after$GH_active_days))
})
