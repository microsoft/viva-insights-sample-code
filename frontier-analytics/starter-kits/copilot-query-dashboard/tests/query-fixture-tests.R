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
test("all nine synthetic exports have independently specified ordered headers", {
  stopifnot(length(query_export_headers) == 9L)
  validate_query_bundle(bundle)
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
end <- grep("^people_meta <-", lines)[1] - 1L
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
eval(parse(text = helper[1:(grep("^pq <-", helper)[1] - 1L)]), github)
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
