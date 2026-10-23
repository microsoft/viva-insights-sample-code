# Base-R regression tests. All generated data below is synthetic.
# Usage: Rscript tests/run-tests.R <existing-private-test-output-parent> [--render-demos]
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
kit <- normalizePath(file.path(dirname(script), ".."), winslash = "/", mustWork = TRUE)
source(file.path(kit, "lib", "runtime.R"))
cli <- commandArgs(trailingOnly = TRUE)
if (!length(cli)) stop("Supply an existing test-output parent outside the repository.")
parent <- normalizePath(cli[1], winslash = "/", mustWork = TRUE)
repo <- normalizePath(file.path(kit, "..", "..", ".."), winslash = "/", mustWork = TRUE)
if (inside(parent, repo)) stop("Test output must be outside the repository.")
root <- file.path(parent, paste0("dashboard-tests-", format(Sys.time(), "%Y%m%d-%H%M%S")))
stopifnot(!file.exists(root), dir.create(root))
setwd(root)
need("jsonlite")
passed <- character()
test <- function(name, code) {
  force(code)
  passed <<- c(passed, name)
  cat("PASS:", name, "\n")
}
blocked <- function(code, pattern) {
  err <- tryCatch({ force(code); NULL }, dashboard_error = function(e) conditionMessage(e))
  stopifnot(!is.null(err), grepl(pattern, err, ignore.case = TRUE))
}
people <- data.frame(PeopleHistoricalId = sprintf("synthetic-history-%02d", 1:20),
                     IsCopilotLicensed = rep(c("true", "false"), 10),
                     Organization = rep(c("Synthetic Alpha", "Synthetic Beta"), each = 10),
                     check.names = FALSE)
activity <- expand.grid(PersonId = sprintf("synthetic-person-%02d", 1:20),
                         ServiceId = c("Synthetic Service A", "Synthetic Service B"),
                         MetricDate = c("2026-06-01", "2026-06-02"), stringsAsFactors = FALSE)
activity$PeopleHistoricalId <- people$PeopleHistoricalId[match(activity$PersonId,
                                                            sprintf("synthetic-person-%02d", 1:20))]
activity[["Total Copilot Credits used"]] <- rep(1:20, 4)
activity[["User limit"]] <- 999
activity[["Spending policy limit"]] <- 99999
write.csv(activity, "synthetic-activity.csv", row.names = FALSE)
write.csv(people, "synthetic-people.csv", row.names = FALSE)
cfg <- list(report = "consumption", mode = "real", repo_root = repo,
            output_dir = file.path(root, "success"),
            inputs = list(activity = "synthetic-activity.csv", people = "synthetic-people.csv"),
            mappings = list(activity = list(person_id = "PersonId", service_id = "ServiceId",
                                           date = "MetricDate", people_id = "PeopleHistoricalId",
                                           credits = "Total Copilot Credits used"),
                            people = list(people_id = "PeopleHistoricalId",
                                          licensed = "IsCopilotLicensed", group = "Organization")),
            mapping_approval = list(approved = TRUE, evidence = "Synthetic regression fixture contract"),
            granularity = "day", group = TRUE, privacy_min = 10)
write_json(cfg, "config.json")
loaded <- read_config("config.json", kit)
tables <- lapply(loaded$inputs, read_csv)
valid <- validate_consumption(loaded, tables)
test("relative input paths resolve from config, not working directory", {
  stopifnot(loaded$inputs$activity == file.path(root, "synthetic-activity.csv"))
})
test("credits aggregate by explicit period/service/group with distinct people", {
  agg <- aggregate_credits(valid, 10)
  stopifnot(agg$status == "ready", nrow(agg$table) == 8,
            sum(agg$table$credits) == 840, all(agg$table$people == 10),
            setequal(agg$table$credits, c(55, 155)))
})
test("real build independently produces self-contained aggregate dashboard", {
  run_dashboard("build", "config.json", kit)
  stopifnot(file.exists(file.path(loaded$output_dir, "build", "dashboard.html")),
            file.exists(file.path(loaded$output_dir, "build", "provenance.json")))
  prov <- jsonlite::read_json(file.path(loaded$output_dir, "build", "provenance.json"))
  stopifnot(identical(prov$synthetic, FALSE), nchar(prov$repository_revision) == 40,
            length(prov$input_hashes) == 2)
})
test("no invented tokens, sessions, currency or policy-limit totals in aggregates", {
  csv <- read.csv(file.path(loaded$output_dir, "build", "aggregates.csv"))
  stopifnot(identical(names(csv), c("date", "service", "group", "people", "credits")),
            !any(grepl("synthetic-person|synthetic-history", unlist(csv))))
})
test("inspect independent of build, no customer rows or values printed", {
  text <- capture.output(run_dashboard("inspect", "config.json", kit))
  stopifnot(!any(grepl("Synthetic Alpha|Synthetic Beta|synthetic-person|synthetic-history|Synthetic Service", text)))
})
test("unapproved inspection validates structure and provides scope before user approval", {
  for (approval in list(NULL, list(approved = FALSE, evidence = "Pending decision"),
                       list(approved = TRUE, evidence = " "))) {
    pending <- loaded
    pending$mapping_approval <- approval
    pending$output_dir <- file.path(root, paste0("pending-", length(list.files(root))))
    capture.output(report <- inspect_dashboard(pending, kit))
    stopifnot(report$status == "awaiting_approval", report$structural_validation == "passed",
              report$privacy_status == "ready", length(report$supported_panels) == 2L,
              length(report$excluded_panels) >= 4L,
              grepl("Only after their decision", report$next_action))
    blocked(build_dashboard(pending, kit), "approval")
    stopifnot(!dir.exists(file.path(pending$output_dir, "build")))
  }
})
test("unapproved inspection reports key errors instead of an approval blocker", {
  duplicate <- rbind(activity, activity[1, ])
  write.csv(duplicate, "unapproved-duplicate.csv", row.names = FALSE)
  pending <- loaded
  pending$mapping_approval <- NULL
  pending$inputs$activity <- file.path(root, "unapproved-duplicate.csv")
  pending$output_dir <- file.path(root, "pending-invalid")
  capture.output(report <- inspect_dashboard(pending, kit))
  stopifnot(report$status == "blocked", grepl("Duplicate activity key", report$blocker),
            grepl("structural validation", report$next_action))
})
test("inspection excludes all panels if privacy fails before approval", {
  write.csv(activity[-1, ], "unapproved-small.csv", row.names = FALSE)
  pending <- loaded
  pending$mapping_approval <- NULL
  pending$inputs$activity <- file.path(root, "unapproved-small.csv")
  pending$output_dir <- file.path(root, "pending-small")
  capture.output(report <- inspect_dashboard(pending, kit))
  stopifnot(report$status == "awaiting_approval", report$privacy_status == "suppressed",
            length(report$supported_panels) == 0L,
            grepl("approval does not override privacy", report$next_action))
})
test("real HTML header records selected observed and generation windows", {
  selected <- loaded
  selected$dates <- list(start = "2026-05-01", end = "2026-06-30")
  html <- dashboard_html(aggregate_credits(valid, 10), selected, "2026-09-11T16:00:00Z")
  stopifnot(grepl("2026-05-01 to 2026-06-30", html),
            grepl("Observed period window:</strong> 2026-06-01 to 2026-06-02", html, fixed = TRUE),
            grepl("Generated (UTC):</strong> 2026-09-11T16:00:00Z", html, fixed = TRUE))
  withheld <- dashboard_html(aggregate_credits(valid[-1, ], 10), selected)
  stopifnot(grepl("Observed period window:</strong> Withheld", withheld, fixed = TRUE),
            grepl("2026-05-01 to 2026-06-30", withheld, fixed = TRUE),
            !grepl("2026-06-01|2026-06-02|Synthetic Alpha|Synthetic Beta|Synthetic Service", withheld))
  empty <- dashboard_html(aggregate_credits(valid[FALSE, ], 10), selected)
  stopifnot(grepl("No observations in the selected window", empty, fixed = TRUE))
  prov <- jsonlite::read_json(file.path(loaded$output_dir, "build", "provenance.json"))
  generated <- paste(readLines(file.path(loaded$output_dir, "build", "dashboard.html")), collapse = "")
  stopifnot(grepl(prov$generated_utc, generated, fixed = TRUE))
})
test("output overwrite is refused", blocked(run_dashboard("build", "config.json", kit), "already exists"))
test("duplicate activity keys rejected", {
  bad <- tables
  bad$activity <- rbind(bad$activity, bad$activity[1, ])
  blocked(validate_consumption(loaded, bad), "Duplicate activity")
})
test("duplicate metadata rejected rather than keeping first", {
  bad <- tables
  bad$people <- rbind(bad$people, bad$people[1, ])
  blocked(validate_consumption(loaded, bad), "Duplicate PeopleHistoricalId")
})
test("unmatched metadata rejected", {
  bad <- tables
  bad$people <- bad$people[-1, ]
  blocked(validate_consumption(loaded, bad), "unmatched")
})
test("one historical key cannot represent two people", {
  bad <- tables
  bad$activity$PeopleHistoricalId[2] <- bad$activity$PeopleHistoricalId[1]
  blocked(validate_consumption(loaded, bad), "multiple PersonIds")
})
test("negative, missing, NaN, infinite and nonnumeric credits rejected", {
  for (value in c("-1", NA_character_, "NaN", "Inf", "1e999", "invalid", "1,000")) {
    bad <- tables
    bad$activity[["Total Copilot Credits used"]][1] <- value
    blocked(validate_consumption(loaded, bad), "Credits")
  }
})
test("invalid dates rejected without coercion", {
  for (value in c("2026-02-30", "06/01/2026", "", "not a date", NA_character_)) {
    bad <- tables
    bad$activity$MetricDate[1] <- value
    blocked(validate_consumption(loaded, bad), "Dates")
  }
})
test("missing IDs and group attributes rejected", {
  for (field in c("PersonId", "ServiceId", "PeopleHistoricalId")) {
    bad <- tables
    bad$activity[[field]][1] <- NA_character_
    blocked(validate_consumption(loaded, bad), "identifiers")
  }
  bad <- tables
  bad$people$Organization[1] <- ""
  blocked(validate_consumption(loaded, bad), "attributes")
})
test("licensing never inferred from positive activity", {
  bad <- tables
  bad$people$IsCopilotLicensed[1] <- NA_character_
  blocked(validate_consumption(loaded, bad), "licens")
})
test("unapproved, incomplete or ambiguous mappings block build", {
  bad <- loaded
  bad$mapping_approval$approved <- FALSE
  blocked(validate_consumption(bad, tables), "approval")
  bad <- loaded
  bad$mappings$activity$credits <- "No such column"
  blocked(validate_consumption(bad, tables), "absent")
  bad$mappings$activity$credits <- "PersonId"
  blocked(validate_consumption(bad, tables), "same source")
})
test("granularity required and preserved rather than guessed", {
  bad <- loaded
  bad$granularity <- NULL
  blocked(validate_consumption(bad, tables), "granularity")
  for (grain in c("day", "week", "month")) {
    bad$granularity <- grain
    stopifnot(identical(validate_consumption(bad, tables)$date, valid$date))
  }
})
test("privacy uses distinct people, not records or historical versions", {
  repeated <- valid[rep(1, 20), ]
  stopifnot(aggregate_credits(repeated, 10)$status == "suppressed")
})
test("one small cell withholds every cell and every composition label", {
  small <- valid[-1, ]
  out <- aggregate_credits(small, 10)
  stopifnot(out$status == "suppressed", nrow(out$table) == 0)
  html <- dashboard_html(out, loaded)
  stopifnot(!grepl("Synthetic Alpha|Synthetic Beta|Synthetic Service", html))
})
test("empty date window produces no invented zeros", {
  bad <- loaded
  bad$dates <- list(start = "2025-01-01", end = "2025-01-31")
  out <- aggregate_credits(validate_consumption(bad, tables), 10)
  stopifnot(out$status == "no_data", nrow(out$table) == 0)
})
test("normalization collisions and malformed rows rejected", {
  writeLines(c("PersonId,Person_Id", "a,b"), "collision.csv")
  blocked(read_csv("collision.csv"), "collision")
  writeLines(c("a,b", "1,2,3"), "bad-width.csv")
  blocked(read_csv("bad-width.csv"), "header width")
})
test("real GitHub inspect only with actionable build blocker", {
  gh <- loaded
  gh$report <- "github"
  gh$dates <- gh$group <- gh$granularity <- gh$mappings <- gh$mapping_approval <- NULL
  gh$output_dir <- file.path(root, "github-inspect")
  report <- inspect_dashboard(gh, kit)
  stopifnot(report$status == "inspect_only", grepl("unverified", report$blocker))
  blocked(build_dashboard(gh, kit), "unsupported")
})
test("real GitHub inspection rejects unapplied analytical settings", {
  for (field in c("dates", "group", "granularity", "mappings", "mapping_approval")) {
    gh <- list(report = "github", mode = "real", repo_root = repo,
               output_dir = file.path(root, paste0("github-", field)),
               inputs = list(activity = "synthetic-activity.csv"), privacy_min = 10)
    gh[[field]] <- switch(field,
      dates = list(start = "2026-06-01", end = "2026-06-02"),
      group = TRUE,
      granularity = "day",
      mappings = cfg$mappings,
      mapping_approval = cfg$mapping_approval)
    write_json(gh, paste0("github-", field, ".json"))
    blocked(read_config(paste0("github-", field, ".json"), kit), "does not accept analytical")
  }
})
test("real mode cannot reproduce and demo mode cannot build", {
  blocked(reproduce_dashboard(loaded, kit), "demo-only")
  demo <- loaded
  demo$mode <- "demo"
  blocked(build_dashboard(demo, kit), "real-only")
  write_json(demo, "mixed-config.json")
  blocked(read_config("mixed-config.json", kit), "does not accept")
})
test("privacy below ten and output inside source repo rejected", {
  bad <- cfg
  bad$privacy_min <- 9
  write_json(bad, "bad-privacy.json")
  blocked(read_config("bad-privacy.json", kit), ">= 10")
  bad <- cfg
  bad$output_dir <- file.path(repo, "do-not-create")
  write_json(bad, "bad-output.json")
  blocked(read_config("bad-output.json", kit), "outside")
  stopifnot(!dir.exists(bad$output_dir))
})
test("nonexistent output tails cannot traverse into the source repository", {
  outside <- file.path(dirname(repo), paste0("uncreated-dashboard-path-", Sys.getpid()))
  stopifnot(!file.exists(outside))
  bad <- cfg
  bad$output_dir <- file.path(outside, "new-folder", "..", "..", basename(repo))
  write_json(bad, "traversal-output.json")
  blocked(read_config("traversal-output.json", kit), "dot segment|outside")
  stopifnot(!file.exists(outside), !file.exists(file.path(outside, "new-folder")))
  bad$output_dir <- file.path(outside, "new-folder", ".", "result")
  write_json(bad, "dot-output.json")
  blocked(read_config("dot-output.json", kit), "dot segment")
  stopifnot(!file.exists(outside))
})
test("HTML attribute values escaped and CSV formulas neutralized", {
  escaped <- aggregate_credits(valid, 10)
  escaped$table$group <- "<script>alert(1)</script>"
  stopifnot(!grepl("<script>", dashboard_html(escaped, loaded), fixed = TRUE),
            grepl("&lt;script&gt;", dashboard_html(escaped, loaded), fixed = TRUE))
  formula <- people
  formula$Organization <- "=1+1"
  write.csv(formula, "formula-people.csv", row.names = FALSE)
  bad <- loaded
  bad$inputs$people <- file.path(root, "formula-people.csv")
  bad$output_dir <- file.path(root, "formula-build")
  build_dashboard(bad, kit)
  csv <- read.csv(file.path(bad$output_dir, "build", "aggregates.csv"))
  stopifnot(all(csv$group == "'=1+1"))
})
# Policy-aware Consumption: synthetic people-service-days split across spending policies,
# with metadata that carries no HR attributes (group=false).
policy_people <- people[c("PeopleHistoricalId", "IsCopilotLicensed")]
policy_activity <- expand.grid(PersonId = sprintf("synthetic-person-%02d", 1:20),
                               SpendingPolicyId = c("synthetic-policy-a", "synthetic-policy-b"),
                               stringsAsFactors = FALSE)
policy_activity$ServiceId <- "Synthetic Service A"
policy_activity$MetricDate <- "2026-06-01"
policy_activity$PeopleHistoricalId <- people$PeopleHistoricalId[match(policy_activity$PersonId,
                                                                   sprintf("synthetic-person-%02d", 1:20))]
policy_activity[["Total Copilot Credits used"]] <- c(1:20, rep(0, 20))
write.csv(policy_activity, "policy-activity.csv", row.names = FALSE)
write.csv(policy_people, "policy-people.csv", row.names = FALSE)
policy_cfg <- loaded
policy_cfg$inputs <- list(activity = file.path(root, "policy-activity.csv"),
                          people = file.path(root, "policy-people.csv"))
policy_cfg$mappings$activity$policy_id <- "SpendingPolicyId"
policy_cfg$mappings$people$group <- NULL
policy_cfg$group <- FALSE
policy_tables <- lapply(policy_cfg$inputs, read_csv)
with_credits <- function(values) {
  out <- policy_tables
  out$activity[["Total Copilot Credits used"]] <- as.character(values)
  out
}
overlap_credits <- c(1:20, 5, 5, 5, rep(0, 17))
disjoint <- list(confirmed = TRUE, evidence = "Synthetic export-owner confirmation of disjoint policies")
test("policy rows with one positive and other zero rows pass and sum per person-service-date", {
  data <- validate_consumption(policy_cfg, policy_tables)
  agg <- aggregate_credits(data, 10)
  stopifnot(attr(data, "multi_policy_positive_keys") == 0L, nrow(data) == 40L,
            agg$status == "ready", nrow(agg$table) == 1L, agg$table$people == 20L,
            agg$table$credits == 210, agg$table$group == "All people")
  built <- policy_cfg
  built$output_dir <- file.path(root, "policy-zero-build")
  capture.output(build_dashboard(built, kit))
  csv <- read.csv(file.path(built$output_dir, "build", "aggregates.csv"))
  prov <- jsonlite::read_json(file.path(built$output_dir, "build", "provenance.json"))
  stopifnot(csv$credits == 210, csv$people == 20L,
            !any(grepl("synthetic-policy|synthetic-person|synthetic-history", unlist(csv))),
            identical(prov$mappings$activity$policy_id, "SpendingPolicyId"),
            "policy_additivity" %in% names(prov), is.null(prov$policy_additivity))
})
test("multiple positive policy rows block build without additivity confirmation", {
  err <- tryCatch({ validate_consumption(policy_cfg, with_credits(overlap_credits)); NULL },
                  dashboard_error = function(e) conditionMessage(e))
  stopifnot(!is.null(err), grepl("Policy additivity unconfirmed: 3 ", err, fixed = TRUE),
            grepl("export owner", err), !grepl("synthetic", err, ignore.case = TRUE))
  overlap <- policy_activity
  overlap[["Total Copilot Credits used"]] <- overlap_credits
  write.csv(overlap, "policy-overlap.csv", row.names = FALSE)
  blocked_cfg <- policy_cfg
  blocked_cfg$inputs$activity <- file.path(root, "policy-overlap.csv")
  blocked_cfg$output_dir <- file.path(root, "policy-overlap-build")
  blocked(build_dashboard(blocked_cfg, kit), "Policy additivity unconfirmed")
  stopifnot(!dir.exists(file.path(blocked_cfg$output_dir, "build")))
  blocked_cfg$policy_additivity <- list(confirmed = FALSE, evidence = "Pending export-owner reply")
  blocked(validate_consumption(blocked_cfg, with_credits(overlap_credits)), "Policy additivity unconfirmed")
})
test("confirmed policy additivity with evidence allows build and sums policy rows", {
  confirmed <- policy_cfg
  confirmed$policy_additivity <- disjoint
  data <- validate_consumption(confirmed, with_credits(overlap_credits))
  agg <- aggregate_credits(data, 10)
  stopifnot(attr(data, "multi_policy_positive_keys") == 3L, agg$status == "ready",
            agg$table$people == 20L, agg$table$credits == 225)
  confirmed$inputs$activity <- file.path(root, "policy-overlap.csv")
  confirmed$output_dir <- file.path(root, "policy-confirmed-build")
  capture.output(build_dashboard(confirmed, kit))
  csv <- read.csv(file.path(confirmed$output_dir, "build", "aggregates.csv"))
  prov <- jsonlite::read_json(file.path(confirmed$output_dir, "build", "provenance.json"))
  stopifnot(csv$credits == 225, identical(prov$policy_additivity$confirmed, TRUE),
            identical(prov$policy_additivity$evidence, disjoint$evidence))
})
test("confirmed policy additivity with blank evidence still blocks", {
  for (evidence in c("", " ")) {
    blank <- policy_cfg
    blank$policy_additivity <- list(confirmed = TRUE, evidence = evidence)
    blocked(validate_consumption(blank, with_credits(overlap_credits)), "Policy additivity unconfirmed")
  }
})
test("policy-aware key rejects duplicates, blank policies and loose mappings", {
  duplicate <- policy_tables
  duplicate$activity <- rbind(duplicate$activity, duplicate$activity[1, ])
  blocked(validate_consumption(policy_cfg, duplicate), "SpendingPolicyId, MetricDate")
  for (value in c(NA_character_, "", " synthetic-policy-a")) {
    bad <- policy_tables
    bad$activity$SpendingPolicyId[1] <- value
    blocked(validate_consumption(policy_cfg, bad), "identifiers")
  }
  bad <- policy_cfg
  bad$mappings$activity$unknown_field <- "IsCopilotLicensed"
  blocked(validate_consumption(bad, policy_tables), "exactly the documented")
  bad <- policy_cfg
  bad$mappings$activity$credits <- NULL
  blocked(validate_consumption(bad, policy_tables), "exactly the documented")
  bad <- policy_cfg
  bad$mappings$activity$policy_id <- "PersonId"
  blocked(validate_consumption(bad, policy_tables), "same source")
  bad <- policy_cfg
  bad$mappings$activity$policy_id <- "No such column"
  blocked(validate_consumption(bad, policy_tables), "absent")
})
test("unmapped policy column with duplicated three-column key suggests SpendingPolicyId", {
  unmapped <- policy_cfg
  unmapped$mappings$activity$policy_id <- NULL
  err <- tryCatch({ validate_consumption(unmapped, policy_tables); NULL },
                  dashboard_error = function(e) conditionMessage(e))
  stopifnot(!is.null(err), grepl("Duplicate activity key", err),
            grepl("map it as activity policy_id", err, fixed = TRUE),
            grepl("confirm policy additivity", err, fixed = TRUE))
  unmapped$policy_additivity <- disjoint
  blocked(validate_consumption(unmapped, policy_tables), "applies only when")
})
test("policy_additivity validated and rejected in demo and real GitHub modes", {
  demo <- list(report = "consumption", mode = "demo", repo_root = repo,
               output_dir = file.path(root, "demo-policy"), privacy_min = 10,
               policy_additivity = list(confirmed = FALSE, evidence = "Synthetic"))
  write_json(demo, "demo-policy.json")
  blocked(read_config("demo-policy.json", kit), "fixed reference")
  gh <- list(report = "github", mode = "real", repo_root = repo,
             output_dir = file.path(root, "github-policy"),
             inputs = list(activity = "synthetic-activity.csv"), privacy_min = 10,
             policy_additivity = disjoint)
  write_json(gh, "github-policy.json")
  blocked(read_config("github-policy.json", kit), "does not accept analytical")
  for (shape in list(list(confirmed = "yes", evidence = "Synthetic"),
                     list(confirmed = TRUE),
                     list(confirmed = TRUE, evidence = "Synthetic", extra = 1),
                     list(confirmed = TRUE, evidence = 1), "confirmed")) {
    bad <- cfg
    bad$policy_additivity <- shape
    write_json(bad, "bad-policy.json")
    blocked(read_config("bad-policy.json", kit), "policy_additivity requires")
  }
  good <- cfg
  good$policy_additivity <- disjoint
  write_json(good, "good-policy.json")
  stopifnot(identical(read_config("good-policy.json", kit)$policy_additivity, disjoint))
})
test("inspect reports policy grain and bounded multi-policy count without identifiers", {
  pending <- policy_cfg
  pending$inputs$activity <- file.path(root, "policy-overlap.csv")
  pending$mapping_approval <- NULL
  pending$output_dir <- file.path(root, "policy-inspect")
  text <- capture.output(report <- inspect_dashboard(pending, kit))
  stopifnot(report$status == "awaiting_policy_confirmation",
            report$structural_validation == "passed",
            report$grain == "PersonId + ServiceId + SpendingPolicyId + MetricDate; day export periods",
            identical(report$multi_policy_positive_keys, 3L),
            report$policy_additivity_status == "awaiting_confirmation",
            grepl("export owner", report$next_action), grepl("Before deciding approval", report$next_action),
            any(grepl("Spending policy breakdowns", report$excluded_panels)),
            !any(grepl("synthetic-person|synthetic-history|synthetic-policy|Synthetic Service", text)))
  saved <- jsonlite::read_json(file.path(pending$output_dir, "inspect", "readiness.json"))
  stopifnot(saved$multi_policy_positive_keys == 3L,
            !any(grepl("synthetic-person|synthetic-policy", unlist(saved[names(saved) != "provenance"]))))
  pending$mapping_approval <- loaded$mapping_approval
  pending$output_dir <- file.path(root, "policy-inspect-approved")
  capture.output(report <- inspect_dashboard(pending, kit))
  stopifnot(report$status == "awaiting_policy_confirmation")
  pending$policy_additivity <- disjoint
  pending$output_dir <- file.path(root, "policy-inspect-confirmed")
  capture.output(report <- inspect_dashboard(pending, kit))
  stopifnot(report$status == "ready", report$policy_additivity_status == "confirmed")
  zero <- policy_cfg
  zero$mapping_approval <- NULL
  zero$output_dir <- file.path(root, "policy-inspect-zero")
  capture.output(report <- inspect_dashboard(zero, kit))
  stopifnot(report$status == "awaiting_approval", report$multi_policy_positive_keys == 0L,
            report$policy_additivity_status == "not_required")
  capture.output(report <- inspect_dashboard(modifyList(loaded, list(
    mapping_approval = NULL, output_dir = file.path(root, "unmapped-policy-inspect"))), kit))
  stopifnot(report$grain == "PersonId + ServiceId + MetricDate; day export periods",
            is.null(report$multi_policy_positive_keys), is.null(report$policy_additivity_status))
})
test("real Consumption example maps SpendingPolicyId and inspects metadata without HR attributes", {
  example <- jsonlite::read_json(file.path(kit, "configs", "real-consumption.example.json"))
  stopifnot(basename(example$inputs$activity) == "PersonM365CreditsMetrics.csv",
            identical(example$mappings$activity$policy_id, "SpendingPolicyId"),
            identical(example$policy_additivity$confirmed, FALSE),
            grepl("^REPLACE", example$policy_additivity$evidence),
            identical(example$group, FALSE), is.null(example$mappings$people$group))
  example$repo_root <- repo
  example$output_dir <- file.path(root, "example-policy")
  example$inputs <- list(activity = "policy-overlap.csv", people = "policy-people.csv")
  write_json(example, "example-policy.json")
  capture.output(report <- run_dashboard("inspect", "example-policy.json", kit))
  stopifnot(report$status == "awaiting_policy_confirmation", report$multi_policy_positive_keys == 3L)
  blocked(build_dashboard(read_config("example-policy.json", kit), kit), "approval")
})
source(file.path(kit, "tests", "query-fixture-tests.R"))
test("demo source isolation preflight: tracked references and exact copy targets", {
  for (report in c("consumption", "github")) {
    demo <- list(report = report, mode = "demo", repo_root = repo,
                 output_dir = file.path(root, paste0(report, "-demo")), privacy_min = 10, group = FALSE)
    refs <- reference_files(demo)
    expected <- if (report == "consumption") c(
      "copilot-consumption-ways-of-working-simulation.Rmd",
      "copilot-super-panel-helpers.R",
      "_data/person-query/PersonQuery.csv",
      "_data/consumption-query/PeopleMetaData.csv",
      "_data/consumption-query/PersonM365CreditsMetrics.csv",
      "_data/consumption-query/PersonGitHubCreditsMetrics.csv",
      "_data/github-query/PersonGitHubActivityMetrics.csv") else c(
        "github-copilot-developer-productivity-simulation.Rmd",
        "github-developer-experience-helpers.R",
        "copilot-super-panel-helpers.R",
        paste0("_data/", names(query_export_headers)))
    utility <- file.path(repo, "examples", "utility-r")
    stopifnot(setequal(gsub("\\\\", "/", unname(refs)), file.path(utility, expected)),
              length(refs) == length(expected),
              all(vapply(refs, function(x) inside(x, repo), logical(1))),
              !inside(demo$output_dir, repo))
  }
})
test("entry point succeeds from an unrelated working directory", {
  command <- file.path(R.home("bin"), "Rscript")
  config <- cfg
  config$output_dir <- file.path(root, "cli-smoke")
  write_json(config, "cli-config.json")
  result <- system2(command, c(shQuote(file.path(kit, "dashboard.R")), "build",
                               shQuote(file.path(root, "cli-config.json"))),
                    stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(result, "status")),
            file.exists(file.path(config$output_dir, "build", "dashboard.html")))
})
test("entry point rejects unsupported GitHub before producing build output", {
  command <- file.path(R.home("bin"), "Rscript")
  config <- list(report = "github", mode = "real", repo_root = repo,
                 output_dir = file.path(root, "cli-github-block"),
                 inputs = list(activity = "synthetic-activity.csv"), privacy_min = 10)
  stderr_file <- tempfile("github-build-stderr-", tmpdir = root)
  write_json(config, "cli-github.json")
  result <- suppressWarnings(system2(command, c(shQuote(file.path(kit, "dashboard.R")), "build",
                                                shQuote(file.path(root, "cli-github.json"))),
                                     stdout = TRUE, stderr = stderr_file))
  # Windows R does not consistently merge stderr into system2()'s return value.
  unlink(stderr_file)
  stopifnot(attr(result, "status") == 1L,
            !dir.exists(config$output_dir))
})
test("supplied examples resolve repo root and require real mapping approval", {
  for (name in c("demo-consumption.json", "demo-github.json")) {
    example <- read_config(file.path(kit, "configs", name), kit)
    stopifnot(example$repo_root == repo, !inside(example$output_dir, repo),
              example$mode == "demo")
  }
  example <- jsonlite::read_json(file.path(kit, "configs", "real-consumption.example.json"))
  stopifnot(identical(example$mapping_approval$approved, FALSE))
})
test("demo does not silently ignore custom privacy or filters", {
  demo <- list(report = "github", mode = "demo", repo_root = repo,
               output_dir = file.path(root, "demo-custom"), privacy_min = 20)
  write_json(demo, "demo-custom.json")
  blocked(read_config("demo-custom.json", kit), "fixed reference")
  demo$privacy_min <- 10
  demo$dates <- list(start = "2026-01-01", end = "2026-06-30")
  write_json(demo, "demo-custom.json")
  blocked(read_config("demo-custom.json", kit), "fixed reference")
})
if ("--render-demos" %in% cli) {
  for (report in c("consumption", "github")) {
    test(paste(report, "full isolated self-contained reproduction; source hashes unchanged"), {
      demo <- list(report = report, mode = "demo", repo_root = repo,
                   output_dir = file.path(root, paste0(report, "-demo")), privacy_min = 10, group = FALSE)
      write_json(demo, paste0(report, "-demo.json"))
      hashes <- file_hashes(reference_files(demo))
      run_dashboard("reproduce", paste0(report, "-demo.json"), kit)
      html <- file.path(demo$output_dir, "reproduce", "dashboard.html")
      stopifnot(file.exists(html), file.info(html)$size > 100000,
                identical(hashes, file_hashes(reference_files(demo))))
      contents <- readLines(html, warn = FALSE)
      stopifnot(!any(grepl('<script[^>]+src="https?://|<link[^>]+href="https?://', contents)))
    })
  }
}
writeLines(c(paste("Passed:", length(passed)), passed), "test-results.txt")
cat("\n", length(passed), " tests passed. Artifacts: ", root, "\n", sep = "")
