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
  gh$output_dir <- file.path(root, "github-inspect")
  report <- inspect_dashboard(gh, kit)
  stopifnot(report$status == "inspect_only", grepl("unverified", report$blocker))
  blocked(build_dashboard(gh, kit), "unsupported")
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
test("demo source isolation preflight: tracked references and exact copy targets", {
  for (report in c("consumption", "github")) {
    demo <- list(report = report, mode = "demo", repo_root = repo,
                 output_dir = file.path(root, paste0(report, "-demo")), privacy_min = 10, group = FALSE)
    refs <- reference_files(demo)
    stopifnot(length(refs) == if (report == "consumption") 6 else 2,
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
  config <- cfg
  config$report <- "github"
  config$output_dir <- file.path(root, "cli-github-block")
  write_json(config, "cli-github.json")
  result <- suppressWarnings(system2(command, c(shQuote(file.path(kit, "dashboard.R")), "build",
                                                shQuote(file.path(root, "cli-github.json"))),
                                     stdout = TRUE, stderr = TRUE))
  stopifnot(attr(result, "status") == 1L, any(grepl("unsupported", result)),
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
