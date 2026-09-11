fail <- function(message) {
  stop(structure(list(message = message, call = NULL),
                 class = c("dashboard_error", "error", "condition")))
}

need <- function(packages) {
  absent <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent)) fail(paste("Missing R packages:", paste(absent, collapse = ", ")))
}

scalar_text <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(trimws(x))
}

resolve_path <- function(path, base, existing = FALSE) {
  if (!scalar_text(path)) fail("Every path must be a nonempty string.")
  absolute <- grepl("^([A-Za-z]:[/\\\\]|[/\\\\])", path)
  path <- if (absolute) path else file.path(base, path)
  if (file.exists(path)) return(normalizePath(path, winslash = "/", mustWork = TRUE))
  if (existing) fail("A configured input or repository path does not exist.")
  if (basename(path) %in% c(".", ".."))
    fail("Unresolved output path contains a dot segment. Use an absolute path without '.' or '..' after nonexistent folders.")
  parent <- dirname(path)
  if (identical(parent, path)) fail("Cannot resolve output path.")
  file.path(resolve_path(parent, base, FALSE), basename(path))
}

inside <- function(path, parent) {
  path <- tolower(gsub("\\\\", "/", path))
  parent <- sub("/+$", "", tolower(gsub("\\\\", "/", parent)))
  identical(path, parent) || startsWith(path, paste0(parent, "/"))
}

strict_date <- function(x) {
  shape <- !is.na(x) & grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", x)
  result <- suppressWarnings(as.Date(ifelse(shape, x, NA_character_), "%Y-%m-%d"))
  if (any(!shape | is.na(result)) || any(format(result, "%Y-%m-%d") != x))
    fail("Dates must be valid ISO YYYY-MM-DD values.")
  result
}

read_config <- function(path, kit) {
  need("jsonlite")
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  cfg <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  allowed <- c("report", "mode", "repo_root", "output_dir", "inputs", "mappings",
               "mapping_approval", "granularity", "dates", "group", "privacy_min")
  if (!is.list(cfg) || is.null(names(cfg)) || anyDuplicated(names(cfg)) ||
      any(!names(cfg) %in% allowed)) fail("Unknown or duplicate configuration fields.")
  if (!scalar_text(cfg$report) || !cfg$report %in% c("consumption", "github"))
    fail("report must be consumption or github.")
  if (!scalar_text(cfg$mode) || !cfg$mode %in% c("demo", "real"))
    fail("mode must be demo or real.")
  base <- dirname(path)
  cfg$repo_root <- resolve_path(cfg$repo_root, base, TRUE)
  cfg$output_dir <- resolve_path(cfg$output_dir, base)
  actual_repo <- normalizePath(file.path(kit, "..", "..", ".."), winslash = "/", mustWork = TRUE)
  if (inside(cfg$output_dir, cfg$repo_root) || inside(cfg$output_dir, actual_repo))
    fail("output_dir must be outside both the source repository and repo_root.")
  cfg$privacy_min <- if (is.null(cfg$privacy_min)) 10L else cfg$privacy_min
  if (!is.numeric(cfg$privacy_min) || length(cfg$privacy_min) != 1L ||
      !is.finite(cfg$privacy_min) || cfg$privacy_min < 10 ||
      cfg$privacy_min != floor(cfg$privacy_min)) fail("privacy_min must be an integer >= 10.")
  cfg$group <- if (is.null(cfg$group)) FALSE else cfg$group
  if (!is.logical(cfg$group) || length(cfg$group) != 1L || is.na(cfg$group))
    fail("group must be true or false.")
  if (!is.null(cfg$granularity) && (!scalar_text(cfg$granularity) ||
      !cfg$granularity %in% c("day", "week", "month")))
    fail("granularity must be day, week, or month.")
  if (!is.null(cfg$dates)) {
    if (!is.list(cfg$dates) || !setequal(names(cfg$dates), c("start", "end")))
      fail("dates requires start and end.")
    if (!scalar_text(cfg$dates$start) || !scalar_text(cfg$dates$end))
      fail("dates requires ISO strings.")
    bounds <- strict_date(c(cfg$dates$start, cfg$dates$end))
    if (bounds[1] > bounds[2]) fail("dates.start must not exceed dates.end.")
  }
  if (cfg$report == "github" && cfg$mode == "real" &&
      any(c("dates", "group", "granularity", "mappings", "mapping_approval") %in% names(cfg)))
    fail("Real GitHub inspection does not accept analytical filters, mappings, or approval settings because no verified adapter exists.")
  if (cfg$mode == "demo" && (!is.null(cfg$inputs) || !is.null(cfg$mappings)))
    fail("Demo mode does not accept real inputs or mappings.")
  if (cfg$mode == "demo" && (cfg$privacy_min != 10L || cfg$group ||
      !is.null(cfg$dates) || !is.null(cfg$granularity) || !is.null(cfg$mapping_approval)))
    fail("Demo reproduces fixed reference settings (privacy_min=10); real-data filters, approvals and custom thresholds do not apply.")
  if (!is.null(cfg$inputs)) {
    if (!is.list(cfg$inputs) || is.null(names(cfg$inputs)) ||
        anyDuplicated(names(cfg$inputs)) || any(!names(cfg$inputs) %in% c("activity", "people")))
      fail("inputs accepts only activity and people CSV paths; Person Query is not supported.")
    cfg$inputs <- lapply(cfg$inputs, resolve_path, base = base, existing = TRUE)
    unsafe_input <- vapply(cfg$inputs, function(p)
      inside(p, cfg$output_dir) || inside(p, cfg$repo_root) || inside(p, actual_repo), logical(1))
    if (any(unsafe_input))
      fail("Input files must be outside output_dir and both source repositories.")
  }
  cfg
}

output_folder <- function(cfg, action) {
  dest <- file.path(cfg$output_dir, action)
  if (file.exists(dest)) fail("Action output already exists. Choose a new output_dir; nothing is overwritten.")
  if (!dir.create(dest, recursive = TRUE, showWarnings = FALSE))
    fail("Cannot create action output directory.")
  dest
}

write_json <- function(value, path) {
  jsonlite::write_json(value, path, pretty = TRUE, auto_unbox = TRUE, null = "null", na = "null")
}

file_hashes <- function(paths) {
  paths <- unlist(paths, use.names = TRUE)
  values <- unname(tools::md5sum(paths))
  if (anyNA(values)) fail("Cannot hash an input file.")
  as.list(setNames(values, names(paths)))
}

provenance <- function(cfg, action, inputs, kit) {
  revision <- suppressWarnings(system2("git", c("-C", shQuote(cfg$repo_root), "rev-parse", "HEAD"),
                                      stdout = TRUE, stderr = FALSE))
  if (!length(revision) || !grepl("^[a-f0-9]{40}$", revision[1]))
    fail("repo_root must be a readable Git checkout with a recorded revision.")
  runtime <- c(entry = file.path(kit, "dashboard.R"), runtime = file.path(kit, "lib", "runtime.R"))
  versions <- vapply(loadedNamespaces(), function(p) as.character(utils::packageVersion(p)), character(1))
  list(action = action, report = cfg$report, mode = cfg$mode,
       synthetic = identical(cfg$mode, "demo"), repository_revision = revision[1],
       file_hash_algorithm = "MD5 (reproducibility checksum, not authenticity)",
       input_hashes = file_hashes(inputs), runtime_hashes = file_hashes(runtime),
       R = R.version.string, packages = as.list(versions),
       locale = Sys.getlocale(), timezone = Sys.timezone(),
       generated_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
       granularity = cfg$granularity, privacy_min = cfg$privacy_min,
       group = cfg$group, dates = cfg$dates,
       mapping_approval = cfg$mapping_approval, mappings = cfg$mappings)
}

read_csv <- function(path) {
  if (dir.exists(path)) fail("CSV inputs must be files.")
  fields <- tryCatch(suppressWarnings(utils::count.fields(path, sep = ",", quote = "\"", comment.char = "",
                                       blank.lines.skip = FALSE)),
                     error = function(e) fail("Cannot parse CSV input."))
  if (!length(fields) || is.na(fields[1]) || any(fields[!is.na(fields)] != fields[1]))
    fail("Malformed CSV: every record must match the header width.")
  data <- tryCatch(suppressWarnings(utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE,
                                  colClasses = "character", na.strings = "", strip.white = FALSE,
                                  fileEncoding = "UTF-8-BOM", comment.char = "", fill = FALSE)),
                   error = function(e) fail("Cannot parse CSV input."))
  normalized <- tolower(gsub("[^[:alnum:]]", "", names(data)))
  if (!ncol(data) || any(!nzchar(normalized)) || anyDuplicated(normalized))
    fail("Schema contains empty names, duplicate names, or normalization collisions.")
  data
}

mapped <- function(data, mapping, required) {
  if (!is.list(mapping) || is.null(names(mapping)) || anyDuplicated(names(mapping)) ||
      !setequal(names(mapping), required) ||
      !all(vapply(mapping, scalar_text, logical(1))))
    fail("Explicit mappings must contain exactly the documented required fields.")
  columns <- unlist(mapping, use.names = FALSE)
  if (anyDuplicated(columns)) fail("Two mapped fields cannot use the same source column.")
  if (!all(columns %in% names(data))) fail("A mapped source column is absent; review the schema.")
  data <- data[, columns, drop = FALSE]
  names(data) <- names(mapping)
  data[, required, drop = FALSE]
}

require_ids <- function(data, columns) {
  for (column in columns) {
    x <- data[[column]]
    if (anyNA(x) || any(!nzchar(trimws(x))) || any(trimws(x) != x))
      fail("Required identifiers/attributes are missing, blank, or padded with whitespace.")
  }
}

has_mapping_approval <- function(cfg) {
  approval <- cfg$mapping_approval
  is.list(approval) && identical(approval$approved, TRUE) && scalar_text(approval$evidence)
}

validate_consumption <- function(cfg, tables, require_approval = TRUE) {
  if (is.null(cfg$granularity)) fail("Declare export granularity explicitly: day, week, or month.")
  if (require_approval && !has_mapping_approval(cfg))
    fail("mapping_approval requires approved=true and a nonempty evidence reference.")
  if (!is.list(cfg$mappings) || !setequal(names(cfg$mappings), c("activity", "people")))
    fail("Consumption requires explicit activity and people mappings.")
  if (is.null(tables$activity) || is.null(tables$people))
    fail("Consumption requires both activity and people metadata CSVs.")
  a <- mapped(tables$activity, cfg$mappings$activity,
              c("person_id", "service_id", "date", "people_id", "credits"))
  required <- c("people_id", "licensed", if (cfg$group) "group")
  p <- mapped(tables$people, cfg$mappings$people, required)
  require_ids(a, c("person_id", "service_id", "people_id"))
  require_ids(p, c("people_id", if (cfg$group) "group"))
  a$date <- strict_date(a$date)
  if (anyDuplicated(a[c("person_id", "service_id", "date")]))
    fail("Duplicate activity key: expected one row per PersonId, ServiceId, MetricDate.")
  if (anyDuplicated(p$people_id)) fail("Duplicate PeopleHistoricalId in metadata; join is ambiguous.")
  links <- unique(a[c("people_id", "person_id")])
  if (anyDuplicated(links$people_id))
    fail("One PeopleHistoricalId maps to multiple PersonIds; join is ambiguous.")
  if (anyNA(a$credits) ||
      any(!grepl("^[+]?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$", a$credits)))
    fail("Credits must be nonnegative finite numbers with a decimal point, not locale separators.")
  a$credits <- suppressWarnings(as.numeric(a$credits))
  if (any(!is.finite(a$credits)) || any(a$credits < 0))
    fail("Credits must be nonnegative finite numbers.")
  if (anyNA(p$licensed) || any(!tolower(p$licensed) %in% c("true", "false")))
    fail("IsCopilotLicensed must contain explicit true/false values; activity does not establish licensing.")
  match_id <- match(a$people_id, p$people_id)
  if (anyNA(match_id)) fail("Activity contains unmatched people metadata; obtain a complete matching export.")
  if (cfg$group) a$group <- p$group[match_id] else a$group <- "All people"
  # Privacy counts people, never metadata versions or activity rows.
  if (!is.null(cfg$dates)) {
    start <- strict_date(cfg$dates$start)
    end <- strict_date(cfg$dates$end)
    a <- a[a$date >= start & a$date <= end, , drop = FALSE]
  }
  a
}

aggregate_credits <- function(data, threshold) {
  empty <- data.frame(date = character(), service = character(), group = character(),
                      people = integer(), credits = numeric(), stringsAsFactors = FALSE)
  if (!nrow(data)) return(list(status = "no_data", table = empty))
  keys <- unique(data[c("date", "service_id", "group")])
  result <- lapply(seq_len(nrow(keys)), function(i) {
    rows <- data$date == keys$date[i] & data$service_id == keys$service_id[i] &
      data$group == keys$group[i]
    data.frame(date = as.character(keys$date[i]), service = keys$service_id[i],
               group = keys$group[i], people = length(unique(data$person_id[rows])),
               credits = sum(data$credits[rows]), stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, result)
  if (any(!is.finite(out$credits))) fail("Aggregated credits overflow; no output is publishable.")
  # Conservative whole-release suppression prevents subtraction across displayed slices.
  if (any(out$people < threshold)) return(list(status = "suppressed", table = empty))
  out <- out[order(out$date, out$service, out$group), , drop = FALSE]
  rownames(out) <- NULL
  list(status = "ready", table = out)
}

schema_summary <- function(data) {
  cols <- head(names(data), 80L)
  clean <- function(x) substr(gsub("[[:cntrl:]]", "", x), 1L, 120L)
  list(rows = nrow(data), columns = ncol(data), displayed_columns = clean(cols),
       schema_truncated = ncol(data) > 80L,
       missing_cells = sum(is.na(data) | data == ""))
}

reference_files <- function(cfg) {
  utility <- file.path(cfg$repo_root, "examples", "utility-r")
  if (cfg$report == "github") {
    relative <- c("github-copilot-developer-productivity-simulation.Rmd",
                  "github-developer-experience-helpers.R")
  } else {
    relative <- c("copilot-consumption-ways-of-working-simulation.Rmd",
                  file.path("_data", "consumption", c(
                    "reference/people-snapshot.csv",
                    "consumption-query/consumption-weekly.csv",
                    "consumption-query/consumption-task-types.csv",
                    "person-query/person-query-weekly.csv",
                    "person-query/network-monthly.csv")))
  }
  paths <- file.path(utility, relative)
  if (!all(file.exists(paths))) fail("Reference report or supplied synthetic fixtures are missing.")
  # Only clean tracked reference files may execute as synthetic code/data.
  for (path in paths) {
    rel <- substring(gsub("\\\\", "/", path), nchar(cfg$repo_root) + 2L)
    tracked <- suppressWarnings(system2("git", c("-C", shQuote(cfg$repo_root), "ls-files",
                                                "--error-unmatch", "--", shQuote(rel)),
                                       stdout = TRUE, stderr = FALSE))
    dirty <- suppressWarnings(system2("git", c("-C", shQuote(cfg$repo_root), "status",
                                              "--porcelain", "--", shQuote(rel)),
                                     stdout = TRUE, stderr = FALSE))
    if (!length(tracked) || !is.null(attr(tracked, "status")) || length(dirty))
      fail("Demo reference files must be tracked and unmodified; do not replace fixtures with customer data.")
  }
  setNames(paths, relative)
}

inspect_dashboard <- function(cfg, kit) {
  inputs <- if (cfg$mode == "demo") reference_files(cfg) else cfg$inputs
  if (is.null(inputs) || !length(inputs)) fail("Real inspection requires inputs.activity (and optionally people).")
  before <- file_hashes(inputs)
  csv_inputs <- if (cfg$mode == "demo") inputs[grepl("\\.csv$", inputs, ignore.case = TRUE)] else inputs
  tables <- lapply(csv_inputs, read_csv)
  report <- list(report = cfg$report, mode = cfg$mode, synthetic = cfg$mode == "demo",
                 schema = lapply(tables, schema_summary), status = "inspect_only")
  if (cfg$mode == "demo") {
    report$next_action <- "reproduce: isolated synthetic reference; this is not a verified real-data adapter."
  } else if (cfg$report == "github") {
    report$blocker <- "Real GitHub schema/semantics are unverified. Obtain an approved export contract, grain, coverage and mappings; build is unsupported."
  } else {
    validation <- tryCatch(validate_consumption(cfg, tables, require_approval = FALSE),
                           dashboard_error = function(e) e)
    if (inherits(validation, "dashboard_error")) {
      report$status <- "blocked"
      report$blocker <- conditionMessage(validation)
      report$next_action <- "Correct the structural validation blocker, then inspect again before deciding approval."
    } else {
      release <- aggregate_credits(validation, cfg$privacy_min)
      approved <- has_mapping_approval(cfg)
      report$status <- if (approved) release$status else "awaiting_approval"
      report$structural_validation <- "passed"
      report$approval_status <- if (approved) "approved" else "awaiting_approval"
      report$privacy_status <- release$status
      report$supported_panels <- if (release$status == "ready")
        c("Copilot credits by source period and service",
          if (cfg$group) "Copilot credits by configured metadata group") else character()
      report$excluded_panels <- c(
        "Tokens and task types: absent from the verified Consumption contract",
        "Sessions, user limits and spending policy limits: not implemented by this credits-only adapter",
        "Person Query associations and GitHub panels: no verified real-data adapter",
        "Licensed population denominators, individual ranks, causal/ROI/productivity/burnout claims: unsupported",
        if (release$status != "ready") "All aggregate panels: withheld by privacy or unavailable in the selected window")
      report$grain <- paste("PersonId + ServiceId + MetricDate;", cfg$granularity, "export periods")
      report$metric_unit <- "Copilot credits, as explicitly mapped; not currency. Confirm source semantics before approval."
      report$rows_in_window <- nrow(validation)
      report$distinct_people <- length(unique(validation$person_id))
      report$date_range <- if (nrow(validation)) as.character(range(validation$date)) else character()
      report$next_action <- if (release$status == "suppressed") {
        "All aggregate panels are withheld. Review an appropriately broader source scope and inspect again; approval does not override privacy."
      } else if (release$status == "no_data") {
        "Review source coverage and the selected period window, then inspect again; no panels are available."
      } else if (!approved) {
        "Review the validated scope, declared units/granularity and excluded panels with the user. Only after their decision, set mapping_approval.approved=true with nonblank evidence, then build."
      } else {
        "Build the approved supported panels; no values/identifiers are included in this inspection."
      }
    }
  }
  report$provenance <- provenance(cfg, "inspect", inputs, kit)
  if (!identical(before, report$provenance$input_hashes))
    fail("Inputs changed during inspection; rerun against stable exports.")
  dest <- output_folder(cfg, "inspect")
  write_json(report, file.path(dest, "readiness.json"))
  # Output is bounded; the full schema is already capped and contains no record samples.
  printable <- report
  printable$provenance <- NULL
  cat(jsonlite::toJSON(printable, pretty = TRUE, auto_unbox = TRUE), "\n")
  invisible(report)
}

escape_html <- function(x) {
  x <- gsub("&", "&amp;", as.character(x), fixed = TRUE)
  x <- gsub("<", "&lt;", x, fixed = TRUE)
  x <- gsub(">", "&gt;", x, fixed = TRUE)
  x <- gsub('"', "&quot;", x, fixed = TRUE)
  gsub("'", "&#39;", x, fixed = TRUE)
}

dashboard_html <- function(release, cfg,
                           generated_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")) {
  text <- switch(release$status,
                 ready = "Credits by export period, service and group",
                 suppressed = "All aggregates withheld: at least one cell has fewer than the privacy minimum.",
                 no_data = "No records in the selected date window.")
  rows <- if (nrow(release$table)) apply(release$table, 1L, function(row)
    paste0("<tr><td>", paste(escape_html(row), collapse = "</td><td>"), "</td></tr>")) else character()
  selected_window <- if (is.null(cfg$dates)) "All supplied source period labels (no date filter)" else
    paste(cfg$dates$start, "to", cfg$dates$end, "(inclusive source period labels)")
  observed_window <- if (release$status == "suppressed") "Withheld under the privacy policy" else
    if (release$status == "ready" && nrow(release$table))
      paste(range(release$table$date), collapse = " to ") else
      "No observations in the selected window"
  paste0('<!doctype html><html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>Copilot credit consumption</title><style>',
         'body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:1rem;color:#253047}',
         'table{border-collapse:collapse;width:100%}td,th{padding:.7rem;text-align:left;border-bottom:1px solid #ddd}',
         '.scroll{overflow:auto}aside{background:#eef2ff;padding:1rem}small{color:#475569}',
         '</style></head><body><h1>Copilot credit consumption</h1>',
         '<p><strong>Real-data aggregate · Provisional · Not a productivity assessment</strong></p>',
         '<p><strong>Selected period window:</strong> ', escape_html(selected_window),
         '<br><strong>Observed period window:</strong> ', escape_html(observed_window),
         '<br><strong>Generated (UTC):</strong> ', escape_html(generated_utc), '</p>',
         '<p>', escape_html(text), '</p><p>Export granularity: ', cfg$granularity,
         '. MetricDate is the source period label; periods are not rebucketed or inferred.',
         ' Privacy minimum: ', cfg$privacy_min, ' distinct people per displayed cell.</p>',
         '<aside>Credits are consumption units, not currency. Positive activity does not establish licensing.',
         ' No token counts, task types, policy-limit totals, individual rankings, causal effects, ROI,',
         ' productivity or burnout claims are calculated. Counts describe people present in activity,',
         ' not the eligible or licensed population. Missing dates are not filled with zeros.',
         ' Review partial periods and source coverage before sharing.</aside>',
         '<div class="scroll"><table><caption>Released aggregates only; no marginal totals</caption>',
         '<thead><tr><th scope="col">MetricDate</th><th scope="col">ServiceId</th>',
         '<th scope="col">Group</th><th scope="col">Distinct people</th>',
         '<th scope="col">Copilot credits</th></tr></thead><tbody>',
         paste(rows, collapse = ""), '</tbody></table></div>',
         '<p><small>Read provenance.json for source hashes, repository revision, configuration and package versions.',
         ' This release uses whole-table suppression; separate releases must be disclosure-reviewed together.</small></p>',
         '</body></html>')
}

build_dashboard <- function(cfg, kit) {
  if (cfg$mode != "real") fail("build is real-only. Use reproduce for synthetic references.")
  if (cfg$report == "github")
    fail("Real GitHub build is unsupported: inspect the export and obtain an approved schema, grain and coverage contract.")
  if (is.null(cfg$inputs$activity) || is.null(cfg$inputs$people))
    fail("Consumption build requires inputs.activity and inputs.people.")
  before <- file_hashes(cfg$inputs)
  tables <- lapply(cfg$inputs, read_csv)
  data <- validate_consumption(cfg, tables)
  release <- aggregate_credits(data, cfg$privacy_min)
  prov <- provenance(cfg, "build", cfg$inputs, kit)
  if (!identical(before, prov$input_hashes))
    fail("Inputs changed during build; rerun against stable exports.")
  prov$release_status <- release$status
  prov$privacy_policy <- "Withhold the entire table if any date/service/group cell is below privacy_min. No marginal totals."
  dest <- output_folder(cfg, "build")
  # Do not serialize pre-suppression counts, raw rows, keys, or group labels.
  csv <- release$table
  for (column in c("service", "group")) {
    dangerous <- grepl("^[[:space:]]*[=+@-]|[\\r\\n\\t]", csv[[column]])
    csv[[column]][dangerous] <- paste0("'", csv[[column]][dangerous])
  }
  utils::write.csv(csv, file.path(dest, "aggregates.csv"), row.names = FALSE,
                   fileEncoding = "UTF-8")
  writeLines(dashboard_html(release, cfg, prov$generated_utc),
             file.path(dest, "dashboard.html"), useBytes = TRUE)
  write_json(prov, file.path(dest, "provenance.json"))
  cat("Build completed:", release$status, "(aggregate outputs only).\n")
  invisible(release)
}

reproduce_dashboard <- function(cfg, kit) {
  if (cfg$mode != "demo") fail("reproduce is demo-only and never accepts real inputs.")
  required <- c("rmarkdown", "flexdashboard", "dplyr", "tidyr", "ggplot2",
                "scales", "knitr", "stringr", "vivainsights",
                if (cfg$report == "consumption") "ggrepel")
  need(required)
  if (!rmarkdown::pandoc_available()) fail("Pandoc is missing. Set RSTUDIO_PANDOC to an existing Pandoc folder.")
  inputs <- reference_files(cfg)
  before <- file_hashes(inputs)
  dest <- output_folder(cfg, "reproduce")
  work <- file.path(dest, "work")
  dir.create(work)
  for (i in seq_along(inputs)) {
    target <- file.path(work, names(inputs)[i])
    dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
    if (!file.copy(inputs[i], target, overwrite = FALSE)) fail("Cannot copy reference into isolated work folder.")
  }
  input <- file.path(work, names(inputs)[1])
  old <- getwd()
  on.exit(setwd(old), add = TRUE)
  setwd(work)
  rmarkdown::render(input, output_file = "dashboard.html", output_dir = dest,
                    intermediates_dir = work, knit_root_dir = work,
                    output_options = list(self_contained = TRUE, mathjax = NULL),
                    envir = new.env(parent = globalenv()), quiet = TRUE)
  if (!identical(before, file_hashes(inputs))) fail("Source reference changed during reproduction.")
  prov <- provenance(cfg, "reproduce", inputs, kit)
  prov$pandoc <- as.character(rmarkdown::pandoc_version())
  prov$notice <- "Synthetic demonstration only. Reference metrics and illustrative narratives do not verify a real-data contract."
  generated <- list.files(work, recursive = TRUE, full.names = TRUE)
  names(generated) <- substring(generated, nchar(work) + 2L)
  prov$work_hashes <- file_hashes(generated)
  write_json(prov, file.path(dest, "provenance.json"))
  cat("Synthetic reference rendered in the isolated reproduce folder. Source files unchanged.\n")
  invisible(prov)
}

run_dashboard <- function(action, config_path, kit) {
  cfg <- read_config(config_path, kit)
  switch(action, inspect = inspect_dashboard(cfg, kit),
         reproduce = reproduce_dashboard(cfg, kit), build = build_dashboard(cfg, kit),
         fail("Unknown action."))
}
