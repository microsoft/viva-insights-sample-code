# Base R only. This is a bounded synthetic illustration, not a real-data engine.
require_ok <- function(condition, code) {
  if (!isTRUE(condition)) stop(code, call. = FALSE)
}

read_contract_csv <- function(path, fields) {
  require_ok(file.exists(path) && !dir.exists(path), "input_or_output_io")
  require_ok(file.info(path)$size <= 2000000, "file_too_large")
  widths <- count.fields(path, sep = ",", quote = "\"", comment.char = "",
                         blank.lines.skip = FALSE)
  require_ok(length(widths) >= 2 && all(widths == length(fields)), "invalid_csv_shape")
  rows <- read.csv(path, colClasses = "character", check.names = FALSE,
                   na.strings = NULL, strip.white = FALSE, comment.char = "",
                   blank.lines.skip = FALSE, fileEncoding = "UTF-8")
  require_ok(identical(names(rows), fields), "invalid_columns")
  require_ok(nrow(rows) > 0 && nrow(rows) <= 10000, "invalid_row_count")
  rows
}

numeric_field <- function(values, missing = FALSE) {
  unknown <- values == "" & missing
  pattern <- "^[+-]?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][+-]?[0-9]+)?$"
  require_ok(all(unknown | grepl(pattern, values)), "invalid_numeric")
  result <- rep(NA_real_, length(values))
  result[!unknown] <- as.numeric(values[!unknown])
  require_ok(all(is.finite(result[!unknown]) & abs(result[!unknown]) <= 1000000),
             "numeric_out_of_bounds")
  result
}

analyze <- function(data_path, metadata_path) {
  require_ok(grepl("^synthetic-.*\\.csv$", basename(data_path)), "synthetic_filename_required")
  meta_rows <- read_contract_csv(metadata_path, c("key", "value"))
  require_ok(!anyDuplicated(meta_rows$key), "duplicate_metadata_key")
  meta <- setNames(meta_rows$value, meta_rows$key)
  fixed <- c(generator = "viva-causal-synthetic-v1", synthetic_only = "true",
             row_grain = "person-month", exposure_field = "CollaborationHours",
             outcome_field = "CustomOutcome", outcome_units = "score points")
  require_ok(all(meta[names(fixed)] == fixed), "unsupported_metadata")
  meanings <- c("exposure_meaning", "count_meaning", "outcome_meaning",
                "identity_meaning", "measurement_meaning", "design")
  require_ok(all(!is.na(meta[meanings]) & nzchar(trimws(meta[meanings]))), "missing_field_meaning")
  cases <- c("effect", "confounding", "annual", "static", "quarterly", "missing",
             "mechanical", "one-period")
  case <- unname(meta["case"])
  require_ok(case %in% cases, "unsupported_case")
  grain <- if (case %in% c("annual", "static", "quarterly")) case else "monthly"
  mechanical <- if (case == "mechanical") "true" else "false"
  truth <- if (case == "confounding") "0" else if (case == "mechanical") "not_causal" else
    if (grain != "monthly") "not_identified" else "2"
  require_ok(all(meta[c("outcome_grain", "outcome_derived_from_exposure", "known_effect")] ==
                   c(grain, mechanical, truth)), "inconsistent_metadata")
  fields <- c("PersonId", "MetricDate", "CollaborationHours", "CopilotActions",
              "CustomOutcome", "OutcomePeriod")
  rows <- read_contract_csv(data_path, fields)
  people <- rows$PersonId
  periods <- rows$MetricDate
  require_ok(all(grepl("^synthetic-P[0-9]{4}$", people)), "invalid_synthetic_identity")
  require_ok(all(grepl("^[0-9]{4}-[0-9]{2}-01$", periods)), "invalid_date")
  dates <- as.Date(periods, format = "%Y-%m-%d")
  require_ok(all(!is.na(dates)) && all(format(dates, "%Y-%m-%d") == periods), "invalid_date")
  years <- as.integer(substr(periods, 1, 4))
  require_ok(all(years >= 2000 & years <= 2100), "invalid_date")
  require_ok(!anyDuplicated(rows[c("PersonId", "MetricDate")]), "duplicate_person_period")
  expected <- switch(grain, monthly = substr(periods, 1, 7), annual = substr(periods, 1, 4),
                     static = rep("static", nrow(rows)),
                     quarterly = paste0(years, "-Q", (as.integer(substr(periods, 6, 7)) - 1) %/% 3 + 1))
  require_ok(all(rows$OutcomePeriod == expected), "measurement_period_mismatch")
  x <- numeric_field(rows$CollaborationHours, missing = TRUE)
  count <- numeric_field(rows$CopilotActions)
  y <- numeric_field(rows$CustomOutcome)
  require_ok(all(is.na(x) | (x >= 0 & x <= 744)), "invalid_hours")
  require_ok(all(count >= 0 & count == floor(count)), "invalid_count")
  measurement_key <- paste(people, expected, sep = "|")
  require_ok(all(vapply(split(y, measurement_key), function(v) length(unique(v)) == 1,
                        logical(1))), "inconsistent_repeated_outcome")
  result <- data.frame(
    status = "blocked", reason = "", rows = nrow(rows), persons = length(unique(people)),
    periods = length(unique(periods)), outcome_measurements = length(unique(measurement_key)),
    missing_exposures = sum(is.na(x)), pooled_slope = "", twfe_slope = "",
    stringsAsFactors = FALSE
  )
  if (mechanical == "true") {
    result$reason <- "mechanical_outcome"
  } else if (grain %in% c("annual", "static")) {
    result$reason <- "no_within_person_measurement"
  } else if (grain == "quarterly") {
    result$status <- "readiness_only"
    result$reason <- "quarterly_reaggregation_required"
  } else if (result$periods < 2 || result$persons < 2) {
    result$reason <- "insufficient_panel"
  } else if (result$missing_exposures > 0) {
    result$reason <- "unknown_exposure"
  } else if (nrow(rows) != result$persons * result$periods) {
    result$reason <- "unbalanced_panel"
  } else {
    demean <- function(v) v - ave(v, people, FUN = mean) - ave(v, periods, FUN = mean) + mean(v)
    xd <- demean(x)
    yd <- demean(y)
    denominator <- sum(xd * xd)
    if (denominator <= 1e-10) {
      result$reason <- "no_within_exposure_variation"
    } else {
      pooled <- sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)
      beta <- sum(xd * yd) / denominator
      result$status <- "synthetic_demo"
      result$reason <- "not_real_data_inference"
      result$pooled_slope <- sprintf("%.10f", pooled)
      result$twfe_slope <- sprintf("%.10f", beta)
    }
  }
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 3) {
    cat("ERROR usage: Rscript run_example.R INPUT.csv METADATA.csv NEW_OUTPUT_DIRECTORY\n",
        file = stderr())
    return(2L)
  }
  tryCatch({
    require_ok(!file.exists(args[3]) && !dir.exists(args[3]), "output_exists")
    result <- analyze(args[1], args[2])
    require_ok(dir.create(args[3], showWarnings = FALSE), "input_or_output_io")
    write.table(result, file.path(args[3], "summary.csv"), sep = ",", row.names = FALSE,
                col.names = TRUE, quote = TRUE, na = "", eol = "\n")
    cat(paste0(result$status, ": ", result$reason, "\n"))
    0L
  }, error = function(error) {
    # Only our fixed error codes may reach stderr; parser errors can contain rows.
    code <- conditionMessage(error)
    allowed <- c("input_or_output_io", "file_too_large", "invalid_csv_shape", "invalid_columns",
                 "invalid_row_count", "invalid_numeric", "numeric_out_of_bounds",
                 "synthetic_filename_required", "duplicate_metadata_key", "unsupported_metadata",
                 "missing_field_meaning", "unsupported_case", "inconsistent_metadata",
                 "invalid_synthetic_identity", "invalid_date", "duplicate_person_period",
                 "measurement_period_mismatch", "invalid_hours", "invalid_count",
                 "inconsistent_repeated_outcome", "output_exists")
    if (!(code %in% allowed)) code <- "input_or_output_io"
    cat(paste0("ERROR ", code, "\n"), file = stderr())
    2L
  })
}

# Convert parser warnings to sanitized errors rather than printing input content.
options(warn = 2)
quit(save = "no", status = main())
