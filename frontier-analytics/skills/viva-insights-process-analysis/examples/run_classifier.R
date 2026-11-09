# Matched R checks of the fabricated scope cases, not a real-export adapter.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
here <- dirname(normalizePath(script))
source(file.path(here, "..", "..", "viva-insights-analysis", "examples", "meeting_quality.R"))
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install jsonlite to run the R classifier checks.")
p <- jsonlite::fromJSON(file.path(here, "fixtures", "synthetic-audit-policy.json"))
d <- read.csv(file.path(here, "fixtures", "synthetic-audit-cases.csv"),
              stringsAsFactors = FALSE, check.names = FALSE, fileEncoding = "UTF-8")
for (column in c("quality_pass", "prior_in_scope", "expected"))
  d[[column]] <- normalise_flag(d[[column]])
if (!setequal(names(p$year_codes), as.character(p$years)))
  stop("Every comparison year needs an explicit supported code")
if (anyDuplicated(d$MeetingId) || any(!nzchar(d$MeetingId))) stop("Invalid event keys")
if (!is.logical(d$quality_pass) || anyNA(d$quality_pass)) stop("Invalid prepared quality flags")
if (any(!d$decision %in% c("", "Keep", "Remove", "Pending"))) stop("Invalid manual decision")
if (any(d$decision == "Pending" & is.na(d$prior_in_scope)))
  stop("Pending review requires its recorded prior disposition")
dates <- as.Date(d$Date)
if (anyNA(dates) || any(dates < as.Date(p$coverage_start) | dates > as.Date(p$coverage_through)))
  stop("Event outside configured source coverage")
match_unicode <- function(pattern, text)
  grepl(paste0("(*UTF)(*UCP)", pattern), enc2utf8(text), perl = TRUE, ignore.case = TRUE)
out <- vapply(seq_len(nrow(d)), function(i) {
  r <- d[i, ]
  if (!r$quality_pass || format(as.Date(r$Date), "%m-%d") < p$annual_start ||
      format(as.Date(r$Date), "%m-%d") > p$annual_end) return(FALSE)
  if (r$decision == "Remove") return(FALSE)
  if (r$decision == "Keep") return(TRUE)
  if (r$decision == "Pending") {
    if (is.na(r$prior_in_scope)) stop("Pending review requires its recorded prior disposition")
    return(r$prior_in_scope)
  }
  if (!r$decision %in% c("", "Keep", "Remove", "Pending")) stop("Invalid manual decision")
  codes <- paste0("\\b(?:", paste(unname(p$year_codes), collapse = "|"), ")\\b")
  topic <- any(vapply(c(p$categories, codes), match_unicode, TRUE, text = r$Subject))
  topic && !match_unicode(p$exclude_pattern, r$Subject)
}, TRUE)
stopifnot(identical(unname(out), d$expected))
cat("Fabricated audit: all", nrow(d), "R Unicode, title and decision cases passed.\n")
