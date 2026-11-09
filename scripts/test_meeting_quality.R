# Base-R flag and duration regressions, also executed by the R PR job.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
root <- dirname(dirname(normalizePath(script)))
source(file.path(root, "frontier-analytics", "skills", "viva-insights-analysis",
                 "examples", "meeting_quality.R"))
frame <- data.frame(
  Number_of_attendees = rep(3, 6), Intended_participant_count = rep(3, 6),
  Cancelled = c("False", " No ", "N", "0", "false", "0.0"),
  All_Day_Meeting = rep("No", 6), Duration = rep(1, 6)
)
before <- frame
stopifnot(all(quality_mask(frame)), identical(frame, before))
stopifnot(all(normalise_flag(c("TRUE", "Yes", "y", "1", "1.0"))))
stopifnot(all(is.na(normalise_flag(c(NA, "", " #N/A ", "null")))))
fails <- function(action, pattern) {
  result <- tryCatch({ action(); NULL }, error = identity)
  stopifnot(inherits(result, "error"), grepl(pattern, conditionMessage(result)))
}
bad <- frame
bad$Cancelled[1] <- "maybe"
fails(function() quality_mask(bad), "Unrecognised")
frame$Duration <- c(1 / 60, 8, 0.01, 8.01, NA, 1)
frame$Intended_participant_count[6] <- 151
stopifnot(identical(quality_mask(frame), c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE)))
stopifnot(quality_mask(frame, max_intended = 200)[6])
fails(function() quality_mask(frame[names(frame) != "Duration"]), "including Duration")
cat("Meeting quality: R flag, missingness and boundary checks passed.\n")
