# Run from the repository root: Rscript scripts/test_segmentation.R
suppressMessages(library(vivainsights))
example <- file.path("frontier-analytics", "skills", "viva-insights-analysis",
                     "examples", "r", "03_copilot_segmentation.R")
panel <- function() data.frame(
  PersonId = rep("synthetic", 30),
  MetricDate = seq(as.Date("2026-01-05"), by = "week", length.out = 30),
  Total_Copilot_enabled_days = rep(5, 30),
  Copilot_actions_taken_in_Teams = rep(10, 30)
)
run <- function(input) {
  env <- new.env()
  env$pq_data <- input
  invisible(capture.output(suppressWarnings(source(example, local = env))))
  env$seg
}
for (column in "Total_Copilot_enabled_days") {
  for (invalid in c(0, NA)) {
    input <- panel()
    input[13, column] <- invalid
    result <- run(input)
    stopifnot(nrow(result) == nrow(input))
    for (width in c(4L, 12L)) {
      columns <- c(paste0("UsageSegments_", width, "w"), paste0("IsHabit", width, "w"),
                   paste0("target_metric_l", width, "w"))
      stopifnot(all(is.na(result[seq_len(width - 1L), columns])),
                all(is.na(result[13:(12L + width), columns])),
                !anyNA(result[(13L + width):30, columns]),
                !anyNA(result[12, columns]))
    }
  }
}
input <- panel()
input$Copilot_actions_taken_in_Teams[13] <- NA
result <- run(input)
for (width in c(4L, 12L)) {
  column <- paste0("UsageSegments_", width, "w")
  stopifnot(all(is.na(result[13:(12L + width), column])),
            !anyNA(result[(13L + width):30, column]))
}
licensed <- panel()
unlicensed <- panel()
unlicensed$PersonId <- "unlicensed"
unlicensed$Total_Copilot_enabled_days <- NA_real_
unlicensed$Copilot_actions_taken_in_Teams <- NA_real_
result <- run(rbind(licensed, unlicensed))
excluded <- result[result$PersonId == "unlicensed", ]
retained <- result[result$PersonId == "synthetic", ]
stopifnot(!any(excluded$is_licensed),
          all(is.na(excluded$UsageSegments_4w)),
          all(is.na(excluded$UsageSegments_12w)),
          !anyNA(retained$UsageSegments_12w[12:30]))
input <- panel()
input$licence_source_matched <- TRUE
input$licence_source_matched[13] <- FALSE
result <- run(input)
stopifnot(is.na(result$is_licensed[13]))
for (width in c(4L, 12L)) {
  column <- paste0("UsageSegments_", width, "w")
  stopifnot(all(is.na(result[13:(12L + width), column])),
            !anyNA(result[(13L + width):30, column]))
}
input <- panel()[-13, ]
result <- run(input[rev(seq_len(nrow(input))), ])
stopifnot(nrow(result) == 29L)
for (width in c(4L, 12L)) {
  column <- paste0("UsageSegments_", width, "w")
  dates <- panel()$MetricDate
  stopifnot(all(is.na(result[result$MetricDate %in% dates[14:(12L + width)], column])),
            !anyNA(result[result$MetricDate == dates[13L + width], column]))
}
input <- panel()
input$Copilot_actions_taken_in_Teams <- 0
result <- run(input)
stopifnot(all(is.na(result$UsageSegments_12w[1:11])),
          all(as.character(result$UsageSegments_12w[12:30]) == "Non-user"))
cat("R segmentation regression fixtures passed.\n")
