# Deprecated compatibility entrypoint; the renderer's default output is the new HTML.
script_args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", script_args[grepl("^--file=", script_args)])
if (!length(script)) stop("Run this compatibility wrapper with Rscript.")
utility <- dirname(normalizePath(script, mustWork = TRUE))
renderer <- file.path(utility, "render-ai-usage-dashboard.R")
args <- commandArgs(trailingOnly = TRUE)
status <- system2(file.path(R.home("bin"), "Rscript"),
                  c(shQuote(renderer), vapply(args, shQuote, character(1))),
                  stdout = "", stderr = "")
if (is.null(status)) status <- 0L
quit(save = "no", status = status, runLast = FALSE)
