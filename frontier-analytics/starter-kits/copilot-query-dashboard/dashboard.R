#!/usr/bin/env Rscript
# Deterministic entry point; synthetic reference code is reachable only through reproduce.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
if (length(script) != 1L) stop("Run this entry point with Rscript.", call. = FALSE)
kit <- dirname(normalizePath(script, winslash = "/", mustWork = TRUE))
source(file.path(kit, "lib", "runtime.R"), local = TRUE)
status <- tryCatch({
  cli <- commandArgs(trailingOnly = TRUE)
  if (length(cli) != 2L || !cli[1] %in% c("inspect", "reproduce", "build"))
    fail("Usage: Rscript dashboard.R <inspect|reproduce|build> <config.json>")
  run_dashboard(cli[1], cli[2], kit)
  0L
}, dashboard_error = function(e) {
  cat("BLOCKED: ", conditionMessage(e), "\n", sep = "", file = stderr())
  1L
}, error = function(e) {
  # Parser and package exceptions can include source records or identifiers.
  cat("BLOCKED: execution failed. Check CSV syntax, file access, and documented dependencies locally.\n",
      file = stderr())
  1L
})
quit(status = status)
