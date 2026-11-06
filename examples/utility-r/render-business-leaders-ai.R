# Run from any directory. The default output is beside the R Markdown source.
# Optional arguments: an alternative HTML output path, and --minimum=N to set the
# minimum group size (default 10, never below 5).
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
if (!length(script)) stop("Run this renderer with Rscript.")
utility <- dirname(normalizePath(script, mustWork = TRUE))
cli <- commandArgs(trailingOnly = TRUE)
minimum_flag <- grepl("^--minimum=", cli)
if (sum(minimum_flag) > 1) stop("Supply --minimum only once.")
minimum <- if (any(minimum_flag))
  suppressWarnings(as.numeric(sub("^--minimum=", "", cli[minimum_flag]))) else NULL
cli <- cli[!minimum_flag]
if (length(cli) > 1) stop("Usage: render-business-leaders-ai.R [output.html] [--minimum=N]")
output <- if (length(cli))
  file.path(normalizePath(dirname(cli[1]), mustWork = TRUE), basename(cli[1])) else
  file.path(utility, "business-leaders-ai-adoption.html")
required <- c("knitr", "rmarkdown", "dplyr", "tidyr", "jsonlite")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("Missing packages: ", paste(missing, collapse = ", "))
if (!rmarkdown::pandoc_available()) stop("Pandoc is required.")
if (!dir.exists(dirname(output))) stop("Output parent directory does not exist.")
setwd(utility)
temporary <- tempfile("business-leaders-", fileext = ".md")
environment <- new.env(parent = globalenv())
environment$params <- rmarkdown::yaml_front_matter("business-leaders-ai-adoption.Rmd")$params
if (!is.null(minimum)) environment$params$minimum <- minimum
tryCatch({
  knitr::knit("business-leaders-ai-adoption.Rmd", output = temporary,
              envir = environment, quiet = TRUE)
  rmarkdown::pandoc_convert(temporary, to = "html5", from = "markdown",
    output = output, options = c("--standalone", "--embed-resources", "--wrap=none",
                                "--metadata=pagetitle:Business Leaders AI Usage"))
}, finally = unlink(temporary))
if (!file.exists(output) || file.info(output)$size > 200 * 1024)
  stop("Dashboard output is missing or exceeds the 200 KB aggregate-file budget.")
cat("Rendered:", output, "\n")
cat("Minimum group size:", environment$payload$minimum, "\n")
cat("People:", dplyr::n_distinct(environment$inputs$panel$PersonId),
    "| Weeks:", length(environment$inputs$weeks),
    "| HTML bytes:", file.info(output)$size, "\n")
