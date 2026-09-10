# Run with Rscript render-github-developer-experience.R from this folder,
# or pass this script's full path to Rscript from any working directory.
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
if(length(script)) setwd(dirname(normalizePath(script)))
required <- c('rmarkdown','flexdashboard','dplyr','tidyr','ggplot2','scales',
              'knitr','stringr','vivainsights')
missing <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop('Missing packages: ',paste(missing,collapse=', '))
if(!rmarkdown::pandoc_available()) stop('Pandoc is required. Set RSTUDIO_PANDOC to its folder.')
env <- new.env(parent=globalenv())
rmarkdown::render('github-copilot-developer-productivity-simulation.Rmd', envir=env)
print(env$validation_summary)
print(env$joint_counts)
cat('R:',R.version.string,'\n')
print(vapply(required,function(p) as.character(packageVersion(p)),character(1)))
