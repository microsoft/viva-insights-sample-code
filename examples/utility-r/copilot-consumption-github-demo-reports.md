# Art-of-the-possible reports: Copilot Consumption query + GitHub Copilot query

## Overview

Two new flexible query types are available in the Viva Insights Analyst portal that this
repository did not previously have example code for:

- **Consumption query** – Copilot credit consumption data (the same metrics behind the AI
  Cost Dashboard), at the `PersonId` x `ServiceId` x date level.
- **GitHub query** – GitHub Copilot usage data (code completions, chat requests, agent
  adoption, language and model mix), at the `PersonId` x date level.

Both are custom flexible queries, created the same way as a Person Query or Meeting Query:
in the Viva Insights web app, select **Create analysis** > **Create custom query**, then pick
the relevant query template (**Consumption query** or **GitHub query**). See
[Create a custom consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query)
on Microsoft Learn for the consumption query setup steps (time period, grouping, HR
attributes, output channel).

This folder contains two demo reports that show what becomes possible when either query is
combined with a standard Person Query, answering questions neither query can answer alone:

| Report | File | Focuses on |
|---|---|---|
| Copilot Consumption and Ways of Working | [`copilot-consumption-ways-of-working-simulation.Rmd`](copilot-consumption-ways-of-working-simulation.Rmd) | Whether credit/token consumption is associated with collaboration volume and mix, workload (after-hours hours, collaboration span), and network metrics (internal/external network size) |
| GitHub Copilot and Developer Productivity | [`github-copilot-developer-productivity-simulation.Rmd`](github-copilot-developer-productivity-simulation.Rmd) | Whether GitHub Copilot usage is associated with meeting load, focus time, and ways-of-working metrics, specifically for the developer population |

Both are R Markdown [flexdashboard](https://pkgs.rstudio.com/flexdashboard/) reports. Knit
either file in RStudio, or from the command line:

```r
rmarkdown::render("copilot-consumption-ways-of-working-simulation.Rmd")
rmarkdown::render("github-copilot-developer-productivity-simulation.Rmd")
```

## The data is entirely synthetic

**No customer or production data of any kind is used anywhere in this folder.** Every query
export referenced by these reports is generated inline by the Rmd itself, from a fixed
random seed, to resemble the shape and column names a real query export would have. This
lets the two report templates run standalone, with no dependency on a live tenant, while
still demonstrating the full join/analysis workflow an analyst would follow with real
exports.

The generated CSVs are written under [`_data/`](_data) as a working example of the expected
input shape, split by report:

```
_data/
├── consumption/
│   ├── consumption-query/    # consumption-weekly.csv, consumption-task-types.csv
│   ├── person-query/         # person-query-weekly.csv, network-monthly.csv
│   └── reference/            # people-snapshot.csv
└── github/
    ├── github-query/         # activity-daily.csv, language-mix.csv, model-mix.csv
    ├── person-query/         # person-query-weekly.csv
    └── reference/            # people-snapshot.csv
```

To adapt either report to a real tenant, replace the synthetic data-generation chunks near
the top of the Rmd with `vivainsights::import_query()` calls against your own Consumption
query / GitHub query / Person Query exports, keeping the downstream column names consistent
with what's read from `_data/` today.

## What each report covers

Both reports move all simulation/methodology detail to an appendix page, so the front pages
read as if built on real data:

- An overview page with population-level headline metrics
- A distribution/concentration page (are results driven by a minority of users?)
- A team/function drill-down with a baseline comparison and a 3-dimension bubble or heatmap
  view
- Metric-specific pages for the report's focus area (ways of working, meeting load, focus
  time)
- A rigour-checks page (composition adjustment, within-person design) that keeps claims
  framed as association, not causation
- A methods and appendix page with the full simulation and join methodology
