# Copilot demo reports

Two self-contained R Markdown flexdashboards demonstrate different analysis questions.
All data used by these two demos are generated synthetic data. Neither report establishes
causal effects, wellbeing or overall productivity from tool activity.

| Report | Source | Rendered report | Scope |
|---|---|---|---|
| Copilot Consumption and Ways of Working | [Rmd](copilot-consumption-ways-of-working-simulation.Rmd) | [HTML](copilot-consumption-ways-of-working-simulation.html) | Consumption credits/tokens alongside collaboration patterns. This report and its data are unchanged by the developer-experience rebuild. |
| Developer Experience and Copilot | [Rmd](github-copilot-developer-productivity-simulation.Rmd) | [HTML](github-copilot-developer-productivity-simulation.html) | Baseline working conditions for all 900 developers, with separate GitHub Copilot and M365 Copilot use, coverage-aware comparisons and an evaluation framework. |

The developer report keeps its canonical filename so existing links continue to work.

## Developer Experience and Copilot

The manager-facing report contains six pages and an appendix:

1. **Developer landscape:** the 900-developer population, six teams, simulated role,
   seniority and tenure, working-week variation and compact joint-product footprint.
2. **Focus and coordination:** team medians and interquartile ranges, separate meeting
   characteristics, and inline distinctions between calendar availability and uninterrupted time.
3. **Sustainable workload:** after-hours distribution and persistence over observed weeks.
   The configurable convention is illustrative, not a health or burnout threshold.
4. **AI use:** joint use for both-eligible, fully covered developers, separate licence and
   coverage exceptions, product-specific frequency, feature use, acceptance and concentration.
5. **Working patterns:** three selected descriptive comparisons on a common eligible and
   observed population, with role and team context. No metric fishing or individual rankings.
6. **Trends and change:** 26 weekly windows with explicit valid denominators, followed by
   an evaluation framework requiring a documented intervention, comparator and balancing measures.
7. **Appendix:** source definitions, units, grains, joins, privacy, reproducibility,
   simulation assumptions, research references and missing evidence.

Pages 1 to 5 use **10 May to 4 July 2026**, the final eight complete weeks of the
**4 January to 4 July 2026** source window. Week starts are Sundays and dates use a UTC
convention. The previous end label omitted the final week’s six remaining days.
The report’s historical synthetic window is explicit and is not presented as current telemetry.

### Rerun

Knit the Rmd in RStudio, or run this file-based command from `examples/utility-r`:

```powershell
Rscript render-github-developer-experience.R
```

The render script also works when passed by full path from another working directory.
It checks existing packages and Pandoc, renders the HTML and prints data-validation results.
If Pandoc is not found, set `RSTUDIO_PANDOC` to its installed folder before running the script.
The script does not install packages.

The Rmd sources [github-developer-experience-helpers.R](github-developer-experience-helpers.R).
The helper contains deterministic simulation, source-key and population assertions, coverage-aware
aggregation, CSV exports and plot/table helpers. Seed `20260910` reproduces the exports.
The package’s `create_boxplot(..., mingroup = 10, return = "table")` supplies person-level
percentile summaries. Custom logic handles the extended joint-product coverage contract.

Dependencies: R, Pandoc, `rmarkdown`, `flexdashboard`, `knitr`, `dplyr`, `tidyr`,
`ggplot2`, `scales`, `stringr` and `vivainsights`. Runtime package versions appear in the appendix.

For the unchanged consumption demo, knit its Rmd or call
`rmarkdown::render("copilot-consumption-ways-of-working-simulation.Rmd")` from an R script.

## Synthetic files and schema boundaries

```
_data/
  consumption/                 # unchanged, separate report
    consumption-query/
    person-query/
    reference/
  github/
    README.md                  # extended illustrative schema manifest
    github-query/              # daily activity, full-window model/language allocations
    m365-query/                # separate illustrative M365 daily activity
    person-query/              # all 1,080 roster people, 26 weekly records each
    reference/                 # shared roster and weekly eligibility/coverage
```

See [_data/github/README.md](_data/github/README.md) for exact file keys, fields and source contracts.
The CSVs are ordinary readable files, with no zip dependency. The generated HTML embeds its charts,
styles and scripts and does not embed person identifiers or local paths.

The developer data are an **extended illustrative schema**, not a drop-in representation of a real
Viva Insights flexible query. In particular, the M365 feature actions, separate eligibility flags
and completeness reference are invented demonstration contracts. They are not Copilot consumption
credits or tokens. The helper generates product propensities independently, with no assumed advantage
for people who use both tools. Model/language allocations are secondary details, not maturity measures.

### Coverage and privacy

The roster includes 1,080 synthetic people, of whom 900 are developers. All 900 contribute to
working-condition baselines regardless of Copilot use. Daily feeds are sparse. The report aggregates
them to person-week before joining to the roster-led reference, with key and join-cardinality assertions.
Missing activity becomes zero only for product-eligible, fully covered weeks. Unresolved observation
remains missing, while ineligibility stays distinct from non-use. Joint categories require both
products’ eligibility and coverage throughout the eight-week window.

Every published group requires at least 10 distinct people. A composition breakdown with any cell
below 10 is withheld for that whole parent group. Charts contain aggregates rather than individual
points. The highest-volume decile’s aggregate share is shown separately for each product without
publishing identities or individual ranks. No charts combine M365 actions with GitHub suggestions,
chat requests or agent days into a common volume.

### Adapting to real data

Replace generation only after validating the actual export contracts. `vivainsights::import_query()`
can import supported flexible queries, but column renaming alone does not establish semantic parity.
Confirm population scope, person identifiers, date grains, licence history, expected coverage, metric
units, query filters and privacy before using the joins. Real sparse exports do not by themselves
prove non-use. Do not put customer exports in this sample repository or in shareable HTML.

## Rebuild validation

- RMarkdown rendered with R 4.6.1 and the package versions recorded in the appendix.
  A second render reproduced all seven CSV exports and the HTML byte-for-byte.
- Independent CSV checks confirmed unique keys, 900 developers, 23,400 developer-weeks,
  7,200 baseline developer-weeks, eligibility/coverage reconciliation, acceptance bounds,
  feature totals, model/language allocations and after-hours persistence.
- Microsoft Edge browser QA covered all seven pages at widths of 1440, 1280 and 390 pixels.
  All ten embedded plot images were visually inspected. Navigation and expandable tables
  worked, with no JavaScript errors, external asset requests or page-level horizontal overflow.
  Wide charts and tables intentionally scroll inside their cards on mobile.
- SHA-256 checks confirmed the consumption Rmd, HTML and five CSVs were unchanged.
  `git diff --check` passed. Temporary QA scripts and screenshots remain outside the repository.

## Research and measurement gaps

- [SPACE](https://queue.acm.org/detail.cfm?id=3454124): multidimensional productivity across
  satisfaction and wellbeing, performance, activity, communication and collaboration, efficiency and flow.
- [DevEx](https://queue.acm.org/detail.cfm?id=3595878): feedback loops, cognitive load and flow,
  combining developer feedback with system measures.
- [DORA](https://dora.dev/guides/dora-metrics/): current five-metric delivery framework at the
  application or service level, covering throughput and instability.
- [Viva Insights metric definitions](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics):
  primary source for paraphrased meeting, focus and after-hours definitions.

Learn, DORA and the [vivainsights function index](https://microsoft.github.io/vivainsights/llms.txt)
were checked during the rebuild. The ACM pages returned HTTP 403, so the report records that a fresh
full-text review was unavailable.

Survey, pull-request, delivery and service-quality outcomes are absent. No outcome fields were
fabricated to complete the framework. Calendar availability does not measure coding or psychological
flow. After-hours collaboration does not measure total work time or diagnose burnout. Collaboration
span was removed from the developer simulation and report rather than assigned an unsupported definition.

**Next step for real use:** the manager, analyst and delivery owner should agree the workflow question,
measurement owners, comparison design and success criteria before running a pilot. Synthetic values
require no operational response.
