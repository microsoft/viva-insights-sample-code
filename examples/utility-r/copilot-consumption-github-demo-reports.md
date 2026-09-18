# Copilot demo reports

Two self-contained R Markdown flexdashboards demonstrate different analysis questions.
All data used by these two demos are generated synthetic data. Neither report establishes
causal effects, wellbeing or overall productivity from tool activity.

Both reports are built around the Consumption query and the GitHub Copilot query, both
accessible from the **Customised query** tab under **Create analysis** in the Viva Insights
analyst experience. See the
[AI cost query documentation](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query)
and the
[metric reference](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
for the official contracts.

| Report | Source | Rendered report | Scope |
|---|---|---|---|
| Copilot Consumption and Ways of Working | [Rmd](copilot-consumption-ways-of-working-simulation.Rmd) | [HTML](copilot-consumption-ways-of-working-simulation.html) | Copilot credit consumption alongside collaboration patterns, by service, organisation and function, including credit concentration and heavy-user consumption patterns. |
| Developer Experience and Copilot | [Rmd](github-copilot-developer-productivity-simulation.Rmd) | [HTML](github-copilot-developer-productivity-simulation.html) | Baseline working conditions for the developer population, including collaboration network breadth, with GitHub Copilot activity, feature/model/language breakdowns, M365 Copilot credit consumption and an evaluation framework. |

## Reading consumption intensity

The Consumption report separates **how much** a person consumes from **how
consistently** they consume it. Credit concentration reports the share of each
product's observed credits used by its highest-volume people, ranked within
that product; M365 and GitHub credits are never pooled into a shared
denominator. The heavy-user pattern split then divides heavy users into
sustained and intermittent consumers.

Every threshold is a parameter declared in the report setup
(`CONCENTRATION_CUTS`, `HEAVY_USER_SHARE`, `HIGH_WEEK_PERCENTILE`,
`SUSTAINED_WEEK_SHARE`) and the published rule text is generated from those
constants, so a threshold change cannot leave a stale rule on the page.
Narrower heavy-user cuts are permitted and are withheld automatically whenever
a resulting group falls below the publication floor — the highest-volume people
are also the fewest. Consistency is measured inside the 13-week product window
only: it is a lookback description, not a trend, and unobserved weeks are
excluded from both the threshold and the per-person counts.

## Navigating Developer Experience
Start with **Overview**, then explore **Collaboration**, **Focus**, **After-hours**
and **AI**. The **More** menu holds GitHub feature/model/language breakdowns,
working-pattern comparisons and methods. The overview's pillar cards are links;
exact values, composition and longer definitions expand in place.

**Collaboration covers both load and breadth.** Collaboration, meeting, email,
chat and call hours describe how much coordination happens and through which
channels. Network measures — internal and external network size, strong ties,
diverse ties and network outside the organisation — describe how many distinct
people that coordination reaches. These count people rather than hours, and
Viva Insights derives them over a trailing window, so they are standing levels
rather than weekly flows: they are never summed across weeks and should not be
read as week-on-week movement. A larger network is not inherently better. The
working-patterns page repeats the network comparison across recorded-GitHub-use
groups, where the differences remain unadjusted and descriptive.

The report uses embedded SVG charts, a reflowing small-multiple grid and separate
wide-chart renders for desktop, tablet and mobile. Long tables show ten rows first
with an expandable remainder; on phones, column labels stay beside each value.
These presentation changes do not equate missing observation with zero or weaken
the privacy floor. "Recorded GitHub use" is a subset of developers with complete
GitHub activity observation, not another name for that observed population.

## Simulated values, real schema

The `_data/` folder contains **fully synthetic values in real export schemas**. Every file
name, column name and key matches a real Viva Insights query export. This is a deliberate
correction: an earlier version of these demos simulated the schema as well as the values,
and carried columns that exist in no export — token counts, a task-type breakdown, an
eligibility/coverage reference file, and an M365 actions feed with no corresponding
query.

Sample code written against an invented schema cannot be pointed at a real export without
rework, so the fixtures were rebuilt against the published contracts. The single
deterministic generator is [simulate-query-exports.R](simulate-query-exports.R), and
[_data/README.md](_data/README.md) is the authoritative manifest for keys, fields, grains
and join rules.

```
_data/
  person-query/
    PersonQuery.csv
  consumption-query/
    PeopleMetaData.csv
    PersonM365CreditsMetrics.csv
    PersonGitHubCreditsMetrics.csv
  github-query/
    PersonGitHubActivityMetrics.csv
    GitHubActivityBreakdownByFeatureMetrics.csv
    GitHubActivityBreakdownByLanguageFeatureMetrics.csv
    GitHubActivityBreakdownByLanguageModelMetrics.csv
    GitHubActivityBreakdownByModelFeatureMetrics.csv
```

One folder per query, mirroring an unpacked download. All three outputs describe one
shared population, so cross-query joins are meaningful exactly as they would be in a real
tenant. The CSVs are ordinary readable files with no zip dependency. The generated HTML
embeds its charts, styles and scripts, and embeds no person identifiers or local paths.

### The rendered HTML is excluded from CodeQL

Because the rendered reports are self-contained, `rmarkdown` inlines vendored
third-party JavaScript into each one — minified Bootstrap 3, the jQuery
`stickyTableHeaders` plugin and flexdashboard's own bundle. CodeQL raises DOM-XSS
alerts inside that vendored code. None of it is written here, flexdashboard 0.6.3
still ships Bootstrap 3 so there is no upgrade path, and editing minified vendor code
inside a generated file would be discarded on the next render.

`.github/codeql/codeql-config.yml` therefore scopes CodeQL away from
`examples/**/*.html`. Only the generated build artifacts are out of scope: the
authored sources that produce them (`.Rmd`, `.R`), the authored JavaScript under
`assets/js/` and `scripts/`, and the Jekyll templates all remain scanned. The config
becomes active only when the repository property `github-codeql-config-file` is set
to `.github/codeql/codeql-config.yml`; the repository uses CodeQL default setup, so
no `.github/workflows/codeql.yml` should be added.

Regenerate everything with:

```powershell
cd examples/utility-r
Rscript simulate-query-exports.R
```

The generator is seeded and validates the whole ordered schema bundle and synthetic
allocation invariants before writing.
It fails rather than exporting a broken file.

### Points where real exports catch people out

- **`PeopleHistoricalId`, not `PersonId`, joins the Consumption activity files to
  `PeopleMetaData`.** The metadata file has no `PersonId` column at all, and the key is
  **opaque**: read the `PersonId` → `PeopleHistoricalId` crosswalk out of the activity
  files rather than reconstructing it. A real download may coincidentally show the same
  constant numeric suffix on every row; that is not a contract. These fixtures mint the
  key independently of `PersonId`, so reconstruction fails here instead of in production.
- **`PersonM365CreditsMetrics` is one row per person _per service_ per day.** Code that
  assumes one row per person per day silently over-counts.
- **Consumption metric columns contain spaces** (`Total Copilot Credits used`,
  `Session count`). Read with `check.names = FALSE`.
- **Empty policy/limit cells need a verified interpretation.** This recipe uses empty
  values alongside an all-zero policy ID for no policy; a real empty value alone
  does not establish absence of policy.
- **The GitHub activity file contains explicit zero rows.** A zero row is observed
  inactivity; a missing row has unknown status, not proof of no provisioning.
- **`PersonGitHubCreditsMetrics` is sparse, but `PersonGitHubActivityMetrics` is not.**
  The credit file carries a row only where billable usage occurred, so activity coverage
  and credit coverage are two different questions. Requiring five credit rows a week to
  accept an activity week discards every fully observed but partly inactive week — in
  these fixtures that was 89% of them. Resolve credit coverage on its own terms: the
  set of credit dates must equal the set of observed billable dates, and a week with
  no billable day is a measured zero. Compare the dates rather than their counts — a
  missing credit date offset by a spurious one leaves the counts equal.
- **Category membership in the breakdown files is persistent, not per-day noise.**
  Developers keep stable languages, models and surfaces, so cohorts share the same
  breakdown cells. A simulation that re-rolls those categories each day gives almost
  everybody a unique membership signature, and a linked cross-tab family built on that
  can never be published under a group-size floor.
- **In this simulation only, the four breakdowns are margins of one allocation.**
  Equality with activity and completion-model attribution are illustrative synthetic
  invariants, not verified real-export semantics. Confirmed headers alone never justify
  enforcing equality. The daily two-dimensional exports have no `Share` column.
- **Product history is shorter than collaboration history.** Here the person query covers 26
  weeks and the product feeds the final 13. Do not assume the windows align.

## Deriving eligibility and coverage

No Viva Insights query produces an eligibility or coverage file, so these signals must be
kept separate by evidence class:

| Question | Real signal |
|---|---|
| M365 eligible in this week? | `Total_Copilot_enabled_days > 0`; static `IsCopilotLicensed` is context, not an override for zero-enabled weeks |
| Provisioned for GitHub Copilot? | Independent provisioning evidence; a row establishes observation only and an absent row remains unknown |
| Was this GitHub week observed? | The weekday activity rows, including their measured zeros — not the sparse credit rows |
| Are this week's GitHub credits resolved? | The credit dates match the observed billable dates exactly, with no missing and no unexpected date; a week with no billable day resolves to a measured zero |
| Used Copilot on this day? | A row with a non-zero measure, not the absence of a row |

Fill an absent activity record with zero **only** when eligibility is independently
established. Against a real export you also need evidence of ingestion completeness and
expected coverage before absence can be read as non-use. A sparse export does not by itself
prove non-use. M365 completeness is unknown by default in the helper; no filename or
schema match certifies it. Consumption missing person-weeks remain unknown; recorded
sums divided by analysis weeks are lower bounds, not verified complete-period totals.

M365 Copilot credits and GitHub AI credits remain separate units. No common-unit
conversion, combined total or cross-product share is established.

## Privacy

Every published group requires at least 10 distinct people. A composition breakdown with any
cell below 10 is withheld with all linked margins. Positive contributors, overlapping
membership patterns and complements also meet the floor. Charts contain aggregates rather than
individual points. Identifiers in `_data/` are synthetic GUIDs that exist only to demonstrate
joins.

Do not put customer exports in this sample repository or in shareable HTML.

## Rerun

Knit either Rmd in RStudio, or from `examples/utility-r`:

```powershell
Rscript render-github-developer-experience.R
Rscript -e "rmarkdown::render('copilot-consumption-ways-of-working-simulation.Rmd')"
```

The render script also works when passed by full path from another working directory.
It checks existing packages and Pandoc, renders the HTML and prints data-validation results.
If Pandoc is not found, set `RSTUDIO_PANDOC` to its installed folder before running the script.
The script does not install packages.

The developer report sources
[github-developer-experience-helpers.R](github-developer-experience-helpers.R), which reads
the committed CSVs and performs derivation, validation, aggregation and presentation. It no
longer generates data; that responsibility belongs to `simulate-query-exports.R`.

Dependencies: R, Pandoc, `rmarkdown`, `flexdashboard`, `knitr`, `dplyr`, `tidyr`,
`ggplot2`, `scales`, `stringr`, `htmltools` and `vivainsights`. The developer report's
SVG device also needs Cairo graphics support (`capabilities("cairo")` in R).
Runtime package versions appear in each report's methods.

After rendering Developer Experience, its responsive layout and navigation can be
checked with the same Playwright and system Edge setup used for report screenshots:

```powershell
node scripts\check-developer-report-layout.js
node scripts\capture-report-screenshots.js --github-only
```

Run these from the repository root. The layout check covers all eight pages at
desktop, tablet and phone widths, including expanded content and keyboard controls.
Pass an output directory to the layout script to retain screenshots.

## Adapting to real data

Never replace committed fixtures with customer files. Use a separate approved workspace.
The supplied real v1 runner is credits-only; sessions, Person Query associations and
GitHub panels need separately implemented and verified adapters. Matching fixture headers
is not a like-for-like real build or semantic certification. Confirm population scope, person identifiers, date grains, licence history, expected
coverage, metric units, query filters and privacy before relying on the joins.
`vivainsights::import_query()` can import supported flexible queries.

## Research and measurement gaps

- [SPACE](https://queue.acm.org/detail.cfm?id=3454124): multidimensional productivity across
  satisfaction and wellbeing, performance, activity, communication and collaboration, efficiency and flow.
- [DevEx](https://queue.acm.org/detail.cfm?id=3595878): feedback loops, cognitive load and flow,
  combining developer feedback with system measures.
- [DORA](https://dora.dev/guides/dora-metrics/): current five-metric delivery framework at the
  application or service level, covering throughput and instability.
- [Viva Insights metric definitions](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics):
  primary source for paraphrased meeting, focus and after-hours definitions.

Survey, pull-request, delivery and service-quality outcomes are absent. No outcome fields were
fabricated to complete the framework. Calendar availability does not measure coding or
psychological flow. After-hours collaboration does not measure total work time and does not
diagnose burnout. Credit consumption measures resource use, not the quality or value of the
output.

**Next step for real use:** the manager, analyst and delivery owner should agree the workflow
question, measurement owners, comparison design and success criteria before running a pilot.
Synthetic values require no operational response.
