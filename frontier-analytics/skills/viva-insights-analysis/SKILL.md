---
name: viva-insights-analysis
description: >
  Analyze Microsoft Viva Insights data with the open-source vivainsights R and
  Python packages. Use this skill whenever working with Viva Insights query
  exports (Person Query, Meeting Query, person-to-person or group-to-group
  network queries) or the vivainsights packages: importing and validating a
  query, computing metrics, segmenting Copilot usage, building the standard
  visualisations, running network or information-value analysis, and avoiding
  the well-known export data pitfalls (IsManager strings, "#N/A" text, non-English
  locales, privacy thresholds, trailing-window metric plateaus, holiday weeks).
  It also points to the viva-insights-sample-code repo and its Frontier
  Analytics toolkit for ready-made prompts, starter kits, and schema docs.
---

## What this skill is for

Microsoft Viva Insights lets analysts export "flexible queries" as CSVs from the
Analyst portal. The open-source **`vivainsights`** packages (R and Python) read,
validate, analyse, and visualise those exports in a consistent, best-practice way.

Use this skill when a task involves any of:

- Importing or validating a Viva Insights query export.
- Computing or visualising collaboration, meeting, email, focus, after-hours, or
  Copilot metrics.
- Segmenting a licensed population by Copilot usage.
- Person-to-person or group-to-group network analysis.
- Information Value / driver analysis, key-metrics scans, interrupted time-series.
- Debugging why a Viva Insights export loaded or aggregated strangely.

This skill is **customer-agnostic**. It contains no organisation-specific paths,
scopes, or dates. Adapt column lists, populations, and time windows to the data
in front of you.

## The packages at a glance

| | R | Python |
|---|---|---|
| Package | `vivainsights` | `vivainsights` |
| Docs | https://microsoft.github.io/vivainsights/ | https://microsoft.github.io/vivainsights-py/ |
| Source | `microsoft/vivainsights` | `microsoft/vivainsights-py` |
| Install | `install.packages("vivainsights")` | `pip install vivainsights` |
| Coverage | Fuller API (metric-family wrappers, text mining, ITSA, survival) | Core subset (import, identify, viz, IV, network) |
| Return arg | `return = "plot"` / `"table"` / `"data"` | `return_type = "plot"` / `"table"` / `"data"` |

Both packages are MIT-licensed. R is the more complete of the two; the Python
package mirrors its design for the most common workflows. **The single most
common cross-language bug is the return argument name: R uses `return`, Python
uses `return_type`.**

## Find the right function before writing custom code

Both packages now ship a **machine-readable function index** (`llms.txt`) and a
**function-discovery guide**, generated directly from the package and built
specifically so agents check them before reimplementing aggregation or
visualisation logic:

| | R | Python |
|---|---|---|
| `llms.txt` (machine-readable, task -> function -> privacy note) | https://microsoft.github.io/vivainsights/llms.txt | https://microsoft.github.io/vivainsights-py/llms.txt |
| Function-discovery guide (human-readable task index) | https://microsoft.github.io/vivainsights/articles/function-discovery.html | https://microsoft.github.io/vivainsights-py/function-discovery.html |

**Preference order when looking for a function:** fetch the relevant `llms.txt`
first. It is generated from the installed package and reflects the exact
current API surface. Fall back to `reference/packages.md` (this skill's
hand-curated, grouped inventory, which adds context `llms.txt` doesn't carry:
the R-only metric-family wrapper suffixes, sample-dataset shapes, export/theming
helpers, and the runtime-introspection snippets to confirm a signature). If they
disagree, `llms.txt` wins, since it is closer to the source of truth.

## How to use the reference files

Read the specific reference file for the task at hand rather than loading
everything:

| File | Read it when you need |
|---|---|
| `reference/packages.md` | The function inventory, grouped by purpose, with R/Python parity and verified signatures. A curated complement to the live `llms.txt` (see above). |
| `reference/query-schemas.md` | The shape of each query type (grain, key columns, raw vs imported column names) and the meeting quality filter. |
| `reference/data-pitfalls.md` | To diagnose or pre-empt a loading / aggregation problem (IsManager, "#N/A", locales, privacy threshold, trailing windows, holidays). |
| `reference/analysis-conventions.md` | To write honest, defensible claims (association vs causation), segment definitions, and period framing. |
| `reference/ecosystem.md` | The canonical upstream: package docs, the sample-code repo, and the Frontier Analytics toolkit to defer to. |
| `reference/environment.md` | Practical run tips: performant loads, headless plotting, script-file vs inline execution. |

## Worked examples

The `examples/` folder has matched Python and R scripts that run against the
built-in sample datasets (no customer data), each verified end-to-end. See
`examples/README.md` for the index. They cover import and validation, metric
ranking, Copilot usage segmentation, the Meeting Query quality filter, and
person-to-person network analysis.

## Minimal workflow

```r
# R
library(vivainsights)
pq <- import_query("person_query.csv")   # validates + cleans column names
check_query(pq)                           # structure + variable-type summary
create_rank(pq, metric = "Collaboration_hours", hrvar = "Organization")
```

```python
# Python
import vivainsights as vi
pq = vi.import_query("person_query.csv")
vi.create_rank(pq, metric="Collaboration_hours", hrvar="Organization",
               return_type="plot")
```

Both packages ship built-in sample datasets, so examples and tests never need
real customer data: `pq_data`, `mt_data`, `p2p_data`, `g2g_data` (R), and the
`load_pq_data()`, `load_mt_data()`, `load_p2p_data()`, `load_g2g_data()` loaders
(Python).

## Guardrails

- **Never embed customer data or identifiers** in scripts, examples, or committed
  files. Use the built-in sample datasets for illustration.
- **Respect privacy thresholds.** Do not report groups below the minimum
  aggregation size (default 5). See `reference/data-pitfalls.md`.
- **Prefer association over causation** in written claims unless a designed
  experiment supports otherwise. See `reference/analysis-conventions.md`.
- **Prefer an existing package function over new code.** Check the relevant
  `llms.txt` / function-discovery guide (see "Find the right function before
  writing custom code" above) or `reference/packages.md` before writing custom
  aggregation, filtering, or plotting logic. Both packages are built to be
  used this way, and a hand-rolled equivalent is more likely to miss a privacy
  threshold or an edge case the package already handles.
