# Copilot query dashboard — executable R starter

Three independent commands support inspection, isolated reference reproduction,
and a deliberately modest real-data credit report. No model-generated report code,
network data connections, synthetic fallback, or edits to source reports are needed.

```powershell
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R inspect <config.json>
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R reproduce <config.json>
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R build <config.json>
```

Run from the repository root, or use the entry point's absolute path. On Windows,
replace `Rscript` with `& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'` if needed.

## Dependencies

Inspect/build use base R plus **jsonlite** and Git. Reproduction additionally
uses the reference reports' installed packages: rmarkdown, flexdashboard, dplyr,
tidyr, ggplot2, scales, knitr, stringr, vivainsights, and **ggrepel** for Consumption.
Pandoc is required only for reproduction; rmarkdown discovers it, or set
`RSTUDIO_PANDOC` to the folder containing an existing `pandoc` executable.
The CLI reports missing dependencies but never installs them automatically.

## Choose an action

| Action | Mode | Support | Outputs below `output_dir` |
|---|---|---|---|
| `inspect` | demo | Checks clean reference inputs; no synthetic generator runs | `inspect/readiness.json` |
| `inspect` | real | Bounded schema/count summary; Consumption mapping/readiness validation; GitHub blocker | `inspect/readiness.json` |
| `reproduce` | demo **only** | Complete supplied Consumption or GitHub reference dashboard | `reproduce/dashboard.html`, `reproduce/provenance.json`, isolated `reproduce/work/` |
| `build` | real **only** | Consumption credits by period, service, optional group | `build/dashboard.html`, `build/aggregates.csv`, `build/provenance.json` |

Each action works without another action's artifacts. Each action directory must
not already exist: the CLI refuses to overwrite it, even after a failed run.
Inspect then build can use the same `output_dir`; reruns require a new directory.
Outputs must be **outside the repository** and cannot contain configured inputs.
Existing path ancestors are canonicalized before containment checks. Dot segments
(`.` or `..`) after nonexistent folders are rejected: use an absolute output path
without unresolved dot segments. Rejection happens before creating any directory.
Keep outputs in a private, access-controlled workspace; inspection and provenance
are local analyst artifacts, not automatically approved for distribution.

### Demo journeys

```powershell
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R reproduce frontier-analytics/starter-kits/copilot-query-dashboard/configs/demo-consumption.json
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R reproduce frontier-analytics/starter-kits/copilot-query-dashboard/configs/demo-github.json
```

The supplied configs put results in a sibling `copilot-dashboard-work` directory.
They reproduce the complete **synthetic** references, not the smaller real adapter.
Demo configuration accepts no data filters or mapping approvals and requires the
references' fixed privacy minimum of 10; it never pretends custom settings were applied.
Consumption copies only its five supplied fixture CSVs and Rmd. GitHub copies
its Rmd and synthetic helper, which writes generated CSVs only inside the isolated
work folder. Only tracked, unmodified reference files are accepted; never replace
demo fixtures with customer data. Rendering and intermediate files stay in the
work/output folder, and original hashes are checked after rendering.

The full reference reports contain illustrative measures and narratives that are
**not** verified real-query fields or real-world findings. In particular, the
Consumption demo's tokens/task types cannot be inferred from the official export.

## Exact JSON configuration

All paths resolve **relative to the JSON file**, never to the shell working
directory. Absolute paths are accepted. JSON Windows paths need escaped backslashes
(`"C:\\private\\activity.csv"`); forward slashes also work in JSON paths.
If you move an example config, update its paths. Do not commit private configs/data.

| Field | Contract |
|---|---|
| `report` | Required: `"consumption"` or `"github"` |
| `mode` | Required: `"demo"` or `"real"` |
| `repo_root` | Required path to the Git checkout containing the references |
| `output_dir` | Required private folder outside both `repo_root` and this runtime's repository |
| `inputs` | Real only. Object with `activity` and (for Consumption) `people` CSV paths. GitHub inspection accepts activity plus optional people. No Person Query adapter. |
| `mappings` | Real Consumption only: exact source-column mappings below; no normalization or fuzzy guessing |
| `mapping_approval` | Build requires `{"approved":true,"evidence":"reviewed contract or approval reference"}`. Inspection validates structure without approval. |
| `granularity` | Consumption validation/build requires `"day"`, `"week"` or `"month"` matching the export; no inference/rebucketing |
| `dates` | Optional inclusive **period-label** filter: `{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}` |
| `group` | Optional boolean, default `false`; `true` requires the metadata group mapping |
| `privacy_min` | Optional integer >=10, default 10; demo references require exactly 10 |

Required Consumption mappings:

```json
{
  "mappings": {
    "activity": {
      "person_id": "PersonId",
      "service_id": "ServiceId",
      "date": "MetricDate",
      "people_id": "PeopleHistoricalId",
      "credits": "Total Copilot Credits used"
    },
    "people": {
      "people_id": "PeopleHistoricalId",
      "licensed": "IsCopilotLicensed",
      "group": "Organization"
    }
  }
}
```

Omit `mappings.people.group` when `group` is false. Mapping objects must have exactly
the required fields, each pointing to a distinct, exact existing source column.
`configs/real-consumption.example.json` deliberately starts with approval **false**.
Inspect it with local paths and explicit mappings while approval is false or
absent. Inspection validates numeric/date values, keys, joins and declared grain,
then reports `status: "awaiting_approval"`, `structural_validation: "passed"`,
`supported_panels`, `excluded_panels`, `privacy_status` and `next_action`.
Structural validation does not independently establish a source's business semantics:
review the export contract, declared credits unit and selected group/grain/window.
Only **after the user's decision**, set approval to true and record the decision
reference in `evidence` before build. Inspection never changes the approval.

Invalid structure produces `status: "blocked"` even when approval is absent.
If privacy withholds the release or the window has no records, supported panels
are empty and the next action is to review scope/coverage, not to publish.
Approval never bypasses privacy. Inspection writes blockers into readiness.json
rather than pretending an adapter is ready. Its successful exit means inspection
completed, not build approval.

## Real Consumption contract and limitations

Grounding: [Microsoft Learn: custom consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query)
(checked 11 September 2026). The activity key is **PersonId + ServiceId + MetricDate**.
Activity joins metadata on **PeopleHistoricalId**, not PersonId. Metadata must
have unique PeopleHistoricalId and explicit **IsCopilotLicensed** (`true`/`false`,
case-insensitive). This adapter validates licensing metadata but does not report
licensed denominators or infer licensing from activity.

The current official metrics include **Total Copilot Credits used**, **Session
count**, **User limit**, and **Spending policy limit**. This version deliberately
uses only credits. It does not sum limits, infer tokens/task types, or treat credits
as currency. Extra unmapped columns are ignored, not reverse-engineered.

Build rejects duplicate activity or metadata keys, one historical ID assigned to
multiple people, unmatched metadata (even without grouping), missing/padded IDs,
missing groups, invalid dates, negative/nonfinite/malformed credits, ambiguous
mappings and schema-normalization collisions. CSVs must be UTF-8, comma separated
with headers; empty cells are missing, credits use decimal points and optional
scientific notation, and dates are strict ISO dates. There is no `keep first`,
imputation, automatic deduplication, metric alias inference or demo fallback.
Validation covers the entire input before applying the date filter.

Output grain is **source MetricDate label × ServiceId × group** (or all people).
Date filtering does not split weekly/monthly periods. Ensure full periods and
coverage in the source export: missing periods are not silently filled with zeros.
No individual ranks, causal/ROI/productivity/burnout claims, or Person Query joins
are produced. The HTML is a self-contained, accessible aggregate table, not a
recreation of the synthetic report's seven-page story.
Its header displays the selected period-label window, observed released period
range, and generation timestamp in UTC (the same timestamp as provenance).
For suppressed releases, the observed range is withheld too; an empty window is
explicitly labeled as having no observations.

### Privacy and output safety

Every displayed cell requires at least `privacy_min` **distinct PersonIds**,
including people with zero credits present in that source cell. Activity rows and
historical metadata versions are never treated as distinct people. If **any** cell
is too small, the **entire release table** is withheld, including its labels,
counts and credits. There are no grand totals/marginal tables that expose hidden
composition by subtraction. Empty/suppressed runs still produce an explanatory
HTML, a header-only CSV and provenance.

This is a conservative single-release rule, **not differential privacy**.
Review overlapping windows, separate reports and external information together
before publication; repeated releases can enable differencing. Inspect prints
only bounded column names (80 columns, 120 characters each), counts, key/grain
description and date range, never sample records, IDs, service values or HR values.
HTML escapes labels; spreadsheet-formula-like CSV service/group labels receive a
leading apostrophe. The CSV contains only the released aggregates.

## Real GitHub: inspect only

Use `configs/real-github-inspect.example.json` with your local export. Its schema
is **not assumed to match the synthetic helper**. Build stops with an actionable
blocker until an approved adapter establishes fields, grain, identity, eligibility,
coverage and semantics. No fabricated “verified GitHub adapter” is provided.

## Reproducibility and tests

Readiness embeds provenance; build/reproduce write it separately. It records
explicit real/synthetic mode, repository HEAD, **MD5 content checksums** of input
and runtime files, loaded package versions, R version, configuration mappings,
approval reference, privacy rule and UTC execution time. Demo provenance also
records Pandoc and isolated work-file hashes. MD5 is a reproducibility checksum,
not an authenticity/security guarantee. Real raw records and input absolute paths
are not copied into provenance. Do not put customer values into approval text.

Run existing base-R assertions (no new test framework):

```powershell
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/tests/run-tests.R C:\private\test-results
# Optional: also render both references and verify original file hashes:
Rscript frontier-analytics/starter-kits/copilot-query-dashboard/tests/run-tests.R C:\private\test-results --render-demos
```

The output parent must already exist outside the repository. Tests generate small,
explicitly synthetic CSV cases there, exercise validation, aggregate totals,
whole-table suppression, output isolation, escaping and mode separation, and write
`dashboard-tests-<timestamp>/test-results.txt`. Fixtures and generated test artifacts
are never placed in the source tree.
