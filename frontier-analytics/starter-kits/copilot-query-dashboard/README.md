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

The full reference reports contain illustrative narratives and synthetic values that
are **not** real-world findings. Their fields do match the real query schemas, but
schema parity is not semantic parity, and tokens and task types are absent from both
the demo and the official export.

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
| `policy_additivity` | Real Consumption only, and only when `mappings.activity.policy_id` is mapped: `{"confirmed":true,"evidence":"export owner's confirmation reference"}`. Exactly these two fields. Required before build when several spending policy rows carry positive credits for the same person, service and date |
| `granularity` | Consumption validation/build requires `"day"`, `"week"` or `"month"` matching the export; no inference/rebucketing |
| `dates` | Optional inclusive **period-label** filter: `{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}` |
| `group` | Optional boolean, default `false`; `true` requires the metadata group mapping. Keep it false when the metadata has no HR attributes |
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
The activity mapping also accepts one optional field, `policy_id`, for exports whose
native key includes **SpendingPolicyId**. No other optional or unknown fields are accepted:

```json
{
  "mappings": {
    "activity": {
      "person_id": "PersonId",
      "service_id": "ServiceId",
      "policy_id": "SpendingPolicyId",
      "date": "MetricDate",
      "people_id": "PeopleHistoricalId",
      "credits": "Total Copilot Credits used"
    }
  },
  "policy_additivity": {
    "confirmed": false,
    "evidence": "REPLACE with the export owner's confirmation reference"
  }
}
```

`configs/real-consumption.example.json` follows the policy-aware layout observed in
real exports in October 2026. It reads `PersonM365CreditsMetrics.csv`, maps
`policy_id`, sets `group` to false because that metadata carried no HR attributes,
and deliberately starts with approval and additivity confirmation **false**.
Inspect it with local paths and explicit mappings while approval is false or
absent. Inspection validates numeric/date values, keys, joins and declared grain,
then reports `status: "awaiting_approval"`, `structural_validation: "passed"`,
`supported_panels`, `excluded_panels`, `privacy_status` and `next_action`.
When `policy_id` is mapped it also reports `multi_policy_positive_keys` and
`policy_additivity_status`. The count covers all supplied rows, before the `dates`
filter, because validation always covers the whole input. It can therefore be above
zero when `rows_in_window` counts none of the affected rows. If the count is above
zero and additivity is not confirmed, the status is `awaiting_policy_confirmation`
and `next_action` asks you to confirm with the export owner that the policy rows are
disjoint before approval. When privacy also withholds the release or the window is
empty, `next_action` gives the privacy or coverage action first and the confirmation
request after it.
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

Real exports observed in October 2026 showed a policy-aware variant. Their
`PersonM365CreditsMetrics.csv` had the native key **PersonId + ServiceId +
SpendingPolicyId + MetricDate**, and some people had more than one row for the
same person, service and day under different SpendingPolicyId values. Without a
`policy_id` mapping such an export fails the three-column key check, and the error
suggests mapping SpendingPolicyId. With `policy_id` mapped, the adapter:

- requires a nonblank, unpadded SpendingPolicyId on every row
- rejects duplicates of the four-column key
- counts the person-service-date keys that have more than one row with positive credits
- blocks build while that count is above zero, unless `policy_additivity` has
  `confirmed: true` and nonblank `evidence`
- sums credits across policy rows only after that gate, so output keeps the
  three-column grain and never reports per-policy values

A unique four-column key does not prove that the policy rows can be added. Two
positive rows for one person, service and day may describe disjoint policies or
overlapping views of the same use, and only the export owner can say which. One
positive row alongside zero rows is not ambiguous and passes without confirmation.
Errors and inspection report only the number of affected keys, never identifiers.
`policy_additivity` is rejected in demo mode, for real GitHub inspection, and when
`policy_id` is not mapped, so it is never silently ignored.

The same observed exports had a `PeopleMetaData.csv` with only PeopleHistoricalId and
IsCopilotLicensed. When the metadata has no HR attributes, set `group` to false and
omit `mappings.people.group`. Organisation grouping from Person Query remains
unsupported by this adapter.

The current official metrics include **Total Copilot Credits used**, **Session
count**, **User limit**, and **Spending policy limit**. This version deliberately
uses only credits. It does not sum limits, infer tokens/task types, or treat credits
as currency. Extra unmapped columns are ignored, not reverse-engineered.

Build rejects duplicate activity or metadata keys, unconfirmed positive credits on
several policy rows for one person-service-date, one historical ID assigned to
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
approval reference, policy additivity confirmation, privacy rule and UTC execution
time. Demo provenance also
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
policy-aware keys and the additivity gate,
whole-table suppression, output isolation, escaping and mode separation, and write
`dashboard-tests-<timestamp>/test-results.txt`. Fixtures and generated test artifacts
are never placed in the source tree.
