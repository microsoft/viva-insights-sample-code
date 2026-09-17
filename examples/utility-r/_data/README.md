# Synthetic query exports — source manifest

These CSV files are **fully synthetic**. They contain no customer data and no real
identifiers.

The values are simulated. **The schema is not.** Every file name, every column name
and every key in this folder matches a real Microsoft Viva Insights query export.
Nothing here is invented, with one deliberate exception noted under
[Organisational attributes](#organisational-attributes).

Regenerate everything with:

```bash
cd examples/utility-r
Rscript simulate-query-exports.R
```

The script is deterministic (`SEED <- 20260916`) and fails rather than exporting if
any ordered fixture schema or synthetic allocation invariant breaks. The independent
header vectors in `query-export-contracts.R` specify all eight fixed exports,
the selected HR attributes and this custom Person Query recipe. All nine outputs
are validated before any file is written; changing the selected custom metrics or
HR attributes requires an explicit recipe-contract update.

## Why this matters

An earlier version of this folder simulated the schema as well as the values. It
carried columns that do not exist in any export — token counts, task-type
breakdowns, a coverage/eligibility reference file, and an M365 actions feed with no
corresponding query. Sample code written against an invented schema cannot be pointed
at a real export without rework, so the folder has been rebuilt against the published
contracts.

## Layout

One folder per query, mirroring what you get when you unpack a real download.

```text
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

All three query outputs describe **one shared population of 480 people**, 360 of
whom are developers across six teams of 60. Joining across folders is therefore
meaningful, exactly as it would be in a real tenant.

## Population and period

- Person query: 26 Sunday-start weeks, `2026-01-04` to `2026-07-04`.
- Consumption and GitHub feeds: the final 13 weeks, `2026-04-05` onward, at daily
  grain, weekdays only. A shorter product history than collaboration history is the
  normal real-world case — do not assume the windows always align.
- Dates are ISO `YYYY-MM-DD` on a UTC convention. They are not a record of personal
  time zones.
- Decimal point, no thousands separators. Empty policy/limit cells mean no policy
  in this simulation. A real empty value does not by itself establish policy absence.

## Consumption query

Source: [Create a custom consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query).

A real run produces a single `.zip` containing the files below. `PeopleHistoricalId`
is the join key between the activity files and the metadata file — **not** `PersonId`.

`PeopleHistoricalId` is **opaque**. Read the `PersonId` → `PeopleHistoricalId`
crosswalk out of the activity files; never reconstruct it. A real export may
coincidentally show the same constant numeric suffix on every row of one download,
but that is a coincidence, not a contract, and code that relies on it fails
silently against a real tenant. These fixtures mint the historical key
independently of `PersonId` — no string operation on `PersonId` reproduces it —
so a reconstruction bug fails here instead of passing the demo.

### `PeopleMetaData.csv`

One row per person. `PeopleHistoricalId` and `IsCopilotLicensed` are always present
and cannot be removed. Every other column is an HR attribute chosen at query setup,
so this file's width varies from query to query.

| Column | Type | Notes |
|---|---|---|
| `PeopleHistoricalId` | string | Join key to the activity files |
| `IsCopilotLicensed` | boolean | `true` / `false` |
| `Organization`, `FunctionType`, `LevelDesignation`, `IsManager`, `Team`, `Role`, `Seniority`, `Tenure`, `IsDeveloper` | string | Selected HR attributes |

### `PersonM365CreditsMetrics.csv`

One row per `PersonId` × `ServiceId` × `MetricDate`. A person who used three services
on one day has three rows for that day — code that assumes one row per person per day
will silently over-count.

`PersonId, ServiceId, ServiceName, SpendingPolicyId, MetricDate, Session count, Spending policy limit, Total Copilot Credits used, User limit, PeopleHistoricalId`

In this fixture `SpendingPolicyId` is the all-zero GUID when no policy applies,
and `Spending policy limit` and `User limit` are empty. Validate real null semantics
against the supplied query contract rather than inferring them from this recipe.

### `PersonGitHubCreditsMetrics.csv`

One row per `PersonId` × `MetricDate`.

`PersonId, MetricDate, Total GitHub AI Credits used, PeopleHistoricalId`

**Units are separate:** M365 Copilot credits and GitHub AI credits have no verified
common-unit conversion here. Do not add them, calculate cross-product shares, or
use GitHub AI credits in an M365 credits-per-session denominator. The Consumption
report uses M365-only volume bands and a separately labelled GitHub panel.

## GitHub Copilot query

Five files, no `PeopleMetaData`. HR attributes for this population come from the
Person query or the Consumption query's metadata file.

### `PersonGitHubActivityMetrics.csv`

One row per `PersonId` × `MetricDate`.

`PersonId, MetricDate, Agent adoption, Code completions accepted, Code completions suggested, User-initiated chat requests`

The fixture includes explicit **zero rows** for simulated provisioned users on days
with no activity. A measured zero is observed inactivity; a missing row is unknown,
not proof of no provisioning. Presence demonstrates observation, not a licence record.
`Code completions accepted` never exceeds `Code completions suggested`.

### Breakdown files

| File | Key | Measure |
|---|---|---|
| `GitHubActivityBreakdownByFeatureMetrics.csv` | `PersonId`, `MetricDate`, `Feature` | `Feature Usage Count` |
| `GitHubActivityBreakdownByLanguageFeatureMetrics.csv` | `PersonId`, `MetricDate`, `Language`, `Feature` | `Usage count by language and feature` |
| `GitHubActivityBreakdownByLanguageModelMetrics.csv` | `PersonId`, `MetricDate`, `Language`, `Model` | `Usage count by language and model` |
| `GitHubActivityBreakdownByModelFeatureMetrics.csv` | `PersonId`, `MetricDate`, `Model`, `Feature` | `Usage count by model and feature` |

In this simulation only, all four are exact margins of one underlying
Feature × Language × Model allocation, so they reconcile with each other, and the feature margin reconciles to
`Code completions accepted` + `User-initiated chat requests` in the activity file.
The generator asserts this synthetic allocation invariant before writing. Model
attribution, including to completion counts, is illustrative. Confirmed headers
do not establish these equalities or attribution rules for real exports; the report
loader does not reject replacement data for failing them.

Feature values use the real vocabulary: `code_completion`, `chat_inline`,
`chat_panel_ask_mode`, `chat_panel_edit_mode`, `chat_panel_agent_mode`,
`chat_panel_plan_mode`, `chat_panel_custom_mode`, `chat_panel_unknown_mode`,
`agent_edit`, `copilot_cli`, `copilot_app`.

## Person query

`person-query/PersonQuery.csv` — one row per `PersonId` × `MetricDate` (Sunday week
start), 26 weeks.

The person query is a **custom** template, so its column set legitimately varies by
tenant and by analyst. What does not vary is that every selected metric must be a
real one. Each metric in this file was verified against the
[metric reference](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics):
collaboration and mode hours, meeting characteristics, after-hours and weekend
collaboration, `Available_to_focus_hours` / `Uninterrupted_hours` / `Interrupted_hours`,
`Open_1_hour_block`, network size and tie metrics, and `Total_Copilot_enabled_days`.

`Available_to_focus_hours` is the hours remaining during working hours after excluding
meetings and scheduled calls, and `Uninterrupted_hours` plus `Interrupted_hours`
partition it. The working-hours basis is configurable per tenant through metric rules,
so the 40-hour basis used here is not universal. Available-to-focus hours measure
calendar availability; they do not measure coding or psychological flow.

The three meeting-characteristic metrics overlap and **must not be stacked
additively**.

## Deriving eligibility and coverage

There is no eligibility or coverage export file. Keep these evidence classes separate:

| Question | Real signal |
|---|---|
| Was this person eligible for M365 in this week? | `Total_Copilot_enabled_days > 0`; a static `IsCopilotLicensed` snapshot cannot override a zero-enabled week |
| Was this person provisioned for GitHub Copilot? | Independent provisioning evidence; missing activity means unknown and M365 licensing is not GitHub licensing |
| Was this person observed in GitHub? | Presence of a measured row; the demo checks observed weekday rows, not verified ingestion completeness |
| Is the sparse M365 export complete? | Unknown by default, even with observed positive rows; independent ingestion evidence is required |
| Did this person use Copilot on this day? | A row with a non-zero measure, not the absence of a row |

Fill an absent activity record with zero **only** when period eligibility and independent
ingestion completeness/expected coverage are established. The helper accepts an
analyst-side evidence object through `derive_m365_coverage`; this is not a fabricated
query-export column or file. Without it, M365 complete-week and joint-product panels
remain unavailable. Sparse source totals are labelled observed records, not full-window totals.

## Organisational attributes

HR attribute names are defined by each customer's own people data, so the specific
attributes here (`Organization`, `FunctionType`, `Team`, `Seniority`, and so on) are
illustrative. The mechanism — analyst-selected attributes appended to each output row
— is real. Every non-attribute column in this folder is a genuine export column.

## Privacy and reuse

Identifiers are synthetic GUIDs generated by the script and exist only to demonstrate
joins. `PeopleHistoricalId` is minted independently of `PersonId`, so the crosswalk
must be read rather than reconstructed. Published aggregates in the accompanying
reports require at least 10 distinct people. Before adapting any of this against real
data, validate the source contract and your organisation's privacy requirements.
