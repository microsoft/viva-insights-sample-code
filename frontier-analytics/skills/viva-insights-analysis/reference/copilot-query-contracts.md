# Consumption, GitHub and Agent query contracts

For current Person, Consumption, GitHub and Agent integration, use
[Joining Viva Insights queries](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/)
and its downloadable JSON contract. That guide supersedes older schemas,
cadence detection, joins and missing-row guidance here. Recommend exporting
all selected queries grouped by week. Assume supplied Viva Insights data are
complete within each query's configured population, date window and product
filters, with omitted inactive person-days contributing zero activity.
This does not establish product eligibility or turn unavailable queries into zeros.

This reference retains historical layouts, synthetic-demo assumptions and
additional analytical context. Use the dated sections to reproduce earlier
examples or explain a layout difference, rather than silently adapting the
current contract. For reproducing or customising the sample dashboards, the
optional `viva-insights-copilot-dashboards` skill owns the workflow. The new guide
does not extend the existing runners' supported inputs or supply a real GitHub
dashboard adapter.

## Evidence boundary

Checked against public documentation on 2026-09-11. Record the documentation
date and the actual export contract used in every adaptation. Do not silently
replace this contract when the website or an export changes. The table below
is the September 2026 boundary. Layouts observed in real exports in October
2026 are described separately in the
[observed Consumption variant](#consumption-observed-four-file-variant-october-2026)
and [Agent query](#agent-query-observed-layout-and-dated-variant) sections.

| Source | Established here | Not established here |
|---|---|---|
| [Consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query) | Activity at PersonId, ServiceId, MetricDate; daily/weekly/monthly query grouping; people metadata joined by PeopleHistoricalId; credit and session metrics | Tokens, delegated task types, currency cost, or a complete licensed population denominator |
| GitHub reference demo | Synthetic values in confirmed daily activity/breakdown headers, Person Query and roster | Real breakdown metric equalities/model attribution, provisioning or ingestion completeness inferred from sparse activity |
| Person Query | Person-period collaboration metrics, subject to the selected query and metric definitions | Coding time, developer output, health, or a causal effect |

The Consumption documentation is not evidence for a GitHub query schema.
An observed header confirms a name, not its meaning or aggregation rule.
Classify each mapping as documented, confirmed by a supplied source contract,
synthetic-only, or unresolved. Unresolved required mappings block computation;
unresolved optional mappings exclude the corresponding panel with a reason.

## Consumption: documented two-file layout (checked 2026-09-11)

This is the layout described by public documentation in September 2026. It
remains a valid dated variant. It is not the only layout in use, so compare a
supplied export against this section and the observed variant below.

Download and extract the ZIP locally. The documented two files are:

- Activity: `PersonId`, `ServiceId`, `ServiceName`, `MetricDate`,
  `PeopleHistoricalId`, `Total Copilot Credits used`, `Session count`,
  `SpendingPolicyId`, `Spending policy limit`, and `User limit`.
- People metadata: unique `PeopleHistoricalId`, `IsCopilotLicensed`, and
  whichever HR attributes were selected.

Confirm the configured period grain, week boundaries, selected population and
filtering. The source is daily, but an exported query can be grouped by week or
month. Never expand weekly totals into invented daily observations.

Assert unique activity keys `(PersonId, ServiceId, MetricDate)` and unique
metadata keys before a many-to-one metadata join. Keep historical metadata
attached through `PeopleHistoricalId`, not a latest-person snapshot. Report
unmatched key counts without printing identifiers; do not silently drop them.
If that three-part key is duplicated and the export carries `SpendingPolicyId`,
check whether the export follows the policy-aware key described below before
treating the rows as duplicates.

Credit totals are additive only within the same verified unit over non-overlapping activity records.
M365 Copilot credits and GitHub AI credits remain separate; no common-unit conversion
or cross-product share is established here.
Session counts are not distinct people. Limits are not consumption and must
not be summed across service rows. Credits are not dollars. Token intensity
requires verified compatible token counts; never derive tokens from credits.
For a supported ratio, document whether it is a ratio of sums or a mean of
person ratios, its population and its zero-denominator rule.

## Consumption: observed four-file variant (October 2026)

Some real exports observed in October 2026 contained four CSV files instead
of the documented two:

| File | Observed key and grain | Role |
|---|---|---|
| `PersonM365CreditsMetrics.csv` | `PersonId`, `ServiceId`, `SpendingPolicyId`, `MetricDate`. Daily | M365 Copilot credits, sessions and limits |
| `PersonGitHubCreditsMetrics.csv` | `PersonId`, `MetricDate`. Daily | GitHub AI credits |
| `PeopleMetaData.csv` | Unique `PeopleHistoricalId` | Historical licence evidence |
| `M365SpendingPolicyMetaData.csv` | Unique `SpendingPolicyId` | Spending policy metadata and reference-integrity checks |

Both fact files carry `PeopleHistoricalId` alongside `PersonId`. Treat this as
an observed variant to match against, rather than a fixed file count. Other
exports may omit a companion file or add columns.

### People metadata without HR attributes

In these exports `PeopleMetaData.csv` contained only `PeopleHistoricalId` and
`IsCopilotLicensed`. It had no `PersonId` and no HR attributes. Join each fact
file to it many-to-one through the `PeopleHistoricalId` carried in that fact
file. Do not assert `PersonId` uniqueness in this file or expect grouping
attributes there.

Organisational attributes then come from the Person Query, joined on `PersonId`
and a compatible reporting period. Declare a dated-assignment rule before the
join, for example the first observed value inside the analysis window, and
record it. Attributes can change within a window, so state when a later
snapshot is attached to earlier periods. An assignment rule chosen for
description is not automatically a verified pre-exposure covariate.

Aggregate daily facts to the Person Query's week definition, such as
Sunday-start weeks, before that join. Never expand weekly Person Query rows
into invented daily records.

### Key uniqueness and additivity are separate checks

With `SpendingPolicyId` in the native key, one person can have several M365
rows for the same service and date. Assert uniqueness on
`(PersonId, ServiceId, SpendingPolicyId, MetricDate)`. A passing check
establishes structure. It does not establish that credits can be added across
policy rows.

Before summing credits across policies to a person-service-date or coarser grain:

1. Count person-service-date keys with more than one positive-credit policy
   row. Report the count only, without identifiers.
2. A key with one positive row and the remaining rows at zero sums to the
   positive amount and raises no additivity question.
3. Keys with more than one positive policy row need a documented decision.
   Ask the data owner whether those rows record disjoint consumption, and
   record the confirmation with its evidence reference.
4. Without that confirmation, stop before summing. With the user's approval of
   a narrower scope, the affected people can be excluded from the primary total
   and the exclusion reported. Summing the rows or keeping the largest row are
   sensitivity checks rather than verified deduplication rules, and agreement
   between them does not establish the correct total.

An additivity confirmation for one export does not carry over to another
(shared rule 11).

Check that `M365SpendingPolicyMetaData.csv` has unique `SpendingPolicyId` values
and that fact `SpendingPolicyId` values resolve to it. Report unmatched counts.
Confirm the meaning of empty or placeholder policy identifiers against the
supplied contract. Never add `Spending policy limit` or `User limit` across
policy rows, and do not infer a cap without matching its scope, period and
semantics.

Keep `PersonGitHubCreditsMetrics.csv` separate from M365 credits. For current
exports, apply the join guide's complete-data and omitted-inactivity defaults
within the configured query scope, rather than requiring a row on every day.

### Earlier Consumption layouts

The two-file layout above, with a combined activity file such as
`PersonServiceCreditsMetrics.csv` and HR attributes in the people metadata,
was the documented variant in September 2026. The synthetic fixtures at
`examples/utility-r/_data/consumption-query/` in the sample-code repository use
a three-file layout without a policy metadata file and with HR attributes in
`PeopleMetaData.csv`. Both remain dated variants. Recheck the repository
revision and the supplied export before adapting either.

## GitHub: demonstration versus real adaptation

The reference demo's manifest lives at
`examples/utility-r/_data/README.md` in the sample-code repository. Its
fixtures carry synthetic values in dated export-shaped schemas, rather than
establishing the current layout of every download. Weekly M365 eligibility uses
enabled days. The synthetic demos retain their own coverage flags and checks;
these do not override the current join guide's complete-data default. GitHub
row presence describes activity observation, not provisioning. Feature, model and language
breakdowns are daily two-dimension files with no exported share column. Their
cross-breakdown equality and model allocation (including completions) are synthetic
invariants only, not verified real-export validation rules.

For real data, record identifiers, units, configured dates and population,
product eligibility and any model/language allocation periods. Apply the
current guide's missing-row defaults for supplied Viva Insights exports.
Do not substitute the GitHub public API schema for a Viva Insights query.
Where those facts remain unknown, return a readiness assessment and the
specific missing evidence, not a supposedly equivalent dashboard.
The real exports observed in October 2026 did not include a dedicated GitHub
activity query, so this guidance remains untested against real exports.

## Agent query: observed layout and dated variant

Availability and release stage vary by tenant and may change. Record the export
date and preserve the original headers.

Some real exports observed in October 2026 used this layout:

| File | Observed key and grain | Role |
|---|---|---|
| `PersonAgentActivityMetrics.csv` | `PersonId`, `AgentId`, `MetricDate`. Weekly, with a Sunday-start `MetricDate` in the observed exports | `Conversational agent sessions`, `Returning Agent 365 user (most recent and prior 7 days)` and `Returning Agent 365 user (most recent and prior 28 days)` |
| `AgentMetadata.csv` (optional) | Unique `AgentId` | Descriptive agent fields |

`AgentMetadata.csv` was present in some exports and absent in others. Enrich
facts with it only when the file is supplied and its `AgentId` values are
unique. Its absence does not block fact-level population, key and aggregation
checks. State the unavailable enrichment instead of inventing agent attributes.
Keep `AgentId` as the analytical key. Agent names and creator fields can change,
and must not be used to reconstruct identities.

Assert uniqueness on `(PersonId, AgentId, MetricDate)`. Sum
`Conversational agent sessions` across agents to the person-week before joining
the Person Query. Keep each returning-user indicator at its stated 7-day or
28-day window. Do not sum these indicators across agents or weeks, or read them
as new activity or daily retention. A person-level version, such as returning
for at least one agent, is a derived definition to state explicitly.

Rows can hold explicit zeros, so the presence of an Agent record does not
establish positive use or complete capture. Sessions are not distinct people or
completed tasks. Agent sessions may overlap measured M365 activity, so do not
add them to M365 sessions or treat them as an extra tool group without evidence
that they are distinct. This layout has no agent credits or responses, and none
should be derived.

### Earlier Agent layout

The synthetic Agent sample in `examples/utility-r/_data/github/agent-query/`
of the sample-code repository follows a September 2026 description with three
files: `AgentMetadata.csv`, `PersonAgentCreditsRetentionMetrics.csv` and
`PersonAgentResponsesMetrics.csv`. That layout is a dated variant. If those fact
files are supplied, verify each file's full native key before combining them,
and confirm any additive relationship independently. Credits, responses and
sessions remain different measures. None establishes agent quality, task
completion or monetary cost.

## Shared rules

1. **Keys:** keep IDs as strings. Do not lowercase opaque IDs or fuzzy-match
   people on HR attributes. Use an authorised identity bridge where necessary.
2. **Grain:** aggregate additive activity to the target person-period before
   joining Person Query metrics. Network snapshots and rolling metrics are not
   additive. Assert join cardinality and reconcile totals before and after.
3. **Duplicates:** report counts and stop for a resolution rule. Do not keep
   the first arbitrary row, even when that makes the join appear to work.
   Rows that look duplicated under a partial key can be distinct under the
   full native key, so confirm the key variant first. Unique keys do not
   establish that amounts can be added. Check additivity separately.
4. **Missingness:** follow the current join guide for supplied Viva Insights
   exports. Assume completeness within the configured query scope and treat
   omitted inactive activity rows as zero. Keep unavailable sources, out-of-scope
   people or periods, null cells and known export problems separate. Licence
   eligibility remains a separate attribute and needs product-specific evidence
   for adoption denominators. Historical synthetic demos retain their own
   coverage logic and are not real-data adapters.
5. **Populations:** preserve the developer roster for working-condition
   baselines; do not select only Copilot users. Joint-product categories
   require both products' eligibility and coverage over the declared window.
6. **Privacy:** for these dashboards, use at least 10 distinct people per
   published group, or a stricter organisational requirement. Check every
   filter and time slice. Withhold a whole breakdown if publishing the
   remaining cells or totals would reveal a suppressed cell by subtraction.
   No individual ranking or identifiers in HTML, tables, hidden JSON or logs.
7. **Claims:** associations are not causal effects. Acceptance is not code
   quality; uninterrupted calendar time is not coding; after-hours activity
   is not burnout. Do not infer ROI or productivity from tool consumption.
8. **Partial results:** a credits-only report is valid if labelled honestly.
   Never synthesize a missing metric, licence flag or coverage record in a
   real-data run. Synthetic examples belong in a separate labelled run.
9. **Integration and cohort selection:** integrating sources and choosing the
   analysis population are separate, declared steps. The Person Query can be
   the integration spine without every analysis using the intersection of all
   queries or only positive-credit people. An adoption report may need the
   broader eligible population. Absence from an optional query can reflect
   coverage, eligibility or filtering rather than non-use.
10. **Measurement windows:** record each variable's measurement window and
    frequency as well as the row grain. Weekly rows do not make network sizes
    weekly flows. In some exports `Internal network size` and
    `External network size` repeat within person-months. Average such stocks
    rather than summing them, and note that a boundary-month value can include
    contacts outside the selected weeks.
11. **Confirmations do not transfer:** a coverage, licence or policy-additivity
    confirmation applies to the export it was given for. Matching headers in
    another dataset do not carry it over. Obtain and record a new confirmation.
12. **Observed tool universe:** before comparing tool mixes, classifications or
    labels between datasets, record which products, services and agents were in
    scope and observed, and the capture status of each (confirmed, unknown or
    absent). Labels derived from different tool universes are not directly
    comparable.

## Minimum evidence record

Keep local source fingerprints, schema versions, selected period and grain,
mapping evidence, inclusion rules, joins and reconciliation, per-panel units
and denominators, suppression rules, and excluded panels with reasons.
Also record the matched layout variant and its date, the policy-additivity
decision and evidence reference, the source and assignment rule for
organisational attributes, variable measurement windows, the observed tool
universe with capture status, and any optional companion file that was absent.
This record is private operational context, not automatically publishable.
Use only aggregate summaries in an approved agent environment; do not paste
employee rows into prompts or commit real exports to the sample repository.
