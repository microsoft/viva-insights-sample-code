# Consumption and GitHub query contracts

Use this reference for analysis of these query types. For reproducing or
customising the sample dashboards, the optional `viva-insights-copilot-dashboards`
skill owns the workflow; this reference owns the analytical rules.

## Evidence boundary

Checked against public documentation on 2026-09-11. Record the documentation
date and the actual export contract used in every adaptation. Do not silently
replace this contract when the website or an export changes.

| Source | Established here | Not established here |
|---|---|---|
| [Consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query) | Activity at PersonId, ServiceId, MetricDate; daily/weekly/monthly query grouping; people metadata joined by PeopleHistoricalId; credit and session metrics | Tokens, delegated task types, currency cost, or a complete licensed population denominator |
| GitHub reference demo | Synthetic daily activity, full-window model/language allocations, Person Query, roster, eligibility and coverage fixtures | A verified real GitHub flexible-query export schema; completeness or identity mappings inferred from sparse activity |
| Person Query | Person-period collaboration metrics, subject to the selected query and metric definitions | Coding time, developer output, health, or a causal effect |

The Consumption documentation is not evidence for a GitHub query schema.
An observed header confirms a name, not its meaning or aggregation rule.
Classify each mapping as documented, confirmed by a supplied source contract,
synthetic-only, or unresolved. Unresolved required mappings block computation;
unresolved optional mappings exclude the corresponding panel with a reason.

## Consumption: public export contract

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

Credit totals are additive only over non-overlapping activity records.
Session counts are not distinct people. Limits are not consumption and must
not be summed across service rows. Credits are not dollars. Token intensity
requires verified compatible token counts; never derive tokens from credits.
For a supported ratio, document whether it is a ratio of sums or a mean of
person ratios, its population and its zero-denominator rule.

## GitHub: demonstration versus real adaptation

The reference demo's manifest lives at
`examples/utility-r/_data/github/README.md` in the sample-code repository.
Its separate M365 feature counts, eligibility flags and completeness reference
are illustrative contracts, not certified exports. Its model/language mixes
cover the entire synthetic window and cannot be reused as weekly observations.

For real data, require evidence for identifiers, units, dates, population,
product eligibility and coverage, and any model/language allocation periods.
Do not substitute the GitHub public API schema for a Viva Insights query.
Where those facts remain unknown, return a readiness assessment and the
specific missing evidence, not a supposedly equivalent dashboard.

## Shared rules

1. **Keys:** keep IDs as strings. Do not lowercase opaque IDs or fuzzy-match
   people on HR attributes. Use an authorised identity bridge where necessary.
2. **Grain:** aggregate additive activity to the target person-period before
   joining Person Query metrics. Network snapshots and rolling metrics are not
   additive. Assert join cardinality and reconcile totals before and after.
3. **Duplicates:** report counts and stop for a resolution rule. Do not keep
   the first arbitrary row, even when that makes the join appear to work.
4. **Missingness:** distinguish observed zero, ineligible and unknown coverage.
   A missing row becomes zero only with independent eligibility and complete
   coverage evidence. Activity is not proof of licence status.
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

## Minimum evidence record

Keep local source fingerprints, schema versions, selected period and grain,
mapping evidence, inclusion rules, joins and reconciliation, per-panel units
and denominators, suppression rules, and excluded panels with reasons.
This record is private operational context, not automatically publishable.
Use only aggregate summaries in an approved agent environment; do not paste
employee rows into prompts or commit real exports to the sample repository.
