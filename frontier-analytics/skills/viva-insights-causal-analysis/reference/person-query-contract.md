# Person Query and organisational outcome contract

## Source precedence and mappings

Start from actual headers and query settings, verified metric definitions and
supplied outcome documentation. Use repository examples as illustrations, not
proof of a tenant's field meaning. Keep the raw-to-imported mapping explicit,
including localised headers and `import_query()` name cleaning.

Authoritative starting points:

- [Person Query setup](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query)
- [Metric definitions](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
- [Metric rules](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/metric-rules)

Person Query supports daily, weekly and monthly grouping. Do not assume all
exports are weekly, that a week always starts on Monday, or that export cadence
equals a metric's measurement frequency. Record the source contract and revision
or access date. Configured grouping and measurement windows are required;
date distributions diagnose alignment/coverage, not cadence by themselves.

Keep opaque IDs as strings. Validate stability across exports. Join external
outcomes only with an authorised identity bridge and a verified time key or
effective-date interval. Assert expected cardinality and reconcile counts and
totals. Duplicate person-period keys require a resolution rule, not an arbitrary
"keep first".

## Data audit

Record eligible people, observed people, joined people, usable person-periods and
each model's sample. Check:

- Query conditions, including `IsActive`, organisational filters and changing roles.
- Actual date coverage, freshness and completeness separately for each metric.
- Nulls versus recorded zero, licence eligibility, coverage and partial periods.
- Within-person exposure/outcome variation and number of independent measurements.
- Metric-rule changes, locale changes, holidays and leave.
- Whether a selected attribute is a baseline confounder, mediator, outcome,
  sampling condition or identifier. It is not a control merely because it exists.

Restricting to active people can change the estimand and select on consequences
of exposure. Flag inactivity and holidays first rather than automatically
deleting those rows. A balanced panel is not automatically more representative.

Copilot licences, engagement and record availability are distinct. Prefer
Copilot Enabled User or Total Copilot enabled days. Native nulls in licensed-only
M365 Copilot metrics mean unlicensed, while zero activity means licensed but
inactive under Microsoft's documented rules. Do not extend these semantics to
unlicensed Copilot Chat or other products. Verify all relevant activity before
calling someone "untreated", because no M365 Copilot licence does not imply
no exposure to unlicensed Copilot Chat. Licence evidence alone does not establish
complete telemetry capture.

Apply the paired core skill's shared policy: in a supplied complete query's
configured scope, absent activity rows contribute zero. Native licensed-only
Copilot nulls retain their unlicensed meaning. Failed joins, unavailable sources,
out-of-scope periods, partial boundary weeks and known export problems remain
unresolved with reasons. Zero activity alone does not establish a causal control
group, even when it establishes licensed inactivity for the selected metric.

## Variable suitability

For every candidate exposure and outcome, record:

| Property | Required evidence |
|---|---|
| Raw field and meaning | Verified metric or custom-field definition, units and source |
| Statistical type | Continuous, count, proportion, binary, ordinal or time-to-event |
| Measurement interval | Start/end of the window being measured |
| Availability/effective time | When the value was known and when the attribute applies |
| Aggregation rule | Sum, weighted mean, ratio of sums, distinct count, stock snapshot or another justified rule |
| Variation | Across-person and within-person support at the intended grain |
| Derivation | Whether the value includes or is computed from the proposed exposure |

Numeric support is necessary for some models but does not establish interpretability:

- An annual ordinal rating encoded as 1-5 is not a weekly count.
- A component of total collaboration should not be interpreted as independently
  causing its containing total without addressing that mechanical relationship.
- Usage-derived assisted hours/value are not independent productivity outcomes.
- Network sizes and stock/rolling metrics are not additive. Check their actual
  windows and repeated values before pooling or lagging.
- Overlapping rolling exposure and outcome windows can leak the same events
  into leads, lags and placebo comparisons.

## Organisational outcome decision table

An organisational attribute's presence on every row does not make it a repeated
measurement. Ask the data owner for frequency, period, effective date and whether
the export is historical or a current snapshot.

| Outcome supplied | Safe starting route | Unsupported shortcut |
|---|---|---|
| Independently measured weekly outcome | Person-week model after verifying timing/support | Assume same-week association is causal |
| Monthly/quarterly outcome copied onto weekly rows | One observation per person-outcome period, exposure aggregated over a justified aligned or preceding window | Count copies as independent outcomes |
| One annual score or fixed shortlist flag | One person-level outcome, explicit exposure window and sample | Person fixed-effects outcome model or invented prior outcomes |
| Repeated survey waves | One person-wave outcome, exposure before each wave, inspect response selection | Forward-fill responses as newly observed scores |
| Team outcome copied to every team member | Team-period design and dependence assumptions | Individual-level effect with multiplied sample size |
| Latest organisational snapshot attached to historical rows | Establish historical applicability or narrow the scope | Interpret current attributes as known baseline covariates |

The repeated-value pattern alone does not identify frequency. Distinct quarterly
measurements may happen to have equal values. Use verified measurement keys and
metadata, not deduplication by value.

If a quarterly outcome has only one observed quarter per person after alignment,
within-person analysis is still unavailable. Changing units or clustering cannot
create missing temporal information. A person-level follow-up association may be
possible, but requires a separate approved contrast and time window.

## Public examples versus real exports

The bundled examples are fabricated, Person Query-shaped tests with custom
organisational outcome names. They are not normative export contracts.
Prefer the vivainsights packages' built-in samples when demonstrating import and
standard metrics. Controlled synthetic effects are useful here because their
known construction tests whether the workflow recognises confounding, copied
outcomes and missing exposure. Never represent their coefficients as empirical
Viva Insights findings.
