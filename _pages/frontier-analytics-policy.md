---
layout: page
title: "Shared analysis policy"
permalink: /frontier-analytics-policy/
---
<!-- Generated from the core Skill reference. Do not edit. -->

# Shared analysis policy

Read this before computing or publishing results. Query schemas and workflow
contracts still own their specific keys, metrics and supported methods.
Copyable prompts retain the rules needed without this checkout.

## Copyable summary

<!-- analysis-policy:start -->
Use explicit product-specific eligibility evidence independently of activity; otherwise retain unknown and omit adoption rates with unknown denominators.
Absent activity inside a supplied complete query's configured scope contributes zero; observed nulls, unavailable sources, out-of-scope periods and known coverage failures remain unknown with reasons.
Read configured grouping, week start and measurement windows; date spacing is diagnostic only. State grain, units and estimand, count distinct people and do not sum stocks or overlapping windows.
Use max(organisation requirement, workflow-specific privacy floor), protect complementary cells and embedded data, and never publish individual rankings.
Treat low-activity flags as review candidates, not verified holidays, employment status or job classifications; value scenarios need explicit conversion and realisation assumptions.
Choose a supported identifying design before causal claims and compatible estimators; default to association, retain unavailable checks and label synthetic evidence.
Inspect installed package versions, signatures and return types; separate executed evidence from expected behaviour and do not silently fork upstream bugs.
<!-- analysis-policy:end -->

## Eligibility and activity are separate

Use documented product-specific enabled-days or licence records for the relevant
person-period. Positive activity, a zero or a null does not establish licensing.
Without eligibility evidence, retain **unknown** and report observed activity,
not an adoption rate with an invented denominator. Explicit negative eligibility
evidence establishes ineligibility only for its documented product and period.

## Missingness has reasons

For the [current join contract](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/),
assume a supplied query is complete within its configured population, date
window and product filters unless a known problem contradicts that assumption.
An absent activity row there contributes zero, including an entirely absent
in-scope person-week. This does not establish eligibility.

An observed null cell is unknown, not zero. Keep unavailable sources, out-of-scope
people or periods, partial boundary weeks and known coverage failures distinct
from inactivity. Preserve the reason. Do not blanket-fill a panel after a join.
Legacy runners may require explicit completeness configuration before building
a non-user panel; record that implementation gate rather than changing the
meaning of nulls or reusing another export's evidence.

## Cadence, grain and estimand

Read configured grouping, week start and metric measurement windows first.
Observed date gaps diagnose alignment and coverage, but cannot establish cadence:
sparse daily activity can look weekly. If configuration is unknown, stop the
affected aggregation or label it unresolved. Do not expand totals into invented
daily observations.

State date window, grain, metric/unit, eligible population and estimand on each
published chart or table. Distinguish a person-weighted summary from a
person-period-weighted mean. Count distinct people in denominators. Do not sum
stocks or overlapping windows. Verify key uniqueness and additivity separately
before joining or summing.

## Privacy and interpretation

Use `max(organisation requirement, workflow floor)` distinct people per
published group: generic analysis has a floor of **5**; joined-query,
dashboard and causal workflows have a floor of **10**. A workflow may require
more. Apply the same rule to tables, charts, embedded data and downloadable
aggregates. Suppress complementary cells or entire breakdowns if totals reveal
small cells. Pseudonymisation is not permission to publish individual rankings.
Keep person-level working data local and approved.

Low-collaboration, holiday-week and inactive-week helper flags are review
candidates, not verified holidays, employment status or job classifications.
Use approved calendar/HR evidence before asserting those facts. Document any
exclusion and its effect on the population.

## Claims and methods

Default to association for observational data. Choose a supported identifying
design before estimators or causal claims. Select compatible diagnostics, not
every estimator. Readiness-only, unavailable checks and null findings are valid
outputs. Fixed effects, DML or a passing placebo do not by themselves establish
causality. Label fabricated data and injected effects prominently.

Assisted hours, actions, credits and collaboration are not measured time saved,
productivity or realised ROI. Value scenarios require explicit conversion,
realisation, price and population assumptions.

## Package and evidence boundary

Inspect installed package versions, signatures and return types before using a
recipe. Hosted documentation may describe a different release. Do not silently
fork an upstream bug. Record executed commands and observed results separately
from expected agent behaviours or historical validation notes.

See [workflow support]({{ site.baseurl }}/frontier-analytics-support/) for executable versus guided,
synthetic-only, inspect-only and unsupported capabilities.

[Back to Skills]({{ site.baseurl }}/frontier-analytics-skills/)
