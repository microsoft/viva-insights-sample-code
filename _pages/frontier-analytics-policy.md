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
Use product-specific licence evidence. For native M365 Copilot metrics covered by Microsoft's documented rules, null means unlicensed and zero activity means licensed but inactive. Prefer Copilot Enabled User or Total Copilot enabled days when present.
Absent activity inside a supplied complete query's configured scope contributes zero. Preserve unresolved joins, unavailable sources, out-of-scope periods and known coverage failures as unknown with reasons. Do not extend licensed-only M365 Copilot null semantics to unlicensed Copilot Chat or Consumption credits.
Read configured grouping, week start and measurement windows; date spacing is diagnostic only. State grain, units and estimand, count distinct people and do not sum stocks or overlapping windows.
Use max(organisation requirement, workflow-specific privacy floor), protect complementary cells and embedded data, and never publish individual rankings.
Treat low-activity flags as review candidates, not verified holidays, employment status or job classifications; value scenarios need explicit conversion and realisation assumptions.
Choose a supported identifying design before causal claims and compatible estimators; default to association, retain unavailable checks and label synthetic evidence.
Inspect installed package versions, signatures and return types; separate executed evidence from expected behaviour and do not silently fork upstream bugs.
<!-- analysis-policy:end -->

## Eligibility and activity are separate

For M365 Copilot, prefer **Copilot Enabled User**, **Total Copilot enabled days**
or dated licence records for the relevant person-period. Positive enabled days
establish that a person was enabled for at least part of the period. Zero enabled
days establish no enabled days, rather than a licensed inactive week.

[Microsoft's Copilot calculation rules](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics#notes-on-copilot-metric-calculations)
state that native null values indicate no M365 Copilot licence, while zero
activity indicates a licensed person with no activity for that metric and period.
Apply these documented semantics to the relevant native export fields, including
null enabled days. If explicit licence fields are absent, a verified licensed-only
metric with these semantics can establish licence status without an external
licence list. Do not infer licensing from an arbitrary column containing "Copilot".

These rules do not apply to unlicensed Copilot Chat metrics, Consumption credits,
GitHub Copilot or other products without their own documented contract. Keep
licence status **unknown** when the source or field semantics are genuinely
unresolved. Investigate conflicting evidence, such as positive enabled days with
unexpectedly null licensed-only activity, before publishing the affected rate.
Adoption denominators include licensed people with zero activity and exclude
unlicensed people. Withhold an overall rate only where licence status or activity
among licensed people remains unresolved.

## Missingness has reasons

For the [current join contract](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/),
assume a supplied query is complete within its configured population, date
window and product filters unless a known problem contradicts that assumption.
An absent activity row there contributes zero, including an entirely absent
in-scope person-week. This does not establish eligibility.

Distinguish native export nulls from nulls introduced by processing. A native
licensed-only M365 Copilot null means unlicensed, so exclude that person-period
from licensed adoption and usage-segment denominators. Do not turn it into a
licensed Non-user, an inactive week or a churn event.

A failed join, unavailable source, out-of-scope person or period, partial boundary
week or known coverage failure remains unresolved. A join-created null does not
inherit the native export's unlicensed meaning. Preserve match indicators and
missingness reasons before classifying values. Do not blanket-fill a panel after a join.
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
dashboard, causal and process workflows have a floor of **10**. A workflow may require
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
