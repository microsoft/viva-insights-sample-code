# External Collaboration Patterns

## Purpose

Create a static HTML report of external collaboration participation, intensity, group differences and changes over time.

## Audience

People analytics teams, functional leaders and analysts studying work across organisational boundaries.

## When to use

When a Person Query includes external collaboration hours and you want to understand how widely and intensively the population collaborates externally.

## Required inputs

- Person Query exports in the workspace's `data` folder with `PersonId`, `MetricDate`, `External collaboration hours` and `Collaboration hours`.
- Available organisational attributes, especially a function or team field for group comparisons.
- Configured date grouping, metric windows, date range and population scope. One complete period supports a snapshot. Two or more comparable periods are needed for a trend.
- ![External Collaboration query icon](../../../assets/images/queries/external-collaboration.svg) **External Collaboration query (optional)** or other approved counterpart data if external organisations, domains or partner groups need to be described. Verify the supplied fields and keys before adding counterpart views. The Person Query remains the primary source for participation and hours and does not identify those counterparts.

## Assumptions

- Participation means positive recorded external collaboration hours during the stated period.
- Intensity among external collaborators and average hours across the whole population have different denominators.
- A person's internal function or team is not the identity of the external party they collaborate with.
- "External" follows the export's documented boundary. It must not be assumed to mean customer-facing, sales-generating or outside the person's team.
- This is a guided analysis prompt, without a completed end-to-end data trial or a verified report adapter.

## Recommended output

A self-contained `output/external-collaboration.html` report, with reproducible R or Python source, privacy-protected aggregate tables and a validation log. Separate participation, intensity and group differences, with counterpart views only when supported.

## Quick prompt (short version)

```
Use the viva-insights-analysis skill to create a self-contained static HTML report on external collaboration. Read the Person Query from the workspace's data folder and save all generated files under its output folder, not filesystem-root paths.
Answer: what share of people collaborate externally, how much time external collaborators spend doing so, which internal groups have the most external collaboration, and how patterns differ across functions and over time.
Prioritise External collaboration hours and Collaboration hours, mapping the actual raw headers and imported names. Inspect configured grouping, metric windows, population, duplicates and missingness before calculating.
Define period participation as distinct people with positive External collaboration hours divided by distinct people with resolved external activity in that in-scope period. Report the full population and unresolved activity separately. For whole-window participation, count each person once and distinguish known positive, complete observed zero and unresolved coverage.
Report external hours among active external collaborators separately from the population-wide mean. Use comparable observation windows, show denominators and do not rank small or incomplete groups.
Compare internal groups by participation rate and mean external hours per active collaborator, showing total volume separately so large groups do not automatically appear most intensive. Use available function and other HR fields, omitting unsupported breakdowns.
The Person Query cannot identify external counterpart organisations or domains. Only add "who we collaborate with" views when suitable approved external collaboration data and verified keys are supplied. Otherwise state this question is unanswered. Do not infer counterpart identities from internal HR fields or scrape email domains.
Only calculate external hours as a share of Collaboration hours if their definitions and windows establish a valid subset relationship. Distinguish the ratio of sums from the mean of person-level ratios and omit undefined denominators. Do not sum stock metrics or overlapping windows.
Use the stricter organisational minimum or 10 distinct people for every cell and its complement, including active-collaborator cohorts, partner intersections, embedded data and downloads. Never publish person-level rankings.
Produce a summary-first HTML report with participation, active-collaborator intensity, group comparisons, compatible trends, an optional counterpart section and an evidence ledger. Embed all assets, verify offline desktop/mobile rendering, and log skipped analyses explicitly.
```

## Prompt

```
Build an External Collaboration Patterns report using the viva-insights-analysis skill and its shared analysis policy. Prefer supported vivainsights functions after checking installed R or Python signatures.

QUESTIONS AND SCOPE
1. What share of people in the population collaborates externally?
2. Among those who do, how much external collaboration time is recorded?
3. Which internal groups have the highest participation, intensity and total external collaboration?
4. How do these patterns differ across functions and over time?
5. Which external groups receive that collaboration, only if suitable counterpart data are supplied?
Use "participation" for the proportion with recorded external activity. Do not imply a clinical or epidemiological incidence measure.

FILES AND VALIDATION
Read exports from data relative to the workspace root. Save all generated files under output relative to that root, including external-collaboration.html, reproducible scripts or notebooks, aggregate CSVs and a validation log. Do not modify source exports.
Inspect actual headers and map External collaboration hours and Collaboration hours to their imported names. Preserve raw header names in the report. Record file inventory, row counts, distinct people, configured grouping, date range, native key uniqueness, units, metric windows and missingness.
Confirm what "external" means in this export using the metric definition. External collaboration must not be equated to collaboration outside the person's internal team.
Resolve duplicate person-period keys before aggregation. Keep missing or failed-import values distinct from measured zero. Do not fill absent behavioural Person Query observations with zero without a verified applicable source rule. Exclude partial periods from complete-period comparisons and disclose those exclusions.

PARTICIPATION
At the configured period grain, mark external participation where External collaboration hours is greater than zero. Count distinct PersonId values.
For each period, show the full in-scope population, the population with resolved external activity, the active external collaborators and unresolved activity, subject to suppression. Calculate participation as active external collaborators divided by people with resolved external activity. Label this observed-population denominator and show its coverage of the full population.
For whole-window participation, a positive observation establishes observed participation. Classify no participation only when all expected in-scope periods are resolved and zero. Keep the remainder unresolved and count each person once. Do not sum weekly user counts into a whole-window headcount.
Where coverage differs materially by group or period, avoid unqualified population-wide comparisons. Report coverage differences alongside the results.

INTENSITY AND SHARE
Report mean and median external hours among active external collaborators for each period, separately from mean external hours across everyone with resolved activity. Include distinct-person denominators and units.
For a whole-window intensity summary, first calculate each person's mean external hours per complete observed period, then summarise equally across people with positive external activity. State whether zero-activity observed periods are included. Avoid silently giving longer-observed people more weight.
Show the distribution using privacy-protected bins or quantiles, without person-level extremes or rankings.
Only calculate External collaboration hours divided by Collaboration hours if verified definitions and aligned windows establish that the numerator is a subset of the denominator. Otherwise omit the share and explain why.
If valid, distinguish the group ratio of summed external hours to summed collaboration hours from the mean of valid person-level shares. Exclude zero or unresolved denominators, show exclusions and investigate values outside the expected range instead of clipping them.
Do not add external hours to total collaboration if already included. Sum across periods only for verified additive, non-overlapping flows. Treat any network-size metric as a stock at its native measurement grain.

GROUPS AND TRENDS
Discover HR fields actually present, prioritising function and organisation where available. If none suit grouping, retain the overall report and explain the missing breakdown.
For each publishable group and period, compare participation rate, mean external hours among active collaborators and population-wide mean external hours. Show total volume as a separate measure with group size so scale is not mistaken for intensity.
Use dated group assignments for period trends. State the assignment rule for window summaries and account for moves without silently counting a person in several window-level groups.
Use at least two comparable complete periods for a time trend. Keep period definitions, population coverage and units aligned. Show a stable-cohort sensitivity when changing composition could explain a trend.
Describe differences and associations without claiming external collaboration causes sales, productivity or business value.

EXTERNAL COUNTERPARTS, IF AVAILABLE
Inspect any supplied external collaboration export for its actual grain, measures, participant keys, group definitions and privacy constraints before joining. Do not assume it follows the Person Query schema or infer a join from similar names.
Only produce counterpart group or domain summaries where documented keys and measures support the question. Use separately aggregated facts to avoid multiplying Person Query rows. Explain whether counterpart-linked hours reconcile to the Person Query total and why differences remain.
If supported, show internal-group by external-group comparisons with the appropriate participant denominators and privacy protection. Collaboration can be reciprocal, so do not imply sender/recipient direction without evidence.
If counterpart data are absent or unsuitable, state that the Person Query supports internal group comparisons but cannot identify which external groups people collaborate with. Do not manufacture counterpart names, infer them from internal HR fields or inspect email content to fill the gap.

REPORT AND CHECKS
Create a single self-contained HTML file with inline assets and no external runtime or server. Lead with concise answers, then participation, intensity, two-column group comparisons, time trends and counterpart evidence or its explicit absence. Put detailed methods behind progressive disclosure.
Use clear units, dates and denominators, a restrained palette and contextual explanations. Keep an evidence ledger linking headlines to source fields, filters and calculations.
Use at least 10 distinct people, or the stricter organisational minimum, for every published cell. Apply the same rule to active and inactive cohorts, counterpart intersections, chart bins, embedded data and aggregate downloads. Suppress complementary cells or whole breakdowns where totals disclose a small group. Never publish individual rankings.
Reconcile active, resolved-zero and unresolved populations, check distinct-user counts and row counts after joins, verify share denominators and aggregation grain, and inspect offline desktop and mobile output.
Save executed validation results and skipped questions separately. Missing counterpart evidence must remain visible as a limitation, not an invented analysis.

<!-- analysis-policy:start -->
Use product-specific licence evidence. For documented native licensed-only M365 Copilot metrics, null means unlicensed and zero activity means licensed but inactive. Prefer Copilot Enabled User or Total Copilot enabled days. Do not extend these semantics to unlicensed Copilot Chat or Consumption credits.
Investigate conflicting licence evidence. Keep affected eligibility or activity unresolved, and withhold affected rates or estimates until resolved.
Within a supplied complete query's configured scope, absent activity rows contribute zero. Keep failed joins, unavailable sources, out-of-scope periods and known coverage failures unresolved with reasons.
Use configured grouping, week start and metric windows; date gaps do not establish cadence. State grain, units and estimand. Count distinct people; do not sum stocks or overlapping windows.
Use max(organisation requirement, workflow floor): 5 distinct people for general analysis and 10 for joined-query, dashboard, causal or process workflows. Protect complementary cells and embedded data; never publish individual rankings.
Use only HR attributes present in the export. If none suit a requested breakdown, omit it, explain why, and continue with supported overall results.
Treat low-activity flags as review candidates, not verified holidays, employment status or job classifications. Value scenarios require explicit conversion and realisation assumptions.
Default to association. Make causal claims only with a supported design and compatible estimators. Record unavailable checks and label synthetic data.
Check installed API versions, signatures and return types. Separate executed results from expectations and do not silently fork upstream bugs.
<!-- analysis-policy:end -->
```

## Adaptation notes

- Replace generic group names with available function, team or region fields.
- Optional meeting or email metrics can explain the recorded channel mix if their definitions support the requested comparison.
- Supply an approved external collaboration query for counterpart analysis. Do not require it for the Person Query report.
- State the time weighting if using a person-period rather than a person-weighted whole-window summary.
- Increase privacy thresholds for partner-level intersections or other sensitive breakdowns.

## Common failure modes

- **Internal groups are presented as external counterparts.** Person Query HR fields describe the person, not their collaborators.
- **Whole-window headcount is inflated.** Count each person once instead of adding period user counts.
- **Active-only and population-wide averages are mixed.** They answer different questions and need separate labels.
- **Large groups are called the most intensive.** Separate total volume, participation rate and per-person intensity.
- **The external share has an incompatible denominator.** Verify definitions and windows before calculating a ratio.
- **Small collaborator or partner cells are disclosed.** Apply suppression to active cohorts, intersections, complements and embedded data.
