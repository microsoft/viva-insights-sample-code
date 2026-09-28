# Reporting and verification

## Output contract

For an executed analysis, retain a local preparation/analysis script, an audit
record, machine-readable results and a self-contained HTML report. For blocked
work, return readiness and missing evidence rather than generating placeholder
effect estimates.

The audit records source hashes, source-to-panel mappings, query settings,
measurement windows, eligibility and coverage, joins, exclusions, approved
assumptions, dependencies, seeds and reproduction commands. It is private
operational context, not automatically publishable.

For each model or diagnostic, record:

- Stable ID, plain question, comparison design and estimator.
- Exact formula, raw/derived outcome/exposure names, units and adjustments.
- Population, outcome period, exposure window, rows, people and effective clusters.
- Original-unit contrast, estimate scale and full-precision estimate.
- Interval bounds/level/method and p-value/method separately.
- Fit status, warnings, dropped units, unavailable checks and provenance.

Use explicit `blocked`/`unavailable` status and a reason instead of zero-valued
substitutes. Never claim success because a model emitted a file.

## Report reading order

1. A concise question and evidence summary, with visible uncertainty.
2. Model-labelled headline estimates, interval and inference method.
3. Explicit variables, using literal names in code styling with plain meanings.
4. Two-column bullets for evidence versus limitations. Collapse to one column
   on smaller screens.
5. Brief comparison with prior work, naming changed populations, definitions,
   measurement periods or estimands.
6. Collapsible method guide, full tables, diagnostics and detailed interpretation.
7. Prioritised questions or additional evidence needed to change the conclusion.

Keep provenance warnings, material missingness and non-causal status visible.
Retain details in disclosures rather than deleting them. Keep tables full width,
with horizontal scrolling confined to their containers. Reuse the same content
components for standalone and combined reports.

## Accessible explanations and filters

Explain odds/probability ratios, intervals, p-values, PPML, fixed effects and
permutation at their point of use. Model tags identify comparisons, not a
quality rating. Use plain language such as "the test did not detect an
association" instead of "the placebo passed".

Tooltips should support hover, keyboard focus and tap, fit within the viewport,
escape scroll clipping, be dismissible with Escape and remain readable while
hovered. Use accessible names and `aria-describedby`. Essential interpretation
must remain in visible text.

Optional significance filters must use unrounded numeric p-values and show
threshold, method, current row count, empty state and reset. Default to all
results, retain null results and state that significance does not establish
causality. Show tiny p-values with a bound rather than `0.0000`.

Expand all relevant disclosures and include all result rows for printing.
Preserve section titles in print, then restore screen state. Scope "Expand all"
to every section the label promises. Render core content without JavaScript.

## Privacy and portability

Use at least 10 distinct people per published group, or the stricter applicable
organisational rule. Check every filter, time period and subgroup. Withhold
complementary cells or totals that reveal suppressed groups by subtraction.

Publish no individual rankings, raw employee rows, identifiers or person-specific
effect estimates, including hidden HTML/JSON. IDs removed from one visible table
may still remain elsewhere in the artefact, so inspect the embedded content.
Keep inference sample counts separate from privacy-safe published breakdowns.

Reports run offline with embedded CSS/scripts and no remote dependencies.
Escape all input-derived text. Identical input snapshots should produce
identical report content. Store execution timestamps in the audit separately.

## Verification gates

| Gate | Evidence |
|---|---|
| Provenance and sample | Grain, source mappings, coverage and model-specific counts reconcile |
| Numerical | Estimate, interval, sign, units, denominator and displayed text match result records |
| Methodological | Adjustment timing, support, dependence, test nulls and claim boundaries documented |
| Privacy | Distinct-person thresholds, complementary disclosure and embedded records reviewed |
| Packaging | Offline operation, no remote assets or unapproved data calls |
| Browser | Default/expanded/filtered states, keyboard/touch help, mobile overflow and complete print output |

Run browser checks after generation completes. Screenshots should cover default
and expanded long-content states. If a browser is unavailable, mark visual and
interaction behaviour unverified. A rendered report is not validation of a
business interpretation.
