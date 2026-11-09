---
name: viva-insights-process-analysis
description: >
  Analyse collaboration around a named business process, such as an annual
  audit, financial close, planning cycle or review programme, using Viva
  Insights Meeting Query and optional purpose-specific Person Query exports.
  Use when defining which process meetings count, reviewing title/date rules,
  comparing matched cycles, explaining changes in volume or meeting format,
  or refreshing a process report. Owns scope discovery, approved reporting,
  reconciliation and descriptive interpretation. Do not trigger for Purview
  audit-log parsing, compliance certification, event-log process mining,
  generic Copilot dashboards or an unrelated software audit.
metadata:
  version: "1.0.0"
---

# Collaboration around a business process

Help the analyst establish what qualifies, how much collaboration is observed,
where it is organised and how comparable periods differ. Meeting titles are
proxies for process activity. They do not prove process completion, quality,
ownership or intervention impact.

This is an agent-guided local R/Python workflow. The bundled Python rehearsal
and R classifier checks use fabricated audit events. They are not universal
real-export adapters, a process-mining engine or an inference/report renderer.

## Shared contracts and installation

Install alongside `viva-insights-analysis` from the same repository revision.
Read its [analysis policy](../viva-insights-analysis/reference/analysis-policy.md),
[Meeting Query reference](../viva-insights-analysis/reference/meeting-query.md),
[analysis conventions](../viva-insights-analysis/reference/analysis-conventions.md)
and [report verification](../viva-insights-analysis/reference/report-verification.md).
Use its package discovery and installed-API checks for imports and analysis.
Read the [workflow support](../viva-insights-analysis/reference/workflow-support.md)
before claiming a maintained execution path.

If paired files are unavailable, ask for the installed core folder or a local
sample-code checkout. Stop the affected execution if neither is supplied.
Do not silently substitute personal skills or private services. No telemetry,
credentials or live-product connector is required. Do not install dependencies,
download data or publish files without the user's approval.

## 1. Agree the question and reporting contract

Recover supplied decisions before asking for the process definition, audience,
R/Python preference, local inputs, output location, scope approver and comparison.
Distinguish descriptive footprint/monitoring from an intervention-effect question.
Record unknown intervention dates or participation without blocking a useful
descriptive assessment.

Read [scope contract](reference/scope-contract.md). Record source fingerprints,
query purpose/configuration, grain, fields/units, population, dates, cycle-year
labels, coverage evidence, organisational lens, privacy and optional event-base
rules. Keep raw coverage, complete months, complete weeks and process phases
distinct. A filename or latest event is not completeness evidence.

**Output:** a short contract, open questions and readiness per requested result.
Confirm material scope and comparison choices before publishing headline totals.

## 2. Explore scope, then freeze an approved definition

Read [scope contract](reference/scope-contract.md) and
[review and precedence](reference/review-and-precedence.md).
Apply the core's reviewed quality profile and shared exclusions before topical
analysis, keeping raw exports unchanged. Inspect fields locally and keep
sensitive subjects/identifiers out of prompts and shared examples.

Create separate candidate and approved flags. Review positive, negative and
borderline examples across years, languages, periods and organisational cuts.
Make keywords, aliases, supported year codes, exclusions and annual gates
explicit. Do not assume a generic word has the same business meaning everywhere.
Verify Unicode/word-boundary behaviour in the actual R/Python regex backends.

Agree exact decision precedence. An occurrence-level decision never becomes a
title-wide rule without approval. Preserve pending states and unmatched or
ambiguous reviews. Never let a Keep bypass mandatory quality/date gates.
Record whether a change applies retrospectively and show its membership impact.

**Gate:** unresolved scope permits diagnostics, not an approved process total.
Retain an approved rule version and test worked examples before report building.

## 3. Build comparable analytical evidence

Read [comparisons](reference/comparisons.md). Reconcile duplicate keys and source
vintages under an explicitly approved policy. Use matched observed dates/phases
across years. Compute each standalone category and an exact event-ID union,
retaining intersections and outside-scope reconciliation privately.

Describe event count, attendee hours, hours per meeting, seasonality and supported
format/behaviour measures. Use exported attendee hours. State the valid base and
weighting for every metric. Separate count from time per event using the exact
volume identity in the shared conventions.

Organisational fields in MQ describe the organiser's recorded reporting line.
Report missing attribution, changed labels and unavailable comparisons. Do not
infer attendee composition, headcount or a process owner from those fields.
Apply the stricter organisational rule and a floor of **10 distinct people** for
this workflow. Verify the applicable person base; meeting counts and summed
attendance cannot establish it. Apply any event-count stability threshold separately.

A purpose-specific PQ provides a separate employee-side lens with its own scope,
window, eligibility and weighting. Do not sum it with MQ or assert that a custom
process-hour metric implements the subject classifier. Joining needs a verified
key, compatible definition and cardinality audit.

**Output:** reusable aggregate evidence, valid bases, coverage and reconciliations.

## 4. Challenge the conclusion

Separate source revision, scope revision, period changes and organisational
composition from observed activity changes. Run agreed scope/window sensitivities.
Show effect sizes and practical importance before significance labels.

If inference is requested, choose independent units, temporal dependence,
test family and compatible methods first. Do not inherit a particular estimator,
block width or significance threshold from the example. There is no bundled
Meeting Query inferential engine.

For an intervention-effect request, require an identifying design and exposure
information. A supported Person Query exposure-outcome design belongs to
`viva-insights-causal-analysis`. A Meeting Query event-level design needs its
own inspected plan and implementation. Descriptive before/after differences,
fixed effects or significant tests alone do not establish impact.

## 5. Report, review and refresh

Use the shared report-verification workflow. Maintain a claim-level evidence
ledger and concise commentary-led views, with visible dates, bases, denominator
and caveats. Retain complete eligible supporting detail. Reconcile category
overlaps and keep unavailable values distinct from observed zero.

Keep the analytical evidence fixed during editorial changes. Preserve manual
edits when revising a publication, verify the candidate and source drift, remove
only resolved comments, and publish only to the authorised target with rollback.
When data/scope/windows change, rebuild dependent views and produce a reconciled
bridge against the previous approved evidence.

## References and rehearsal

| Read | When |
|---|---|
| [Scope contract](reference/scope-contract.md) | Intake, readiness and discovery-to-approved reporting |
| [Review and precedence](reference/review-and-precedence.md) | Keywords, review packs, exact decisions and rule changes |
| [Comparisons](reference/comparisons.md) | Matched cycles, denominators, organisational/PQ lenses and refreshes |
| [Evaluation scenarios](reference/evaluation.md) | Checking routing, safeguards and publication boundaries |
| [Fabricated audit rehearsal](examples/README.md) | Exercising classification, reconciliation, privacy and partial periods locally |

Never transfer an engagement's aliases, dates, targets, exclusions, personal
names or data into the public skill. The example's vocabulary and calendar are
fictional choices that must not become defaults for another organisation.
