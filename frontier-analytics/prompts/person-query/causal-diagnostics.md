# Person Query: exposure-outcome analysis and causal diagnostics

## Purpose

Assess whether a Person Query and a selected outcome can support the proposed
analysis, then run compatible diagnostics and produce an evidence-led report.

## Audience

Viva Insights analysts, people analytics practitioners and HR analytics leads.

## When to use

Use for a user-selected Viva Insights exposure and another metric or custom
organisational outcome. Use the separate
[Copilot Causal Toolkit prompt](../copilot-adoption/copilot-causal-toolkit.md)
when explicitly requesting that toolkit's prescribed Copilot/DML workflow.

## Required inputs

- A local Person Query export, usually keyed by `PersonId` and `MetricDate`,
  with an explicit mapping if imported/localised names differ.
- Exposure and outcome definitions, units and actual measurement windows.
- Query grouping, filters, metric rules and coverage/eligibility evidence.
- For custom outcomes, when and how each value was measured and attached.
- Intended population, decision, privacy threshold and R/Python preference.
- A local sample-code checkout if using the paired skill's references/examples.

No fixed number of weeks guarantees identification. The data must support
the particular contrast. A static outcome is not made longitudinal by appearing
on every weekly row.

## Assumptions

Sources are read locally in an approved environment. Unknowns remain unknown.
Numeric fields are candidate variables rather than automatically valid causal
exposures or outcomes. Inference defaults to observational claims unless a
defensible identifying design has been established.

## Recommended output

A readiness assessment first. After an approved supported analysis, reproducible
scripts, audit/results records and a self-contained HTML report with model labels,
two-column evidence summaries, explicit variables and expandable technical detail.

## Prompt

```text
Help me assess and analyse an exposure-outcome relationship in a Viva Insights
Person Query. Start in readiness mode and ask for missing input locations,
R/Python preference, question, population and selected variable definitions.
If a local sample-code checkout is available, read
frontier-analytics/skills/viva-insights-causal-analysis/SKILL.md and its required
references, using the paired viva-insights-analysis guidance for schema/import.
If it is unavailable, request it before executing that workflow. Do not download
or install anything without approval.

Inspect actual headers and bounded aggregate summaries locally. Do not print
employee rows. Confirm PersonId/MetricDate mappings, daily/weekly/monthly grain,
metric rules, eligibility, missing-versus-zero semantics, joins and privacy.
For a custom organisational outcome, ask when it was measured, what period it
describes, and whether repeated values are new measurements or copied attributes.
Do not treat an annual score or quarterly total repeated on weekly rows as a
weekly outcome. Require actual measurement grain and a defensible exposure
window. Reject ambiguous joins and mechanical exposure-derived outcomes.

Propose the primary contrast, estimator, controls and only those diagnostics
the data supports. Within-person, prior-outcome and timing-placebo models are
not mandatory. A permutation test needs an explicit exchangeability argument.
Do not scan arbitrary metric pairs or report only significant results.
Confirm material design choices with me before fitting models.

Use at least 10 distinct people in published groups, or the stricter applicable
organisational requirement. Do not publish person-level scores, identifiers or
hidden raw data. Treat null findings and unsupported checks as valid outputs.
Explain units, interval methods, dependence and limitations in plain language.
Do not call model agreement proof of causality or a non-significant placebo
proof of no confounding.

If the analysis is blocked, return the exact missing evidence and the narrower
analysis that remains possible. Otherwise generate reproducible results and an
offline report, verify the numbers and UI separately, and compare with any
prior analysis without assuming its conclusions are correct.
```

## Adaptation notes

- Any documented metric can be proposed as an exposure or outcome, but its
  support, timing and derivation determine whether a model is meaningful.
- Keep the outcome's measurement period even when the query has finer grain.
- A non-Copilot exposure belongs here rather than being silently substituted
  into the existing fixed-treatment Copilot Causal Toolkit.
- Use the paired skill's R and Python synthetic examples to learn the gates.
  They are demonstrations, not real-export adapters.

## Common failure modes

- **Copied outcomes inflate sample size:** count independent measurement periods,
  not the number of times an attribute is repeated.
- **Missing usage becomes a control group:** require eligibility and capture
  evidence before interpreting missing exposure as zero.
- **Rolling windows leak into placebos:** establish non-overlapping exposure and
  outcome windows, not just different row dates.
- **Numeric means suitable:** inspect ordinal codes, rates, stock measures and
  exposure-derived outcomes separately.
- **Sophisticated estimator means causal:** record the identifying assumptions
  and unresolved confounding regardless of the estimator's name.
