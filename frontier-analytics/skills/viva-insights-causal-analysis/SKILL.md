---
name: viva-insights-causal-analysis
description: >
  Assess and analyse exposure-outcome relationships in Viva Insights Person
  Query exports using design-first panel diagnostics. Use when an analyst asks
  whether a workplace behaviour, Copilot metric or intervention affects another
  metric or a supplied organisational outcome, or requests within-person,
  prior-outcome, placebo or sensitivity comparisons. Verify actual outcome
  measurement frequency before using repeated organisational attributes as
  panel outcomes. Supports R and Python guidance with bounded synthetic
  examples. Do not use for routine descriptive dashboards, prediction-only
  tasks, arbitrary metric-pair scans or pipeline validation. An explicit request
  for the existing Copilot Causal Toolkit follows that toolkit's workflow.
metadata:
  version: "1.0.0"
---

# Person Query analysis and causal diagnostics

Own the design and diagnostic workflow, while keeping evidence claims within
what the export can establish. An observational estimate is not automatically
causal because it uses fixed effects or several checks.

This is a guided skill, not a drop-in real-data modelling engine. Its bundled
R and Python examples demonstrate readiness checks and a limited balanced-panel
linear model on fabricated data. They do not implement a general PPML, bootstrap,
DML or quasi-experimental estimator.

## Locate the shared rules

Install alongside `viva-insights-analysis` from the same repository revision.
Read its [analysis conventions](../viva-insights-analysis/reference/analysis-conventions.md)
and [data pitfalls](../viva-insights-analysis/reference/data-pitfalls.md) directly,
rather than launching a second workflow for the same task. If those files are
absent, ask for the local sample-code checkout and read them there. If neither
location is available, stop execution and explain the paired installation.

Use the shared skill's package discovery guidance for imports, aggregation and
plots. Verify functions against installed package versions. Do not assume the
live package documentation describes the installed environment. Existing
segmentation recipes do not authorise blanket zero-filling for inference.

No private services, personal skills, telemetry or credentials are required.
Do not install packages, clone repositories, publish reports or send data
elsewhere without the user's approval.

## Workflow

### 1. Scope the question and choose a mode

For discussion or readiness requests, return an assessment without fitting
models. For execution, recover supplied choices before asking for:

- Local input files, R/Python preference and output location.
- Audience, decision and proposed exposure-outcome contrast.
- Raw field names, units and definitions for exposure and outcome.
- Export grain and the **actual measurement window** of each selected variable.
- Eligible population, query filters, licence/coverage evidence and identity keys.
- Baseline controls, intervention timing and known missingness.
- Required privacy threshold and whether standalone or combined reports are wanted.

If an answer is skipped, record it as unknown. Do not invent outcome timing or
turn a real-data request into a synthetic demonstration to make it succeed.

**Output:** a short analysis contract and open questions. Confirm material design
choices before estimation. An approved plan need not be approved repeatedly.

### 2. Establish the Person Query and outcome contract

Read [Person Query contract](reference/person-query-contract.md).
Inspect bounded summaries locally. Keep employee rows and identifiers out of
prompts, logs and shareable files.

Validate `(PersonId, MetricDate)` or a verified mapping, query grain, source
coverage, measurement rules and join cardinality. A numeric column is only a
candidate variable. Audit its support, within-person variation, denominator,
derivation and temporal alignment.

**Gate:** do not run weekly within-person outcome models on copied annual scores,
monthly stock values or quarterly totals. Do not treat missing telemetry as zero.
Stop an affected analysis when source definitions, joins or timing are unresolved.
An explicit method rehearsal may proceed with a visible synthetic/unverified label.

**Output:** mapping and coverage audit plus per-analysis readiness status. Retain
the eligible denominator, and record the reason for every sample restriction.

### 3. Select compatible comparisons and inference

Read [Design and inference](reference/design-and-inference.md).
Choose the primary model and compatible diagnostics rather than automatically
running every M1/M2/M3/P1 model. A static organisational outcome needs a
person-level outcome design, not weekly replication.

Record literal variables, formula, exposure contrast, outcome window, estimation
sample and identifying assumptions. Use maintained statistical packages for
real-data estimation, checking convergence, support, separation and exclusions.
Do not reuse the synthetic runner as a validated real-data adapter.

**Gate:** a causal claim requires an identifying design and explicit assumptions.
Otherwise proceed only with an agreed observational association. A missing
design-specific inference implementation blocks that inferential claim, not a
clearly labelled descriptive summary.

### 4. Challenge the finding

Run justified timing, sample, scale, influence and missingness checks. Record
the null and exchangeability assumption before any permutation test. Distinguish
different estimands from contradictory evidence.

Check controls for post-exposure measurement, mechanical overlap and leakage.
Report unperformed diagnostics as unavailable, with reasons. Do not select the
most favourable p-value or subgroup. A non-significant placebo does not rule
out confounding.

**Output:** model results and an evidence table separating support, conflicts,
unavailable checks and unresolved provenance.

### 5. Produce and verify the deliverable

Read [Report and verification](reference/report-and-verification.md).
Return the readiness assessment alone if blocked. After a supported analysis,
generate reproducible scripts, audit/results records and a self-contained HTML
report unless another format is requested.

Preserve technical detail through disclosure panels. Show plain-language
questions, exact variables, model-labelled results, two-column evidence bullets,
visible caveats and hover/focus/tap definitions. Generate standalone and combined
reports from shared content.

**Gate:** verify numerical consistency, privacy, offline packaging and browser
behaviour separately. If visual verification cannot run, report it as unverified.
The user reviews the claim and disclosure boundaries before external sharing.

## Routing boundaries

- An explicit request to use the existing Copilot Causal Toolkit stays with its
  documented notebooks and prescribed treatment. See the repository's
  [toolkit prompt](../../prompts/copilot-adoption/copilot-causal-toolkit.md) when
  working from a checkout. For an installed-only skill, locate that prompt in
  `repo_root` first. Do not modify its helpers or generalise its treatment silently.
- For DML, difference-in-differences, event studies, IV or discontinuity requests,
  propose the necessary design-specific plan. V1 does not automatically execute
  those designs or endorse existing example implementations without inspection.
- Dashboard-only tasks belong to the relevant dashboard skill. Prediction-only
  tasks need held-out evaluation and should not inherit causal labels.

## References and examples

| Read | When |
|---|---|
| [Person Query contract](reference/person-query-contract.md) | Before mapping or aggregating an export or organisational outcome |
| [Design and inference](reference/design-and-inference.md) | Before selecting models, controls or diagnostic tests |
| [Report and verification](reference/report-and-verification.md) | Before reporting or sharing results |
| [Evaluation scenarios](reference/evaluation.md) | When testing or extending the skill |
| [R and Python examples](examples/README.md) | To run the synthetic readiness and balanced-panel demonstration |

## Guardrails

- Use only approved, local data access. Do not reconstruct identity mappings
  through names, email guesses or combinations of organisational attributes.
- Apply the organisation's minimum reporting size, with a floor of 10 distinct
  people for this skill's published groups, or a stricter applicable requirement.
  Suppress complementary cells when totals would reveal a small group.
- Never publish individual performance rankings, identifiers or person-specific
  effect estimates. Pseudonymised records remain sensitive.
- Do not translate collaboration activity directly into productivity, burnout,
  performance or ROI. Explain the measured behaviour and its limitations.
- No statistical check establishes data provenance. Fractional values or
  tier-shaped patterns justify questions, not a claim that outcomes are synthetic.
