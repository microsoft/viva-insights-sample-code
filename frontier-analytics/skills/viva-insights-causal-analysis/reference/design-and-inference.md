# Choosing comparisons and inference

## Define the contrast

State the exposure change, outcome window and target population in one sentence.
For example, a ten-action increase and subsequent measured collaboration hours,
or mean exposure before a survey wave and its score. Custom organisational
outcomes may be binary, ordinal or continuous. Do not default to count models
because the source is a Person Query.

Separate treatment assignment (such as licence rollout) from subsequent chosen
usage. Randomised access does not make observed usage random. Draw a simple
causal ordering or diagram to select baseline controls and avoid adjusting for
mediators, post-exposure selection or outcome components.

## Observational menu

| Label | Question | Required support and caveat |
|---|---|---|
| M1 | Do people with different exposure have different outcomes? | Justified population/adjustments, vulnerable to stable and changing confounding |
| M2 | Does the same person's outcome change with their exposure? | Multiple actual measurement periods and within-person variation, time-varying confounding remains |
| M3 | Does the association persist after accounting for earlier outcomes/exposure? | Genuine baseline measurement, no future information, history may change the estimand |
| P1 | Is later exposure associated with an earlier outcome? | Non-overlapping, justified windows, persistence/anticipation/reverse causality need interpretation |

Use exact model-specific fields and samples. Calendar fixed effects remove
common additive shocks in a linear model, not every seasonal or organisational
difference. A static outcome repeated across rows is not suitable for M2.
Lagged outcomes in short fixed-effects panels can introduce dynamic panel bias.
These labels are optional, not a required four-model checklist.

Fixed effects do not diagnose stable selection they difference out. Baseline
comparisons of future adopters and appropriate untreated units can address that
question more directly. A timing placebo that is not significant does not
establish no selection, especially with low power.

Joint lead/current/lag terms and separate marginal regressions answer different
questions. Document the choice, inspect autocorrelation/collinearity and do not
infer reverse causality solely from an unstable sign in a joint model.

## Estimators and adjustment

- Continuous outcomes: a justified linear or nonlinear mean model. Inspect
  support, influence and dependence. Numeric rating codes are not automatically
  interval scales.
- Non-negative counts or amounts: PPML may be appropriate for a multiplicative
  conditional mean. Integer support and equal mean/variance are not required for
  PPML consistency with a correctly specified mean. Inspect separation, convergence
  and all-zero/dropped units.
- Binary outcomes: logit/risk models with interpretable contrasts. An odds ratio
  is not a probability ratio. Check separation and sparse events.
- Rates/proportions: verify numerator/denominator and justified offset or binomial
  formulation. Zero denominators are not zero outcomes.
- Ordinal, censored or time-to-event outcomes: propose a suitable specialist
  model or limit the analysis. Do not force the balanced OLS example onto them.

For real data, check existing package functions and maintained estimators.
Record package versions and verify arguments before use. The examples only
demonstrate balanced two-way demeaning and OLS on specified synthetic cases.
They are not an unbalanced-panel estimator or a production inference library.

Cluster or otherwise account for dependence at the relevant unit/assignment
level. A large number of weekly rows does not compensate for few people or
teams. Few-cluster bootstrap methods are estimator- and design-specific. Verify
implementation, null restrictions and applicability rather than copying a custom
bootstrap from another project. Report the interval and p-value methods separately.

## Credibility checks

Predeclare the primary outcome/model, then choose diagnostics with an explicit
null, scope and interpretation:

- Scale and functional-form alternatives, with original-unit contrasts.
- Influence of extreme observations or units, without selective removal.
- Alternative samples and documented missingness assumptions.
- Timing placebos or negative-control outcomes with defensible causal meaning.
- Permutations only when exchangeability or the assignment scheme supports them.
  Preserve dependence, blocks and time structure as necessary.

A Monte Carlo permutation p-value includes ties and uses
`(extreme_draws + 1) / (draws + 1)`. A random seed makes it reproducible, not valid.
If exposure reshuffling destroys confounding or serial structure, do not call it
calibrated causal inference. Mark it unavailable or explicitly heuristic.

An OLS change statistic and a within-person PPML coefficient may use different
weights, samples, scales and estimands. Agreement is not independent confirmation,
and disagreement is not automatically a failed model.

Do not scan arbitrary exposure/outcome/subgroup combinations and report only
significant ones. Label exploration, preserve null findings and address
multiplicity or validation on independent data.

## Scale integrity

`LogExposure = log2(1 + Exposure)` changes by one when **1 + Exposure** doubles.
It is only approximately per doubling of exposure at larger positive values.
For a raw-unit change x0 -> x1:

`delta = log2((1 + x1) / (1 + x0))`

- Linear outcome model: `outcome difference = beta * delta`.
- Log-link mean model: `mean ratio = exp(beta * delta)`.
- Logit model: `odds ratio = exp(beta * delta)`.

Multiplying exposure by ten changes `log2(1 + Exposure)` nonlinearly near zero.
Do not claim invariance to unit conversion. A log-transformed outcome also
requires explicit retransformation assumptions and original-scale interpretation.

## Coexistence with other repository methods

The Copilot Causal Toolkit is a separate DML workflow with a prescribed
`Total_Copilot_actions_taken` treatment and its own notebooks and helpers.
The workbench does not replace, edit or automatically wrap those helpers.
Inspect the selected notebook's actual grain, preprocessing, controls, estimand
and inference before combining results or claiming methodological equivalence.

The existing methodology materials describe both longitudinal residualisation
and person-level aggregation. Resolve that distinction against the executed
notebook rather than assuming all toolkit estimates are within-person.
DML, cross-fitting and sensitivity statistics do not remove the need for
unconfoundedness, overlap and valid control timing. Do not inherit universal
E-value cut-offs or claim that adding more contemporaneous metrics guarantees
confounder control.

Difference-in-differences, event studies, instruments and discontinuities need
their own design-specific review. V1 returns a proposed plan for those requests.
Existing examples are resources to inspect, not automatic evidence of validity.

## Claim language

Default to association for observational exposure. State effect size, precision,
identification and provenance separately. Do not equate a p-value with the
probability of no effect, or a non-significant estimate with equivalence.
More external collaboration is not necessarily more sales, and less after-hours
activity is not a measured reduction in burnout.
