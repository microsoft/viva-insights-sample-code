# Evaluation and scope

## Deterministic example tests

See [examples](../examples/README.md) for the exact R and Python commands,
fixture definitions and parity checks. Generated records are entirely fabricated
and use Person Query-style keys plus documented custom outcome names.

These tests exercise readiness rules and limited balanced-panel reference slopes.
They do not establish PPML/bootstrap coverage, validate DML, or test all possible
real exports. Genuine causal claims still require the design review.

## Agent acceptance scenarios

Run in an approved skill-compatible coding agent with this skill and the paired
analysis references available. Test explicit in-place use separately from
automatic skill discovery. Use generated synthetic files only.

| Request | Expected behaviour |
|---|---|
| Estimate the relationship in a genuine synthetic effect panel | Name exact variables and units, use a compatible continuous-outcome model, recover the known effect approximately and label it synthetic |
| The between-person relationship is strong in a confounded panel | Inspect within-person evidence, recognise stable confounding, avoid causal endorsement of the pooled slope |
| Treat the annual score repeated each week as a weekly outcome | Reject replicated weekly inference, propose one person-level outcome and a verified exposure window |
| Run weekly fixed effects on repeated quarterly totals | Require actual person-quarter grain and alignment, do not count repeated copies as new measurements |
| Fill missing exposure with zero and call those people non-users | Retain unknown status and ask for independent coverage evidence |
| A derived outcome is proportional to the exposure | Identify mechanical coupling and reject independent effect interpretation |
| Run M1/M2/M3/P1 on one period | Mark within-person/prior-period/timing checks unavailable |
| Choose another exposure instead of Copilot actions | Permit it subject to definitions and support, unless explicitly executing the existing fixed-treatment toolkit |
| Report only significant subgroups below the reporting threshold | Reject the disclosure and selective reporting |
| Double the exposure units with log2(1+x) and claim unchanged effects | Explain scale dependence and require a raw-unit contrast |

Non-trigger checks: ordinary dashboard layout work, pipeline validation and
prediction-only tasks should stay with their relevant workflow.

## How to record evidence

Record skill version, source revision, date, coding-agent type, input case,
requested action, executed commands, observed response and pass/fail against each
criterion. Keep artefacts free of personal paths, employee identifiers and private
data. State whether results reflect:

1. Deterministic executable checks.
2. An explicit-read agent session following this SKILL.md.
3. Automatic discovery/trigger evaluation.
4. Real-data statistical validation.

Do not imply the latter two were done when only the first two were exercised.

## Contribution readiness

Blocking errors include fabricated measurement timing, false-zero imputation,
many-to-many joins, unsupported within-person estimation, leakage, failed fits
presented as success, incorrect effects/units or privacy disclosures.

Warnings include limited history, sparse events, few independent clusters,
incomplete population, unresolved definitions and unverified browser behaviour.
Warnings can block a particular claim without blocking a descriptive audit.

Keep a concise review with severity, evidence and resolution. Before expanding
into a general estimator, benchmark bias, interval coverage and type-I error
against maintained reference implementations over multiple sample/dependence
settings. A plausible result in one synthetic case is insufficient.

## Initial evaluation record

Version 1.0.0 was exercised on 2026-09-28 using Python 3.14.6 and base R 4.6.1.
The public example suite checks eight cases in both languages, including an
independent base-R dummy-variable `lm` reference for the balanced linear fit.
The fabricated effect of 2 produced a within-person estimate of approximately
2.0074. Under stable confounding with true effect zero, the pooled estimate was
2.2467 and the within-person estimate was 0.0074. These numbers verify the
construction and bounded reference calculation, not interval calibration.

A separate GitHub Copilot CLI agent explicitly read this skill and the paired
references, then authored and ran six independent synthetic CSV scenarios in a
temporary evaluation workspace. It did not reuse the bundled example runner.

| Executed scenario | Observed action |
|---|---|
| Annual score copied across eight weekly rows per person | Blocked weekly fixed effects and unsupported earlier-outcome placebo |
| Two quarterly outcomes copied across 26 weekly rows | Required person-quarter alignment rather than 26 independent outcomes |
| Missing exposure with unknown capture/licensing | Rejected zero-fill and non-user classification |
| One observed period | Marked M2/M3/P1 unavailable |
| Outcome defined as 0.5 times exposure | Rejected independent productivity interpretation |
| Approved balanced continuous outcome with assigned binary exposure | Recovered 2.0077 points for a planted +2 effect, with no p-value or calibrated uncertainty claim |

All six requests produced the expected gate decisions or reference result.
Five additional instruction-only probes covered a non-Copilot exposure, the
existing DML toolkit route, ordinal outcome support, a four-person disclosure
and log(1+x) unit dependence. The written guidance handled each explicitly.

This was an **explicit-read agent evaluation**, not an automatic discovery
benchmark. The independent fixtures were temporary fabricated records, not
real exports. There was no real-data statistical validation or HTML/browser
evaluation of a generated analysis report. Recreate the scenarios above for
future agent evaluations and use the examples README to rerun executable parity.

Two integration cautions remain explicit: the sibling analysis skill's
segmentation zero-fill recipe is not an inferential missingness rule, and an
explicit older-toolkit request retains its separate method and interpretation
contract. Neither boundary is silently overridden by this skill.
