# Build with AI: Developer Experience and Copilot

## Purpose

Reproduce and customise the synthetic developer dashboard with its existing R
implementation, and assess real GitHub query inputs without guessing their contract.

## Audience

Engineering managers and people analytics practitioners studying working
conditions alongside GitHub and Microsoft 365 Copilot use.

## When to use

Choose this journey for the Developer Experience reference report, not direct
GitHub API analytics or an evaluation of developer productivity.

## Required inputs

- A local sample-code checkout containing the shared dashboard runner.
- R and the dependencies listed by the runner; Pandoc for demo rendering.
- Demo: no customer files; the copied helper creates synthetic data.
- Real readiness assessment: approved local CSVs and available documentation
  for grain, identity, metrics, eligibility and coverage.

## Assumptions

The demo uses an extended illustrative schema. M365 feature actions and
eligibility/completeness flags are not a verified flexible-query export.
Real GitHub data can be inspected, but v1 has no verified real build adapter.

## Recommended output

Demo mode produces a self-contained HTML report, copied source and provenance.
Real mode produces an input-readiness assessment and missing-evidence list;
it does not promise a complete report.

## Quick prompt (short version)

```text
Help me reproduce or customise the Developer Experience and Copilot dashboard.
Recover choices already supplied, then ask only for missing checkout, demo versus real mode, output folder and intended changes.
Read frontier-analytics/skills/viva-insights-copilot-dashboards/SKILL.md in that checkout as workflow instructions, without requiring skill installation.
Use its GitHub manifest and shared R runner; preserve coverage, privacy and synthetic labels.
For real GitHub files, inspect only and explain the missing contract evidence; v1 has no verified real build adapter.
Do not infer productivity, code quality, burnout, licensing or non-use from activity alone.
```

## Prompt

```text
Use the Developer Experience and Copilot reference in my local sample-code checkout.
Recover choices already supplied, then use prompt mode for checkout/output locations,
demo or real, audience, available files, period, grouping, privacy policy and intended changes.

Read frontier-analytics/skills/viva-insights-copilot-dashboards/SKILL.md, its selected mode
and GitHub manifest, and the shared analytical contract. Use the shared runner README
at frontier-analytics/starter-kits/copilot-query-dashboard/ for exact commands/config.

In demo mode, reproduce the copied Rmd/helper in an isolated workspace. Keep the seed,
synthetic window, full developer baseline, eligibility and coverage distinctions.
In real mode, run bounded local inspection. Identify source grain, join keys, units,
coverage, licence evidence and model/language allocation periods. Headers alone do not
verify semantics. Return supported observations and missing evidence for a future adapter.
Do not source the synthetic helper on real inputs or bypass the unsupported-build gate.

For customisation, edit the copied source, not bundled HTML or the canonical example.
Presentation changes should preserve aggregate values. Population or metric changes
require renewed inspection and my approval. No arbitrary duplicate removal, fuzzy
employee joins, individual rankings or converting unresolved coverage to zero.
Keep at least 10 distinct people per published group or a stricter organisational rule.
Do not interpret acceptance as quality, available time as coding, or after-hours as burnout.

Record revision, inputs and configuration. Avoid loading whole CSVs or generated HTML
into model context. After two unsuccessful repairs of one failure, report the blocker.
Return the report or readiness assessment, rerun command and limitations; do not publish.
```

## Adaptation notes

Try changing the panel order or explanatory language on the synthetic demo
first. Real-data support needs a source contract and a separately reviewed
adapter. Do not rename public GitHub API fields into an assumed Viva contract.

## Common failure modes

- **Missing activity becomes zero:** require independent coverage and eligibility.
- **Only Copilot users form the baseline:** retain the full developer population.
- **Full-window mixes become weekly trends:** preserve their actual period.
- **Demo runs over customer data:** use distinct modes and output workspaces.

## Resources

- [Starter kit](../../starter-kits/developer-experience-dashboard/README.md)
- [Shared runner](../../starter-kits/copilot-query-dashboard/README.md)
- [Optional skill installation](../../skills/README.md#dashboard-skill-installation)
- [Synthetic schema manifest](../../../examples/utility-r/_data/github/README.md)
