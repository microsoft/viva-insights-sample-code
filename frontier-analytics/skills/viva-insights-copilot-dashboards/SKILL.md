---
name: viva-insights-copilot-dashboards
description: >
  Reproduce and customise the Viva Insights Consumption and Ways of Working
  or Developer Experience and Copilot reference dashboards with reusable R
  code. Use for building either demo, adapting Consumption or GitHub query
  exports into these dashboards, or changing their layout, grouping or
  periods. Do not use for general Person Query analysis, unrelated dashboards,
  direct GitHub API analytics, or causal impact estimation.
metadata:
  version: "1.0.0"
---

# Copilot query dashboards

Own the dashboard workflow, not general Viva Insights analysis. Use the
existing R implementation before generating new code. Both reference demos
are synthetic; they are not verified drop-in adapters for real exports.

## Locate the assets

The user needs a local checkout of `microsoft/viva-insights-sample-code`
containing `frontier-analytics/starter-kits/copilot-query-dashboard/dashboard.R`.
Ask for `repo_root` if it is not known. Do not silently clone, install packages,
or use private services. Record the checkout revision; check installed package
versions rather than assuming live package documentation matches them.

If installing, install this skill alongside `viva-insights-analysis` from the
same checkout. Prompt-only use reads these files in place; no installation is needed.
Read its [shared contract](../viva-insights-analysis/reference/copilot-query-contracts.md)
directly; do not recursively launch that skill for the same task. If the sibling
file is absent, read the corresponding file in `repo_root`. If neither exists,
stop and explain how to obtain the paired skills. Never fall back to invented
schema or licensing rules.

## Choose one mode, then load only its reference

| Request | Reference |
|---|---|
| Open or reproduce a supplied synthetic demo | [Reproduce](reference/reproduce.md) |
| Use the user's real exports | [Adapt](reference/adapt.md) |
| Change an existing report's style, panels, grouping or period | [Customise](reference/customise.md) |

Then read only the selected report's manifest:
[Consumption](reference/consumption.md) or [Developer Experience](reference/github.md).
Read the runner README in `repo_root` for its config schema and exact commands.
Do not load both reports, their generated HTML, or all CSVs into model context.

## Decisions belong to the user

Recover previously supplied choices first. In prompt mode (a structured
question tool when available, concise questions otherwise), collect report,
demo versus real, audience, local input/output locations, date window, grouping,
privacy minimum and intended changes. A demo may use its fixed sample defaults.
For real data, show supported panels, excluded panels and reasons, and mapping
evidence before asking for approval. Ask again only when the scope or evidence
changes. Do not change "real" to "demo" to make a blocked run succeed.

## Output contract

For a successful render, return the local HTML location, reproducible
configuration/command, validation summary, supported versus excluded sections,
and evidence limitations. For inspection or a blocked build, return only the
readiness assessment, missing evidence and next supported action; do not invent
an HTML location. The runner produces provenance for successful reports.
Do not imply that rendering validates business meaning. No commits, uploads
or publication without explicit approval.

## Efficiency and safety

- Execute the runner locally; return bounded summaries, never raw employee
  rows, identifiers, full error dumps containing records, or full HTML bundles.
- Inspect once per input fingerprint. Reinspect when inputs or mapping changes.
- Use a fresh output directory for each run; preserve prior results and source
  fixtures. Edit copies, not the canonical reference.
- Validate aggregates before rendering. For presentation-only changes, reuse
  the approved aggregate data and compare it before/after.
- After two unsuccessful repairs of the same failure, stop with the cause,
  evidence and next required input; do not regenerate the whole project.
- Keep synthetic and real paths separate. All analytical and privacy rules
  come from the shared contract, including the minimum group size of 10.
