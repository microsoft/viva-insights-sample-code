---
layout: page
title: "Build with AI: Consumption Dashboard"
eyebrow: "Frontier"
description: "Reproduce the Consumption demo or build a scoped credits dashboard with an AI coding agent and reusable R code."
permalink: /frontier-analytics-prompt-consumption-dashboard/
---
{% include prompt-styles.html %}

[Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) · [View dashboard examples]({{ site.baseurl }}/copilot-dashboards/)

## Purpose

Reproduce the synthetic seven-page R demo, then customise it or build a narrower
credits dashboard from documented Consumption exports without starting over.

## Audience

People analytics practitioners and analysts exploring Copilot consumption.

## When to use

Choose this journey for the Consumption query, not the general Person Query
adoption dashboard. A coding agent can follow the workflow without installing
a skill; frequent users can install the optional dashboard skill.

## Required inputs

- A local checkout of `microsoft/viva-insights-sample-code` containing the
  shared runner at `frontier-analytics/starter-kits/copilot-query-dashboard/`.
- R and the dependencies listed by the runner; Pandoc for the reference demo.
- Demo: supplied synthetic fixture files only.
- Real: locally extracted activity and people-metadata CSVs, query grain,
  date window, chosen HR group, mapping evidence and privacy requirements.

## Assumptions

The demo's tokens and delegated task types are illustrative. The real v1
adapter supports documented credit consumption, not the full demo. Missing
tokens do not block a credits-only report; they exclude token-based panels.

## Recommended output

A self-contained HTML report, reproducible config and source, local provenance,
validation summary, and explicit supported/excluded analyses. Real data stays
outside the repository and is never pasted into the prompt.

## Quick prompt (short version)

```text
Help me reproduce or customise the Consumption and Ways of Working dashboard.
Recover choices already supplied, then ask only for missing checkout, demo versus real mode, output folder and intended changes.
Read frontier-analytics/skills/viva-insights-copilot-dashboards/SKILL.md in that checkout as workflow instructions, without requiring skill installation.
Use its Consumption manifest and the shared R runner; do not regenerate the dashboard from scratch.
For real exports, inspect locally, show supported and excluded panels, and get my approval before building.
Do not invent tokens, licensing, coverage, identity mappings or causal conclusions.
```

## Prompt

```text
Build a Consumption dashboard using the local microsoft/viva-insights-sample-code checkout.
Recover any inputs I already supplied. Use prompt mode for the remaining choices:
checkout path, demo or real, audience, input files, output directory, period, grouping,
privacy threshold (at least 10 distinct people), and requested customisations.

Read frontier-analytics/skills/viva-insights-copilot-dashboards/SKILL.md and only
the selected mode plus Consumption manifest. Read the shared query contract it links.
Use frontier-analytics/starter-kits/copilot-query-dashboard/README.md for the config
schema and dashboard.R commands. Record the checkout revision and source fingerprints.

For demo mode, reproduce the reference using copied synthetic fixtures in a new output
workspace. Keep the fixed seed/window and synthetic labels. Do not alter canonical source files.
For real mode, inspect the extracted CSVs locally and return bounded schema summaries,
not rows or identifiers. Verify keys, grain, units and historical metadata joins.
Present supported/excluded panels and mapping evidence; wait for my approval to build.
Support a credits-only output when tokens or task types are unavailable. Do not generate
synthetic replacements, infer licences from activity, or sum policy limits.

Use the existing R code. Check dependencies before requesting installation. Do not read
the generated HTML or whole CSVs into model context. No individual rankings or unsupported
productivity, ROI, quality, burnout or causal claims. Suppress small groups in every output.
After two unsuccessful repairs of the same failure, stop with the specific blocker.
Return the local report, rerun command, config/provenance and limitations. Do not publish.
```

## Adaptation notes

Start with "change the colours and section order" for a presentation-only
iteration. Changing dates, grouping or sources requires renewed data and
privacy checks. Person Query associations require a separately verified adapter.

## Common failure modes

- **Assuming credit exports contain tokens:** build only supported panels.
- **Joining service rows directly to Person Query:** aggregate to compatible
  person-period grain first; a joined real-data adapter is not supplied in v1.
- **Rendering in the examples folder:** use the isolated runner workspace.
- **Calling credits money or value:** retain credit units and descriptive claims.

## Resources

- [Starter kit](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/starter-kits/consumption-dashboard)
- [Shared runner](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/starter-kits/copilot-query-dashboard)
- [Optional skill installation](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#dashboard-skill-installation)
- [Public Consumption schema](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query)

{% include responsible-use.html %}
