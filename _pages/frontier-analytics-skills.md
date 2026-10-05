---
layout: page
title: "Frontier — Agent Skills"
eyebrow: "Frontier"
description: "Choose a Viva Insights agent Skill for analysis, query dashboards, or Person Query causal diagnostics."
permalink: /frontier-analytics-skills/
---

# Agent Skills for Viva Insights

An agent Skill is a folder containing a `SKILL.md` file and supporting references. A compatible coding agent can load the Skill when your request matches its description, applying its guidance without you pasting instructions each time. The Skills in this library are for working with Viva Insights exports in R or Python. They are separate from [People Skills data analysis]({{ site.baseurl }}/skills-data-join/).

For a single task without installation, use the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/). Check your agent's documentation for Skill support and installation locations.

## Choose a Skill

| Skill | Use it for | What to expect |
|---|---|---|
| [Viva Insights analysis](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills/viva-insights-analysis) | Importing and checking exports, computing and visualising metrics, Copilot usage segmentation, network analysis, and common R or Python deliverables. | Shared analytical conventions and references. Start here for general Viva Insights work. |
| [Copilot query dashboards](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills/viva-insights-copilot-dashboards) | Reproducing or customising the Consumption and Developer Experience reference dashboards, or assessing whether real query exports support them. | An optional R-first workflow. Install it alongside Viva Insights analysis from the same checkout revision. The demos are synthetic; real Consumption support is scoped to documented credits, and a real GitHub adapter is not yet verified. |
| [Person Query causal analysis](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills/viva-insights-causal-analysis) | Checking whether a selected exposure and outcome support within-person analysis and diagnostics. | An optional design-first workflow. Install it alongside Viva Insights analysis from the same checkout revision. Its R and Python examples use fabricated data and are not a general real-data modelling engine. |

The two optional Skills read references from the shared analysis Skill. Keep the paired folders intact and from the same revision. For the full list of files and installation instructions, see the [Skills guide on GitHub](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md).

## Get started

1. Choose the Skill that matches your task and check the [installation guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#installing-a-skill) for your agent's skills directory. For either optional Skill, follow its [dashboard](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#dashboard-skill-installation) or [causal-analysis](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#person-query-causal-analysis-skill-installation) pairing instructions.
2. Keep a local checkout of the sample-code repository when using a workflow that reads its runner or reference assets. Installing a Skill does not install R, Python, or their packages.
3. Open your coding agent and describe your task, data, and intended output. For example: *"Assess whether my Person Query and quarterly outcome can support within-person analysis before running models."* Review the agent's proposed approach and check its results against your actual export.

For one-off dashboard or causal-diagnostics work, you can instead use the [dashboard prompts]({{ site.baseurl }}/frontier-analytics-prompts/#query-dashboards) or the [Person Query prompt](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/prompts/person-query/causal-diagnostics.md) with a local checkout, without installing a Skill. If your agent does not support Skills, the [context file](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) is another starting point.

{% include responsible-use.html %}

[Back to Frontier]({{ site.baseurl }}/frontier-analytics/) · [Browse the Skills source](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills)
