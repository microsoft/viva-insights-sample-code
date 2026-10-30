---
layout: page
title: "Frontier — Agent Skills"
eyebrow: "Frontier"
description: "Choose a Viva Insights agent Skill for analysis, query dashboards, or Person Query causal diagnostics."
permalink: /frontier-analytics-skills/
css: /assets/css/skill-catalogue.css
---

# Agent Skills for Viva Insights

An agent Skill is a folder containing a `SKILL.md` file and supporting references. A compatible coding agent can load the Skill when your request matches its description, applying its guidance without you pasting instructions each time. The Skills in this library are for working with Viva Insights exports in R or Python. They are separate from [People Skills data analysis]({{ site.baseurl }}/skills-data-join/).

For a single task without installation, use the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/). Check your agent's documentation for Skill support and installation locations.

## Choose a Skill

Select **Start using** on a card for setup steps, a starter prompt, the questions each Skill helps answer and the results to expect. The two optional Skills read references from the shared analysis Skill, so their steps install both.

{% include skill-catalogue.html %}

<span id="get-started" aria-hidden="true"></span>

## Before you begin

Installing a Skill does not install R, Python or their packages. For the full list of files and manual installation steps for any agent, see the [Skills guide on GitHub](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md). Review the agent's proposed approach and check its results against your actual export.

For one-off dashboard or causal-diagnostics work, you can instead use the [dashboard prompts]({{ site.baseurl }}/frontier-analytics-prompts/#query-dashboards) or the [Person Query prompt](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/prompts/person-query/causal-diagnostics.md) with a local checkout, without installing a Skill. If your agent does not support Skills, the [context file](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) is another starting point.

{% include responsible-use.html %}

[Back to Frontier]({{ site.baseurl }}/frontier-analytics/) · [Browse the Skills source](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills)

<script src="{{ '/assets/js/skill-dialog.js' | relative_url }}"></script>
