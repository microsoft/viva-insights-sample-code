---
layout: page
title: "Frontier"
eyebrow: "Frontier"
description: "Explore synthetic Viva Insights dashboards, or use prompts, Skills and schema guides to build and review your own analysis."
permalink: /frontier-analytics/
---
<span id="frontier"></span>

Frontier brings together dashboard examples, prompts, agent Skills and schema
guides for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/).
Explore a synthetic report in your browser, or use a coding agent such as
[GitHub Copilot](https://github.com/features/copilot) or
[Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
to build an analysis from your exports. Review the inputs, calculations and
limitations before sharing the output.

> **Note:** Everything in this section is sample code and starter assets. It is not production software. Outputs require review, validation, and adaptation to your environment before use.

---

## Which asset should I use?

Frontier offers four ways to bring this work into a coding agent. Pick a card to jump straight to what you need, or see the full comparison below for more nuance.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="{{ site.baseurl }}/frontier-analytics-prompts/">
    <span class="vi-card-icon">📋</span>
    <span class="vi-card-title">Prompt Library</span>
    <span class="vi-card-desc">Copy a ready-made prompt into your agent for a one-off analysis. No setup required.</span>
    <span class="vi-card-more">Browse prompts →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/frontier-analytics-skills/">
    <span class="vi-card-icon">🧠</span>
    <span class="vi-card-title">Skills</span>
    <span class="vi-card-desc">Your agent loads the right conventions automatically for ongoing work. Needs a Skill-compatible agent, for example GitHub Copilot CLI or Claude Code.</span>
    <span class="vi-card-more">Explore Skills →</span>
  </a>
  <a class="vi-card" href="https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md">
    <span class="vi-card-icon">📄</span>
    <span class="vi-card-title">Context file</span>
    <span class="vi-card-desc">Paste once at the start of a session. A lighter fallback when your agent doesn't support Skills.</span>
    <span class="vi-card-more">View on GitHub →</span>
  </a>
  <a class="vi-card" href="https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/mcp">
    <span class="vi-card-icon">🔌</span>
    <span class="vi-card-title">MCP (concept only)</span>
    <span class="vi-card-desc">A protocol-level integration that would let an agent query prompts and schemas directly from a server. Not implemented yet.</span>
    <span class="vi-card-more">Read the design intent →</span>
  </a>
</div>

### Full comparison

| Mechanism | What it is | Status | Use it when |
|-----------|------------|--------|-------------|
| [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) | Structured text you copy and paste into any coding agent for a single analysis task. | Available | Your agent does not support Skills or MCP, or you want a one-off analysis without any setup. |
| [Agent Skills]({{ site.baseurl }}/frontier-analytics-skills/) | A packaged capability that a compatible coding agent loads automatically, so it applies the right conventions without you pasting anything. | Available | Your coding agent supports the Skill format (for example GitHub Copilot CLI or Claude Code) and you want ongoing, repeated Viva Insights work to follow consistent conventions. |
| [vivainsights-context.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) | A single context file you paste once at the start of a session. | Available | Your agent does not support Skills, but you still want to avoid repeating setup instructions in every prompt. |
| [mcp/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/mcp) | A protocol-level integration that would let an agent query prompts, schemas, and tools directly from a server. | Concept only, no server implemented | Not yet. Read the folder for the design intent and to track progress. |

---

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

---

## What's inside

### Explore the dashboard demos

Open any of these three reports in your browser without installing anything.
They use synthetic data and include R Markdown source and rendering requirements.
Select **Start using** for the links and capability limits.

**Business Leaders AI Usage** brings together M365 Copilot, Cowork, Agent,
Consumption and GitHub Copilot fixtures. Explore reach, usage and trends over
four, eight or thirteen weeks by organisation, function and team. It is a synthetic
reference dashboard, with no verified real-export adapter or dedicated prompt
journey. Descriptive comparisons do not establish causal effects.

{% include dashboard-catalogue.html %}

### Build or customise the query dashboards

These two guided journeys reproduce the existing Consumption and Developer
Experience R demos. The agent inspects real inputs and asks you to approve
supported analyses before building. Missing metrics and coverage remain explicit.

| Journey | Demo | Real-data support |
|---|---|---|
| [Consumption Dashboard]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/) | Full synthetic reference | Documented credit consumption; token/task-type and Person Query panels excluded |
| [Developer Experience Dashboard]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/) | Full synthetic reference | Input-readiness assessment only; real adapter not yet verified |

[Browse the dashboard examples]({{ site.baseurl }}/copilot-dashboards/) or
[install the optional dashboard skill](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#dashboard-skill-installation)
for repeated work. Prompt-only use requires no skill installation.

| Section | Description |
|---------|-------------|
| [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) | Data dictionaries for person query exports, Purview audit logs, join patterns, and common data pitfalls. |

Additional resources available on GitHub:

| Folder | Description |
|--------|-------------|
| [examples/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/examples) | Sample output specifications describing what a finished deliverable looks like. |
| [templates/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates) | Templates for contributing new prompt cards, skills, and schema docs. |

---

## Getting started

### Prerequisites

Viewing the synthetic demos above needs only a browser. To run a prompt or
render a report, check its specific requirements. The general analysis prompts
use R or Python. The dashboard templates require R, Pandoc, R Markdown,
flexdashboard and the packages listed in their rendering requirements.

For analysis of your own exports, prepare:

1. **Exported Viva Insights data.** Use the query types required by the selected workflow. A Person Query has rows keyed by `PersonId` and `MetricDate` (person-week or person-day granularity). HR attributes such as organisation, function, geography and level are included as columns. Cross-query dashboards may need several exports.

2. **An R or Python environment.** You need one of the following set up locally:
   - **R** (4.1+) with [vivainsights](https://microsoft.github.io/vivainsights/) installed: `install.packages("vivainsights")`
   - **Python** (3.9+) with [vivainsights](https://microsoft.github.io/vivainsights-py/) installed: `pip install vivainsights`

3. **A coding agent.** Any AI-assisted coding tool that can run R or Python:
   - [GitHub Copilot](https://github.com/features/copilot) (in VS Code, JetBrains, or CLI)
   - [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
   - Other coding agents with code execution capabilities

### Don't have a coding agent yet?

- **GitHub Copilot** is free to start with a personal GitHub account. Install the Copilot extension in [VS Code](https://code.visualstudio.com/), or install the [GitHub Copilot CLI](https://github.com/features/copilot) and sign in from a terminal.
- **Claude Code** is a terminal-based coding agent from Anthropic. Its [getting started guide](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) covers installation.
- Use your organisation's approved coding agent and data-handling environment for real HR data.

Once installed, open the agent in a folder that has R or Python available, then continue with the workflow below.

### Workflow

1. **Export your data.** Run a person query (or other query type) from the Viva Insights Analyst portal and download the CSV.
2. **Pick a prompt card.** Browse the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) for the analysis task that matches your use case.
3. **Review the schema docs.** Check [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) to understand the structure of your exported data: column definitions, expected granularity, and common pitfalls.
4. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or a similar tool in a workspace with R or Python available.
5. **Paste the prompt.** Copy the prompt text from the card, point it at your data file, and let the agent generate the output.
6. **Review and iterate.** Check the output against the documented failure modes and adaptation notes. Refine as needed.

### Tips for working with coding agents

1. **Be specific about your data.** Tell the agent the file name, column names, and date range. The more context you give, the better the output.
2. **Iterate in small steps.** If the output is not right, ask the agent to fix one thing at a time rather than re-generating everything.
3. **Validate the output.** Spot-check row counts, date ranges, and aggregation logic. Coding agents can make plausible-looking mistakes.
4. **Use the vivainsights packages.** The R and Python packages handle common data validation and visualization tasks. Prompts that reference these functions tend to produce cleaner code.

See "Responsible use & data privacy" below before you work with real HR data.

---

{% include responsible-use.html %}

---

## Quick links

- [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [Agent Skills]({{ site.baseurl }}/frontier-analytics-skills/)
- [Browse on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics)
- [Contributing](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md)
- [Changelog](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CHANGELOG.md)

---

## Contributing

See the [Contributing guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md) for how to add prompt cards and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.
