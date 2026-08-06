---
layout: page
title: "Frontier"
eyebrow: "Frontier"
description: "Turn a Viva Insights export into a finished dashboard, executive deck, or ROI analysis by pasting a prompt into a coding agent. Includes reusable prompts, schema guides, and example outputs."
permalink: /frontier-analytics/
---
# Frontier

**Turn a Viva Insights export into a finished dashboard, executive deck, or ROI analysis by pasting a prompt into a coding agent.**

Frontier is a collection of ready-to-use prompts, schema guides, and example outputs for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/). Instead of writing analysis code from scratch, you export your data, pick a prompt, and let a coding agent such as [GitHub Copilot](https://github.com/features/copilot) or [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) generate the deliverable for you to review and adapt.

> **Note:** Everything in this section is sample code and starter assets, not production software. Outputs require review, validation, and adaptation to your environment before use.

{% include responsible-use.html %}

---

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

---

## Which asset should I use?

Frontier offers four ways to bring this work into a coding agent. They solve overlapping problems, so use this table to pick the right one before browsing further.

| Mechanism | What it is | Status | Use it when |
|-----------|------------|--------|-------------|
| [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) | Structured text you copy and paste into any coding agent for a single analysis task. | Available | Your agent does not support Skills or MCP, or you want a one-off analysis without any setup. |
| Skills | A packaged capability that a compatible coding agent loads automatically, so it applies the right conventions without you pasting anything. See the [skills folder on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills). | Available | Your coding agent supports the Skill format (for example GitHub Copilot CLI or Claude Code) and you want ongoing, repeated Viva Insights work to follow consistent conventions. |
| [vivainsights-context.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) | A single context file you paste once at the start of a session. | Available | Your agent does not support Skills, but you still want to avoid repeating setup instructions in every prompt. |
| [mcp/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/mcp) | A protocol-level integration that would let an agent query prompts, schemas, and tools directly from a server. | Concept only, no server implemented | Not yet. Read the folder for the design intent and to track progress. |

---

## What's inside

| Section | Description |
|---------|-------------|
| [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) | Structured, ready-to-paste prompts for coding agents. Covers Copilot adoption tracking, user segmentation, ROI estimation, and Purview audit log analysis. |
| [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) | Data dictionaries for person query exports, Purview audit logs, join patterns, and common data pitfalls. |

Additional resources available on GitHub:

| Folder | Description |
|--------|-------------|
| [skills/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills) | Agent Skills. Packaged capabilities that a compatible coding agent loads automatically for ongoing Viva Insights work. |
| [examples/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/examples) | Sample output specifications describing what a finished deliverable looks like. |
| [templates/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates) | Templates for contributing new prompt cards, skills, and schema docs. |
| [mcp/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/mcp) | Concept and sample configuration for Model Context Protocol (MCP) integration. Not yet implemented: there is no MCP server today, see the folder's README for the design intent. |

---

## Getting started

### Prerequisites

Before you start, make sure you have:

1. **Exported Viva Insights data.** Typically a person query CSV exported from the Viva Insights Analyst portal. Person query data has a panel structure with rows keyed by `PersonId` and `MetricDate` (person-week or person-day granularity). HR attributes such as organization, function, geography, and level are included as columns.

2. **An R or Python environment.** You need one of the following set up locally:
   - **R** (4.1+) with [vivainsights](https://microsoft.github.io/vivainsights/) installed: `install.packages("vivainsights")`
   - **Python** (3.9+) with [vivainsights](https://microsoft.github.io/vivainsights-py/) installed: `pip install vivainsights`

3. **A coding agent.** Any AI-assisted coding tool that can run R or Python:
   - [GitHub Copilot](https://github.com/features/copilot) (in VS Code, JetBrains, or CLI)
   - [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
   - Other coding agents with code execution capabilities

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
5. **Keep your data private.** Do not paste raw data into cloud-based agents unless your organization's data policies allow it. Use local or enterprise-hosted agents when working with sensitive HR data.

---

## Quick links

- [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [Skills library](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/skills)
- [Browse on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics)
- [Contributing](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md)
- [Changelog](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CHANGELOG.md)

---

## Contributing

See the [Contributing guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md) for how to add prompt cards and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.
