---
layout: page
title: "Frontier Analytics"
permalink: /frontier-analytics/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide any default Minima navigation that might appear */
.site-header .site-nav,
.trigger,
.page-link:not(.dropdown-toggle):not(.btn) {
  display: none !important;
}

/* Ensure our custom navigation is visible */
.custom-nav {
  display: block !important;
}
</style>

# Frontier Analytics

Frontier Analytics is an **export-first, self-service analytics toolkit** for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/). It provides reusable prompts, starter kits, schema documentation, and example specifications that you can combine with a coding agent to produce analytics outputs from exported Viva Insights data.

> **Note:** Everything in this section is sample code and starter assets — not production software. Outputs require review, validation, and adaptation to your environment before use.

---

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

---

## What's inside

| Section | Description |
|---------|-------------|
| [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) | Structured, ready-to-paste prompts for coding agents. Covers Copilot adoption tracking, user segmentation, ROI estimation, and Purview audit log analysis. |
| [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/) | Bundled workflows that combine a use case, required inputs, prompts, and expected outputs into a single package. Start here if you want an end-to-end walkthrough. |
| [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) | Data dictionaries for person query exports, Purview audit logs, join patterns, and common data pitfalls. |
| [Quickstart]({{ site.baseurl }}/frontier-analytics-quickstart/) | Step-by-step guide to getting your first output from Frontier Analytics. |

Additional resources available on GitHub:

| Folder | Description |
|--------|-------------|
| [examples/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/examples) | Sample output specifications describing what a finished deliverable looks like. |
| [templates/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates) | Templates for contributing new prompt cards, starter kits, and schema docs. |
| [mcp/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/mcp) | Concepts and sample configuration for Model Context Protocol (MCP) integration. |

---

## How to use this with a coding agent

The intended workflow:

1. **Export your data.** Run a person query (or other query type) from the Viva Insights Analyst portal and download the CSV.
2. **Pick a starter kit or prompt card.** Browse [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/) for end-to-end workflows or the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) for individual analysis tasks.
3. **Review the schema docs.** Check [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) to understand the structure of your exported data — column definitions, expected granularity, and common pitfalls.
4. **Open your coding agent.** Launch [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar tool in a workspace with R or Python available.
5. **Paste the prompt.** Copy the prompt text from the card, point it at your data file, and let the agent generate the output.
6. **Review and iterate.** Check the output against the documented failure modes and adaptation notes. Refine as needed.

### Recommended packages

These prompts are designed to work with the open-source Viva Insights packages:

- **R:** [vivainsights](https://microsoft.github.io/vivainsights/) — `install.packages("vivainsights")`
- **Python:** [vivainsights](https://microsoft.github.io/vivainsights-py/) — `pip install vivainsights`

The packages provide helper functions for reading, validating, and visualizing Viva Insights data. Prompts in this toolkit may reference package functions where appropriate.

---

## Quick links

- [Quickstart Guide]({{ site.baseurl }}/frontier-analytics-quickstart/)
- [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [Browse on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics)
- [Contributing](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md)
- [Changelog](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CHANGELOG.md)

---

## Contributing

See the [Contributing guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md) for how to add prompt cards, starter kits, and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.
