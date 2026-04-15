---
layout: page
title: "Frontier Analytics — Quickstart"
permalink: /frontier-analytics-quickstart/
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

# Quickstart

This guide walks you through using Frontier Analytics assets to produce an analytics output from your Viva Insights data.

---

## What is Frontier Analytics in this repo?

Frontier Analytics is a section of the [Viva Insights Sample Code Library]({{ site.baseurl }}/) containing reusable assets for self-service analytics on exported Viva Insights data. It is not a standalone tool or application. The assets — prompt cards, starter kits, schema docs, and templates — are designed to be used _with_ a coding agent to generate dashboards, reports, and analysis notebooks.

---

## Prerequisites

Before you start, make sure you have:

1. **Exported Viva Insights data.** Typically a person query CSV exported from the Viva Insights Analyst portal. Person query data has a panel structure with rows keyed by `PersonId` and `MetricDate` (person-week or person-day granularity). HR attributes such as organization, function, geography, and level are included as columns.

2. **An R or Python environment.** You need one of the following set up locally:
   - **R** (4.1+) with [vivainsights](https://microsoft.github.io/vivainsights/) installed: `install.packages("vivainsights")`
   - **Python** (3.9+) with [vivainsights](https://microsoft.github.io/vivainsights-py/) installed: `pip install vivainsights`

3. **A coding agent.** Any AI-assisted coding tool that can run R or Python:
   - [GitHub Copilot](https://github.com/features/copilot) (in VS Code, JetBrains, or CLI)
   - [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
   - Other coding agents with code execution capabilities

---

## Step-by-step workflow

### 1. Choose a starter kit

Browse the [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/) to find a workflow that matches your use case. Each kit bundles a use case, required inputs, prompts, and expected outputs.

| Starter kit | Description | Complexity | Output |
|-------------|-------------|------------|--------|
| [Copilot Adoption Dashboard]({{ site.baseurl }}/frontier-analytics-starter-kits/#copilot-adoption-dashboard) | Interactive HTML dashboard tracking Copilot adoption metrics, usage trends, and user segmentation | Intermediate | HTML dashboard |
| [Executive Summary Report]({{ site.baseurl }}/frontier-analytics-starter-kits/#executive-summary-report) | One-page memo summarizing key collaboration and Copilot metrics for leadership | Beginner | Markdown / HTML memo |

If no starter kit fits your scenario, browse the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) for individual analysis prompts.

### 2. Prepare your data files

- Export the required query from the Viva Insights Analyst portal (the starter kit specifies which query types).
- Place the CSV file(s) in your working directory.
- If combining with Purview audit logs, export those separately and have them available.

> **Tip:** Missing values in person query data may indicate unlicensed users. Check the [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) for details on how to handle gaps.

### 3. Review the schema docs

Read the relevant sections in [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) to understand:

- What each column in your export means
- The granularity of the data (person-week vs. person-day)
- How to join person query data with other sources (e.g., Purview audit logs)
- Common data pitfalls and how to avoid them

### 4. Open the relevant prompt cards

Each starter kit references specific prompt cards from the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/). Open those cards and read through:

- **Purpose** — what the prompt produces
- **Required inputs** — what data the prompt expects
- **Assumptions** — what the prompt assumes about your data
- **Common failure modes** — known issues and workarounds

### 5. Paste the prompt into your coding agent

1. Copy the full text from the **Prompt** section of the card.
2. Open your coding agent in a workspace that contains your data files and has R or Python available.
3. Paste the prompt. If the prompt asks for a file path, prepend it or provide it when the agent asks.
4. Let the agent generate the code and run it.

### 6. Iterate on the output

- Compare the output against the **expected output** section in the starter kit.
- Check the **common failure modes** listed on the prompt card.
- Use the **adaptation notes** to customize for your organization's HR attributes, date ranges, or metric selections.
- Re-run or adjust as needed. Coding agents work best with specific, incremental feedback.

---

## What outputs to expect

Depending on the starter kit or prompt card, you may get:

- **HTML dashboards** — self-contained interactive dashboards you can open in a browser or share
- **Markdown / HTML reports** — formatted summary documents suitable for leadership review
- **Jupyter notebooks / R Markdown** — reproducible analysis documents with code and commentary
- **Data tables and charts** — intermediate outputs for further analysis

---

## Tips for working with coding agents on analytics tasks

1. **Be specific about your data.** Tell the agent the file name, column names, and date range. The more context you give, the better the output.
2. **Start with the starter kit prompts.** They are tested and structured. Freeform prompts work too, but structured prompts produce more consistent results.
3. **Iterate in small steps.** If the output is not right, ask the agent to fix one thing at a time rather than re-generating everything.
4. **Validate the output.** Spot-check row counts, date ranges, and aggregation logic. Coding agents can make plausible-looking mistakes.
5. **Use the vivainsights packages.** The R and Python packages handle common data validation and visualization tasks. Prompts that reference these functions tend to produce cleaner code.
6. **Keep your data private.** Do not paste raw data into cloud-based agents unless your organization's data policies allow it. Use local or enterprise-hosted agents when working with sensitive HR data.

---

## Next steps

- [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/) — full details on available kits
- [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) — browse all available prompts
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) — understand your data
- [Back to Frontier Analytics]({{ site.baseurl }}/frontier-analytics/)
