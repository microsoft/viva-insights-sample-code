# Executive Summary Report — Starter Kit

## Overview

This starter kit helps you generate a **polished executive summary document** from Viva Insights person query data. The output is a professional memo — formatted in Markdown or HTML — that distills Copilot adoption metrics (or broader workforce collaboration patterns) into key findings, trends, and recommendations suitable for senior leadership.

You don't write the analysis yourself. The kit provides a structured **prompt card** that you paste into a coding agent. The agent reads your data, computes the metrics, identifies noteworthy patterns, and drafts the memo.

## Use case

People analytics teams regularly need to report to senior leadership on:

- **Copilot adoption progress**: How many users are active? Is adoption growing? Which organizations lead?
- **Workforce collaboration patterns**: How is collaboration time distributed? Are after-hours trends improving?
- **Program effectiveness**: Is a deployment wave achieving its targets? What needs attention?

These reports typically go to VP/C-suite executives, IT steering committees, or HR leadership on a monthly or quarterly cadence. This kit automates the data analysis and first-draft writing so you can focus on interpretation and storytelling.

## What you'll get

A **1–3 page executive summary** containing:

1. **Headline metrics**: 3–5 key numbers (e.g., adoption rate, active users, average actions per user) with week-over-week or period-over-period comparisons
2. **Key findings**: 3–5 bullet points highlighting the most important patterns in the data — what's growing, what's lagging, what's changed
3. **Trend analysis**: A brief narrative on how metrics have evolved over the reporting period, with references to specific inflection points
4. **Organizational highlights**: Which departments or functions are leading, which are behind, and any notable outliers
5. **Recommendations**: 2–3 actionable suggestions based on the data (e.g., "Expand targeted enablement in the Finance organization where adoption lags 15 points behind the company average")
6. **Areas of concern**: Any red flags — declining engagement, high churn, concentrated usage among a small group

The tone is concise, data-driven, and action-oriented — written for a busy executive who needs the bottom line in under 2 minutes.

## Prerequisites

| Requirement | Details |
|---|---|
| **Person query CSV** | Exported from the Viva Insights Analyst portal with `PersonId`, `MetricDate`, Copilot metrics, and HR attributes. See the [Copilot Adoption Dashboard required inputs](../copilot-adoption-dashboard/required-inputs.md) for the full column specification. |
| **R or Python environment** | A local or cloud environment where the coding agent can execute code (Python with `pandas` is most common). |
| **Coding agent** | [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar AI coding assistant. |
| **8+ weeks of data** | 12+ weeks preferred for robust trend analysis. 4 weeks is a workable minimum. |

## Workflow

### Step 1 — Export your data

Export a person query from the Viva Insights Analyst portal that includes Copilot activity metrics and HR attributes. This is the same export used for the [Copilot Adoption Dashboard](../copilot-adoption-dashboard/README.md) — if you've already generated a dashboard, you can reuse the same CSV.

### Step 2 — Review the schema docs

Confirm that your column names match expected conventions. See the [schemas directory](../../schemas/) for reference. Note any differences — you'll communicate them to the agent.

### Step 3 — Use the executive summary prompt

Open the [Executive Summary prompt card](../../prompts/copilot-adoption/executive-summary.md) and copy the full text from the **Prompt** section.

In your coding agent, prepend a context line:

```
My person query CSV is at ./data/person-query.csv.
The reporting period is Q3 2024 (July–September).
The audience is the CTO and IT leadership team.
Focus on Copilot adoption metrics.

[paste the full Executive Summary prompt here]
```

> **Tip:** Tell the agent who the audience is and what time period to cover. This shapes the memo's framing and level of detail.

### Step 4 — Review and refine the output

The agent will produce a draft memo. Review it for:

- **Accuracy**: Do the numbers look right given what you know about your data?
- **Tone**: Is it appropriately formal for the audience?
- **Completeness**: Are the most important findings captured?
- **Caveats**: Are any claims overly strong? (The prompt instructs caution, but always double-check.)

Common refinement instructions:

- _"Make the recommendations more specific — reference actual organization names."_
- _"Shorten the trend analysis section to 2 sentences."_
- _"Add a comparison to the previous quarter."_
- _"Remove any references to individual-level data."_

### Step 5 — Export to desired format

The default output is **Markdown** or **HTML**. To convert:

- **PDF**: Open the HTML in a browser and print to PDF.
- **Word**: Copy the Markdown into a Word document, or ask the agent to generate a `.docx` file directly (requires `python-docx` or `officer` in R).
- **PowerPoint**: Ask the agent: _"Convert the key findings and metrics into a 3-slide PowerPoint summary."_ (requires `python-pptx` or `officer` in R).
- **Email body**: Copy the Markdown or HTML directly into an Outlook email.

## Included prompts

| Prompt Card | Description |
|---|---|
| [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) | Generate a concise executive memo with headline metrics, key findings, trend analysis, and recommendations. |

> **Tip:** You can also use the [Dashboard Overview](../../prompts/copilot-adoption/dashboard-overview.md) prompt first to build visual charts, then ask the agent to write an executive summary narrative referencing those charts.

## Expected output

A professional memo formatted roughly as:

```
COPILOT ADOPTION — EXECUTIVE SUMMARY
Reporting period: [date range]
Prepared for: [audience]

HEADLINE METRICS
• Licensed users: X,XXX
• Active users (most recent week): X,XXX (XX% adoption rate)
• Average Copilot actions per active user: XX.X
• Week-over-week change: +X.X%

KEY FINDINGS
1. [Finding with supporting data]
2. [Finding with supporting data]
3. [Finding with supporting data]

ORGANIZATIONAL HIGHLIGHTS
• [Top-performing org] leads at XX% adoption...
• [Lagging org] trails at XX% adoption...

RECOMMENDATIONS
1. [Actionable recommendation]
2. [Actionable recommendation]

AREAS OF CONCERN
• [Concern with supporting data]
```

The actual formatting will be cleaner (with proper headings, bold text, and spacing). The agent may also include a small inline chart if the environment supports it.

## Tips for different audiences

| Audience | Customization |
|---|---|
| **CTO / IT leadership** | Emphasize technical adoption metrics, deployment coverage, and infrastructure readiness. |
| **CHRO / HR leadership** | Frame around employee experience, collaboration patterns, and change management. |
| **CFO / Finance** | Lead with ROI metrics, license utilization, and cost-per-active-user. Pair with the [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) prompt. |
| **CEO / Board** | Keep it to one page. Focus on 3 headline numbers and 2 recommendations. Minimize jargon. |
| **Copilot program team** | Include more operational detail — churn rates, segment shifts, specific org-level data. |

Customize by adding a line to the agent: _"The audience is the CFO. Lead with financial metrics and license ROI."_

## Related resources

- [Prompt Card Library](../../prompts/)
- [Copilot Adoption Dashboard starter kit](../copilot-adoption-dashboard/README.md)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Viva Insights Sample Code Library](../../)
