# Copilot Adoption Dashboard — Starter Kit

## Overview

This starter kit helps you build a **self-contained static HTML dashboard** that tracks Microsoft Copilot adoption across your organization using Viva Insights person query data. The dashboard requires no server infrastructure — it produces a single HTML file you can open in any browser, email to stakeholders, or host on SharePoint.

You don't need to write code yourself. The kit provides structured **prompt cards** that you paste into a coding agent (GitHub Copilot, Claude Code, or similar). The agent generates the code; you provide the data and review the output.

## Use case

Organizations deploying Microsoft 365 Copilot need to answer questions like:

- How many licensed users are actively using Copilot each week?
- Is adoption growing, plateauing, or declining?
- Which departments or functions are leading or lagging?
- Who are the power users, and who has disengaged?
- What is the estimated return on Copilot license investment?

This kit produces a dashboard that answers all of these in a single, shareable artifact.

## What you'll get

A multi-panel HTML dashboard containing:

1. **Summary metrics bar** — headline numbers (total licensed users, active users, adoption rate, average actions per user)
2. **Adoption trend chart** — weekly time series of licensed users, active users, and adoption rate
3. **Usage intensity chart** — distribution of Copilot actions or assisted hours across users, with percentile markers
4. **Organizational breakdown** — adoption rates and average usage by `Organization`, `FunctionType`, or other HR attributes
5. **User segment distribution** — proportions of power users, regular users, light users, and inactive licensed users over time
6. **Top users table** — anonymized leaderboard of highest-usage individuals with their key metrics

The dashboard is styled for professional presentation and includes hover tooltips, responsive layout, and a clean color scheme.

## Prerequisites

| Requirement | Details |
|---|---|
| **Person query CSV** | Exported from the Viva Insights Analyst portal. Must include `PersonId`, `MetricDate`, Copilot metric columns, and HR attributes. See [required-inputs.md](required-inputs.md) for the full specification. |
| **R or Python environment** | A local or cloud environment where the coding agent can execute code. Python with `pandas` and `plotly` (or `matplotlib`) is most common; R with `tidyverse` and `htmlwidgets` also works. |
| **Coding agent** | [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar AI coding assistant that can read files and execute scripts. |
| **8+ weeks of data** | Fewer weeks will still produce output, but trend charts and segmentation become less meaningful. |

## Workflow

### Step 1 — Export your person query data

In the Viva Insights Analyst portal, create or open a person query that includes Copilot activity metrics (`Copilot_Actions`, `Copilot_Assisted_Hours`, `Copilot_Chat_Queries`, etc.) and HR attributes (`Organization`, `FunctionType`, `LevelDesignation`). Export the results as CSV.

> **Tip:** If you're unsure which metrics to include, start with all available Copilot metrics. The prompt will auto-detect columns starting with `Copilot_`.

### Step 2 — Review the data dictionary

Familiarize yourself with the expected schema so you can verify your export matches:

- [Data dictionary / schemas](../../schemas/)

If your column names differ from the defaults (e.g., `Org` instead of `Organization`), note the differences — you'll tell the agent in Step 4.

### Step 3 — Open the prompt cards

This kit uses four prompt cards, designed to be run in sequence:

| Order | Prompt Card | What it produces |
|---|---|---|
| 1 | [Dashboard Overview](../../prompts/copilot-adoption/dashboard-overview.md) | The core dashboard with adoption trends, usage patterns, and org breakdowns |
| 2 | [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) | A text-based executive memo panel summarizing key findings |
| 3 | [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) | User segment classification, transition tracking, and churn analysis |
| 4 | [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) | Return-on-investment estimates and sensitivity analysis |

Start with **Dashboard Overview** — it produces the foundational dashboard. The other three add depth and can be appended to the same HTML file or generated as separate outputs.

### Step 4 — Paste prompts into your coding agent

Open your coding agent in a workspace that has access to your CSV file and can execute R or Python. Then:

1. Copy the full text from the **Prompt** section of the first prompt card.
2. Prepend a context line pointing to your data, for example:

   > _"My CSV is at `./data/person-query.csv`. The date column is `MetricDate` and the org column is `Organization`."_

3. Paste the prompt and let the agent run.
4. Review the output HTML file in your browser.

### Step 5 — Iterate on the output

Check the output against the **Common failure modes** section of each prompt card. Typical adjustments:

- **Column name mismatches**: Tell the agent which columns in your data correspond to expected fields.
- **Date parsing issues**: Specify the date format explicitly (e.g., `"Dates are in YYYY-MM-DD format"`).
- **Privacy thresholds**: Add `"Suppress any group with fewer than 5 people"` if needed.
- **Language/package preference**: Add `"Use R instead of Python"` or `"Use plotly for charts"` as needed.

Repeat Steps 3–5 for each additional prompt card.

## Included prompts

| Prompt Card | Description |
|---|---|
| [Dashboard Overview](../../prompts/copilot-adoption/dashboard-overview.md) | Core dashboard with adoption trends, usage intensity, and organizational breakdowns. |
| [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) | Concise memo distilling Copilot adoption into findings and recommendations for leadership. |
| [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) | Classify users into usage tiers, track segment transitions, and quantify churn rates. |
| [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) | Estimate time savings, monetary value, and license ROI with sensitivity analysis. |

## Expected output

The primary output is a **single self-contained HTML file** (typically 1–5 MB depending on data size and chart complexity). It:

- Opens in any modern browser (Chrome, Edge, Firefox, Safari)
- Requires no internet connection or server
- Embeds all JavaScript, CSS, and chart data inline
- Can be shared as an email attachment, uploaded to SharePoint, or posted in a Teams channel

See [expected-output.md](expected-output.md) for a detailed description of each dashboard panel.

## Tips

- **Start with Dashboard Overview only.** Get that working before adding the other prompt cards. It's easier to debug one output at a time.
- **Keep your data file in the agent's workspace.** If the agent can't find the file, it can't generate charts. Use a relative path or provide the absolute path.
- **Preview frequently.** After each prompt, open the HTML in your browser. Catching issues early saves time.
- **Provide context, not code.** You don't need to write code — just describe your data and what you want. The prompts handle the implementation details.
- **Use the adaptation notes.** Each prompt card includes notes on customizing for your environment. Read them before running.
- **Combine outputs incrementally.** You can ask the agent to merge panels from different prompts into a single HTML file: _"Add the segmentation charts to the existing dashboard HTML."_
- **Check privacy.** Before sharing, verify that no individual-level data is exposed in the output. The prompts include privacy safeguards, but always review.

## Related resources

- [Prompt Card Library](../../prompts/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Copilot Analytics All-in-one Dashboard](https://github.com/microsoft/AI-in-One-Dashboard)
- [Viva Insights Sample Code Library](../../)
