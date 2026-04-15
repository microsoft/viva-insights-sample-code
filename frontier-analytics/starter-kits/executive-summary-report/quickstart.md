# Quickstart — Executive Summary Report

Generate a polished executive summary memo from your Viva Insights data in about 10 minutes.

## What you need

| Item | Minimum requirement |
|---|---|
| **Person query CSV** | Exported from Viva Insights with `PersonId`, `MetricDate`, at least one `Copilot_*` metric, and at least one HR attribute. |
| **Data span** | 8+ weeks recommended (4 weeks minimum). |
| **Coding agent** | GitHub Copilot, Claude Code, or similar — with access to a Python or R environment. |

## Steps

### 1. Open your coding agent

Launch the agent in a workspace where it can execute Python (or R) and access your CSV file.

### 2. Copy the Executive Summary prompt

Open the [Executive Summary prompt card](../../prompts/copilot-adoption/executive-summary.md). Copy the entire text from the **Prompt** section.

### 3. Paste with context

In your coding agent, provide context about your data and audience, then paste the prompt:

```
My person query CSV is at ./data/person-query.csv.
The reporting period is September 2024.
The audience is the VP of IT and the CTO.
Focus on Copilot adoption trends and organizational differences.

[paste the full Executive Summary prompt here]
```

> **Tip:** Be specific about the audience and time period. This directly shapes the memo's framing, level of detail, and recommendations.

### 4. Review the draft

The agent will produce a Markdown or HTML memo containing:

- Headline metrics (adoption rate, active users, average actions)
- Key findings (3–5 bullets)
- Organizational highlights
- Recommendations

Read it critically. The numbers should align with what you know about your organization. Flag anything that seems off.

### 5. Refine

Common follow-up instructions:

- _"Make the tone more formal."_
- _"Shorten to one page."_
- _"Add a comparison to last month."_
- _"Replace Organization names with abbreviations: Engineering → ENG, Sales → SLS."_
- _"Remove the Areas of Concern section — we'll address that separately."_

### 6. Export

- **HTML or Markdown**: Use as-is, or copy into an email.
- **PDF**: Open the HTML in your browser → Print → Save as PDF.
- **PowerPoint**: Ask the agent: _"Convert this into a 3-slide PowerPoint."_

## Expected output in 10 minutes

After one pass through Steps 1–4, you should have a 1–2 page memo with:

- A summary metrics header with 3–5 key numbers
- A findings section with data-backed observations
- Actionable recommendations tied to specific organizational patterns

This draft is typically 80% ready. Step 5 polishes it for your specific audience.

## What to do next

- **Add visual context**: Run the [Dashboard Overview](../../prompts/copilot-adoption/dashboard-overview.md) prompt to generate charts that complement the memo.
- **Deepen the analysis**: Run [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) to add detail on user behavior patterns.
- **Build the business case**: Run [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) if leadership wants to see financial returns.
- **Set up a recurring cadence**: Save your prompt with the context line. Each month or quarter, drop in a fresh CSV and re-run.

For the full guide, see the [README](README.md).
