# Prompt Card Library

This directory contains **prompt cards** — ready-to-use prompts that you can paste directly into a coding agent (such as [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or similar tools) to generate analytics outputs from Viva Insights data.

## What is a prompt card?

A prompt card is a structured document that contains:

- **Purpose**: What the output is designed to answer
- **Audience**: Who the output is for
- **When to use**: The scenario that triggers this analysis
- **Required inputs**: The data you need before running
- **Assumptions**: What the prompt expects about your data
- **Recommended output**: The format of the deliverable
- **Prompt**: The full text you paste into your coding agent
- **Adaptation notes**: How to customize for your environment
- **Common failure modes**: Known pitfalls and how to avoid them

## How to use a prompt card

1. **Prepare your data.** Export the required query from the Viva Insights Analyst portal (or Purview audit logs, depending on the card).
2. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or your preferred AI assistant in a workspace that has access to R or Python.
3. **Copy the prompt.** Navigate to the relevant prompt card below and copy the full text from the **Prompt** section.
4. **Paste and run.** Paste the prompt into the agent. Point it at your data file when asked (or prepend the file path to the prompt).
5. **Review and adapt.** Check the output against the **Common failure modes** section. Use the **Adaptation notes** to customize.

> **Tip:** You can prepend context to any prompt. For example: _"My CSV is at `./data/person-query.csv`. The Organization column is called `Org`."_ followed by the full prompt text.

## Available prompts

### Copilot Adoption

| Prompt Card | Description |
|---|---|
| [Dashboard Overview](copilot-adoption/dashboard-overview.md) | Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns. |
| [Executive Summary](copilot-adoption/executive-summary.md) | Produce a concise executive memo summarizing Copilot adoption metrics, key findings, and recommendations for VP/C-suite audiences. |
| [Segmentation & Churn](copilot-adoption/segmentation-and-churn.md) | Classify users into usage segments, track segment transitions over time, and calculate churn rates. |
| [ROI Estimation](copilot-adoption/roi-estimation.md) | Estimate return on investment for Copilot by quantifying time savings, productivity value, and license cost breakdowns. |

### Purview Augmentation

| Prompt Card | Description |
|---|---|
| [Agent Usage Analysis](purview-augmentation/agent-usage-analysis.md) | Analyze Copilot agent and extension usage patterns from Purview audit logs. |
| [Audit Log Parsing](purview-augmentation/audit-log-parsing.md) | Parse and clean raw Purview audit log exports into analysis-ready flat tables. |

## Tips for adapting prompts

- **Column names vary between tenants.** Always verify your actual column names against what the prompt expects. Prepend a note like _"In my data, the Organization column is called `Org`."_
- **Granularity matters.** Most prompts assume person-week data. If your export is person-day, instruct the agent to aggregate first.
- **Privacy thresholds.** For smaller organizations, add a note requesting minimum group sizes (e.g., suppress segments with fewer than 5 people).
- **Language preference.** Prompts default to Python. If you prefer R, add _"Use R instead of Python"_ before the prompt.
- **Package availability.** Prompts reference the [vivainsights R package](https://microsoft.github.io/vivainsights/) and [vivainsights Python package](https://microsoft.github.io/vivainsights-py/). Install them beforehand if you want the agent to use them.

## Creating new prompt cards

Use the [prompt card template](../templates/) as a starting point when authoring new prompts. Follow the structure defined above to ensure consistency across the library.

## Related resources

- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Viva Insights Sample Code Library](https://microsoft.github.io/viva-insights-sample-code/)
- [Copilot Analytics All-in-one Dashboard](https://github.com/microsoft/AI-in-One-Dashboard)
