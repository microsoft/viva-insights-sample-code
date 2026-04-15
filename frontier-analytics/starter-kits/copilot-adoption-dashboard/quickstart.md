# Quickstart — Copilot Adoption Dashboard

Get a working Copilot adoption dashboard in under 5 minutes.

## What you need

| Item | Minimum requirement |
|---|---|
| **Person query CSV** | Exported from Viva Insights with `PersonId`, `MetricDate`, at least one `Copilot_*` metric, and at least one HR attribute (e.g., `Organization`). |
| **Data span** | 8+ weeks recommended. 4 weeks is the minimum for a usable trend chart. |
| **Licensed users** | 50+ Copilot-licensed users recommended. Fewer will work, but segmentation panels become sparse. |
| **Coding agent** | GitHub Copilot, Claude Code, or similar — with access to an R or Python environment. |

## Steps

### 1. Open your coding agent

Launch your agent in a workspace where it can execute Python (or R) and read your CSV file.

### 2. Start with the Dashboard Overview prompt

Open the [Dashboard Overview prompt card](../../prompts/copilot-adoption/dashboard-overview.md). Copy the entire text from the **Prompt** section.

### 3. Paste with context

In your coding agent, type a brief context line and then paste the prompt:

```
My person query CSV is at ./data/person-query.csv.
The date column is MetricDate and the primary org column is Organization.
Use Python with plotly for charts.

[paste the full Dashboard Overview prompt here]
```

### 4. Let it run

The agent will:
1. Load and validate your CSV
2. Detect Copilot metric columns
3. Classify licensed vs. active users
4. Compute weekly metrics
5. Generate charts and a summary panel
6. Output a self-contained HTML file

### 5. Open the HTML

Open the generated HTML file in your browser. You should see:

- A **summary metrics bar** at the top (total licensed, active, adoption rate)
- An **adoption trend** chart showing weekly active users over time
- A **usage intensity** chart showing the distribution of Copilot actions
- An **organizational breakdown** table or chart

If something looks wrong, check the **Common failure modes** section of the prompt card.

## What to do next

Once the core dashboard is working:

1. **Add segmentation** — Run the [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) prompt to add user segment analysis and churn tracking.
2. **Add an executive summary** — Run the [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) prompt to generate a text panel with key findings.
3. **Add ROI estimation** — Run the [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) prompt to add cost-benefit analysis.
4. **Combine panels** — Ask the agent: _"Merge the segmentation and ROI outputs into the existing dashboard HTML file."_

## Troubleshooting

| Issue | Fix |
|---|---|
| Agent can't find the CSV | Provide the full absolute path to the file. |
| Column names don't match | Prepend: _"In my data, the org column is called `Org` and the date column is `Date`."_ |
| Charts are blank | Ensure `plotly` (Python) or `htmlwidgets` (R) is installed in the environment. |
| Too few data points | Confirm your export covers at least 4 weeks and includes Copilot metric columns. |
| HTML file is very large | Ask the agent to downsample or aggregate data before charting. |

For the full guide, see the [README](README.md).
