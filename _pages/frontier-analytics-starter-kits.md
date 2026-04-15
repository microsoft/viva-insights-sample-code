---
layout: page
title: "Frontier Analytics — Starter Kits"
permalink: /frontier-analytics-starter-kits/
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

A **starter kit** is a bundled workflow that combines a use case, required inputs, prompt cards, and expected outputs into a single package. Each kit gives you everything you need to go from an exported Viva Insights CSV to a finished deliverable using a coding agent.

## Available Starter Kits

| Starter Kit | Description | Complexity | Primary Output |
|-------------|-------------|------------|----------------|
| [Copilot Adoption Dashboard](#copilot-adoption-dashboard) | Interactive dashboard tracking Copilot adoption metrics over time, including usage trends, feature-level breakdowns, and user segmentation by HR attributes. | Intermediate | HTML dashboard |
| [Executive Summary Report](#executive-summary-report) | One-page memo summarizing key collaboration and Copilot metrics for executive audiences. Designed for fast turnaround with minimal customization. | Beginner | Markdown / HTML memo |

## What's in a Starter Kit?

Each starter kit folder contains some or all of the following:

- **README** — Overview of the use case, who it's for, and what it produces
- **Quickstart** — Step-by-step instructions specific to that kit
- **Required inputs** — The data files and parameters you need before starting
- **Recommended files** — Suggested file organization and naming
- **Expected output** — Description or screenshot of what the finished output looks like

Starter kits reference prompt cards from the [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/). The prompts contain the actual text you paste into your coding agent.

## How to Use a Starter Kit

1. **Read the overview.** Understand the use case and check that it matches your scenario.
2. **Check the required inputs.** Make sure you have the necessary data exports (person query CSV, Purview audit logs, etc.) and that your R or Python environment is set up.
3. **Follow the quickstart.** The kit's quickstart walks you through the workflow step by step.
4. **Use the prompts with your coding agent.** Open the referenced prompt cards, copy the prompt text, and paste it into your agent with your data context.
5. **Review the expected output.** Compare your results against the documented output to verify correctness.

For general setup instructions, see the [Quickstart guide]({{ site.baseurl }}/frontier-analytics-quickstart/).

---

## Copilot Adoption Dashboard

> 📂 **Source files**: [View on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/starter-kits/copilot-adoption-dashboard/)

### Overview

This starter kit helps you build a **self-contained static HTML dashboard** that tracks Microsoft Copilot adoption across your organization using Viva Insights person query data. The dashboard requires no server infrastructure — it produces a single HTML file you can open in any browser, email to stakeholders, or host on SharePoint.

You don't need to write code yourself. The kit provides structured **prompt cards** that you paste into a coding agent (GitHub Copilot, Claude Code, or similar). The agent generates the code; you provide the data and review the output.

### Use case

Organizations deploying Microsoft 365 Copilot need to answer questions like:

- How many licensed users are actively using Copilot each week?
- Is adoption growing, plateauing, or declining?
- Which departments or functions are leading or lagging?
- Who are the power users, and who has disengaged?
- What is the estimated return on Copilot license investment?

This kit produces a dashboard that answers all of these in a single, shareable artifact.

### What you'll get

A multi-panel HTML dashboard containing:

1. **Summary metrics bar** — headline numbers (total licensed users, active users, adoption rate, average actions per user)
2. **Adoption trend chart** — weekly time series of licensed users, active users, and adoption rate
3. **Usage intensity chart** — distribution of Copilot actions or assisted hours across users, with percentile markers
4. **Organizational breakdown** — adoption rates and average usage by `Organization`, `FunctionType`, or other HR attributes
5. **User segment distribution** — proportions of power users, regular users, light users, and inactive licensed users over time
6. **Top users table** — anonymized leaderboard of highest-usage individuals with their key metrics

The dashboard is styled for professional presentation and includes hover tooltips, responsive layout, and a clean color scheme.

### Prerequisites

| Requirement | Details |
|---|---|
| **Person query CSV** | Exported from the Viva Insights Analyst portal. Must include `PersonId`, `MetricDate`, Copilot metric columns, and HR attributes. See [Required Inputs](#required-inputs) for the full specification. |
| **R or Python environment** | A local or cloud environment where the coding agent can execute code. Python with `pandas` and `plotly` (or `matplotlib`) is most common; R with `tidyverse` and `htmlwidgets` also works. |
| **Coding agent** | [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar AI coding assistant that can read files and execute scripts. |
| **8+ weeks of data** | Fewer weeks will still produce output, but trend charts and segmentation become less meaningful. |

### Workflow

#### Step 1 — Export your person query data

In the Viva Insights Analyst portal, create or open a person query that includes Copilot activity metrics (`Copilot_Actions`, `Copilot_Assisted_Hours`, `Copilot_Chat_Queries`, etc.) and HR attributes (`Organization`, `FunctionType`, `LevelDesignation`). Export the results as CSV.

> **Tip:** If you're unsure which metrics to include, start with all available Copilot metrics. The prompt will auto-detect columns starting with `Copilot_`.

#### Step 2 — Review the data dictionary

Familiarize yourself with the expected schema so you can verify your export matches:

- [Data dictionary / schemas]({{ site.baseurl }}/frontier-analytics-schemas/)

If your column names differ from the defaults (e.g., `Org` instead of `Organization`), note the differences — you'll tell the agent in Step 4.

#### Step 3 — Open the prompt cards

This kit uses four prompt cards, designed to be run in sequence:

| Order | Prompt Card | What it produces |
|---|---|---|
| 1 | [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) | The core dashboard with adoption trends, usage patterns, and org breakdowns |
| 2 | [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) | A text-based executive memo panel summarizing key findings |
| 3 | [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) | User segment classification, transition tracking, and churn analysis |
| 4 | [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) | Return-on-investment estimates and sensitivity analysis |

Start with **Dashboard Overview** — it produces the foundational dashboard. The other three add depth and can be appended to the same HTML file or generated as separate outputs.

#### Step 4 — Paste prompts into your coding agent

Open your coding agent in a workspace that has access to your CSV file and can execute R or Python. Then:

1. Copy the full text from the **Prompt** section of the first prompt card.
2. Prepend a context line pointing to your data, for example:

   > _"My CSV is at `./data/person-query.csv`. The date column is `MetricDate` and the org column is `Organization`."_

3. Paste the prompt and let the agent run.
4. Review the output HTML file in your browser.

#### Step 5 — Iterate on the output

Check the output against the **Common failure modes** section of each prompt card. Typical adjustments:

- **Column name mismatches**: Tell the agent which columns in your data correspond to expected fields.
- **Date parsing issues**: Specify the date format explicitly (e.g., `"Dates are in YYYY-MM-DD format"`).
- **Privacy thresholds**: Add `"Suppress any group with fewer than 5 people"` if needed.
- **Language/package preference**: Add `"Use R instead of Python"` or `"Use plotly for charts"` as needed.

Repeat Steps 3–5 for each additional prompt card.

### Included prompts

| Prompt Card | Description |
|---|---|
| [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) | Core dashboard with adoption trends, usage intensity, and organizational breakdowns. |
| [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) | Concise memo distilling Copilot adoption into findings and recommendations for leadership. |
| [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) | Classify users into usage tiers, track segment transitions, and quantify churn rates. |
| [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) | Estimate time savings, monetary value, and license ROI with sensitivity analysis. |

### Tips

- **Start with Dashboard Overview only.** Get that working before adding the other prompt cards. It's easier to debug one output at a time.
- **Keep your data file in the agent's workspace.** If the agent can't find the file, it can't generate charts. Use a relative path or provide the absolute path.
- **Preview frequently.** After each prompt, open the HTML in your browser. Catching issues early saves time.
- **Provide context, not code.** You don't need to write code — just describe your data and what you want. The prompts handle the implementation details.
- **Use the adaptation notes.** Each prompt card includes notes on customizing for your environment. Read them before running.
- **Combine outputs incrementally.** You can ask the agent to merge panels from different prompts into a single HTML file: _"Add the segmentation charts to the existing dashboard HTML."_
- **Check privacy.** Before sharing, verify that no individual-level data is exposed in the output. The prompts include privacy safeguards, but always review.

### Related resources

- [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Copilot Analytics All-in-one Dashboard](https://github.com/microsoft/AI-in-One-Dashboard)
- [Viva Insights Sample Code Library]({{ site.baseurl }}/)

### Quickstart

Get a working Copilot adoption dashboard in under 5 minutes.

#### What you need

| Item | Minimum requirement |
|---|---|
| **Person query CSV** | Exported from Viva Insights with `PersonId`, `MetricDate`, at least one `Copilot_*` metric, and at least one HR attribute (e.g., `Organization`). |
| **Data span** | 8+ weeks recommended. 4 weeks is the minimum for a usable trend chart. |
| **Licensed users** | 50+ Copilot-licensed users recommended. Fewer will work, but segmentation panels become sparse. |
| **Coding agent** | GitHub Copilot, Claude Code, or similar — with access to an R or Python environment. |

#### 1. Open your coding agent

Launch your agent in a workspace where it can execute Python (or R) and read your CSV file.

#### 2. Start with the Dashboard Overview prompt

Open the [Dashboard Overview prompt card]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption). Copy the entire text from the **Prompt** section.

#### 3. Paste with context

In your coding agent, type a brief context line and then paste the prompt:

```
My person query CSV is at ./data/person-query.csv.
The date column is MetricDate and the primary org column is Organization.
Use Python with plotly for charts.

[paste the full Dashboard Overview prompt here]
```

#### 4. Let it run

The agent will:
1. Load and validate your CSV
2. Detect Copilot metric columns
3. Classify licensed vs. active users
4. Compute weekly metrics
5. Generate charts and a summary panel
6. Output a self-contained HTML file

#### 5. Open the HTML

Open the generated HTML file in your browser. You should see:

- A **summary metrics bar** at the top (total licensed, active, adoption rate)
- An **adoption trend** chart showing weekly active users over time
- A **usage intensity** chart showing the distribution of Copilot actions
- An **organizational breakdown** table or chart

If something looks wrong, check the **Common failure modes** section of the prompt card.

#### What to do next

Once the core dashboard is working:

1. **Add segmentation** — Run the [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) prompt to add user segment analysis and churn tracking.
2. **Add an executive summary** — Run the [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) prompt to generate a text panel with key findings.
3. **Add ROI estimation** — Run the [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) prompt to add cost-benefit analysis.
4. **Combine panels** — Ask the agent: _"Merge the segmentation and ROI outputs into the existing dashboard HTML file."_

#### Troubleshooting

| Issue | Fix |
|---|---|
| Agent can't find the CSV | Provide the full absolute path to the file. |
| Column names don't match | Prepend: _"In my data, the org column is called `Org` and the date column is `Date`."_ |
| Charts are blank | Ensure `plotly` (Python) or `htmlwidgets` (R) is installed in the environment. |
| Too few data points | Confirm your export covers at least 4 weeks and includes Copilot metric columns. |
| HTML file is very large | Ask the agent to downsample or aggregate data before charting. |

For the full guide, see the [overview](#copilot-adoption-dashboard).

### Required Inputs

This section describes the data you need before running the Copilot Adoption Dashboard prompts.

#### Person query CSV

The primary input is a **person query export** from the Viva Insights Analyst portal. Each row represents one person in one time period (typically one week).

##### Expected columns

| Column | Type | Description | Required? |
|---|---|---|---|
| `PersonId` | String | Anonymized unique identifier for each person. Consistent across weeks. | **Yes** |
| `MetricDate` | Date | Start date of the measurement period (usually the Monday of each week). Format: `YYYY-MM-DD`. | **Yes** |
| `Copilot_Actions` | Numeric | Total number of Copilot actions taken by the user in that week. `NA`/null for unlicensed users. | **Yes** |
| `Copilot_Assisted_Hours` | Numeric | Hours of work where Copilot provided assistance. `NA`/null for unlicensed users. | Recommended |
| `Copilot_Chat_Queries` | Numeric | Number of queries sent to Copilot Chat (Business Chat). `NA`/null for unlicensed users. | Recommended |
| `Copilot_Summarized_Hours` | Numeric | Hours spent on content that Copilot summarized. `NA`/null for unlicensed users. | Optional |
| `Organization` | String | HR attribute indicating the person's organizational unit (e.g., department or division). | **Yes** |
| `FunctionType` | String | HR attribute indicating the person's job function (e.g., Engineering, Sales, Finance). | Recommended |
| `LevelDesignation` | String | HR attribute indicating the person's seniority level (e.g., IC, Manager, Director). | Recommended |
| `SupervisorIndicator` | String | Indicates whether the person is a manager (`Manager`) or individual contributor (`Individual Contributor`). | Optional |
| `City` | String | Person's city (HR attribute). | Optional |
| `Country` | String | Person's country or region (HR attribute). | Optional |
| `Region` | String | Person's geographic region (HR attribute). | Optional |

> **Note:** Column names vary between tenants and query configurations. The prompts auto-detect columns starting with `Copilot_` as Copilot metrics. If your HR attribute columns have different names (e.g., `Org` instead of `Organization`), tell the coding agent when you paste the prompt.

##### Additional standard person query columns

Your export may also include collaboration metrics that are not required for this dashboard but can add context:

- `Collaboration_Hours`, `Meeting_Hours`, `Email_Hours`, `Chat_Hours`
- `After_Hours_Collaboration_Hours`
- `Internal_Network_Size`, `External_Network_Size`
- `Manager_Coaching_Hours_1on1`

The prompts will ignore columns they don't need, so including extra columns is harmless.

#### Minimum data requirements

| Dimension | Minimum | Recommended |
|---|---|---|
| **Time span** | 4 weeks | 8–12+ weeks |
| **Licensed users** | 10 | 50+ |
| **HR attributes** | 1 (e.g., `Organization`) | 3+ (Organization, FunctionType, LevelDesignation) |
| **Copilot metrics** | 1 (e.g., `Copilot_Actions`) | 3+ |

- **Time span**: Fewer than 4 weeks makes trend analysis unreliable. 12+ weeks is ideal for detecting meaningful adoption trajectories.
- **Licensed users**: With fewer than 50 users, segmentation and organizational breakdowns will have very small group sizes. Consider applying privacy thresholds.
- **HR attributes**: More attributes enable richer slicing. At minimum, you need one organizational grouping.

#### How to export from Viva Insights

1. Open the **Viva Insights Analyst** portal at [insights.cloud.microsoft](https://insights.cloud.microsoft/).
2. Navigate to **Analysis** > **Custom queries** > **Person query**.
3. Configure the query:
   - **Time period**: Select a range covering at least 8 weeks.
   - **Granularity**: Weekly (recommended) or Daily.
   - **Metrics**: Add all available `Copilot_*` metrics.
   - **Organizational data**: Include `Organization`, `FunctionType`, `LevelDesignation`, and any other HR attributes you want to analyze.
4. Run the query and wait for it to complete.
5. Download the results as a **CSV** file.

For detailed guidance, see the [Microsoft documentation on person queries](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query-overview).

#### Data format notes

- **Encoding**: CSV files should be UTF-8 encoded. If you encounter character issues, re-save as UTF-8 in Excel or a text editor.
- **Date format**: `MetricDate` should be in `YYYY-MM-DD` format (e.g., `2024-09-02`). If your export uses a different format (e.g., `MM/DD/YYYY`), tell the coding agent so it can parse correctly.
- **Decimal separator**: Use a period (`.`) as the decimal separator. Comma-separated decimals (common in some European locales) will cause parsing errors — convert before loading or tell the agent.
- **Missing values**: Unlicensed users will have `NA`, `null`, or empty cells for Copilot metric columns. This is expected and the prompts handle it automatically.
- **File size**: Person query exports can range from a few MB to several hundred MB depending on the population size and time span. Files over 100 MB may require chunked processing — mention the file size to the agent.

#### Schema reference

For the full data dictionary, see the [schemas directory]({{ site.baseurl }}/frontier-analytics-schemas/).

### Recommended Files

Gather these files and resources before starting. Items marked **required** are necessary to produce the dashboard; everything else improves the output but isn't blocking.

#### Required

| File / Resource | Description | Where to get it |
|---|---|---|
| **Person query CSV** | The exported Viva Insights person query containing `PersonId`, `MetricDate`, Copilot metrics, and HR attributes. This is the sole data input. | Export from the [Viva Insights Analyst portal](https://insights.cloud.microsoft/). See [Required Inputs](#required-inputs) for column details. |
| **Prompt cards** | The structured prompts that you paste into your coding agent. Start with Dashboard Overview. | [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) · [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) · [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) · [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) |

#### Recommended

| File / Resource | Description | Why it helps |
|---|---|---|
| **Data dictionary** | Reference documentation describing each column in the person query export — data types, expected values, and definitions. | Helps you verify that your export matches what the prompts expect. Speeds up troubleshooting when column names differ. Available in the [schemas directory]({{ site.baseurl }}/frontier-analytics-schemas/). |
| **License assignment data** | A list of users who have been assigned a Copilot license, with assignment dates. Typically sourced from Microsoft 365 Admin Center or Microsoft Entra. | The prompts infer licensing from non-null metric values, which is a reasonable approximation. Explicit license data gives more accurate adoption rates, especially for recently assigned users who haven't yet appeared in metrics. |
| **Organizational hierarchy file** | A CSV or Excel file mapping people to custom organizational groupings (e.g., business units, cost centers, or project teams) not captured in standard HR attributes. | Enables custom breakdowns beyond the default `Organization` and `FunctionType` columns. Useful if your standard HR attributes are too broad or too granular. |

#### Optional

| File / Resource | Description | Why it helps |
|---|---|---|
| **Previous dashboard outputs** | HTML files or screenshots from earlier runs of this kit (or similar dashboards). | Provides a comparison baseline. You can ask the agent: _"Compare this week's metrics to the previous dashboard."_ Also useful for validating that trend lines are consistent. |
| **Target / benchmark values** | Internal targets for adoption rate, usage intensity, or ROI thresholds set by leadership or the Copilot deployment team. | Lets you add target lines or conditional formatting to the dashboard (e.g., highlight organizations below the 50% adoption target). Tell the agent: _"Add a horizontal target line at 60% adoption rate."_ |
| **Stakeholder distribution list** | The list of people who will receive the dashboard. | Knowing the audience helps you tailor the level of detail and privacy thresholds. For example, if the dashboard goes to a broad audience, you may want stricter suppression of small groups. |
| **Custom branding assets** | Company logo, brand colors (hex codes), or a style guide. | Ask the agent to apply your branding: _"Use #0078D4 as the primary color and include the logo at `./assets/logo.png`."_ |

#### Checklist

Use this checklist before you begin:

- [ ] Person query CSV is downloaded and accessible in the coding agent's workspace
- [ ] You've reviewed [Required Inputs](#required-inputs) and confirmed your columns match (or noted differences)
- [ ] You've opened the [Dashboard Overview prompt card]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) and are ready to copy the prompt
- [ ] Your coding agent has access to Python (with `pandas`, `plotly`) or R (with `tidyverse`, `htmlwidgets`)
- [ ] (Optional) You have license assignment data or org hierarchy files ready to reference

### Expected Output

The primary output is a **single self-contained HTML file** (typically 1–5 MB depending on data size and chart complexity). It:

- Opens in any modern browser (Chrome, Edge, Firefox, Safari)
- Requires no internet connection or server
- Embeds all JavaScript, CSS, and chart data inline
- Can be shared as an email attachment, uploaded to SharePoint, or posted in a Teams channel

#### Output format

| Property | Details |
|---|---|
| **File type** | `.html` |
| **Typical size** | 1–5 MB (varies with data volume and chart complexity) |
| **Browser support** | Chrome, Edge, Firefox, Safari (any modern browser) |
| **Offline viewing** | Yes — fully self-contained |
| **Responsive layout** | Yes — adapts to different screen widths, though optimized for desktop |

#### Dashboard panels

The dashboard is organized into distinct panels, arranged vertically. Each panel addresses a specific analytical question.

##### Panel 1 — Summary Metrics Bar

A horizontal bar at the top of the dashboard displaying headline numbers at a glance.

**Contents:**
- **Total licensed users**: Count of unique persons classified as Copilot-licensed in the most recent week
- **Active users**: Count of licensed users with at least one Copilot action in the most recent week
- **Adoption rate**: Active users ÷ licensed users, displayed as a percentage
- **Average actions per active user**: Mean `Copilot_Actions` among active users in the most recent week
- **Average assisted hours per active user**: Mean `Copilot_Assisted_Hours` among active users in the most recent week
- **Week-over-week change indicators**: Small arrows or percentage deltas showing change from the prior week

**Design:** Large, bold numbers with labels. Color-coded change indicators (green for increase, red for decrease). Designed to be readable at a glance in a meeting.

##### Panel 2 — Adoption Trend Chart

A time-series line chart showing how Copilot adoption evolves week over week.

**Contents:**
- **X-axis**: `MetricDate` (weekly)
- **Y-axis (primary)**: Count of licensed users and active users (two lines)
- **Y-axis (secondary or overlay)**: Adoption rate as a percentage line
- **Annotations**: Optional markers for key events (e.g., "Copilot rollout to Sales org")

**Design:** Clean line chart with a legend. Hover tooltips show exact values for each week. The adoption rate line uses a distinct style (dashed or different color) to differentiate from count lines.

##### Panel 3 — Usage Intensity Distribution

A chart showing how Copilot usage is distributed across the active user population.

**Contents:**
- **Chart type**: Histogram or box plot of `Copilot_Actions` (or `Copilot_Assisted_Hours`) for active users in the most recent week
- **Percentile markers**: Lines or annotations at the 25th, 50th, 75th, and 90th percentiles
- **Summary statistics**: Median, mean, and standard deviation displayed alongside the chart

**Design:** Helps identify whether usage is concentrated among a few power users or broadly distributed. A right-skewed distribution (common in early adoption) signals that most users are light users while a small group drives most activity.

##### Panel 4 — Organizational Breakdown

A comparative view of adoption and usage across organizational units.

**Contents:**
- **Chart type**: Grouped bar chart or heatmap
- **Rows/categories**: Values of `Organization` (or `FunctionType`, depending on configuration)
- **Metrics per group**: Licensed count, active count, adoption rate, average `Copilot_Actions`
- **Sorting**: By adoption rate (descending) or by group size
- **Privacy threshold**: Groups with fewer than 5 users are suppressed or aggregated into "Other"

**Design:** Enables leadership to see which departments lead adoption and which need attention. Color intensity or bar length encodes the metric value for quick visual scanning.

##### Panel 5 — User Segment Distribution

A stacked area or stacked bar chart showing how the user population breaks down into usage segments over time.

**Contents:**
- **Segments** (defined by percentile thresholds on `Copilot_Actions`):
  - **Power users**: Top quartile of active users
  - **Regular users**: 25th–75th percentile
  - **Light users**: Bottom quartile of active users
  - **Inactive licensed**: Licensed but zero actions that week
- **X-axis**: `MetricDate` (weekly)
- **Y-axis**: Count or proportion of users in each segment
- **Transition indicators**: Optional annotations showing net flows between segments

**Design:** Reveals whether the organization is growing its power-user base or seeing users slip into lighter usage tiers. Stacked format makes proportional shifts easy to see.

##### Panel 6 — Top Users Table

A tabular view of the highest-usage individuals (anonymized).

**Contents:**
- **Columns**: `PersonId` (anonymized), `Organization`, `FunctionType`, total `Copilot_Actions` (summed over the reporting period), average weekly `Copilot_Assisted_Hours`, number of active weeks, usage segment
- **Rows**: Top 20 users by total `Copilot_Actions`
- **Sorting**: Descending by total actions (user can re-sort by clicking column headers, if interactivity is enabled)

**Design:** Useful for identifying Copilot champions who could serve as advocates or trainers. All identifiers are anonymized per the person query privacy model.

#### Additional panels (from supplementary prompts)

If you also run the supplementary prompt cards, the dashboard can include:

| Prompt | Additional panels |
|---|---|
| [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) | A text panel with a formatted executive memo — headline findings, trends, and recommendations |
| [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) | Segment transition matrix (Sankey or heatmap), churn rate trend line, at-risk user list |
| [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) | ROI summary card, time-savings bar chart by organization, sensitivity analysis table, break-even threshold indicator |

#### How to share

The HTML file is portable and easy to distribute:

| Method | Notes |
|---|---|
| **Email attachment** | Attach the `.html` file directly. Recipients open it in their browser. Keep file size under 10 MB for most email systems. |
| **SharePoint / OneDrive** | Upload to a document library. Users can preview in the browser or download. |
| **Microsoft Teams** | Share via a Teams channel or chat. Pin to a channel tab for recurring access. |
| **Embed in a wiki or intranet** | Use an iframe or direct link if your intranet supports HTML embedding. |
| **Print / PDF** | Open in a browser and use the browser's print function to save as PDF. Layout may need minor adjustment for print. |

#### Customization options

You can ask your coding agent to modify the dashboard after generation:

- **Branding**: _"Change the primary color to #0078D4 and add a company logo."_
- **Date range filter**: _"Add a dropdown to filter the dashboard to a specific date range."_
- **Additional breakdowns**: _"Add a panel breaking down adoption by LevelDesignation."_
- **Metric selection**: _"Use Copilot_Assisted_Hours instead of Copilot_Actions as the primary metric."_
- **Privacy thresholds**: _"Suppress all groups with fewer than 10 people."_
- **Export options**: _"Add a button to export chart data as CSV."_
- **Dark mode**: _"Add a toggle for dark mode styling."_

Each customization is a follow-up instruction to your coding agent — no manual coding required.

---

## Executive Summary Report

> 📂 **Source files**: [View on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/starter-kits/executive-summary-report/)

### Overview

This starter kit helps you generate a **polished executive summary document** from Viva Insights person query data. The output is a professional memo — formatted in Markdown or HTML — that distills Copilot adoption metrics (or broader workforce collaboration patterns) into key findings, trends, and recommendations suitable for senior leadership.

You don't write the analysis yourself. The kit provides a structured **prompt card** that you paste into a coding agent. The agent reads your data, computes the metrics, identifies noteworthy patterns, and drafts the memo.

### Use case

People analytics teams regularly need to report to senior leadership on:

- **Copilot adoption progress**: How many users are active? Is adoption growing? Which organizations lead?
- **Workforce collaboration patterns**: How is collaboration time distributed? Are after-hours trends improving?
- **Program effectiveness**: Is a deployment wave achieving its targets? What needs attention?

These reports typically go to VP/C-suite executives, IT steering committees, or HR leadership on a monthly or quarterly cadence. This kit automates the data analysis and first-draft writing so you can focus on interpretation and storytelling.

### What you'll get

A **1–3 page executive summary** containing:

1. **Headline metrics**: 3–5 key numbers (e.g., adoption rate, active users, average actions per user) with week-over-week or period-over-period comparisons
2. **Key findings**: 3–5 bullet points highlighting the most important patterns in the data — what's growing, what's lagging, what's changed
3. **Trend analysis**: A brief narrative on how metrics have evolved over the reporting period, with references to specific inflection points
4. **Organizational highlights**: Which departments or functions are leading, which are behind, and any notable outliers
5. **Recommendations**: 2–3 actionable suggestions based on the data (e.g., "Expand targeted enablement in the Finance organization where adoption lags 15 points behind the company average")
6. **Areas of concern**: Any red flags — declining engagement, high churn, concentrated usage among a small group

The tone is concise, data-driven, and action-oriented — written for a busy executive who needs the bottom line in under 2 minutes.

### Prerequisites

| Requirement | Details |
|---|---|
| **Person query CSV** | Exported from the Viva Insights Analyst portal with `PersonId`, `MetricDate`, Copilot metrics, and HR attributes. See the [Copilot Adoption Dashboard required inputs](#required-inputs) for the full column specification. |
| **R or Python environment** | A local or cloud environment where the coding agent can execute code (Python with `pandas` is most common). |
| **Coding agent** | [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar AI coding assistant. |
| **8+ weeks of data** | 12+ weeks preferred for robust trend analysis. 4 weeks is a workable minimum. |

### Workflow

#### Step 1 — Export your data

Export a person query from the Viva Insights Analyst portal that includes Copilot activity metrics and HR attributes. This is the same export used for the [Copilot Adoption Dashboard](#copilot-adoption-dashboard) — if you've already generated a dashboard, you can reuse the same CSV.

#### Step 2 — Review the schema docs

Confirm that your column names match expected conventions. See the [schemas directory]({{ site.baseurl }}/frontier-analytics-schemas/) for reference. Note any differences — you'll communicate them to the agent.

#### Step 3 — Use the executive summary prompt

Open the [Executive Summary prompt card]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) and copy the full text from the **Prompt** section.

In your coding agent, prepend a context line:

```
My person query CSV is at ./data/person-query.csv.
The reporting period is Q3 2024 (July–September).
The audience is the CTO and IT leadership team.
Focus on Copilot adoption metrics.

[paste the full Executive Summary prompt here]
```

> **Tip:** Tell the agent who the audience is and what time period to cover. This shapes the memo's framing and level of detail.

#### Step 4 — Review and refine the output

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

#### Step 5 — Export to desired format

The default output is **Markdown** or **HTML**. To convert:

- **PDF**: Open the HTML in a browser and print to PDF.
- **Word**: Copy the Markdown into a Word document, or ask the agent to generate a `.docx` file directly (requires `python-docx` or `officer` in R).
- **PowerPoint**: Ask the agent: _"Convert the key findings and metrics into a 3-slide PowerPoint summary."_ (requires `python-pptx` or `officer` in R).
- **Email body**: Copy the Markdown or HTML directly into an Outlook email.

### Included prompts

| Prompt Card | Description |
|---|---|
| [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption) | Generate a concise executive memo with headline metrics, key findings, trend analysis, and recommendations. |

> **Tip:** You can also use the [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) prompt first to build visual charts, then ask the agent to write an executive summary narrative referencing those charts.

### Expected output

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

### Tips for different audiences

| Audience | Customization |
|---|---|
| **CTO / IT leadership** | Emphasize technical adoption metrics, deployment coverage, and infrastructure readiness. |
| **CHRO / HR leadership** | Frame around employee experience, collaboration patterns, and change management. |
| **CFO / Finance** | Lead with ROI metrics, license utilization, and cost-per-active-user. Pair with the [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) prompt. |
| **CEO / Board** | Keep it to one page. Focus on 3 headline numbers and 2 recommendations. Minimize jargon. |
| **Copilot program team** | Include more operational detail — churn rates, segment shifts, specific org-level data. |

Customize by adding a line to the agent: _"The audience is the CFO. Lead with financial metrics and license ROI."_

### Related resources

- [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [Copilot Adoption Dashboard starter kit](#copilot-adoption-dashboard)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Viva Insights Sample Code Library]({{ site.baseurl }}/)

### Quickstart

Generate a polished executive summary memo from your Viva Insights data in about 10 minutes.

#### What you need

| Item | Minimum requirement |
|---|---|
| **Person query CSV** | Exported from Viva Insights with `PersonId`, `MetricDate`, at least one `Copilot_*` metric, and at least one HR attribute. |
| **Data span** | 8+ weeks recommended (4 weeks minimum). |
| **Coding agent** | GitHub Copilot, Claude Code, or similar — with access to a Python or R environment. |

#### 1. Open your coding agent

Launch the agent in a workspace where it can execute Python (or R) and access your CSV file.

#### 2. Copy the Executive Summary prompt

Open the [Executive Summary prompt card]({{ site.baseurl }}/frontier-analytics-prompts/#executive-summary--copilot-adoption). Copy the entire text from the **Prompt** section.

#### 3. Paste with context

In your coding agent, provide context about your data and audience, then paste the prompt:

```
My person query CSV is at ./data/person-query.csv.
The reporting period is September 2024.
The audience is the VP of IT and the CTO.
Focus on Copilot adoption trends and organizational differences.

[paste the full Executive Summary prompt here]
```

> **Tip:** Be specific about the audience and time period. This directly shapes the memo's framing, level of detail, and recommendations.

#### 4. Review the draft

The agent will produce a Markdown or HTML memo containing:

- Headline metrics (adoption rate, active users, average actions)
- Key findings (3–5 bullets)
- Organizational highlights
- Recommendations

Read it critically. The numbers should align with what you know about your organization. Flag anything that seems off.

#### 5. Refine

Common follow-up instructions:

- _"Make the tone more formal."_
- _"Shorten to one page."_
- _"Add a comparison to last month."_
- _"Replace Organization names with abbreviations: Engineering → ENG, Sales → SLS."_
- _"Remove the Areas of Concern section — we'll address that separately."_

#### 6. Export

- **HTML or Markdown**: Use as-is, or copy into an email.
- **PDF**: Open the HTML in your browser → Print → Save as PDF.
- **PowerPoint**: Ask the agent: _"Convert this into a 3-slide PowerPoint."_

#### Expected output in 10 minutes

After one pass through Steps 1–4, you should have a 1–2 page memo with:

- A summary metrics header with 3–5 key numbers
- A findings section with data-backed observations
- Actionable recommendations tied to specific organizational patterns

This draft is typically 80% ready. Step 5 polishes it for your specific audience.

#### What to do next

- **Add visual context**: Run the [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompts/#dashboard-overview--copilot-adoption) prompt to generate charts that complement the memo.
- **Deepen the analysis**: Run [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompts/#segmentation--churn--copilot-adoption) to add detail on user behavior patterns.
- **Build the business case**: Run [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompts/#roi-estimation--copilot-adoption) if leadership wants to see financial returns.
- **Set up a recurring cadence**: Save your prompt with the context line. Each month or quarter, drop in a fresh CSV and re-run.

For the full guide, see the [overview](#executive-summary-report).

---

## How to Contribute a New Starter Kit

We welcome new starter kits. To add one:

1. Check the [templates folder](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates/) for the starter kit template.
2. Create a new folder under [`starter-kits/`](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/starter-kits/) with a descriptive kebab-case name (e.g., `starter-kits/meeting-culture-report/`).
3. Include at minimum: `README.md`, `quickstart.md`, and `required-inputs.md`.
4. Reference existing prompt cards from the [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/) or create new ones following the [prompt card template](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates/).
5. Submit a pull request. See the [contributing guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md) for details.

## Next Steps

- [Quickstart guide]({{ site.baseurl }}/frontier-analytics-quickstart/) — general getting-started instructions
- [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/) — browse all available prompts
- [Schema documentation]({{ site.baseurl }}/frontier-analytics-schemas/) — understand your data
- [Back to Frontier Analytics]({{ site.baseurl }}/frontier-analytics/)
