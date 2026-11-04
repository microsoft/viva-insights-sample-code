# Expected Output — Copilot Adoption Dashboard

This document describes what the finished dashboard looks like, how it's structured, and how to share it.

The [starter README](README.md) owns scope and policy. Omit adoption rates
without explicit eligibility evidence and unsupported metrics rather than
filling panels with guessed values. Apply the dashboard privacy minimum to every
panel, filter, embedded dataset and download.

## Output format

The primary deliverable is a **single self-contained HTML file**. Everything — JavaScript, CSS, chart data, and layout — is embedded inline. No external dependencies, no server, no internet connection required to view.

| Property | Details |
|---|---|
| **File type** | `.html` |
| **Typical size** | 1–5 MB (varies with data volume and chart complexity) |
| **Browser support** | Chrome, Edge, Firefox, Safari (any modern browser) |
| **Offline viewing** | Yes — fully self-contained |
| **Responsive layout** | Yes — adapts to different screen widths, though optimized for desktop |

## Dashboard panels

The dashboard is organized into distinct panels, arranged vertically. Each panel addresses a specific analytical question.

### Panel 1 — Summary Metrics Bar

A horizontal bar at the top of the dashboard displaying headline numbers at a glance.

**Contents:**
- **Total licensed users**: Count of unique persons classified as Copilot-licensed in the most recent week
- **Active users**: Count of licensed users with at least one Copilot action in the most recent week
- **Adoption rate**: Active users ÷ licensed users, displayed as a percentage
- **Average actions per active user**: Mean `Copilot_Actions` among active users in the most recent week
- **Average assisted hours per active user**: Mean `Copilot_Assisted_Hours` among active users in the most recent week
- **Week-over-week change indicators**: Small arrows or percentage deltas showing change from the prior week

**Design:** Large, bold numbers with labels. Color-coded change indicators (green for increase, red for decrease). Designed to be readable at a glance in a meeting.

### Panel 2 — Adoption Trend Chart

A time-series line chart showing how Copilot adoption evolves week over week.

**Contents:**
- **X-axis**: `MetricDate` (weekly)
- **Y-axis (primary)**: Count of licensed users and active users (two lines)
- **Y-axis (secondary or overlay)**: Adoption rate as a percentage line
- **Annotations**: Optional markers for key events (e.g., "Copilot rollout to Sales org")

**Design:** Clean line chart with a legend. Hover tooltips show exact values for each week. The adoption rate line uses a distinct style (dashed or different color) to differentiate from count lines.

### Panel 3 — Usage Intensity Distribution

A chart showing how Copilot usage is distributed across the active user population.

**Contents:**
- **Chart type**: Histogram or box plot of `Copilot_Actions` (or `Copilot_Assisted_Hours`) for active users in the most recent week
- **Percentile markers**: Lines or annotations at the 25th, 50th, 75th, and 90th percentiles
- **Summary statistics**: Median, mean, and standard deviation displayed alongside the chart

**Design:** Helps identify whether usage is concentrated among a few power users or broadly distributed. A right-skewed distribution (common in early adoption) signals that most users are light users while a small group drives most activity.

### Panel 4 — Organizational Breakdown

A comparative view of adoption and usage across organizational units.

**Contents:**
- **Chart type**: Grouped bar chart or heatmap
- **Rows/categories**: Values of `Organization` (or `FunctionType`, depending on configuration)
- **Metrics per group**: Licensed count, active count, adoption rate, average `Copilot_Actions`
- **Sorting**: By adoption rate (descending) or by group size
- **Privacy threshold**: max(organisation requirement, 10 distinct people), with complementary-cell checks; pooling into "Other" must itself meet the threshold

**Design:** Enables leadership to see which departments lead adoption and which need attention. Color intensity or bar length encodes the metric value for quick visual scanning.

### Panel 5 — User Segment Distribution

A stacked area or stacked bar chart showing how the user population breaks down into usage segments over time.

**Contents:**
- **Segments**: `identify_usage_segments()` Power, Habitual, Novice, Low and
  Non-users, using the documented rolling window and thresholds, not percentiles.
- **Unavailable histories**: report separately when licence/activity evidence
  or the required calendar history is unknown.
- **X-axis**: `MetricDate` (weekly)
- **Y-axis**: Count or proportion of users in each segment
- **Transition indicators**: Optional annotations showing net flows between segments

**Design:** Reveals whether the organization is growing its power-user base or seeing users slip into lighter usage tiers. Stacked format makes proportional shifts easy to see.

### Panel 6 — Group Summary Table

A privacy-protected aggregate view of approved HR groups.

**Contents:**
- **Columns**: Group, reporting window, distinct-person n, verified activity
  summaries and their units/estimands
- **Rows**: Groups satisfying the applicable privacy minimum in the displayed cut
- **Disclosure**: No person identifiers, individual rankings, raw tooltips or
  hidden person-level records; withhold the breakdown if totals reveal small cells

**Design:** Supports aggregate enablement planning, not individual evaluation.
Pseudonymisation does not make an individual leaderboard safe to publish.

## Additional panels (from supplementary prompts)

If you also run the supplementary prompt cards, the dashboard can include:

| Prompt | Additional panels |
|---|---|
| [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) | A text panel with a formatted executive memo — headline findings, trends, and recommendations |
| [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) | Protected segment transition matrix, churn trend and at-risk group summaries |
| [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) | Potential-value scenario, approved realisation assumptions, sensitivity table and break-even indicator; not measured time savings or realised ROI |

## How to share

The HTML file is portable and easy to distribute:

| Method | Notes |
|---|---|
| **Email attachment** | Attach the `.html` file directly. Recipients open it in their browser. Keep file size under 10 MB for most email systems. |
| **SharePoint / OneDrive** | Upload to a document library. Users can preview in the browser or download. |
| **Microsoft Teams** | Share via a Teams channel or chat. Pin to a channel tab for recurring access. |
| **Embed in a wiki or intranet** | Use an iframe or direct link if your intranet supports HTML embedding. |
| **Print / PDF** | Open in a browser and use the browser's print function to save as PDF. Layout may need minor adjustment for print. |

## Customization options

You can ask your coding agent to modify the dashboard after generation:

- **Branding**: _"Change the primary color to #0078D4 and add a company logo."_
- **Date range filter**: _"Add a dropdown to filter the dashboard to a specific date range."_
- **Additional breakdowns**: _"Add a panel breaking down adoption by LevelDesignation."_
- **Metric selection**: _"Use Copilot_Assisted_Hours instead of Copilot_Actions as the primary metric."_
- **Privacy thresholds**: _"Suppress all groups with fewer than 10 people."_
- **Export options**: _"Add a button to export chart data as CSV."_
- **Dark mode**: _"Add a toggle for dark mode styling."_

Each customization is a follow-up instruction to your coding agent — no manual coding required.
