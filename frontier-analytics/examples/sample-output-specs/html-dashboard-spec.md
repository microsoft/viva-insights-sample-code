# Output Spec: Static HTML Dashboard

This specification defines the structure, styling, and content expectations for a **self-contained static HTML dashboard** generated from Viva Insights data. Use this spec as context for coding agents when you want them to produce a polished, shareable dashboard.

Apply the [shared analysis policy](../../skills/viva-insights-analysis/reference/analysis-policy.md).
The numeric examples below are illustrative, not executed results.

---

## Target format

| Property | Requirement |
|----------|-------------|
| **File type** | Single `.html` file |
| **Dependencies** | Fully self-contained — no external CSS, JS, fonts, or images. Everything is inline or base64-encoded. |
| **File size** | Under 5 MB (ideally under 2 MB) |
| **Compatibility** | Opens correctly in Chrome, Edge, Firefox, and Safari (current versions) |
| **Interactivity** | Static content only — no running server, no JavaScript frameworks requiring build steps |

> **Why self-contained?** The dashboard will be shared as an email attachment, uploaded to SharePoint, or opened locally. It must render without network access.

---

## Recommended structure

The dashboard should follow this section layout, top to bottom:

### 1. Header

- Dashboard title (e.g., "Copilot Adoption Dashboard")
- Date range of the data (e.g., "Sep 2, 2024 – Nov 24, 2024")
- Generation timestamp (e.g., "Generated on Dec 1, 2024 at 14:30 UTC")
- Optional: logo or organization name

### 2. Summary metrics row

A horizontal row of 4–6 **metric cards**, each showing:
- Metric name (e.g., "Adoption Rate")
- Current value (e.g., "68%")
- Trend indicator (↑ or ↓ with comparison period, e.g., "↑ 5pp vs. prior 4 weeks")
- Color coding: green for positive trends, red for negative, gray for neutral

Example cards:
| Adoption Rate | Active Users | Avg Actions/User/Week | Top Organization |
|:---:|:---:|:---:|:---:|
| **68%** | **2,340** | **34.2** | **Engineering** |
| ↑ 5pp vs. prior 4 wks | ↑ 12% vs. prior 4 wks | ↑ 8% vs. prior 4 wks | 78% adoption |

### 3. Trend charts

Line charts showing metrics over time. Each chart should have:
- Clear title
- X-axis: `MetricDate` (weekly)
- Y-axis: metric value with unit
- Axis labels and gridlines
- Data point markers (optional)

Recommended charts:
- **Adoption rate over time** (% of licensed users who are active)
- **Average Copilot Actions per active user per week**
- **Average Copilot Assisted Hours per active user per week**
- **Licensed user count over time** (total users with Copilot access)

### 4. Segmentation charts

Bar charts or heatmaps showing breakdowns by HR attributes. Include:
- **Adoption rate by Organization** (horizontal bar chart, latest 4-week average)
- **Adoption rate by FunctionType** (horizontal bar chart, latest 4-week average)
- **Adoption rate by LevelDesignation** (horizontal bar chart, latest 4-week average)
- **Heatmap: Adoption rate by Organization × Week** (if ≤15 organizations)

Use max(organisation requirement, dashboard floor of 10 distinct people).
Protect complementary cells, embedded data and downloads.

### 5. Data tables

HTML tables showing underlying detail. Include:
- **Weekly summary table** with columns: Week, Licensed Users, Active Users, Adoption Rate, Avg Actions, Avg Assisted Hours
- **Group summary table**: approved HR group, window, distinct-person n,
  verified activity metrics, units and estimands. No individual rankings.

Tables should be sortable if JavaScript is used, or sorted by default (descending by the primary metric).

### 6. Methodology footer

A brief section explaining:
- Data source (e.g., "Viva Insights person query export")
- How "licensed" and "active" are defined
- Time period and granularity
- Privacy thresholds applied
- Any caveats or limitations

---

## Chart types and when to use them

| Chart Type | Use For | Notes |
|-----------|---------|-------|
| Line chart | Trends over time | Use for weekly metric trends. Include at least 4 data points. |
| Horizontal bar chart | Comparisons across segments | Use for adoption rate by Organization, Function, Level. Sort descending. |
| Heatmap | Two-dimensional breakdowns | Use for Organization × Week adoption. Limit to ≤15 segments for readability. |
| Stacked bar chart | Composition breakdowns | Use sparingly — for showing licensed vs. active vs. inactive composition. |

---

## Styling guidance

```css
/* Recommended styling principles */
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #333;
    background: #fff;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.5;
}

h1 { font-size: 1.8em; color: #0078d4; margin-bottom: 0.3em; }
h2 { font-size: 1.3em; color: #333; border-bottom: 1px solid #e0e0e0; padding-bottom: 0.3em; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
}
th { background: #f5f5f5; text-align: left; padding: 8px 12px; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }

.metric-card {
    display: inline-block;
    text-align: center;
    padding: 16px 24px;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    margin: 8px;
    min-width: 160px;
}
.metric-value { font-size: 2em; font-weight: 600; }
.metric-trend { font-size: 0.85em; margin-top: 4px; }
.trend-up { color: #107c10; }
.trend-down { color: #d13438; }
```

**Key styling principles:**
- Clean and professional — suitable for executive sharing
- Consistent color palette (Microsoft Fluent-inspired blues and grays work well)
- Adequate whitespace between sections
- Responsive layout that works at different screen widths
- Print-friendly (charts should be readable when printed)

---

## Accessibility notes

- Use sufficient color contrast (WCAG AA minimum: 4.5:1 for text)
- Do not rely on color alone to convey information — add text labels to trend indicators
- Include `alt` text for chart images (description of what the chart shows)
- Use semantic HTML (`<h1>`, `<h2>`, `<table>`, `<th>`, `<td>`) for screen reader compatibility
- Ensure tables have proper header cells (`<th>`) for each column

---

## Example HTML skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Copilot Adoption Dashboard</title>
    <style>
        /* Inline CSS — see styling guidance above */
    </style>
</head>
<body>
    <!-- Header -->
    <header>
        <h1>Copilot Adoption Dashboard</h1>
        <p class="subtitle">Data: Sep 2, 2024 – Nov 24, 2024 | Generated: Dec 1, 2024</p>
    </header>

    <!-- Summary Metrics -->
    <section id="summary">
        <div class="metric-cards">
            <div class="metric-card">
                <div class="metric-label">Adoption Rate</div>
                <div class="metric-value">68%</div>
                <div class="metric-trend trend-up">↑ 5pp vs. prior 4 wks</div>
            </div>
            <!-- Additional metric cards -->
        </div>
    </section>

    <!-- Trend Charts -->
    <section id="trends">
        <h2>Trends Over Time</h2>
        <img src="data:image/png;base64,..." alt="Line chart showing weekly adoption rate trending upward from 55% to 68% over 12 weeks." />
        <!-- Additional chart images -->
    </section>

    <!-- Segmentation -->
    <section id="segmentation">
        <h2>Adoption by Segment</h2>
        <img src="data:image/png;base64,..." alt="Bar chart showing adoption rate by organization. Engineering leads at 78%." />
        <!-- Additional segmentation charts -->
    </section>

    <!-- Data Tables -->
    <section id="tables">
        <h2>Weekly Summary</h2>
        <table>
            <thead>
                <tr><th>Week</th><th>Licensed</th><th>Active</th><th>Adoption Rate</th><th>Avg Actions</th></tr>
            </thead>
            <tbody>
                <tr><td>2024-11-18</td><td>3,450</td><td>2,340</td><td>67.8%</td><td>34.2</td></tr>
                <!-- Additional rows -->
            </tbody>
        </table>
    </section>

    <!-- Methodology -->
    <footer id="methodology">
        <h2>Methodology</h2>
        <p><strong>Data source:</strong> Viva Insights person query export (person-week granularity).</p>
        <p><strong>Definitions:</strong> Prefer Copilot Enabled User or Total Copilot enabled days for licensing. Native licensed-only M365 Copilot nulls mean unlicensed and zero activity means licensed but inactive. A verified metric with these semantics is a valid fallback. Adoption rates include licensed zero-activity people and exclude unlicensed people.</p>
        <p><strong>Missingness:</strong> Absent in-scope activity contributes zero. Failed joins, unavailable sources, out-of-scope periods and known coverage failures remain unresolved. Do not extend licensed-only M365 Copilot licence semantics to unlicensed Copilot Chat or Consumption credits.</p>
        <p><strong>Privacy:</strong> Use max(organisation requirement, 10 distinct people), including complementary cells and embedded data. No individual rankings.</p>
    </footer>
</body>
</html>
```

---

## How to instruct a coding agent

When asking a coding agent to produce this output, include context from this spec. Here is a sample instruction snippet you can prepend to any analysis prompt:

```
OUTPUT FORMAT INSTRUCTIONS:
Produce a single self-contained HTML file. All CSS must be inline in a <style> tag. All chart
images must be base64-encoded and embedded as <img> tags. No external dependencies.

Structure the dashboard as:
1. Header with title, date range, generation timestamp
2. Summary metric cards (4-6 key numbers with trend indicators)
3. Trend charts (line charts for weekly metrics)
4. Segmentation charts (bar charts by Organization, FunctionType, LevelDesignation)
5. Protected aggregate tables (weekly summary and HR groups, no person rows)
6. Methodology footer

Use matplotlib or ggplot2 for charts. Convert to base64 PNG for embedding.
Use import_query() from the vivainsights package to load CSV data.
Use extract_hr() to identify organizational attribute columns.
Style with a clean, professional design (Segoe UI font, blue/gray palette).
Confirm configured cadence and measurement windows, not date spacing. Prefer
Copilot Enabled User or Total Copilot enabled days for licensing. Apply documented
native licensed-only M365 Copilot null/zero semantics, with a verified licensed-only
metric as a fallback. Exclude native unlicensed nulls from licensed denominators.
Absent activity inside the supplied complete query's configured scope contributes
zero. Failed joins and out-of-scope/coverage failures remain unresolved.
Target file size under 5 MB. Use max(the organisational requirement, 10 distinct
people) per published group, with complementary-cell and embedded-data checks.
Do not publish individual rankings.
Save as "copilot_dashboard_YYYYMMDD.html".
```
