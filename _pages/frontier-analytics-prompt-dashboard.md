---
layout: page
title: "Prompt — Dashboard Overview"
permalink: /frontier-analytics-prompt-dashboard/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}
{% include prompt-styles.html %}

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

/* Prompt page navigation */
.prompt-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e0e0e0;
}
.prompt-nav a {
  text-decoration: none;
  color: #0366d6;
  font-weight: 500;
}
.prompt-nav a:hover {
  text-decoration: underline;
}
.prompt-nav .nav-disabled {
  color: #999;
  pointer-events: none;
}
</style>

# Dashboard Overview — Copilot Adoption

[← Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)

## Purpose

Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns.

## Audience

People analytics leads, IT deployment managers

## When to use

After exporting a person query with Copilot activity metrics spanning at least 8 weeks of data.

## Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended
- HR attributes for segmentation

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

## Recommended output

A self-contained static HTML file with embedded charts, suitable for sharing via email or SharePoint.

## Prompt

```
You are a people analytics engineer. Your task is to build a self-contained static HTML dashboard that visualizes Microsoft Copilot adoption from a Viva Insights person query export.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.

DATA LOADING AND VALIDATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python). This handles variable name cleaning and type parsing automatically.
2. Ensure PersonId is treated as a string and MetricDate is parsed as a date type.
3. Run `extract_hr(df)` from the `vivainsights` library to identify the available HR / organizational attribute columns in the data. Use the returned list of HR attributes for all segmentation breakdowns instead of hard-coding column names like Organization, FunctionType, or LevelDesignation.
4. Verify the panel structure: each row should represent a unique PersonId × MetricDate combination. If there are duplicates, flag them and keep the first occurrence.
5. Print the shape of the data, the date range covered, and the number of unique persons.
6. List all column names so I can verify the Copilot metric columns and HR attribute columns match what is expected. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their name. Reference the taxonomy at https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary activity metric (it captures all Copilot usage across apps).

IDENTIFYING LICENSED USERS
6. A user is considered "Copilot-licensed" in a given week if they have a non-null, non-zero value in at least one Copilot metric column for that week. Create a boolean column `is_licensed` to flag these rows.
7. Also flag "active" users: licensed users who have Total_Copilot_actions_taken > 0 (the primary activity metric) in that week. Create a boolean column `is_active`.
8. Print a summary: total person-weeks, licensed person-weeks, active person-weeks.

METRIC CALCULATIONS
9. For each week (MetricDate), compute:
   a. Total licensed users (count of distinct PersonId where is_licensed == True)
   b. Total active users (count of distinct PersonId where is_active == True)
   c. Adoption rate = active users / licensed users (as a percentage)
   d. Mean Total_Copilot_actions_taken per active user
   e. Mean Copilot_Assisted_Hours per active user
   f. Median Total_Copilot_actions_taken per active user
10. Store these weekly aggregates in a summary DataFrame called `weekly_summary`.

SEGMENTATION METRICS
11. For each HR attribute (Organization, FunctionType, LevelDesignation), for each week, compute:
    a. Licensed user count
    b. Active user count
    c. Adoption rate
    d. Mean Total_Copilot_actions_taken per active user
12. Store each in a separate DataFrame (e.g., `org_summary`, `function_summary`, `level_summary`).

TOP USERS TABLE
13. Compute a "top users" table: for each PersonId, calculate total Total_Copilot_actions_taken across all weeks, total active weeks, and average Total_Copilot_actions_taken per active week. Rank by total actions descending. Keep the top 20. Include their HR attributes for context.

SUMMARY STATISTICS PANEL
14. Calculate overall summary statistics for the dashboard header:
    a. Latest week adoption rate
    b. Trend direction: compare the last 4 weeks' average adoption rate to the prior 4 weeks
    c. Total unique licensed users across the entire period
    d. Total unique active users across the entire period
    e. Average Total_Copilot_actions_taken per active user per week (grand mean)
    f. Most active organization (highest adoption rate in the latest week)

DASHBOARD GENERATION
15. Create the dashboard as an intermediary document first, then export to HTML:
    - R: Create an RMarkdown file (.Rmd) with ggplot2 charts, then knit to a self-contained HTML file (output: html_document, self_contained: true).
    - Python: Create a Jupyter notebook (.ipynb) with matplotlib/seaborn charts, then export to a self-contained HTML file (e.g., `jupyter nbconvert --to html`). Keep the intermediary .Rmd or .ipynb file alongside the HTML output — it makes troubleshooting and iteration easier. Do NOT use a web framework or server.

16. The HTML dashboard should contain these sections in order:
    a. HEADER: Title ("Copilot Adoption Dashboard"), date range, generation timestamp.
    b. SUMMARY PANEL: Cards showing the key metrics from step 14 (adoption rate, trend, total users, avg actions, top org). Use colored indicators (green for positive trend, red for negative).
    c. TREND CHARTS:
       - Line chart: Weekly adoption rate over time
       - Line chart: Mean Total_Copilot_actions_taken per active user over time
       - Line chart: Mean Copilot_Assisted_Hours per active user over time
    d. SEGMENTATION CHARTS:
       - Grouped bar chart: Adoption rate by Organization (latest 4-week average)
       - Grouped bar chart: Adoption rate by FunctionType (latest 4-week average)
       - Grouped bar chart: Adoption rate by LevelDesignation (latest 4-week average)
       - Heatmap: Adoption rate by Organization × week (if number of orgs <= 15)
    e. TOP USERS TABLE: HTML table of top 20 users from step 13.
    f. METHODOLOGY NOTE: Brief paragraph explaining how adoption rate is calculated, what "licensed" and "active" mean, and the data source.

17. Style the HTML with a clean, professional design. Use a sans-serif font, consistent color palette, and adequate whitespace. The dashboard should look presentable when opened in a browser.

18. Save the HTML file and the intermediary .Rmd or .ipynb to the working directory with descriptive filenames like "copilot_adoption_dashboard_YYYYMMDD.html".

IMPORTANT NOTES
- Do NOT create interactive plots that require a running server (no plotly, no bokeh server). Static charts embedded in the RMarkdown/Jupyter output are preferred.
- Handle missing values gracefully: NaN in Copilot columns means the user is unlicensed that week.
- If any HR attribute column is missing from the data, skip that segmentation chart and note it.
- Use the `vivainsights` package (R or Python) for data loading (`import_query()`) and HR attribute discovery (`extract_hr()`). Use ggplot2 (R) or matplotlib/seaborn (Python) for charting.
- All charts should have clear titles, axis labels, and legends.
- If any segment has fewer than 5 users, suppress it from charts to protect privacy.
```

## Adaptation notes

- The `extract_hr()` function auto-detects organizational attributes, so you typically do not need to manually specify HR column names. If your data has custom attribute columns that `extract_hr()` does not detect, add an instruction specifying them.
- If your data is at person-day granularity, add an instruction: _"Aggregate person-day data to person-week by summing Copilot metrics per PersonId per week."_
- For smaller organizations, increase the privacy threshold (e.g., from 5 to 10 users per segment) or remove segmentation breakdowns entirely.
- Add custom Copilot metrics by extending the list in step 5 (e.g., `Copilot_Edited_Hours`, `Copilot_Rewritten_Hours`).
- If you prefer R over Python, add _"Use R with ggplot2 and R Markdown"_ at the start of the prompt.

## Common failure modes

- **Agent assumes all rows have Copilot data.** The prompt explicitly separates licensed vs. unlicensed users, but some agents may skip this. Verify that the `is_licensed` flag is computed before any metric calculations.
- **Agent creates interactive plots that need a server.** The prompt specifies static HTML with base64-encoded images. If the agent uses plotly or bokeh, ask it to switch to matplotlib/seaborn or export static images.
- **Agent ignores the panel structure and double-counts users.** Ensure adoption rate is calculated per week using distinct `PersonId` counts, not row counts.
- **Metric column names differ between tenants.** Copilot metric columns generally contain the word "Copilot" but do not always start with `Copilot_`. Reference the [metrics taxonomy](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv) to validate detected columns.
- **HTML file is not self-contained.** Check that the output HTML opens correctly with no external dependencies. If charts appear broken, ensure base64 encoding was applied.

<div class="prompt-nav">
  <span class="nav-disabled">← Previous</span>
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-executive-summary/">Next: Executive Summary →</a>
</div>
