# Dashboard Overview — Copilot Adoption

## Purpose

Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns.

## Audience

People analytics leads, IT deployment managers

## When to use

After exporting a person query with Copilot activity metrics spanning at least 8 weeks of data.

## Required inputs

- Person query CSV with columns: `PersonId`, `MetricDate`, Copilot metrics (e.g., `Copilot_Actions`, `Copilot_Assisted_Hours`, `Copilot_Chat_Queries`, `Copilot_Summarized_Hours`), and HR attributes (e.g., `Organization`, `FunctionType`, `LevelDesignation`)
- At least 8 weeks of data recommended
- HR attributes for segmentation (e.g., `Organization`, `FunctionType`, `LevelDesignation`)

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Licensing and coverage require explicit evidence; missing activity alone is inconclusive
- The `vivainsights` R or Python package is available in the environment

## Recommended output

A self-contained static HTML file with embedded charts, suitable for sharing via email or SharePoint.

## Quick prompt (short version)

Use this shorter prompt for a fast first pass that adapts to your data.

```
Help me create a self-contained static HTML dashboard for people analytics and IT leaders that summarizes Copilot adoption trends and group differences from a Viva Insights person query export.
Start by loading the actual CSV and printing the row count, column names, data types, date range, and number of unique people, then map the real metric and HR columns from what is present.
Do not assume exact field names, and ask me to confirm the primary activity metric or any missing HR columns if the mapping is ambiguous.
Separate licensed users from active users using explicit licensing and coverage evidence. If those are unavailable, show observed activity and mark adoption rates unavailable; do not infer licensing from activity or missingness.
Build weekly adoption metrics using distinct people per week, then add the most useful group views from the available HR attributes.
The output must be a single self-contained static HTML file with inline assets and no server or external dependencies.
Suppress any group with fewer than 10 distinct people, or the stricter organisational minimum, and withhold breakdowns that could reveal suppressed cells. Do not publish individual rankings. Note skipped charts or missing segments.
At the end, list any assumptions you made about metric selection, licensing logic, or field mappings.
```

Use the detailed prompt below for a reproducible, exact-structure output. Use the quick prompt above for a faster first pass that you refine through follow-up turns.

## Prompt

```
You are a people analytics engineer. Your task is to build a self-contained static HTML dashboard
that visualizes Microsoft Copilot adoption from a Viva Insights person query export.

DATA LOADING AND VALIDATION
1. Load the person query CSV file into a DataFrame (use pandas in Python or readr/vroom in R).
2. Parse the MetricDate column as a date type. Ensure PersonId is treated as a string.
3. Verify the panel structure: each row should represent a unique PersonId × MetricDate combination.
   If there are duplicates, report their count and stop for an explicit resolution rule.
4. Print the shape of the data, the date range covered, and the number of unique persons.
5. List all column names so I can verify the Copilot metric columns and HR attribute columns
   match what is expected. Auto-detect columns that start with "Copilot_" as Copilot metric columns.

IDENTIFYING LICENSED USERS
6. A user is considered "Copilot-licensed" in a given week using this rule, in order:
   a. If an enabled-days column is present (for example `Total_Copilot_enabled_days`), the user
      is licensed when that column is greater than zero for the week.
   b. Otherwise, use documented licence records for that period. If unavailable, retain
      unknown licensing and omit adoption rates; do not infer licences from activity.
   Create `is_licensed` with unknown values retained. Confirm coverage before filling absent
   activity with zero.
7. Also flag "active" users: licensed users who have Copilot_Actions > 0 (or the equivalent primary
   activity metric) in that week. Create a boolean column `is_active`.
8. Print a summary: total person-weeks, licensed person-weeks, active person-weeks.

METRIC CALCULATIONS
9. For each week (MetricDate), compute:
   a. Total licensed users (count of distinct PersonId where is_licensed == True)
   b. Total active users (count of distinct PersonId where is_active == True)
   c. Adoption rate = active users / licensed users (as a percentage)
   d. Mean Copilot_Actions per active user
   e. Mean Copilot_Assisted_Hours per active user
   f. Median Copilot_Actions per active user
10. Store these weekly aggregates in a summary DataFrame called `weekly_summary`.

SEGMENTATION METRICS
11. For each HR attribute (Organization, FunctionType, LevelDesignation), for each week, compute:
    a. Licensed user count
    b. Active user count
    c. Adoption rate
    d. Mean Copilot_Actions per active user
12. Store each in a separate DataFrame (e.g., `org_summary`, `function_summary`, `level_summary`).

GROUP SUMMARY TABLE
13. Compute activity summaries by an approved HR group using distinct people and the declared
    time window. Apply privacy suppression to every cell and omit the entire breakdown if
    remaining cells or totals could reveal a suppressed value. Do not publish person rows.

SUMMARY STATISTICS PANEL
14. Calculate overall summary statistics for the dashboard header:
    a. Latest week adoption rate
    b. Trend direction: compare the last 4 weeks' average adoption rate to the prior 4 weeks
    c. Total unique licensed users across the entire period
    d. Total unique active users across the entire period
    e. Average Copilot_Actions per active user per week (grand mean)
    f. Most active organization (highest adoption rate in the latest week)

DASHBOARD GENERATION
15. Generate a single self-contained HTML file. Do NOT use a web framework or server. All CSS and
    JavaScript must be inline. Use one of these approaches:
    - Python: use matplotlib/seaborn to create chart images, base64-encode them, and embed in HTML
      using an HTML template string or Jinja2.
    - R: use ggplot2 to create chart images, base64-encode them, and embed in an R Markdown document
      rendered to self-contained HTML (self_contained: true), or build the HTML manually.

16. The HTML dashboard should contain these sections in order:
    a. HEADER: Title ("Copilot Adoption Dashboard"), date range, generation timestamp.
    b. SUMMARY PANEL: Cards showing the key metrics from step 14 (adoption rate, trend, total users,
       avg actions, top org). Use colored indicators (green for positive trend, red for negative).
    c. TREND CHARTS:
       - Line chart: Weekly adoption rate over time
       - Line chart: Mean Copilot_Actions per active user over time
       - Line chart: Mean Copilot_Assisted_Hours per active user over time
    d. SEGMENTATION CHARTS:
       - Grouped bar chart: Adoption rate by Organization (latest 4-week average)
       - Grouped bar chart: Adoption rate by FunctionType (latest 4-week average)
       - Grouped bar chart: Adoption rate by LevelDesignation (latest 4-week average)
       - Heatmap: Adoption rate by Organization × week (if number of orgs <= 15)
    e. GROUP SUMMARY TABLE: privacy-protected aggregate table from step 13, with no identifiers.
    f. METHODOLOGY NOTE: Brief paragraph explaining how adoption rate is calculated, what
       "licensed" and "active" mean, and the data source.

17. Style the HTML with a clean, professional design. Use a sans-serif font, consistent color
    palette, and adequate whitespace. The dashboard should look presentable when opened in a browser.

18. Save the HTML file to the working directory with a descriptive filename like
    "copilot_adoption_dashboard_YYYYMMDD.html".

IMPORTANT NOTES
- Do NOT create interactive plots that require a running server (no plotly, no bokeh server).
  Static images embedded as base64 are preferred.
- Preserve unknown coverage and eligibility. NaN is not proof of an unlicensed user.
- Omit metrics without verified inputs, including assisted hours or licensing-dependent
  summaries. Describe skipped panels rather than estimating missing values.
- If any HR attribute column is missing from the data, skip that segmentation chart and note it.
- Use the vivainsights package for any helper functions it provides, but do not depend on it for
  core logic — the dashboard should work with just pandas/matplotlib or base R/ggplot2.
- All charts should have clear titles, axis labels, and legends.
- Use at least 10 distinct people per published group, or the stricter organisational minimum.
```

## Adaptation notes

- Adjust HR attribute column names to match your export (e.g., `Organization` vs `Org`). Prepend a note to the prompt specifying your actual column names.
- If your data is at person-day granularity, add an instruction: _"Aggregate person-day data to person-week by summing Copilot metrics per PersonId per week."_
- Apply at least 10 distinct people per group, increase this if required by organisational policy, or omit segmentation entirely.
- Add custom Copilot metrics by extending the list in step 5 (e.g., `Copilot_Edited_Hours`, `Copilot_Rewritten_Hours`).
- If you prefer R over Python, add _"Use R with ggplot2 and R Markdown"_ at the start of the prompt.

## Common failure modes

- **Agent assumes all rows have Copilot data.** The prompt explicitly separates licensed vs. unlicensed users, but some agents may skip this. Verify that the `is_licensed` flag is computed before any metric calculations.
- **Agent creates interactive plots that need a server.** The prompt specifies static HTML with base64-encoded images. If the agent uses plotly or bokeh, ask it to switch to matplotlib/seaborn or export static images.
- **Agent ignores the panel structure and double-counts users.** Ensure adoption rate is calculated per week using distinct `PersonId` counts, not row counts.
- **Metric column names differ between tenants.** The prompt includes an auto-detect step (columns starting with `Copilot_`), but always verify column names in your export before running.
- **HTML file is not self-contained.** Check that the output HTML opens correctly with no external dependencies. If charts appear broken, ensure base64 encoding was applied.
