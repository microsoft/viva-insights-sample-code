---
layout: page
title: "Frontier Analytics — Prompt Library"
permalink: /frontier-analytics-prompts/
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

# Prompt Card Library

This page contains **prompt cards** — ready-to-use prompts that you can paste directly into a coding agent (such as [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or similar tools) to generate analytics outputs from Viva Insights data.

Each prompt card includes the purpose, required inputs, assumptions, the full prompt text, adaptation notes, and common failure modes.

## How to use a prompt card

1. **Prepare your data.** Export the required query from the Viva Insights Analyst portal.
2. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or your preferred AI assistant.
3. **Copy the prompt.** Find the relevant card below and copy the full text from the **Prompt** section.
4. **Paste and run.** Paste the prompt into the agent. Point it at your data file.
5. **Review and adapt.** Check the output against the **Common failure modes** section. Use the **Adaptation notes** to customize.

> **Tip:** You can prepend context to any prompt. For example: _"My CSV is at `./data/person-query.csv`. The Organization column is called `Org`."_ followed by the full prompt text.

## Available prompts

### Copilot Adoption

| Prompt Card | Description |
|---|---|
| [Dashboard Overview](#dashboard-overview--copilot-adoption) | Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns. |
| [Executive Summary](#executive-summary--copilot-adoption) | Produce a concise executive memo summarizing Copilot adoption metrics for VP/C-suite audiences. |
| [Segmentation & Churn](#segmentation--churn-analysis--copilot-adoption) | Classify users into usage segments, track transitions, and calculate churn rates. |
| [ROI Estimation](#roi-estimation--copilot-adoption) | Estimate return on investment for Copilot by quantifying time savings and license costs. |

### Purview Augmentation

| Prompt Card | Description |
|---|---|
| [Agent Usage Analysis](#agent-usage-analysis--purview-audit-logs) | Analyze Copilot agent and extension usage patterns from Purview audit logs. |
| [Audit Log Parsing](#audit-log-parsing--purview-audit-logs) | Parse and clean raw Purview audit log exports into analysis-ready flat tables. |

---

## Dashboard Overview — Copilot Adoption

### Purpose

Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns.

### Audience

People analytics leads, IT deployment managers

### When to use

After exporting a person query with Copilot activity metrics spanning at least 8 weeks of data.

### Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended
- HR attributes for segmentation

### Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

### Recommended output

A self-contained static HTML file with embedded charts, suitable for sharing via email or SharePoint.

### Prompt

```
You are a people analytics engineer. Your task is to build a self-contained static HTML dashboard
that visualizes Microsoft Copilot adoption from a Viva Insights person query export.

DATA LOADING AND VALIDATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python).
   This handles variable name cleaning and type parsing automatically.
2. Ensure PersonId is treated as a string and MetricDate is parsed as a date type.
3. Run `extract_hr(df)` from the `vivainsights` library to identify the available HR / organizational
   attribute columns in the data. Use the returned list of HR attributes for all segmentation
   breakdowns instead of hard-coding column names like Organization, FunctionType, or LevelDesignation.
4. Verify the panel structure: each row should represent a unique PersonId × MetricDate combination.
   If there are duplicates, flag them and keep the first occurrence.
5. Print the shape of the data, the date range covered, and the number of unique persons.
6. List all column names so I can verify the Copilot metric columns and HR attribute columns
   match what is expected. Identify Copilot metric columns by checking for columns containing
   the word "Copilot" in their name. Reference the taxonomy at
   https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv
   to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the
   primary activity metric (it captures all Copilot usage across apps).

IDENTIFYING LICENSED USERS
6. A user is considered "Copilot-licensed" in a given week if they have a non-null, non-zero value
   in at least one Copilot metric column for that week. Create a boolean column `is_licensed` to
   flag these rows.
7. Also flag "active" users: licensed users who have Total_Copilot_actions_taken > 0 (the primary
   activity metric) in that week. Create a boolean column `is_active`.
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
13. Compute a "top users" table: for each PersonId, calculate total Total_Copilot_actions_taken across all weeks,
    total active weeks, and average Total_Copilot_actions_taken per active week. Rank by total actions descending.
    Keep the top 20. Include their HR attributes for context.

SUMMARY STATISTICS PANEL
14. Calculate overall summary statistics for the dashboard header:
    a. Latest week adoption rate
    b. Trend direction: compare the last 4 weeks' average adoption rate to the prior 4 weeks
    c. Total unique licensed users across the entire period
    d. Total unique active users across the entire period
    e. Average Total_Copilot_actions_taken per active user per week (grand mean)
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
       - Line chart: Mean Total_Copilot_actions_taken per active user over time
       - Line chart: Mean Copilot_Assisted_Hours per active user over time
    d. SEGMENTATION CHARTS:
       - Grouped bar chart: Adoption rate by Organization (latest 4-week average)
       - Grouped bar chart: Adoption rate by FunctionType (latest 4-week average)
       - Grouped bar chart: Adoption rate by LevelDesignation (latest 4-week average)
       - Heatmap: Adoption rate by Organization × week (if number of orgs <= 15)
    e. TOP USERS TABLE: HTML table of top 20 users from step 13.
    f. METHODOLOGY NOTE: Brief paragraph explaining how adoption rate is calculated, what
       "licensed" and "active" mean, and the data source.

17. Style the HTML with a clean, professional design. Use a sans-serif font, consistent color
    palette, and adequate whitespace. The dashboard should look presentable when opened in a browser.

18. Save the HTML file to the working directory with a descriptive filename like
    "copilot_adoption_dashboard_YYYYMMDD.html".

IMPORTANT NOTES
- Do NOT create interactive plots that require a running server (no plotly, no bokeh server).
  Static images embedded as base64 are preferred.
- Handle missing values gracefully: NaN in Copilot columns means the user is unlicensed that week.
- If any HR attribute column is missing from the data, skip that segmentation chart and note it.
- Use the `vivainsights` package (R or Python) for data loading (`import_query()`) and HR attribute
  discovery (`extract_hr()`). Use pandas/matplotlib or base R/ggplot2 for charting and analysis.
- All charts should have clear titles, axis labels, and legends.
- If any segment has fewer than 5 users, suppress it from charts to protect privacy.
```

### Adaptation notes

- The `extract_hr()` function auto-detects organizational attributes, so you typically do not need to manually specify HR column names. If your data has custom attribute columns that `extract_hr()` does not detect, add an instruction specifying them.
- If your data is at person-day granularity, add an instruction: _"Aggregate person-day data to person-week by summing Copilot metrics per PersonId per week."_
- For smaller organizations, increase the privacy threshold (e.g., from 5 to 10 users per segment) or remove segmentation breakdowns entirely.
- Add custom Copilot metrics by extending the list in step 5 (e.g., `Copilot_Edited_Hours`, `Copilot_Rewritten_Hours`).
- If you prefer R over Python, add _"Use R with ggplot2 and R Markdown"_ at the start of the prompt.

### Common failure modes

- **Agent assumes all rows have Copilot data.** The prompt explicitly separates licensed vs. unlicensed users, but some agents may skip this. Verify that the `is_licensed` flag is computed before any metric calculations.
- **Agent creates interactive plots that need a server.** The prompt specifies static HTML with base64-encoded images. If the agent uses plotly or bokeh, ask it to switch to matplotlib/seaborn or export static images.
- **Agent ignores the panel structure and double-counts users.** Ensure adoption rate is calculated per week using distinct `PersonId` counts, not row counts.
- **Metric column names differ between tenants.** Copilot metric columns generally contain the word "Copilot" but do not always start with `Copilot_`. Reference the [metrics taxonomy](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv) to validate detected columns.
- **HTML file is not self-contained.** Check that the output HTML opens correctly with no external dependencies. If charts appear broken, ensure base64 encoding was applied.

---

## Executive Summary — Copilot Adoption

### Purpose

Generate a concise executive summary memo that distills Copilot adoption metrics into key findings, trend analysis, and actionable recommendations for senior leadership.

### Audience

VP/C-suite executives, senior leadership team, IT steering committee

### When to use

When you need to communicate Copilot adoption progress to senior leadership — typically on a monthly or quarterly cadence after collecting at least 8 weeks of person query data.

### Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended (12+ weeks preferred for trend analysis)
- HR attributes for organizational breakdowns

### Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

### Recommended output

A 1–2 page executive summary in HTML or Markdown, formatted as a professional memo suitable for distribution to VP/C-suite audiences.

### Prompt

```
You are a senior people analytics consultant. Your task is to generate a polished executive summary
memo about Microsoft Copilot adoption, based on a Viva Insights person query export. The memo must
be suitable for a VP or C-suite audience — concise, insight-driven, and action-oriented.

DATA LOADING AND PREPARATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python).
   This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their
   name. Reference the taxonomy at
   https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv
   to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary
   activity metric. Print the detected columns and the date range for verification.
3. Run `extract_hr(df)` from the `vivainsights` library to identify available HR / organizational
   attribute columns. Use the returned list for all organizational breakdowns instead of hard-coding
   column names.
4. Classify each person-week row:
   - "Licensed": has at least one non-null, non-zero Copilot metric value.
   - "Active": is licensed AND has Total_Copilot_actions_taken > 0 (the primary activity metric).
   - "Unlicensed": all Copilot metric values are null or zero.
5. If any expected HR attribute columns are not found by `extract_hr()`, note which ones are
   unavailable and proceed with what is present.

HEADLINE METRICS (compute these for the memo)
5. Current adoption rate: In the most recent complete week, what percentage of licensed users were
   active? Report as "X% of licensed users actively used Copilot in the week of [date]."
6. Adoption trend: Compare the average weekly adoption rate over the last 4 complete weeks to the
   prior 4 weeks. Calculate the percentage-point change. Classify as "improving", "stable" (within
   ±2pp), or "declining".
7. Usage intensity: Average Total_Copilot_actions_taken per active user per week over the last 4 weeks.
   Compare to the prior 4-week period. Report the direction and magnitude of change.
8. Breadth of adoption: Total unique users who have been active at least once in the last 4 weeks
   as a percentage of all licensed users in that period.
9. Top 3 organizations by adoption rate (latest 4-week average). Bottom 3 organizations by adoption
   rate. Only include organizations with at least 10 licensed users.
10. Copilot_Assisted_Hours: Average weekly assisted hours per active user (last 4 weeks). Convert
    to a relatable figure (e.g., "X minutes per user per week").

KEY FINDINGS (synthesize the metrics into 3-5 bullet points)
11. Identify the most important story in the data. Is adoption growing? Plateauing? Declining?
    Which groups are leading and which are lagging?
12. Look for notable patterns:
    - Is there a gap between license deployment and actual usage?
    - Are certain functions or levels adopting faster than others?
    - Is usage intensity growing even if adoption rate is flat (deepening engagement)?
    - Are there signs of churn (users who were active but stopped)?

RECOMMENDATIONS (generate 2-4 actionable recommendations)
13. Based on the findings, generate specific, actionable recommendations. Examples:
    - "Target [specific org] for enablement workshops, as their adoption rate is X pp below average."
    - "Investigate why [function type] has low adoption despite high license deployment."
    - "Celebrate and share practices from [top org], which has achieved X% adoption."
    - "Consider a re-engagement campaign for the estimated N users who were active in weeks 1-4
       but inactive in the most recent 4 weeks."
    Recommendations should reference specific numbers from the data.

AREAS OF CONCERN (flag 1-3 risks or issues)
14. Flag potential concerns:
    - Low overall adoption relative to license spend
    - Declining trends in any major segment
    - Segments with fewer than 5 active users (note privacy limitations)
    - Data quality issues (e.g., missing weeks, unexpected nulls)

MEMO GENERATION
15. Generate the executive summary as a well-formatted HTML file (or Markdown if specified). Use
    the following structure:

    HEADER:
    - Title: "Copilot Adoption: Executive Summary"
    - Subtitle: "Reporting period: [start date] to [end date]"
    - Date generated: [today's date]

    SECTION 1: "At a Glance" (summary statistics in a clean table or card layout)
    - Current adoption rate
    - Adoption trend (with directional arrow or indicator)
    - Active users (last 4 weeks)
    - Average Copilot actions per active user per week
    - Average Copilot-assisted time per active user per week

    SECTION 2: "Key Findings" (3-5 concise bullet points, each 1-2 sentences)

    SECTION 3: "Organizational Breakdown" (a small table showing top and bottom orgs)

    SECTION 4: "Recommendations" (2-4 numbered recommendations, each 2-3 sentences)

    SECTION 5: "Areas of Concern" (1-3 flagged items)

    SECTION 6: "Methodology" (brief paragraph)
    - Explain: data source (Viva Insights person query), how "licensed" and "active" are defined,
      the time period used for trend calculations, and any caveats (correlation ≠ causation,
      missing data handling).

16. Style guidelines for the memo:
    - Use professional, executive-friendly language. Avoid jargon.
    - Lead with the most important finding.
    - Every metric should be accompanied by context (e.g., "up from X% last month").
    - Use bold text for key numbers.
    - Keep the total length to 1-2 pages when printed.
    - Use a clean, professional design with a sans-serif font if HTML.

17. Save the output file with a descriptive name like "copilot_executive_summary_YYYYMMDD.html".

IMPORTANT NOTES
- Do NOT fabricate numbers. Every statistic must be computed directly from the data.
- Handle missing data gracefully: if Copilot metric columns are null for a user-week, that user
  is unlicensed for that week — do not treat nulls as zeros.
- If the dataset has fewer than 8 weeks, note this limitation and adjust trend calculations
  accordingly (e.g., compare last 2 weeks to prior 2 weeks).
- Suppress any organizational segment with fewer than 5 people to protect employee privacy.
- The tone should be balanced — highlight successes but do not oversell. Flag genuine concerns.
```

### Adaptation notes

- If your leadership prefers Markdown over HTML, add _"Output as a Markdown file"_ at the start of the prompt.
- Adjust the organization size threshold (default: 10 licensed users) based on your company size. For very large organizations, increase to 20+.
- Add company-specific context by prepending: _"Our Copilot deployment started on [date] and currently covers [N] licenses across [divisions]."_
- If you want the memo to reference specific business goals (e.g., "target 60% adoption by Q3"), include that context before the prompt.
- Change the trend comparison window (default: 4 weeks vs. prior 4 weeks) to match your reporting cadence.

### Common failure modes

- **Agent fabricates insights not supported by data.** The prompt instructs the agent to compute every statistic from the data, but always verify headline numbers independently.
- **Agent uses overly technical language.** The prompt specifies executive-friendly language, but review the output for jargon like "person-week panel structure" and simplify.
- **Agent double-counts users across weeks for "total active users."** Ensure breadth metrics use distinct `PersonId` counts, not row counts.
- **Agent treats missing Copilot values as zero.** This would inflate denominators and deflate adoption rates. Verify the licensed user logic.
- **Recommendations are too generic.** If the agent produces vague advice ("increase adoption"), ask it to regenerate with specific org names and numbers from the data.

---

## Segmentation & Churn Analysis — Copilot Adoption

### Purpose

Classify Copilot users into behavioral segments based on usage intensity, track how users move between segments over time, and quantify churn — the rate at which users disengage from Copilot after initial adoption.

### Audience

People analytics leads, Copilot program managers, digital adoption teams

### When to use

After at least 8–12 weeks of person query data have been collected, when you need to understand not just whether people are using Copilot, but how their usage patterns evolve over time.

### Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data (12+ weeks recommended for meaningful churn analysis)
- HR attributes for segmentation breakdowns

### Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

### Recommended output

An HTML report or Jupyter/R notebook containing segment distribution charts, transition matrices, churn curves, and at-risk group analysis.

### Prompt

```
You are a behavioral analytics specialist. Your task is to perform a user segmentation and churn
analysis on Microsoft Copilot usage data from a Viva Insights person query export. The goal is to
classify users into usage-based segments, track transitions between segments, and quantify churn.

DATA LOADING AND PREPARATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python).
   This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their
   name. Reference the taxonomy at
   https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv
   to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary
   activity metric. Print detected columns and the date range.
3. Run `extract_hr(df)` from the `vivainsights` library to identify available HR / organizational
   attribute columns. Use the returned list for all segmentation breakdowns instead of hard-coding
   column names.
4. Classify each person-week:
   - "Licensed": at least one non-null, non-zero Copilot metric value.
   - "Unlicensed": all Copilot metrics are null or zero.
5. Filter to only licensed person-weeks for the segmentation analysis.
6. Fill any remaining NaN values in Copilot metric columns with 0 for licensed users (a licensed
   user with a null action count in one metric likely had zero usage of that specific feature).

USER SEGMENTATION
6. Use `identify_usage_segments()` from the `vivainsights` library to classify users into segments.
   This function uses `Total_Copilot_actions_taken` and classifies users based on both usage
   volume and consistency (habit formation over 12 weeks), producing segments such as Power Users
   and Habitual Users. It is preferred over manual percentile-based segmentation.

7. Also compute a "stable segment" for each user based on their most frequent weekly segment
   over the entire period (mode). This gives a single label per person.

SEGMENT DISTRIBUTION OVER TIME
8. For each week, count the number of users in each segment. Create:
   a. A stacked area chart showing the four segments over time (absolute counts).
   b. A stacked area chart showing the four segments as percentages of total licensed users.
   c. A summary table of the latest week's segment distribution.

SEGMENT BREAKDOWN BY HR ATTRIBUTES
9. For each HR attribute (Organization, FunctionType, LevelDesignation), compute the segment
   distribution using the "stable segment" label. Create:
   a. A stacked bar chart showing segment proportions by each HR attribute value.
   b. Only include HR attribute values with at least 5 licensed users.
10. Identify groups with the highest proportion of Power Users and groups with the highest
    proportion of Inactive users. Flag these as "leading" and "at-risk" groups respectively.

TRANSITION ANALYSIS
11. Build a week-over-week transition matrix. For each consecutive pair of weeks, count how many
    users moved from each segment to every other segment. Aggregate across all week transitions.
12. Normalize the matrix to show transition probabilities: given a user is in segment X this week,
    what is the probability they are in segment Y next week?
13. Visualize the transition matrix as a heatmap with annotations showing the probabilities.
14. Highlight key transitions of interest:
    - "Activation": Inactive → any active segment
    - "Churning": any active segment → Inactive
    - "Deepening": Light User → Regular User or Regular User → Power User
    - "Declining": Power User → Regular User or Regular User → Light User

CHURN ANALYSIS
15. Define "churn" as: a user who was active (in any active segment) for at least 2 consecutive
    weeks, then became Inactive for 2 or more consecutive weeks. The churn date is the first
    week of inactivity.
16. For each week, calculate:
    a. The number of users who churned that week.
    b. The churn rate: churned users / active users in the prior week.
17. Create a line chart of weekly churn rate over time.
18. Build a "time to churn" distribution: for users who churned, how many weeks were they active
    before churning? Show as a histogram.
19. Compute overall churn statistics:
    a. Total users who ever churned
    b. Percentage of ever-active users who churned
    c. Median active weeks before churn
    d. Re-activation rate: of those who churned, what percentage became active again later?

AT-RISK USER IDENTIFICATION
20. Identify currently at-risk users: licensed users who were active 4+ weeks ago but have been
    Inactive for the last 2 consecutive weeks (not yet meeting the full churn definition).
21. Summarize at-risk users by HR attribute: which organizations, functions, or levels have the
    highest concentration of at-risk users?
22. Create a table of at-risk user counts by Organization and FunctionType.

REPORT GENERATION
23. Compile all outputs into a single HTML report or notebook with these sections:
    a. "Segment Definitions" — explain the four segments and thresholds used.
    b. "Segment Distribution" — charts from step 8 and summary table.
    c. "Segment Breakdown by Group" — charts from step 9, with leading/at-risk group callouts.
    d. "Transition Analysis" — heatmap from step 13, key transition highlights from step 14.
    e. "Churn Analysis" — churn rate trend, time-to-churn histogram, summary statistics.
    f. "At-Risk Users" — summary table from step 22.
    g. "Key Takeaways" — 3-5 bullet points synthesizing the most important findings.
    h. "Methodology" — brief description of data source, segment definitions, churn definition.

24. Use static charts (matplotlib/seaborn for Python or ggplot2 for R). Embed as base64 images
    if generating a standalone HTML file.
25. Save the report with a descriptive filename like "copilot_segmentation_churn_YYYYMMDD.html".

IMPORTANT NOTES
- Segments are assigned by `identify_usage_segments()` based on both usage volume and consistency
  (habit formation over 12 weeks). Print the segment definitions used so the reader can interpret them.
- Do NOT count unlicensed users in any denominator. They are excluded from the analysis entirely.
- Suppress any HR attribute segment with fewer than 5 users in the visualizations.
- The transition matrix should only include users who appear in both consecutive weeks.
- If the dataset has fewer than 8 weeks, note that churn analysis may not be reliable and
  adjust the churn definition (e.g., reduce the "2 consecutive inactive weeks" requirement).
```

### Adaptation notes

- The `identify_usage_segments()` function provides sensible default thresholds. If the segments do not produce meaningful groupings for your data, consult the function documentation for available parameters to customize segmentation criteria.
- If your organization has a specific definition of "churn" (e.g., 4 weeks of inactivity instead of 2), modify the churn definition in step 15.
- For organizations with multiple Copilot products, consider segmenting by product (e.g., Copilot in Teams vs. Copilot in Word) by using product-specific metrics.
- Add a "New User" segment by tracking each user's first active week and analyzing the onboarding trajectory separately.
- If you want to predict future churn, add a note requesting a logistic regression or survival analysis model.

### Common failure modes

- **Agent uses absolute thresholds instead of percentiles.** The prompt specifies percentile-based thresholds, but an agent may default to arbitrary cutoffs. Verify that it calculates and prints the actual thresholds.
- **Agent counts unlicensed users as "Inactive."** The Inactive segment should only include licensed users with zero activity — not unlicensed users. Verify the filtering step.
- **Transition matrix includes users who appear in only one week.** Ensure the matrix only counts users present in both the "from" and "to" weeks.
- **Agent conflates week-level segments with user-level segments.** The report uses both: per-week segments for trends and a "stable segment" (mode) for HR attribute breakdowns. Ensure both are present.
- **Churn definition is too aggressive.** Two weeks of inactivity may include holidays or vacation. Consider extending to 3–4 weeks for more conservative churn estimates.

---

## ROI Estimation — Copilot Adoption

### Purpose

Estimate the return on investment (ROI) for Microsoft Copilot by quantifying time savings, translating them into monetary value, and comparing against license costs to build a defensible business case.

### Audience

CFO/finance teams, IT leadership, Copilot program sponsors, business case reviewers

### When to use

When stakeholders need a quantified business justification for Copilot investment — typically during budget reviews, renewal decisions, or expansion proposals. Requires at least 8 weeks of data for stable estimates.

### Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended
- Configurable assumptions: hourly cost of employee time (default: $75/hour), annual Copilot license cost per user (default: $360/year)
- Optional: collaboration metrics for licensed vs. unlicensed comparison (e.g., `Collaboration_Hours`, `Meeting_Hours`, `Email_Hours`)

### Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- `Copilot_Assisted_Hours` (or similar) represents time where Copilot contributed to the user's work
- The `vivainsights` R or Python package is available in the environment

### Recommended output

An ROI summary report in HTML or Markdown, containing a value framework, sensitivity analysis, and methodology notes suitable for inclusion in a business case document.

### Prompt

```
You are a people analytics consultant specializing in technology ROI. Your task is to produce a
defensible ROI estimate for Microsoft Copilot based on a Viva Insights person query export. The
output should be suitable for inclusion in a business case presented to finance and IT leadership.

CONFIGURABLE ASSUMPTIONS (define as variables at the top of the script so they are easy to adjust)
- HOURLY_RATE = 75  # Fully loaded cost per employee hour in USD
- ANNUAL_LICENSE_COST = 360  # Annual Copilot license cost per user in USD
- WEEKS_IN_YEAR = 52
- ANALYSIS_WEEKS = 4  # Number of recent weeks to use for annualized projections

DATA LOADING AND PREPARATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python).
   This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their
   name. Reference the taxonomy at
   https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv
   to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary
   activity metric. Print detected columns.
3. Run `extract_hr(df)` from the `vivainsights` library to identify available HR / organizational
   attribute columns. Use the returned list for all organizational breakdowns instead of hard-coding
   column names.
4. Classify each person-week:
   - "Licensed": at least one non-null, non-zero Copilot metric value.
   - "Active": licensed AND Total_Copilot_actions_taken > 0.
   - "Unlicensed": all Copilot metric values are null or zero.
5. Print: total persons, licensed persons, active persons (ever active), date range.

TIME SAVINGS ESTIMATION
5. For each active person-week, extract Copilot_Assisted_Hours (the primary time-savings metric).
   If this column is not available, fall back to estimating from Copilot_Summarized_Hours or a
   fraction of Total_Copilot_actions_taken (note the assumption if using a proxy).
6. Compute per-user weekly time savings:
   a. Mean Copilot_Assisted_Hours per active user per week (over the last ANALYSIS_WEEKS weeks).
   b. Median Copilot_Assisted_Hours per active user per week.
   c. Total Copilot_Assisted_Hours across all active users per week.
7. Annualize the time savings:
   a. Per-user annualized hours saved = mean weekly hours × WEEKS_IN_YEAR.
   b. Total annualized hours saved = total weekly hours × WEEKS_IN_YEAR.
   Print both figures.

MONETARY VALUE ESTIMATION
8. Convert time savings to monetary value:
   a. Per-user annual value = per-user annualized hours × HOURLY_RATE.
   b. Total annual value = total annualized hours × HOURLY_RATE.
9. Compute total annual license cost:
   a. Total licensed users (distinct PersonId with is_licensed in the latest week).
   b. Total annual cost = total licensed users × ANNUAL_LICENSE_COST.
10. Compute ROI metrics:
    a. Net annual value = total annual value - total annual cost.
    b. ROI ratio = total annual value / total annual cost.
    c. Break-even hours: the minimum weekly Copilot_Assisted_Hours per user needed for the
       license to pay for itself = (ANNUAL_LICENSE_COST / WEEKS_IN_YEAR) / HOURLY_RATE.
    d. Percentage of licensed users exceeding the break-even threshold.

SENSITIVITY ANALYSIS
11. Compute ROI under different assumptions. Create a table with rows for different hourly rates
    ($50, $75, $100, $125) and columns for different utilization scenarios:
    a. "Current active users only" — value from only currently active users.
    b. "50% more adoption" — if 50% more licensed users become active at the current average.
    c. "Full adoption" — if all licensed users become active at the current average.
12. This table shows leadership how ROI improves with higher adoption.

LICENSED VS. UNLICENSED COMPARISON (if collaboration metrics are available)
13. If collaboration metrics are present in the data (Collaboration_Hours, Meeting_Hours,
    Email_Hours, Focus_Hours, or similar), compare licensed vs. unlicensed users:
    a. For the most recent ANALYSIS_WEEKS weeks, compute the mean of each collaboration metric
       for licensed-active users, licensed-inactive users, and unlicensed users.
    b. Present as a comparison table.
    c. Include a caveat: this is an observational comparison, NOT a causal estimate. Differences
       may reflect selection effects (e.g., more collaborative employees may adopt Copilot first).
14. If collaboration metrics are not present, skip this section and note its absence.

VALUE BY SEGMENT
15. Break down the annual value estimate by Organization (or the primary HR attribute):
    a. For each organization, compute: licensed users, active users, total Copilot_Assisted_Hours,
       annualized value, license cost, net value.
    b. Present as a sorted table (by net value descending).
    c. Identify organizations with negative net value (license cost exceeds estimated value).

REPORT GENERATION
16. Compile into a professional HTML report with these sections:

    a. "Executive Summary" (3-4 sentences)
       - Lead with the headline ROI number.
       - State the total estimated annual value and cost.
       - Note the break-even threshold and current adoption rate.

    b. "ROI Framework" (a clean summary table)
       | Metric | Value |
       |--------|-------|
       | Licensed users | N |
       | Active users (last 4 weeks) | N |
       | Avg. weekly time saved per active user | X.X hours |
       | Annualized value of time saved | $X |
       | Annual license cost | $X |
       | Net annual value | $X |
       | ROI ratio | X.Xx |
       | Break-even threshold | X.X hours/week |
       | Users above break-even | X% |

    c. "Sensitivity Analysis" — the table from step 11.

    d. "Value by Organization" — the table from step 15.

    e. "Licensed vs. Unlicensed Comparison" (if applicable) — table from step 13.

    f. "Methodology & Caveats"
       - Data source and time period.
       - How time savings are measured (Copilot_Assisted_Hours).
       - The hourly rate assumption and its basis.
       - Explicit caveat: "This analysis estimates the potential value of time savings attributed
         to Copilot-assisted work. It does not establish a causal relationship between Copilot
         usage and productivity gains. Actual ROI depends on how saved time is reallocated.
         Correlation between Copilot usage and collaboration patterns may reflect selection
         effects rather than causal impact."
       - Note: time savings represent Copilot-assisted work, not necessarily net new time freed.
       - Privacy note: segments with fewer than 5 users are suppressed.

    g. "Assumptions" — list all configurable assumptions and their values.

17. Use a clean, professional design. Format currency values with dollar signs and commas.
    Use bold text for headline numbers.
18. Save as "copilot_roi_estimation_YYYYMMDD.html".

IMPORTANT NOTES
- Do NOT overstate the ROI. This is an estimation framework, not a proven causal impact.
  Every section should include appropriate caveats.
- Handle missing Copilot_Assisted_Hours carefully. If null for a licensed user, treat as 0
  (they are licensed but did not use Copilot-assisted features that week).
- The break-even calculation is a useful framing device — it tells leadership the minimum
  usage needed to justify the license cost.
- If the ROI is negative, report it honestly and frame recommendations around increasing adoption.
- Suppress segments with fewer than 5 users.
```

### Adaptation notes

- **Adjust the hourly rate** to match your organization's fully loaded labor cost. Many organizations use $50–$150/hour depending on geography and role mix. If available, use different rates per `LevelDesignation` or `FunctionType`.
- **Adjust the license cost** based on your actual agreement (E3/E5 add-on vs. standalone Copilot license).
- **Add qualitative benefits** by extending the report with a section on "Unquantified Benefits" (e.g., employee satisfaction, reduced context-switching, faster onboarding).
- **Time horizon:** The default projects current weekly usage to an annual figure. For a multi-year business case, add a growth assumption (e.g., 10% quarterly adoption increase).
- If you have time-series data from before and after Copilot deployment, you can add a pre/post comparison section by modifying the prompt.

### Common failure modes

- **Agent treats Copilot_Assisted_Hours as "time saved."** Clarify that Copilot_Assisted_Hours represents time where Copilot assisted the user — the actual time saved may be a fraction of this. Consider adding a "realization factor" (e.g., 50%) for more conservative estimates.
- **Agent ignores selection bias in licensed vs. unlicensed comparison.** The prompt includes a causal caveat, but verify it appears prominently in the output.
- **Agent annualizes from a single week.** The prompt uses the last `ANALYSIS_WEEKS` (4) weeks for stability, but ensure the agent averages across these weeks rather than extrapolating from one.
- **Agent reports ROI without caveats.** Senior leadership will scrutinize the methodology. Ensure every value estimate is accompanied by its assumptions and limitations.
- **Currency formatting issues.** Verify that dollar amounts are formatted with commas and two decimal places for large figures.

---

## Agent Usage Analysis — Purview Audit Logs

### Purpose

Analyze Copilot agent and extension usage patterns from Microsoft Purview audit logs to understand which Copilot features and agents are being used, by whom, and how usage trends over time.

### Audience

IT administrators, Copilot program managers, security and compliance teams, people analytics leads

### When to use

When you have access to Purview audit log exports containing Copilot-related events and want to understand Copilot agent/extension adoption patterns beyond what Viva Insights person query data provides. This is especially useful for tracking specific Copilot interaction types (e.g., chat vs. summarization vs. agent invocations).

### Required inputs

- Purview audit log export (CSV or JSON format)
- Expected fields include: `UserId`, `Operation`, `Workload`, `CreationTime`, and optionally `AuditData` (a JSON field containing event details)
- At least 4 weeks of audit data recommended for trend analysis

### Assumptions

- The audit log export contains Copilot-related events (operations may include terms like "CopilotInteraction", "CopilotQuery", "AIAppInteraction", or similar — exact names vary by tenant and configuration)
- `UserId` identifies the user (may be a UPN/email or anonymized identifier)
- `CreationTime` is a timestamp for each event
- `Workload` indicates the Microsoft 365 application (e.g., "MicrosoftCopilot", "Teams", "Exchange", "SharePoint")
- `Operation` indicates the type of Copilot action
- The audit log schema may vary between tenants — field names and event types should be verified before analysis

### Recommended output

An exploratory HTML report or Jupyter/R notebook with usage trends, operation breakdowns, and user activity distributions.

### Prompt

```
You are a data analyst working with Microsoft Purview audit logs. Your task is to analyze Copilot
agent and extension usage patterns from a Purview audit log export. Because the Purview audit
schema can vary between tenants, this analysis should be exploratory — start by understanding
the data structure before computing metrics.

IMPORTANT CAVEAT: Purview audit log schemas are not standardized across tenants. Field names,
operation types, and event structures may differ from what is described below. The first phase
of this analysis must be data exploration and validation.

DATA LOADING AND EXPLORATION
1. Load the audit log file. Support both CSV and JSON formats — detect the format automatically.
   If the file is JSON, it may be a JSON array or newline-delimited JSON (one object per line).
   If CSV, parse normally with pandas or readr.
2. Print the column names, data types, and the first 5 rows to understand the schema.
3. Print the number of total records and the date range (from CreationTime or the equivalent
   timestamp field).
4. If there is an "AuditData" column that contains JSON strings, note it but do NOT parse it
   yet — we will handle it in a later step if needed.

FIELD IDENTIFICATION
5. Identify key fields by searching column names for common patterns:
   - User identifier: look for "UserId", "UserKey", "User", "UPN"
   - Timestamp: look for "CreationTime", "CreationDate", "Timestamp", "EventTime"
   - Operation: look for "Operation", "Action", "EventType", "Activity"
   - Workload/Application: look for "Workload", "Application", "AppName", "Product"
   Print the identified field mappings and ask for confirmation if ambiguous.
6. Parse the timestamp field as a datetime type. Extract date (day) and week columns.

COPILOT EVENT FILTERING
7. Explore the unique values in the Operation and Workload columns. Print value counts for both.
8. Filter for Copilot-related events. Use a broad filter first:
   - Operation values containing "Copilot", "AI", "Agent", "GPT", "Assist", "Summarize" (case-insensitive)
   - Workload values containing "Copilot", "Microsoft365", "M365" (case-insensitive)
   Print the number of matching events and the operation/workload values that matched.
9. If the filtered dataset is empty, expand the filter or report that no Copilot events were
   found and list all unique Operation and Workload values for manual inspection.
10. If an AuditData column exists and the initial filtering is too broad, parse the JSON in
    AuditData for a sample of 100 rows and look for additional fields that indicate Copilot
    usage (e.g., "AppName", "CopilotEventType", "AgentName", "ExtensionName").

USAGE METRICS
11. Using the filtered Copilot events, compute:
    a. Total events per day and per week.
    b. Unique users per day and per week.
    c. Events per user per week (distribution: mean, median, p25, p75).
12. Create trend charts:
    a. Line chart: daily event count over time (with a 7-day rolling average overlay).
    b. Line chart: weekly unique users over time.
    c. Line chart: weekly events per user over time (mean).

OPERATION TYPE BREAKDOWN
13. For each unique Operation value in the Copilot-filtered data:
    a. Count total events.
    b. Count unique users.
    c. Compute events per user.
14. Create:
    a. A horizontal bar chart of event counts by Operation (top 15 operations).
    b. A horizontal bar chart of unique users by Operation (top 15).
15. If Workload is available and has multiple values, create:
    a. A grouped bar chart showing event counts by Workload.
    b. A heatmap of Operation × Workload (event counts).

USER ACTIVITY DISTRIBUTION
16. Compute a per-user activity summary over the entire period:
    a. Total events per user.
    b. Active days per user.
    c. Active weeks per user.
    d. Most common Operation per user.
17. Create a histogram of total events per user (log scale if distribution is highly skewed).
18. Classify users into activity tiers:
    - "Heavy": top 10% by total events
    - "Moderate": 10th-50th percentile
    - "Light": bottom 50%
    Print the count and percentage in each tier.
19. If any identifier for user department or group is available (from AuditData or a separate
    mapping file), break down activity tiers by group.

AGENT/EXTENSION ANALYSIS (if data is available)
20. If the AuditData or other fields contain information about specific Copilot agents or
    extensions (e.g., "AgentName", "ExtensionId", "PluginName"), extract and analyze:
    a. Top agents/extensions by usage (event count and unique users).
    b. Trend of agent/extension usage over time.
    c. Agent-specific user engagement (events per user per agent).
    If no agent/extension information is found, skip this section and note its absence.

REPORT GENERATION
21. Compile into an HTML report or notebook with these sections:
    a. "Data Overview" — schema summary, date range, total events, Copilot filter criteria used.
    b. "Usage Trends" — trend charts from step 12.
    c. "Operation Breakdown" — charts from steps 14-15.
    d. "User Activity Distribution" — histogram and tier summary from steps 17-18.
    e. "Agent/Extension Usage" — analysis from step 20 (if available).
    f. "Key Findings" — 3-5 bullet points summarizing the most notable patterns.
    g. "Data Notes" — document any field mapping decisions, filter criteria, and schema
       observations for reproducibility.

22. Use static charts (matplotlib/seaborn or ggplot2). Embed as base64 for standalone HTML.
23. Save as "purview_copilot_agent_analysis_YYYYMMDD.html".

IMPORTANT NOTES
- This is an EXPLORATORY analysis. The Purview schema is not standardized — always start by
  inspecting the data rather than assuming specific field names or values.
- Print intermediate outputs (unique values, sample rows) so I can verify the field mappings.
- If the dataset is very large (>1M rows), sample for exploration but use the full data for
  final metrics.
- User identifiers in Purview logs may be email addresses/UPNs. Do not expose raw email
  addresses in the report — if possible, hash or truncate them, or use only aggregate statistics.
- Some operations may be system-generated rather than user-initiated. Look for patterns that
  distinguish user actions from system events.
```

### Adaptation notes

- **Field names will vary.** The most critical adaptation step is verifying your audit log's actual field names. Run the exploration steps first, then update the prompt with your specific field mappings.
- **Copilot event identification** depends on your tenant's audit configuration. The filter in step 8 casts a wide net — narrow it after inspecting the Operation and Workload values in your data.
- **Joining with Viva Insights data:** If you want to enrich the audit log analysis with HR attributes from Viva Insights, add a join step using `UserId` (after normalizing the identifier format between the two sources).
- **Privacy:** Purview logs may contain PII (email addresses). Ensure your analysis complies with your organization's data handling policies. Add anonymization steps if needed.
- **Large datasets:** For audit logs spanning months, consider filtering to a specific date range before loading the full file to reduce memory usage.

### Common failure modes

- **Agent assumes specific Purview field names that don't exist.** The prompt starts with data exploration, but some agents may skip ahead. Ensure the schema inspection step runs first.
- **Agent fails to parse nested JSON in AuditData.** This column often contains a JSON string with nested objects. The agent may need guidance on which nested fields to extract.
- **Agent exposes raw email addresses in the report.** Purview logs use UPNs. Instruct the agent to anonymize or aggregate to avoid PII exposure.
- **Copilot events are not clearly labeled.** In some tenants, Copilot interactions are logged under generic operation names. You may need to inspect AuditData contents to identify them.
- **Agent treats all events equally.** Some operations may be system-level events or duplicates. Review the Operation values to determine which represent genuine user interactions.

---

## Audit Log Parsing — Purview Audit Logs

### Purpose

Parse and clean raw Microsoft Purview audit log exports into a flat, analysis-ready dataset. This is a data engineering prompt that prepares audit log data for downstream analytics (such as the [Agent Usage Analysis](#agent-usage-analysis.md) prompt).

### Audience

Data engineers, people analytics teams, IT administrators preparing data for analysis

### When to use

Immediately after exporting raw audit logs from Purview, before running any analytical prompts. Purview exports often contain nested JSON fields, inconsistent event types, and mixed schemas that need to be normalized before meaningful analysis can be performed.

### Required inputs

- Raw Purview audit log export (CSV or JSON format)
- Expected raw fields include: `CreationTime`, `UserId`, `Operation`, `Workload`, `AuditData` (a JSON string containing event details)
- Optional: a list of Copilot-related operation names to filter for (if known for your tenant)

### Assumptions

- The export is from Microsoft Purview unified audit log
- `AuditData` is a JSON string column containing nested event details
- Event types and field structures may vary across different `Workload` and `Operation` values
- Some records may be malformed or have missing fields
- The `vivainsights` R or Python package is available but not required for this task

### Recommended output

A cleaned, flat CSV file (or DataFrame) with one row per event and consistently named columns, ready for analysis with any downstream prompt or tool.

### Prompt

```
You are a data engineer. Your task is to parse and clean a raw Microsoft Purview audit log export
into a flat, analysis-ready dataset. Purview audit logs contain nested JSON fields and mixed event
schemas, so this task requires careful exploration, parsing, and normalization.

IMPORTANT: Purview audit log schemas vary by tenant and event type. Do NOT assume specific field
names in the AuditData JSON — explore the data first and adapt.

PHASE 1: INITIAL LOADING AND INSPECTION
1. Load the raw audit log file. Auto-detect the format:
   - If CSV: load with pandas (Python) or readr (R). Handle encoding issues (try utf-8, then
     utf-8-sig, then latin-1).
   - If JSON: detect whether it is a JSON array, newline-delimited JSON, or a single object.
     Load accordingly.
2. Print: column names, data types, row count, and the first 3 rows.
3. Identify the core columns:
   - Timestamp: typically "CreationTime" or "CreationDate"
   - User: typically "UserId" or "UserKey"
   - Operation: typically "Operation"
   - Workload: typically "Workload"
   - AuditData: typically "AuditData" — a JSON string with event details
   Print which columns were identified and which are missing.
4. Check for data quality issues:
   a. How many rows have null/empty values in each core column?
   b. How many rows have non-parseable JSON in AuditData (if present)?
   c. Are there duplicate rows (identical across all columns)?
   Print a summary of these findings.

PHASE 2: PARSING THE AUDITDATA JSON COLUMN
5. If an AuditData column exists:
   a. Parse the JSON string for each row into a dictionary/object. Wrap in a try-except to
      handle malformed JSON gracefully — log the count of unparseable rows and skip them.
   b. Inspect the parsed JSON structure for a sample of 50 rows. Print the unique top-level keys
      found and their frequency. Identify which keys are present in most records vs. rare.
   c. Identify common nested objects (keys whose values are dicts or lists). Print examples of
      the nested structures.

6. Extract top-level fields from AuditData into new columns. Prioritize these fields (but use
   whatever is actually present in your data):
   - "Id" or "UniqueId" → event_id
   - "CreationTime" → audit_creation_time (may differ from the outer CreationTime)
   - "Operation" → audit_operation (may be more specific than the outer Operation)
   - "UserId" → audit_user_id
   - "ClientIP" → client_ip
   - "Workload" → audit_workload
   - "ObjectId" → object_id (the resource being accessed)
   - "ItemType" → item_type
   - "AppAccessContext" → extract nested fields like "AADSessionId", "CorrelationId"
   - "CopilotEventData" or similar → extract Copilot-specific details (agent name, plugin name,
     interaction type, prompt length, etc.)
   Create a new column for each extracted field.

7. For nested objects, flatten one level deep. For example, if AuditData contains:
   {"CopilotEventData": {"AgentName": "MyAgent", "InteractionType": "Chat"}}
   Create columns: copilot_agent_name, copilot_interaction_type.
   Do NOT attempt to flatten deeply nested structures (3+ levels) — store them as JSON strings.

PHASE 3: EVENT TYPE CLASSIFICATION
8. Analyze the distinct values in Operation (and audit_operation if different). Print:
   a. All unique Operation values with their counts.
   b. Group operations into categories:
      - Copilot events: operations containing "Copilot", "AI", "Agent", "GPT", "Assist",
        "Summarize", "CopilotInteraction" (case-insensitive)
      - User activity events: sign-in, file access, mail read, etc.
      - Admin events: settings changes, policy updates, etc.
      - Other/unknown
   c. Create a new column `event_category` with these classifications.
9. Print the count of events per category.

PHASE 4: FILTERING FOR COPILOT EVENTS
10. Create a filtered dataset containing only Copilot-related events (event_category == "copilot").
11. If the Copilot filter produces zero results:
    a. Print the top 30 most common Operation values so the user can identify Copilot events
       manually.
    b. Check if any AuditData fields contain Copilot-related values even if the Operation name
       does not indicate it.
    c. Save the full (unfiltered) cleaned dataset and note that manual filtering is needed.
12. If Copilot events are found, print:
    a. Count of Copilot events.
    b. Unique Copilot operation types.
    c. Date range of Copilot events.
    d. Sample of 5 Copilot event rows (all extracted columns).

PHASE 5: DATA CLEANING AND NORMALIZATION
13. Apply the following cleaning steps to the full dataset (or the Copilot-filtered subset):
    a. Parse all timestamp columns to datetime with timezone handling (Purview typically uses UTC).
    b. Normalize UserId:
       - Strip whitespace and convert to lowercase.
       - If UPNs (email format), keep as-is for joining. Optionally create a hashed_user_id
         column for anonymized output.
    c. Remove exact duplicate rows.
    d. Remove rows where both Operation and UserId are null.
    e. Create derived columns:
       - event_date: date extracted from the timestamp (UTC date)
       - event_hour: hour of day (0-23)
       - event_weekday: day of week (Monday=0, Sunday=6)
       - event_week: ISO week start date (Monday)

14. Standardize column names:
    - Use snake_case for all columns.
    - Prefix AuditData-derived columns with their source context (e.g., copilot_agent_name
      rather than just agent_name).
    - Ensure no column name conflicts between outer columns and AuditData-extracted columns.

PHASE 6: OUTPUT
15. Save the cleaned dataset:
    a. Full cleaned dataset → "purview_audit_cleaned_YYYYMMDD.csv"
    b. Copilot events only → "purview_copilot_events_YYYYMMDD.csv"
    c. Print the final schema: column names, data types, non-null counts for each output file.
    d. Print the row counts for each output.

16. Generate a data dictionary as a separate file ("purview_data_dictionary_YYYYMMDD.md"):
    - For each column in the cleaned output, list:
      - Column name
      - Data type
      - Source (outer field, AuditData top-level, AuditData nested)
      - Description (based on observed values)
      - Example values (2-3 examples)
      - Null rate

17. Print a final summary:
    a. Total raw records loaded.
    b. Records with parse errors (skipped).
    c. Duplicate records removed.
    d. Final clean record count.
    e. Copilot event count.
    f. Date range.
    g. Unique users.

IMPORTANT NOTES
- Prioritize robustness over speed. Wrap every JSON parse and type conversion in error handling.
- Do NOT assume the AuditData JSON structure is consistent across all rows — different Operation
  types may have completely different AuditData schemas. Handle this gracefully.
- If the file is very large (>500MB), process in chunks rather than loading all at once.
- Do NOT expose raw UserIds/email addresses in printed output — show only the first few
  characters or use aggregate counts when printing summaries.
- The data dictionary is critical for downstream users. Invest time in making it accurate.
```

### Adaptation notes

- **Your AuditData fields will differ.** The field names listed in step 6 (e.g., `CopilotEventData`, `AppAccessContext`) are examples. The prompt instructs the agent to explore first and adapt — but you may need to guide it if your tenant uses unusual field names.
- **Filtering criteria:** If you know your tenant's Copilot operation names (e.g., from the Purview audit log documentation or prior inspection), prepend them to the prompt: _"In my tenant, Copilot events use Operation values: CopilotInteraction, CopilotQuery."_
- **Large files:** For exports exceeding 1GB, add: _"Process the file in chunks of 100,000 rows using pandas chunked reading."_
- **Multiple export files:** If your audit log is split across multiple files (e.g., one per day), add: _"Load all CSV files from the directory and concatenate them before processing."_
- **Anonymization requirements:** If your organization requires user anonymization before analysis, strengthen step 13b: _"Replace all UserId values with a consistent SHA-256 hash. Do not retain the original identifier."_

### Common failure modes

- **Agent fails on malformed JSON in AuditData.** Some rows may have truncated or malformed JSON. The prompt includes try-except handling, but verify the agent implements it — a single bad row should not crash the entire parse.
- **Agent creates inconsistent column names.** When extracting from AuditData, column naming can become inconsistent across event types. Verify that the final output has standardized snake_case names.
- **Agent loads the entire file into memory.** For very large exports, this may cause out-of-memory errors. Watch for memory warnings and switch to chunked processing if needed.
- **Agent flattens deeply nested JSON, creating hundreds of columns.** The prompt limits flattening to one level deep, but some agents may over-expand. Verify the output schema is manageable (ideally <50 columns).
- **Copilot event filtering is too narrow or too broad.** If zero events match, the prompt instructs the agent to show all Operation values for manual inspection. If too many match, review the filter criteria and narrow them.
- **Encoding issues.** Purview CSV exports may use UTF-8 BOM encoding. If the agent encounters parsing errors on the first column name, instruct it to use `utf-8-sig` encoding.

---

## Tips for adapting prompts

- **Column names vary between tenants.** Always verify your actual column names against what the prompt expects. Prepend a note like _"In my data, the Organization column is called `Org`."_
- **Granularity matters.** Most prompts assume person-week data. If your export is person-day, instruct the agent to aggregate first.
- **Privacy thresholds.** For smaller organizations, add a note requesting minimum group sizes (e.g., suppress segments with fewer than 5 people).
- **Language preference.** Prompts default to Python. If you prefer R, add _"Use R instead of Python"_ before the prompt.
- **Package availability.** Prompts reference the [vivainsights R package](https://microsoft.github.io/vivainsights/) and [vivainsights Python package](https://microsoft.github.io/vivainsights-py/). Install them beforehand.

## Related resources

- [Frontier Analytics Overview]({{ site.baseurl }}/frontier-analytics/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [Starter Kits]({{ site.baseurl }}/frontier-analytics-starter-kits/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)

