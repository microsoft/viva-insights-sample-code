# Segmentation & Churn Analysis — Copilot Adoption

## Purpose

Classify Copilot users into behavioral segments based on usage intensity, track how users move between segments over time, and quantify churn — the rate at which users disengage from Copilot after initial adoption.

## Audience

People analytics leads, Copilot program managers, digital adoption teams

## When to use

After at least 8–12 weeks of person query data have been collected, when you need to understand not just whether people are using Copilot, but how their usage patterns evolve over time.

## Required inputs

- Person query CSV with columns: `PersonId`, `MetricDate`, Copilot metrics (e.g., `Copilot_Actions`, `Copilot_Assisted_Hours`), and HR attributes (e.g., `Organization`, `FunctionType`, `LevelDesignation`)
- At least 8 weeks of data (12+ weeks recommended for meaningful churn analysis)
- A Copilot enabled-days field or another confirmed source of weekly licence status
- HR attributes for segmentation breakdowns

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Licence status can be established for the person-weeks included in the analysis
- The `vivainsights` R or Python package is available in the environment

## Recommended output

An HTML report or Jupyter/R notebook containing segment distribution charts, transition matrices, churn curves, and at-risk group analysis.

## Quick prompt (short version)

Use this shorter prompt for a fast first pass that adapts to your data.

```
Help me create a segmentation and churn analysis for Copilot usage that classifies licensed users, tracks movement between segments over time, and highlights at-risk groups.
Start by loading the actual CSV and printing row count, column names, data types, date range, and unique people so you can map the real activity and HR fields.
Do not assume exact column names, and ask me for the file path, the main activity metric, or any unclear field mapping before you continue.
Use enabled-days or other confirmed licence data to exclude unlicensed users from denominators; do not infer licence status from zero or missing activity alone.
Use `identify_usage_segments()` (vivainsights) to compute the five weekly usage segments. If it is unavailable, clearly label any alternative segmentation and print its thresholds.
Include both week-level segment trends and a stable user segment for group breakdowns, then build a week-to-week transition matrix only for users present in consecutive weeks.
Independently of the rolling segments, define weekly activity from the observed action count. Churn means at least two consecutive active weeks followed by at least two consecutive inactive weeks; date it to the first inactive week once the second confirms it.
Identify currently at-risk users separately: they have one inactive week immediately after at least two active weeks and have not yet met the churn definition. Do not treat unknown or unlicensed weeks as inactive.
Flag if the time span is too short for reliable churn analysis.
Suppress group views below about 5 people for privacy, and note any missing HR attributes or holiday-related caveats that affect interpretation.
End by listing any assumptions you made about thresholds, churn rules, or field mappings.
```

Use the detailed prompt below for a reproducible, exact-structure output. Use the quick prompt above for a faster first pass that you refine through follow-up turns.

## Prompt

```
You are a behavioral analytics specialist. Your task is to perform a user segmentation and churn analysis on Microsoft Copilot usage data from a Viva Insights person query export. The goal is to classify users into usage-based segments, track transitions between segments, and quantify churn.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.

DATA LOADING AND PREPARATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python). This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their name. Reference the taxonomy at https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary activity metric. Print detected columns and the date range.
3. Identify HR / organizational attribute columns with `extract_hr(df, return = "names")` in R or `vi.extract_hr(df, return_type="suggestion")` in Python. Use the returned names for all segmentation breakdowns instead of hard-coding column names.
4. Classify each person-week:
   - "Licensed": an enabled-days column (for example `Total_Copilot_enabled_days`) is greater than zero for the week, or another confirmed licence source establishes coverage.
   - "Unlicensed": the enabled-days column equals zero or another confirmed licence source establishes no licence.
   - "Unknown": licence status cannot be established. Zero or missing activity alone does not establish that a user is unlicensed.
5. Restrict segment and churn denominators to confirmed licensed person-weeks. Report the count with unknown licence status.
6. Align each person to a weekly calendar and derive a separate activity flag from the primary action metric: positive is active, zero is inactive, and missing is unknown unless the query's documented schema establishes that it represents zero. An unlicensed week, or a week with unknown activity or licence status, is not inactive and breaks an activity or inactivity streak.

USER SEGMENTATION
6. Use `identify_usage_segments()` from the `vivainsights` library to classify users into segments. This function uses `Total_Copilot_actions_taken` and classifies users based on both usage volume and consistency (habit formation over 12 weeks), producing the canonical **mutually-exclusive** ladder — Power Users → Habitual Users → Novice Users → Low Users → Non-users (see the [Copilot Usage Segments definitions](https://microsoft.github.io/viva-insights-sample-code/copilot-usage-segments/#formal-definitions)). It is preferred over manual percentile-based segmentation. Use these exact segment names consistently throughout the report. Do not use the rolling segment as the weekly activity flag. Mark a segment unavailable where the required window contains unknown activity or licence status; do not recode unknown activity as zero.

7. Also compute a "stable segment" for each user based on their most frequent weekly segment over the entire period (mode). This gives a single label per person.

SEGMENT DISTRIBUTION OVER TIME
8. For each week, count the number of users in each segment. Create:
   a. A stacked area chart showing the five segments over time (absolute counts).
   b. A stacked area chart showing the five segments as percentages of licensed users with an available segment. Report how many licensed users could not be classified.
   c. A summary table of the latest week's segment distribution.

SEGMENT BREAKDOWN BY HR ATTRIBUTES
9. For each available HR attribute, compute the segment distribution using the "stable segment" label. Create:
   a. A stacked bar chart showing segment proportions by each HR attribute value.
   b. Only include HR attribute values with at least 5 licensed users.
10. Identify groups with the highest proportions of Power Users and Non-users. Label these as segment distributions, not as the separate at-risk churn definition below.

TRANSITION ANALYSIS
11. Build a week-over-week transition matrix. For each consecutive pair of weeks, count how many users moved from each segment to every other segment. Aggregate across all week transitions.
12. Normalize the matrix to show transition probabilities: given a user is in segment X this week, what is the probability they are in segment Y next week?
13. Visualize the transition matrix as a heatmap with annotations showing the probabilities.
14. Highlight key transitions of interest:
    - "Entering use": Non-user → any higher segment
    - "Returning to non-use": any higher segment → Non-user
    - "Deepening": a move up the ladder, e.g. Low User → Novice User, Novice User → Habitual User, or Habitual User → Power User
    - "Declining": a move down the ladder, e.g. Power User → Habitual User, Habitual User → Novice User, or Novice User → Low User

CHURN ANALYSIS
15. Define "churn" from the weekly activity flag, independently of the rolling segment: at least 2 consecutive active weeks immediately followed by at least 2 consecutive inactive weeks. The churn date is the first inactive week, confirmed only after the second. Unknown, missing or unlicensed weeks break the streak. A user may churn again after reactivation.
16. For each week, calculate:
    a. The number of users whose confirmed churn began that week.
    b. The churn rate: confirmed churn onsets / users with at least 2 consecutive active weeks immediately before that week who are licensed and have known activity in both that week and the following week. Do not report a rate for a week without the following week's observation.
17. Create a line chart of weekly churn rate over time.
18. Build a "time to churn" distribution: for users who churned, how many weeks were they active before churning? Show as a histogram.
19. Compute overall churn statistics:
    a. Total users who ever churned
    b. Percentage of ever-active users who churned
    c. Median active weeks before churn
    d. Re-activation rate: among those who churned and have subsequent known licensed weeks, what percentage became active again later? Note incomplete follow-up.

AT-RISK USER IDENTIFICATION
20. Identify currently at-risk users: licensed users with exactly 1 inactive week immediately after at least 2 consecutive active weeks. They have not yet met the churn definition.
21. Summarize at-risk users by HR attribute: which organizations, functions, or levels have the highest concentration of at-risk users?
22. Create a table of at-risk user counts by available HR attributes.

REPORT GENERATION
23. Compile all outputs into an intermediary document first, then export to HTML:
    - R: Create an RMarkdown file (.Rmd), then knit to a self-contained HTML file.
    - Python: Create a Jupyter notebook (.ipynb), then export to a self-contained HTML file. Keep the intermediary file alongside the HTML output for troubleshooting. The report should contain these sections:
    a. "Segment Definitions" — explain the five segments and thresholds used.
    b. "Segment Distribution" — charts from step 8 and summary table.
    c. "Segment Breakdown by Group" — charts and segment-distribution callouts from steps 9 and 10.
    d. "Transition Analysis" — heatmap from step 13, key transition highlights from step 14.
    e. "Churn Analysis" — churn rate trend, time-to-churn histogram, summary statistics.
    f. "At-Risk Users" — summary table from step 22.
    g. "Key Takeaways" — 3-5 bullet points synthesizing the most important findings.
    h. "Methodology" — brief description of data source, segment definitions, churn definition.

24. Use static charts (matplotlib/seaborn for Python or ggplot2 for R).
25. Save the report and intermediary file with descriptive filenames like "copilot_segmentation_churn_YYYYMMDD.html".

IMPORTANT NOTES
- Segments are assigned by `identify_usage_segments()` based on both usage volume and consistency (habit formation over 12 weeks). Print the segment definitions used so the reader can interpret them.
- Do NOT count unlicensed or unknown-licence users in licensed denominators.
- Suppress any HR attribute segment with fewer than 5 users in the visualizations.
- The transition matrix should only include users who appear in both consecutive weeks.
- If the dataset has fewer than 8 weeks, note that churn analysis may not be reliable; do not silently change the churn definition.
```

## Adaptation notes

- The `identify_usage_segments()` function provides default thresholds. If the segments do not produce meaningful groupings for your data, consult its documentation for customization parameters and report any changes.
- If your organization has a specific definition of "churn" (e.g., 4 weeks of inactivity instead of 2), modify the churn definition in step 15.
- For organizations with multiple Copilot products, consider segmenting by product (e.g., Copilot in Teams vs. Copilot in Word) by using product-specific metrics.
- Add a "New User" segment by tracking each user's first active week and analyzing the onboarding trajectory separately.
- If you want to predict future churn, add a note requesting a logistic regression or survival analysis model.

## Common failure modes

- **Agent uses rolling segments as weekly activity.** A user may remain Habitual after two inactive weeks. Compute the churn flag from observed weekly activity instead.
- **Agent treats missing or unlicensed weeks as inactive.** Only a confirmed licensed week with known zero activity is inactive for churn.
- **Transition matrix includes users who appear in only one week.** Ensure the matrix only counts users present in both the "from" and "to" weeks.
- **Agent conflates week-level segments with user-level segments.** The report uses both: per-week segments for trends and a "stable segment" (mode) for HR attribute breakdowns. Ensure both are present.
- **Churn definition is too aggressive.** Two weeks of inactivity may include holidays or vacation. Consider extending to 3–4 weeks for more conservative churn estimates.
