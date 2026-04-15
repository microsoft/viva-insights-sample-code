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
- HR attributes for segmentation breakdowns

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

## Recommended output

An HTML report or Jupyter/R notebook containing segment distribution charts, transition matrices, churn curves, and at-risk group analysis.

## Prompt

```
You are a behavioral analytics specialist. Your task is to perform a user segmentation and churn
analysis on Microsoft Copilot usage data from a Viva Insights person query export. The goal is to
classify users into usage-based segments, track transitions between segments, and quantify churn.

DATA LOADING AND PREPARATION
1. Load the person query CSV. Parse MetricDate as a date. Treat PersonId as a string.
2. Auto-detect Copilot metric columns (columns starting with "Copilot_"). Print detected columns
   and the date range.
3. Classify each person-week:
   - "Licensed": at least one non-null, non-zero Copilot metric value.
   - "Unlicensed": all Copilot metrics are null or zero.
4. Filter to only licensed person-weeks for the segmentation analysis.
5. Fill any remaining NaN values in Copilot metric columns with 0 for licensed users (a licensed
   user with a null action count in one metric likely had zero usage of that specific feature).

USER SEGMENTATION
6. For each PersonId in each week, compute a primary usage score using Copilot_Actions (or the
   main activity metric available). Then classify each person-week into one of four segments:
   - "Power User": Copilot_Actions >= 75th percentile of all active person-weeks
   - "Regular User": Copilot_Actions >= 25th percentile AND < 75th percentile
   - "Light User": Copilot_Actions > 0 AND < 25th percentile
   - "Inactive": Copilot_Actions == 0 (licensed but no activity)

   Calculate the percentile thresholds from the distribution of Copilot_Actions across all
   person-weeks where Copilot_Actions > 0. Print the thresholds used.

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
- Segment thresholds are based on percentiles, not absolute numbers, so they adapt to your data.
  Print the actual threshold values so the reader can interpret them.
- Do NOT count unlicensed users in any denominator. They are excluded from the analysis entirely.
- Suppress any HR attribute segment with fewer than 5 users in the visualizations.
- The transition matrix should only include users who appear in both consecutive weeks.
- If the dataset has fewer than 8 weeks, note that churn analysis may not be reliable and
  adjust the churn definition (e.g., reduce the "2 consecutive inactive weeks" requirement).
```

## Adaptation notes

- Adjust the percentile thresholds (75th/25th) if they do not produce meaningful segments for your data. For very skewed distributions, try 90th/50th instead, or define absolute thresholds (e.g., Power User ≥ 50 actions/week).
- If your organization has a specific definition of "churn" (e.g., 4 weeks of inactivity instead of 2), modify the churn definition in step 15.
- For organizations with multiple Copilot products, consider segmenting by product (e.g., Copilot in Teams vs. Copilot in Word) by using product-specific metrics.
- Add a "New User" segment by tracking each user's first active week and analyzing the onboarding trajectory separately.
- If you want to predict future churn, add a note requesting a logistic regression or survival analysis model.

## Common failure modes

- **Agent uses absolute thresholds instead of percentiles.** The prompt specifies percentile-based thresholds, but an agent may default to arbitrary cutoffs. Verify that it calculates and prints the actual thresholds.
- **Agent counts unlicensed users as "Inactive."** The Inactive segment should only include licensed users with zero activity — not unlicensed users. Verify the filtering step.
- **Transition matrix includes users who appear in only one week.** Ensure the matrix only counts users present in both the "from" and "to" weeks.
- **Agent conflates week-level segments with user-level segments.** The report uses both: per-week segments for trends and a "stable segment" (mode) for HR attribute breakdowns. Ensure both are present.
- **Churn definition is too aggressive.** Two weeks of inactivity may include holidays or vacation. Consider extending to 3–4 weeks for more conservative churn estimates.
