# ROI Estimation — Copilot Adoption

## Purpose

Estimate the return on investment (ROI) for Microsoft Copilot by quantifying time savings, translating them into monetary value, and comparing against license costs to build a defensible business case.

## Audience

CFO/finance teams, IT leadership, Copilot program sponsors, business case reviewers

## When to use

When stakeholders need a quantified business justification for Copilot investment — typically during budget reviews, renewal decisions, or expansion proposals. Requires at least 8 weeks of data for stable estimates.

## Required inputs

- Person query CSV with columns: `PersonId`, `MetricDate`, Copilot metrics (e.g., `Copilot_Assisted_Hours`, `Copilot_Actions`, `Copilot_Summarized_Hours`), and HR attributes (e.g., `Organization`, `FunctionType`, `LevelDesignation`)
- At least 8 weeks of data recommended
- Configurable assumptions: hourly cost of employee time (default: $75/hour), annual Copilot license cost per user (default: $360/year)
- Optional: collaboration metrics for licensed vs. unlicensed comparison (e.g., `Collaboration_Hours`, `Meeting_Hours`, `Email_Hours`)

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- `Copilot_Assisted_Hours` (or similar) represents time where Copilot contributed to the user's work
- The `vivainsights` R or Python package is available in the environment

## Recommended output

An ROI summary report in HTML or Markdown, containing a value framework, sensitivity analysis, and methodology notes suitable for inclusion in a business case document.

## Prompt

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
1. Load the person query CSV. Parse MetricDate as a date. Treat PersonId as a string.
2. Auto-detect Copilot metric columns (columns starting with "Copilot_"). Print detected columns.
3. Classify each person-week:
   - "Licensed": at least one non-null, non-zero Copilot metric value.
   - "Active": licensed AND Copilot_Actions > 0.
   - "Unlicensed": all Copilot metric values are null or zero.
4. Print: total persons, licensed persons, active persons (ever active), date range.

TIME SAVINGS ESTIMATION
5. For each active person-week, extract Copilot_Assisted_Hours (the primary time-savings metric).
   If this column is not available, fall back to estimating from Copilot_Summarized_Hours or a
   fraction of Copilot_Actions (note the assumption if using a proxy).
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

## Adaptation notes

- **Adjust the hourly rate** to match your organization's fully loaded labor cost. Many organizations use $50–$150/hour depending on geography and role mix. If available, use different rates per `LevelDesignation` or `FunctionType`.
- **Adjust the license cost** based on your actual agreement (E3/E5 add-on vs. standalone Copilot license).
- **Add qualitative benefits** by extending the report with a section on "Unquantified Benefits" (e.g., employee satisfaction, reduced context-switching, faster onboarding).
- **Time horizon:** The default projects current weekly usage to an annual figure. For a multi-year business case, add a growth assumption (e.g., 10% quarterly adoption increase).
- If you have time-series data from before and after Copilot deployment, you can add a pre/post comparison section by modifying the prompt.

## Common failure modes

- **Agent treats Copilot_Assisted_Hours as "time saved."** Clarify that Copilot_Assisted_Hours represents time where Copilot assisted the user — the actual time saved may be a fraction of this. Consider adding a "realization factor" (e.g., 50%) for more conservative estimates.
- **Agent ignores selection bias in licensed vs. unlicensed comparison.** The prompt includes a causal caveat, but verify it appears prominently in the output.
- **Agent annualizes from a single week.** The prompt uses the last `ANALYSIS_WEEKS` (4) weeks for stability, but ensure the agent averages across these weeks rather than extrapolating from one.
- **Agent reports ROI without caveats.** Senior leadership will scrutinize the methodology. Ensure every value estimate is accompanied by its assumptions and limitations.
- **Currency formatting issues.** Verify that dollar amounts are formatted with commas and two decimal places for large figures.
