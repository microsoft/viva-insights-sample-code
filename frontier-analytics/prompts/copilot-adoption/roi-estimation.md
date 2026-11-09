# Copilot Adoption ROI Estimation

## Purpose

Build an assumption-based Copilot value scenario from verified activity or assisted-time metrics and licence costs. Assisted time is not measured time saved or realised ROI.

## Audience

CFO/finance teams, IT leadership, Copilot program sponsors, business case reviewers

## When to use

For budget reviews, renewal decisions or expansion proposals that need an assumption-based estimate of Copilot's potential value. Requires at least 8 weeks of data for stable estimates.

## Required inputs

- Person query CSV with columns: `PersonId`, `MetricDate`, Copilot metrics (e.g., `Copilot_Assisted_Hours`, `Copilot_Actions`, `Copilot_Summarized_Hours`), and HR attributes (e.g., `Organization`, `FunctionType`, `LevelDesignation`)
- At least 8 weeks of data recommended
- Configurable assumptions: hourly cost of employee time (default: $75/hour), annual Copilot license cost per user (default: $360/year)
- Optional: collaboration metrics for licensed vs. unlicensed comparison (e.g., `Collaboration_Hours`, `Meeting_Hours`, `Email_Hours`)

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Licensing follows documented native M365 Copilot null/zero semantics, with enabled-user or enabled-days evidence preferred
- `Copilot_Assisted_Hours` (or similar) represents time where Copilot contributed to the user's work
- The `vivainsights` R or Python package is available in the environment

## Recommended output

An ROI summary report in HTML or Markdown, containing a value framework, sensitivity analysis, and methodology notes suitable for inclusion in a business case document.

## Quick prompt (short version)

Use this shorter prompt for a fast first pass that adapts to your data.

```
Help me build an ROI summary report for finance and IT leaders that estimates the value of Copilot usage from a Viva Insights person query export.
Start by loading the actual CSV and printing row count, column names, data types, date range, and unique people so you can identify the real Copilot and collaboration fields.
Adapt to the column names that are present, and ask me for the file path, hourly rate, license cost, or any ambiguous metric mapping instead of guessing.
Prefer Copilot Enabled User or Total Copilot enabled days per person-week. Native licensed-only M365 Copilot nulls mean unlicensed and zero activity means licensed but inactive. A verified metric with those semantics is a valid fallback without explicit fields. Omit licence-dependent estimates only if licensing remains genuinely unresolved.
Read configured cadence and measurement windows, not date spacing. Within a supplied complete query's configured scope, absent activity rows contribute zero. Failed joins, unavailable sources, out-of-scope periods and known coverage failures remain unresolved.
Use a recent multi-week average, distinct-person denominators and an explicit person versus person-week estimand. Assisted hours are not measured time saved: ask me to approve a realisation factor and any proxy conversion before estimating potential value, then make the break-even threshold explicit.
Include a sensitivity analysis under multiple hourly-rate and adoption scenarios, and be explicit that any licensed versus unlicensed comparison is observational and does not prove causation.
Report negative or uncertain value scenarios honestly. Suppress groups below max(the organisational requirement, 5 distinct people), using floor 10 for dashboard/joined outputs. Protect complementary cells and embedded data; never publish individual rankings.
Finish by listing every assumption you had to make about value realization, proxies, and segment definitions.
```

Use the detailed prompt below for a reproducible, exact-structure output. Use the quick prompt above for a faster first pass that you refine through follow-up turns.

## Prompt

```
You are a people analytics consultant specializing in technology ROI. Your task is to produce a defensible ROI estimate for Microsoft Copilot based on a Viva Insights person query export. The output should be suitable for inclusion in a business case presented to finance and IT leadership.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.

CONFIGURABLE ASSUMPTIONS (define as variables at the top of the script so they are easy to adjust)
- HOURLY_RATE = 75  # Fully loaded cost per employee hour in USD
- ANNUAL_LICENSE_COST = 360  # Annual Copilot license cost per user in USD
- WEEKS_IN_YEAR = 52
- ANALYSIS_WEEKS = 4  # Number of recent weeks to use for annualized projections
- REALISATION_FACTOR = None  # Ask for an approved fraction of assisted time assumed saved; do not silently assume 100%. Propose 0.25, 0.5 and 0.75 as a starting range for me to choose from.

DATA LOADING AND PREPARATION
Confirm the query scope, configured grouping and measurement windows before estimating value. Apply the shared analysis policy below to eligibility, missingness and privacy.
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python). This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their name. Reference the taxonomy at https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary activity metric. Print detected columns.
3. Identify HR / organizational attribute columns with `extract_hr(df, return = "names")` in R or `vi.extract_hr(df, return_type="suggestion")` in Python. Use the returned names for all organizational breakdowns instead of hard-coding column names.
4. Classify person-weeks using the shared eligibility policy. Prefer explicit enablement or dated licence evidence, then a verified native licensed-only M365 metric. Mark a person active when licensed and `Total_Copilot_actions_taken` is positive. Keep unresolved eligibility separate and omit affected estimates.
5. Print: total persons, licensed persons, active persons (ever active), date range.

TIME SAVINGS ESTIMATION
5. For each active person-week, use verified Copilot_Assisted_Hours as assisted time, not measured savings. Potential hours saved = assisted hours × approved REALISATION_FACTOR. If Copilot_Assisted_Hours is null despite positive licence or activity evidence, inspect the field definition and source joins; withhold affected estimates until resolved. If the field is unavailable, propose a documented proxy and conversion for approval; do not infer hours from actions or credits. Label potential savings as scenarios.
6. Compute scenario potential time savings, keeping observed assisted hours separate:
   a. Mean potential hours saved per active person-week over the last ANALYSIS_WEEKS complete configured weeks.
   b. Median potential hours saved per active person-week.
   c. Sum potential hours saved within each week, then average the weekly totals for annualisation. State the distinct-person n and missingness for each week.
7. Annualize the time savings:
   a. Per-active-user annualised potential hours saved = mean weekly potential hours × WEEKS_IN_YEAR.
   b. Total annualised potential hours saved = average weekly potential total × WEEKS_IN_YEAR. Do not multiply the multi-week sum by 52.

MONETARY VALUE ESTIMATION
8. Convert time savings to monetary value:
   a. Per-user annual value = per-user annualized hours × HOURLY_RATE.
   b. Total annual value = total annualized hours × HOURLY_RATE.
9. Compute total annual license cost:
   a. Total licensed users (distinct PersonId with is_licensed in the latest week).
   b. Total annual cost = total licensed users × ANNUAL_LICENSE_COST. Align the projected benefit population with this licensed population; if eligibility changes, declare a cohort/window rule instead of mixing unmatched populations.
10. Compute ROI metrics:
    a. Net annual value = total annual value - total annual cost.
    b. ROI ratio = total annual value / total annual cost.
    c. Break-even assisted hours per licensed user per week = ANNUAL_LICENSE_COST / (WEEKS_IN_YEAR × HOURLY_RATE × REALISATION_FACTOR), for positive hourly rate and factor. Report unavailable rather than divide by zero. Distinguish this from break-even assumed saved hours.
    d. Percentage of licensed users exceeding the break-even threshold.

SENSITIVITY ANALYSIS
11. Compute ROI under different assumptions. Create a table with rows for different hourly rates ($50, $75, $100, $125) and columns for different utilization scenarios:
    a. "Current active users only": value from only currently active users.
    b. "50% more adoption": cap projected active people at the eligible population and retain the approved realisation assumption.
    c. "Full adoption": if all licensed users become active at the current average.
12. This table shows leadership how ROI improves with higher adoption.

LICENSED VS. UNLICENSED COMPARISON (if collaboration metrics are available)
13. If collaboration metrics are present in the data (Collaboration_Hours, Meeting_Hours, Email_Hours, Focus_Hours, or similar), compare licensed vs. unlicensed users:
    a. For the most recent ANALYSIS_WEEKS weeks, compute the mean of each collaboration metric for licensed-active users, licensed-inactive users, and unlicensed users.
    b. Present as a comparison table.
    c. Include a caveat: this is an observational comparison, NOT a causal estimate. Differences may reflect selection effects (e.g., more collaborative employees may adopt Copilot first).
14. If collaboration metrics are not present, skip this section and note its absence.

VALUE BY SEGMENT
15. Break down the annual value estimate by Organization (or the primary HR attribute):
    a. For each organization, compute: licensed users, active users, total Copilot_Assisted_Hours, annualized value, license cost, net value.
    b. Present as a sorted table (by net value descending).
    c. Identify organizations with negative net value (license cost exceeds estimated value).

REPORT GENERATION
16. Compile into a professional report. Create an intermediary document first, then export to HTML:
    - R: Create an RMarkdown file (.Rmd), then knit to a self-contained HTML file.
    - Python: Create a Jupyter notebook (.ipynb), then export to a self-contained HTML file. Keep the intermediary file alongside the HTML output for troubleshooting. The report should contain these sections:

    a. "Executive Summary" (3-4 sentences)
       - Lead with the headline ROI number.
       - State the total estimated annual value and cost.
       - Note the break-even threshold and current adoption rate.

    b. "ROI Framework" (a clean summary table)
       | Metric | Value |
       |--------|-------|
       | Licensed users | N |
       | Active users (last 4 weeks) | N |
       | Assumed weekly hours saved per active person-week | X.X hours |
       | Potential annualised time value (scenario) | $X |
       | Annual license cost | $X |
       | Net annual value | $X |
       | ROI ratio | X.Xx |
       | Break-even threshold | X.X hours/week |
       | Users above break-even | X% |

    c. "Sensitivity Analysis": the table from step 11.

    d. "Value by Organization": the table from step 15.

    e. "Licensed vs. Unlicensed Comparison" (if applicable): table from step 13.

    f. "Methodology & Caveats"
       - Data source and time period.
       - How assisted hours are defined, the approved realisation factor and any proxy conversion. Savings are assumed, not measured.
       - The hourly rate assumption and its basis.
       - Explicit caveat: "This analysis estimates the potential value of time savings attributed to Copilot-assisted work. It does not establish a causal relationship between Copilot usage and productivity gains. Actual ROI depends on how saved time is reallocated. Correlation between Copilot usage and collaboration patterns may reflect selection effects rather than causal impact."
       - Note: time savings represent Copilot-assisted work, not necessarily net new time freed.
       - Privacy note: use max(the organisational requirement, 5 distinct people), with floor 10 for dashboard/joined outputs, including complementary cells and embedded data.

    g. "Assumptions": list all configurable assumptions and their values.

17. Use a clean, professional design. Format currency values with dollar signs and commas. Use bold text for headline numbers.
18. Save as "copilot_roi_estimation_YYYYMMDD.html".

IMPORTANT NOTES
- Present ROI as an assumption-based scenario, not a proven causal impact. Do not infer missing assisted hours or replace unresolved values with zero.
- Break-even is conditional on the approved realisation factor and price assumptions, not proof that observed usage recovers the licence cost.
- Report negative or uncertain value scenarios with the assumptions that produced them.

<!-- analysis-policy:start -->
Use product-specific licence evidence. For documented native licensed-only M365 Copilot metrics, null means unlicensed and zero activity means licensed but inactive. Prefer Copilot Enabled User or Total Copilot enabled days. Do not extend these semantics to unlicensed Copilot Chat or Consumption credits.
Investigate conflicting licence evidence. Keep affected eligibility or activity unresolved, and withhold affected rates or estimates until resolved.
Within a supplied complete query's configured scope, absent activity rows contribute zero. Keep failed joins, unavailable sources, out-of-scope periods and known coverage failures unresolved with reasons.
Use configured grouping, week start and metric windows; date gaps do not establish cadence. State grain, units and estimand. Count distinct people; do not sum stocks or overlapping windows.
Use max(organisation requirement, workflow floor): 5 distinct people for general analysis and 10 for joined-query, dashboard, causal or process workflows. Protect complementary cells and embedded data; never publish individual rankings.
Treat low-activity flags as review candidates, not verified holidays, employment status or job classifications. Value scenarios require explicit conversion and realisation assumptions.
Default to association. Make causal claims only with a supported design and compatible estimators. Record unavailable checks and label synthetic data.
Check installed API versions, signatures and return types. Separate executed results from expectations and do not silently fork upstream bugs.
<!-- analysis-policy:end -->
```

## Adaptation notes

- **Adjust the hourly rate** to match your organization's fully loaded labor cost. Many organizations use $50–$150/hour depending on geography and role mix. If available, use different rates per `LevelDesignation` or `FunctionType`.
- **Adjust the license cost** based on your actual agreement (E3/E5 add-on vs. standalone Copilot license).
- **Add qualitative benefits** by extending the report with a section on "Unquantified Benefits" (e.g., employee satisfaction, reduced context-switching, faster onboarding).
- **Time horizon:** The default projects current weekly usage to an annual figure. For a multi-year business case, add a growth assumption (e.g., 10% quarterly adoption increase).
- If you have time-series data from before and after Copilot deployment, you can add a pre/post comparison section by modifying the prompt.

## Common failure modes

- **Agent treats assisted time as measured savings.** Require the approved realisation factor throughout calculations and labels, rather than adding a caveat after a 100% savings assumption. A factor of 0.5 is a common conservative starting point for discussion, but the decision belongs to the person commissioning the analysis.
- **Agent ignores selection bias in licensed vs. unlicensed comparison.** The prompt includes a causal caveat, but verify it appears prominently in the output.
- **Agent annualizes from a single week.** The prompt uses the last `ANALYSIS_WEEKS` (4) weeks for stability, but ensure the agent averages across these weeks rather than extrapolating from one.
- **Agent reports ROI without caveats.** Senior leadership will scrutinize the methodology. Ensure every value estimate is accompanied by its assumptions and limitations.
- **Currency formatting issues.** Verify that dollar amounts are formatted with commas and two decimal places for large figures.
