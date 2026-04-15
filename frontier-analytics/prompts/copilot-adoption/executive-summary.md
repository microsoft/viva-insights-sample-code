# Executive Summary — Copilot Adoption

## Purpose

Generate a concise executive summary memo that distills Copilot adoption metrics into key findings, trend analysis, and actionable recommendations for senior leadership.

## Audience

VP/C-suite executives, senior leadership team, IT steering committee

## When to use

When you need to communicate Copilot adoption progress to senior leadership — typically on a monthly or quarterly cadence after collecting at least 8 weeks of person query data.

## Required inputs

- Person query CSV with columns: `PersonId`, `MetricDate`, Copilot metrics (e.g., `Copilot_Actions`, `Copilot_Assisted_Hours`), and HR attributes (e.g., `Organization`, `FunctionType`, `LevelDesignation`)
- At least 8 weeks of data recommended (12+ weeks preferred for trend analysis)
- HR attributes for organizational breakdowns

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment

## Recommended output

A 1–2 page executive summary in HTML or Markdown, formatted as a professional memo suitable for distribution to VP/C-suite audiences.

## Prompt

```
You are a senior people analytics consultant. Your task is to generate a polished executive summary
memo about Microsoft Copilot adoption, based on a Viva Insights person query export. The memo must
be suitable for a VP or C-suite audience — concise, insight-driven, and action-oriented.

DATA LOADING AND PREPARATION
1. Load the person query CSV into a DataFrame. Parse MetricDate as a date. Treat PersonId as a string.
2. Auto-detect Copilot metric columns (columns starting with "Copilot_"). Print the detected columns
   and the date range for verification.
3. Classify each person-week row:
   - "Licensed": has at least one non-null, non-zero Copilot metric value.
   - "Active": is licensed AND has Copilot_Actions > 0 (or the primary activity metric).
   - "Unlicensed": all Copilot metric values are null or zero.
4. If any HR attribute columns (Organization, FunctionType, LevelDesignation) are missing, note which
   ones are unavailable and proceed with what is present.

HEADLINE METRICS (compute these for the memo)
5. Current adoption rate: In the most recent complete week, what percentage of licensed users were
   active? Report as "X% of licensed users actively used Copilot in the week of [date]."
6. Adoption trend: Compare the average weekly adoption rate over the last 4 complete weeks to the
   prior 4 weeks. Calculate the percentage-point change. Classify as "improving", "stable" (within
   ±2pp), or "declining".
7. Usage intensity: Average Copilot_Actions per active user per week over the last 4 weeks.
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

## Adaptation notes

- If your leadership prefers Markdown over HTML, add _"Output as a Markdown file"_ at the start of the prompt.
- Adjust the organization size threshold (default: 10 licensed users) based on your company size. For very large organizations, increase to 20+.
- Add company-specific context by prepending: _"Our Copilot deployment started on [date] and currently covers [N] licenses across [divisions]."_
- If you want the memo to reference specific business goals (e.g., "target 60% adoption by Q3"), include that context before the prompt.
- Change the trend comparison window (default: 4 weeks vs. prior 4 weeks) to match your reporting cadence.

## Common failure modes

- **Agent fabricates insights not supported by data.** The prompt instructs the agent to compute every statistic from the data, but always verify headline numbers independently.
- **Agent uses overly technical language.** The prompt specifies executive-friendly language, but review the output for jargon like "person-week panel structure" and simplify.
- **Agent double-counts users across weeks for "total active users."** Ensure breadth metrics use distinct `PersonId` counts, not row counts.
- **Agent treats missing Copilot values as zero.** This would inflate denominators and deflate adoption rates. Verify the licensed user logic.
- **Recommendations are too generic.** If the agent produces vague advice ("increase adoption"), ask it to regenerate with specific org names and numbers from the data.
