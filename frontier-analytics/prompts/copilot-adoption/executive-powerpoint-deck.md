# Executive PowerPoint Deck: Copilot Adoption

## Purpose

Create a 10-15 slide PowerPoint deck (.pptx) with editable native charts covering Copilot adoption trends, organisational breakdowns and recommendations.

## Audience

VP/C-suite executives, board presentations, steering committee reviews

## When to use

When a static HTML report or memo is not suitable and stakeholders need a polished, editable PowerPoint deck. For example, for a live presentation, a board pack, or a document that will be further edited by others.

## Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended (12+ weeks preferred)
- HR attributes for organizational breakdowns

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Licensing follows documented native M365 Copilot null/zero semantics, with enabled-user or enabled-days evidence preferred
- The `vivainsights` R or Python package is available in the environment
- A package for creating PowerPoint files with native charts is available (for example `officer` + `mschart` in R, or `python-pptx` in Python)

## Recommended output

A .pptx file with 10-15 slides, using editable native PowerPoint charts (not pasted images) so that recipients can modify the deck as needed.

## Quick prompt (short version)

Use this shorter prompt for a fast first pass that adapts to your data.

```
Help me generate an exec-ready PowerPoint deck that summarizes Copilot adoption for senior leaders from a Viva Insights person query export.
Start by loading the actual CSV and printing row count, column names, data types, date range, and unique people so you can map the real Copilot and HR fields before building slides.
Adapt field names dynamically, use the installed toolchain that best supports native PowerPoint charts, and ask me for the file path, template path, or any ambiguous metric mapping before you proceed.
Prefer Copilot Enabled User or Total Copilot enabled days for licensing. Native licensed-only M365 Copilot nulls mean unlicensed and zero activity means licensed but inactive. Use a verified metric with those semantics as a fallback without explicit fields. Omit rates only when licence status or licensed activity is genuinely unresolved.
Read configured cadence and measurement windows, not date spacing. In a supplied complete query's configured scope, absent activity rows contribute zero. Failed joins, out-of-scope periods and known coverage failures remain unresolved.
Create the deck directly as a .pptx file with native editable chart objects wherever the library supports them, and avoid HTML to PowerPoint conversion pipelines.
If a needed chart type cannot be created natively, simplify it to a supported native chart where possible and clearly note any remaining slide that must fall back to an image.
Use distinct-person denominators and state whether summaries weight people or person-weeks. Suppress groups below max(the organisational requirement, 5 distinct people), using floor 10 in dashboard/joined workflows. Protect complementary cells and embedded chart data; never publish individual rankings. Skip unavailable slides.
Use commentary-led slide titles and keep each number traceable to a local evidence ledger with dates, units, aggregation, bases and denominator. Related charts or tables can share a slide if labels and caveats remain readable.
For an existing user-edited deck, save a rollback, preserve unrelated content and verify the source has not changed before replacement. Remove only comments whose requested edits have been verified.
Verify numbers, embedded chart data, privacy and actual rendering separately. Report unavailable visual checks honestly.
Finish by listing any assumptions you made about chart support, metric mappings, or optional slide substitutions.
```

Use the detailed prompt below for a reproducible, exact-structure output. Use the quick prompt above for a faster first pass that you refine through follow-up turns.

## Prompt

```
You are a senior people analytics consultant. Your task is to generate a PowerPoint deck (.pptx) summarising Microsoft Copilot adoption from a Viva Insights person query export for senior leaders. Use editable native PowerPoint charts so recipients can modify charts and data as needed. Do not paste chart images where a native chart is supported.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.
- R: Use the `officer` and `mschart` packages for native PowerPoint chart generation.
- Python: Use the `python-pptx` package. Its native chart support is more limited. If advanced chart types are needed, R with `mschart` is recommended.

DATA LOADING AND PREPARATION
Read the configured grouping, week start, measurement windows and query scope before calculating metrics. Apply the shared analysis policy below to eligibility, missingness and privacy.
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python). This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their name. Reference the taxonomy at https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary activity metric.
3. Identify HR / organizational attribute columns with `extract_hr(df, return = "names")` in R or `vi.extract_hr(df, return_type="suggestion")` in Python.
4. Classify person-weeks using the shared eligibility policy. Prefer explicit enablement or dated licence evidence, then a verified native licensed-only M365 metric. Mark a person active when licensed and `Total_Copilot_actions_taken` is positive. Keep unresolved eligibility separate and withhold affected rates.
5. Print: date range, total persons, licensed persons, active persons.

METRIC CALCULATIONS
6. Compute the following metrics for the deck:

   Retain a local evidence ledger linking every claim/view to source fingerprints,
   exact dates, metric definitions, aggregation, valid bases and numerator/denominator.
   Generate prose, tables and chart data from the same aggregates.

   Headline metrics (latest 4 complete weeks vs. prior 4 weeks):
   a. Current adoption rate (active / licensed in the latest week)
   b. Adoption trend (percentage-point change between the two 4-week windows)
   c. Average Total_Copilot_actions_taken per active user per week
   d. Average Copilot_Assisted_Hours per active user per week (if available)
   e. Total unique active users in the last 4 weeks

   Trend data:
   f. Weekly adoption rate over the full date range
   g. Weekly mean Total_Copilot_actions_taken per active user

   Segmentation data:
   h. Adoption rate by each HR attribute (latest 4-week average)
   i. Top 3 and bottom 3 organizations by adoption rate (minimum 10 licensed users)

   ROI headline (if Copilot_Assisted_Hours is available):
   j. Average weekly assisted hours per active user, not measured time saved
   k. Potential annualised value only after approval of a realisation factor, hourly rate and any proxy conversion; label it an assumption-based scenario

SLIDE DECK GENERATION
7. Create a PowerPoint deck with the following slides. Prefer R (mschart/officer) for native, editable PowerPoint charts. If you use Python (python-pptx), restrict yourself to its supported native chart types (e.g., clustered/stacked bar and column, line, and pie charts). Avoid combo charts, secondary axes or annotations it cannot render natively. For an unsupported chart type, simplify it to a supported type or fall back to a high-resolution embedded image. Note which slides use images rather than editable charts.

   SLIDE 1: Title slide
   - Title: "Copilot Adoption Review"
   - Subtitle: Reporting period [start date] to [end date]
   - Date generated

   SLIDE 2: Executive summary
   - 3-4 key bullet points summarizing the most important findings
   - Lead with the headline adoption rate and trend direction
   - Use bold text for key numbers

   SLIDE 3: At a Glance (KPI cards)
   - Layout showing key metrics: adoption rate, trend, active users, avg actions/user, avg assisted hours/user
   - Use directional indicators (▲ ▼ ►) for trends
   - Use text boxes with large numbers for these metrics

   SLIDE 4: Adoption trend over time
   - Native line chart: weekly adoption rate over the full period
   - Clear axis labels, chart title, and a trend line if appropriate

   SLIDE 5: Usage intensity over time
   - Native line chart: weekly mean Total_Copilot_actions_taken per active user
   - Optional second series for median if it adds insight

   SLIDE 6: Copilot-assisted hours over time (if metric is available)
   - Native line chart: weekly mean Copilot_Assisted_Hours per active user
   - If this metric is not available, replace with a slide on another relevant metric or skip

   SLIDES 7-9: Organizational breakdowns (one slide per HR attribute)
   - Native horizontal bar chart: adoption rate by segment (latest 4-week average)
   - Sort bars by adoption rate descending
   - Suppress below max(the organisational requirement, 5 distinct licensed people), using floor 10 for dashboard/joined outputs and at least 10 for the top/bottom organisation table
   - If there are more than 12 segments, show only the top and bottom 6

   SLIDE 10: Groups with higher and lower adoption
   - A table or two-column layout showing:
     - Top 3 organizations by adoption rate (with their rates and user counts)
     - Bottom 3 organizations by adoption rate
   - Brief annotation on observed usage differences and relevant coverage/context
   - Adoption differences do not establish productivity or employee performance

   SLIDE 11: ROI summary (if Copilot_Assisted_Hours is available)
   - Average weekly assisted hours and the approved realisation factor
   - Annualized value estimate and cost
   - ROI ratio
   - Break-even threshold
   - Include a caveat that this is an estimation framework, not proven causal impact
   - If ROI data is not available, replace with a usage depth slide

   SLIDE 12: Key recommendations
   - 3-4 specific, data-driven recommendations
   - Each should reference a finding from the data with specific numbers
   - Use action-oriented language

   SLIDE 13: Risks and considerations
   - 2-3 flagged risks or areas of concern
   - Include data quality caveats if relevant

   SLIDE 14: Methodology
   - Brief description of data source, definitions (licensed, active), time period
   - Privacy note on minimum group size suppression
   - Caveat: correlation ≠ causation

   SLIDE 15: Appendix (optional)
   - Full segmentation tables with numbers
   - Any suppressed data notes
   - Data source details

   Adjust the slide count as needed. Aim for 10-15 slides total. Skip slides where data is unavailable rather than leaving them blank.

8. Design guidelines:
   - Use a clean, professional layout with consistent fonts (e.g., Calibri or Segoe UI)
   - Use a consistent color palette across all charts
   - Keep text concise and use bullet points
   - Every chart must have a clear title and takeaway annotation
   - Slide titles should state the specific finding and its important qualification
   - One question may use two tables or several related charts on one slide when readable
   - Keep exact windows, bases, denominator and material caveats visible
   - Retain complete eligible supporting tables in an indexed appendix when condensing
   - Use stable colours and common scales for like-for-like panels; label intentional differences
   - Number all slides

9. Save the deck as "copilot_adoption_deck_YYYYMMDD.pptx".
   For an existing user-edited deck, work from a recoverable live snapshot and an
   unused candidate path. Preserve unrelated manual edits and compare retained
   content before publishing. Refresh affected contents, footers, links and notes.
   Recheck the live source before replacement, retain a rollback and verify the
   persisted file. Remove only resolved comments after their changes pass checks.

10. Verify numerical consistency, privacy and rendering independently. Check
    chart caches and embedded workbooks as well as visible labels. Open and inspect
    the actual PPTX where a compatible renderer is available, fix clipping/wrapping
    and re-render affected slides. XML/ZIP validity alone is insufficient.
    If visual verification cannot run, label it unavailable rather than approved.

IMPORTANT NOTES
- Use native editable PowerPoint charts wherever supported, as described in step 7.
- Do NOT use RMarkdown or Jupyter notebook as an intermediary for this output. Generate the .pptx file directly using the appropriate PowerPoint package.
- Do NOT fabricate numbers. Every statistic must be computed from the data.
- Skip metrics whose inputs are unavailable. For unsupported chart types, follow step 7 and document any image fallback.

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

- **R is recommended** for this prompt due to the `officer` + `mschart` packages, which provide strong native PowerPoint chart support. Python's `python-pptx` has more limited chart types.
- **Adjust the hourly rate** for ROI calculations by modifying the $75/hour default. Add "Use $X/hour for the hourly rate assumption" to the prompt.
- **Custom branding:** If your organization has a PowerPoint template (.potx), add "Use the template file at [path] as the base for the deck." The `officer` package in R supports applying templates.
- **Additional slides:** Add or remove slides by modifying the slide list. For a shorter deck (board summary), keep only slides 1-5 and 12.
- **Localization:** For non-English audiences, add "Generate all slide text and chart labels in [language]."

## Common failure modes

- **Agent pastes images instead of creating native charts.** This is the most common failure. Verify that charts are editable by opening the .pptx and clicking on a chart. You should be able to edit the data. If the agent falls back to images, explicitly instruct it to use `mschart` (R) or chart objects in `python-pptx` (Python).
- **Agent uses an R/Python to HTML to PPTX conversion pipeline.** This produces image-based slides, not native charts. The agent should create the .pptx directly using the PowerPoint package.
- **Chart types not supported.** Some chart types (for example heatmaps) may not be available as native PowerPoint charts. The agent should substitute with a table or the closest available chart type.
- **Slide layout is cluttered or fragmented.** Organise each slide around one question. Combine related comparisons where readable, or paginate/simplify if crowded. Keep supporting detail in an indexed appendix rather than deleting it or shrinking text excessively.
- **A revision overwrites manual edits or leaves stale chart data.** Snapshot the current live deck and use a candidate. Verify unchanged content and both cached/embedded data. Stop on concurrent content changes and rebase before publishing. Delete only verified resolved comments.
- **The file opens but the slide is unreadable.** Render and inspect actual chart labels, legends, wrapping and footnotes. Record visual verification separately from numerical and privacy checks.
- **Agent does not handle missing metrics gracefully.** If `Copilot_Assisted_Hours` or other optional metrics are not in the data, the agent should skip those slides rather than error.
