---
layout: page
title: "Prompt — Executive PowerPoint Deck"
permalink: /frontier-analytics-prompt-powerpoint/
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

# Executive PowerPoint Deck — Copilot Adoption

[← Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)

## Purpose

Generate an exec-ready 10–15 page PowerPoint deck (.pptx) with editable native PowerPoint charts, summarizing Copilot adoption trends, organizational breakdowns, and key recommendations.

## Audience

VP/C-suite executives, board presentations, steering committee reviews

## When to use

When a static HTML report or memo is not suitable and stakeholders need a polished, editable PowerPoint deck — for example, for a live presentation, a board pack, or a document that will be further edited by others.

## Required inputs

- Person query CSV with Copilot metrics and HR attributes
- At least 8 weeks of data recommended (12+ weeks preferred)
- HR attributes for organizational breakdowns

## Assumptions

- Data is at person-week granularity
- `PersonId` is a consistent anonymized identifier
- `MetricDate` is a date field representing the start of each week
- Rows with missing Copilot metric values likely represent unlicensed users
- The `vivainsights` R or Python package is available in the environment
- A package for creating PowerPoint files with native charts is available (e.g., `officer` + `mschart` in R, or `python-pptx` in Python)

## Recommended output

A .pptx file with 10–15 slides, using editable native PowerPoint charts (not pasted images) so that recipients can modify the deck as needed.

## Prompt

```
You are a senior people analytics consultant. Your task is to generate a polished, exec-ready PowerPoint deck (.pptx) summarizing Microsoft Copilot adoption from a Viva Insights person query export. The deck must use editable native PowerPoint charts — NOT pasted images — so that recipients can modify charts and data as needed.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.
- R: Use the `officer` and `mschart` packages for native PowerPoint chart generation.
- Python: Use the `python-pptx` package. Note that native chart support in python-pptx is more limited — if advanced chart types are needed, R with `mschart` is recommended.

DATA LOADING AND PREPARATION
1. Load the person query CSV using `import_query()` from the `vivainsights` library (R or Python). This handles variable name cleaning and type parsing automatically.
2. Identify Copilot metric columns by checking for columns containing the word "Copilot" in their name. Reference the taxonomy at https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv to classify and validate the detected metrics. Use `Total_Copilot_actions_taken` as the primary activity metric.
3. Run `extract_hr(df)` from the `vivainsights` library to identify available HR / organizational attribute columns.
4. Classify each person-week:
   - "Licensed": at least one non-null, non-zero Copilot metric value.
   - "Active": licensed AND Total_Copilot_actions_taken > 0.
5. Print: date range, total persons, licensed persons, active persons.

METRIC CALCULATIONS
6. Compute the following metrics for the deck:

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
   j. Average weekly time saved per active user
   k. Annualized estimated value (using $75/hour default, configurable)

SLIDE DECK GENERATION
7. Create a PowerPoint deck with the following slides. All charts must be native editable PowerPoint charts (created with mschart/officer in R or python-pptx chart objects in Python), NOT static images.

   SLIDE 1: Title slide
   - Title: "Copilot Adoption Review"
   - Subtitle: Reporting period [start date] to [end date]
   - Date generated

   SLIDE 2: Executive summary
   - 3-4 key bullet points summarizing the most important findings
   - Lead with the headline adoption rate and trend direction
   - Use bold text for key numbers

   SLIDE 3: At a Glance — KPI cards
   - Layout showing key metrics: adoption rate, trend, active users, avg actions/user, avg assisted hours/user
   - Use directional indicators (▲ ▼ ►) for trends
   - Use text boxes with large numbers — these do not need to be charts

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
   - Suppress segments with fewer than 5 licensed users
   - If there are more than 12 segments, show only the top and bottom 6

   SLIDE 10: Top and bottom performers
   - A table or two-column layout showing:
     - Top 3 organizations by adoption rate (with their rates and user counts)
     - Bottom 3 organizations by adoption rate
   - Brief annotation on what distinguishes leading and lagging groups

   SLIDE 11: ROI summary (if Copilot_Assisted_Hours is available)
   - Average weekly time saved per active user
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

   Adjust the slide count as needed — aim for 10-15 slides total. Skip slides where data is unavailable rather than leaving them blank.

8. Design guidelines:
   - Use a clean, professional layout with consistent fonts (e.g., Calibri or Segoe UI)
   - Use a consistent color palette across all charts
   - Keep text concise — use bullet points, not paragraphs
   - Every chart must have a clear title and takeaway annotation
   - Number all slides

9. Save the deck as "copilot_adoption_deck_YYYYMMDD.pptx".

IMPORTANT NOTES
- All charts MUST be native editable PowerPoint chart objects, not static images. This is the primary requirement — recipients need to be able to modify charts and update data.
- Do NOT use RMarkdown or Jupyter notebook as an intermediary for this output. Generate the .pptx file directly using the appropriate PowerPoint package.
- Handle missing values correctly: null in Copilot columns = unlicensed, 0 = licensed but inactive.
- Suppress segments with fewer than 5 users to protect privacy.
- Do NOT fabricate numbers. Every statistic must be computed from the data.
- If a chart type is not supported natively by the PowerPoint library, use the closest available chart type and note the limitation.
```

## Adaptation notes

- **R is recommended** for this prompt due to the `officer` + `mschart` packages, which provide excellent native PowerPoint chart support. Python's `python-pptx` has more limited chart types.
- **Adjust the hourly rate** for ROI calculations by modifying the $75/hour default. Add _"Use $X/hour for the hourly rate assumption"_ to the prompt.
- **Custom branding:** If your organization has a PowerPoint template (.potx), add: _"Use the template file at [path] as the base for the deck."_ The `officer` package in R supports applying templates.
- **Additional slides:** Add or remove slides by modifying the slide list. For a shorter deck (board summary), keep only slides 1–5 and 12.
- **Localization:** For non-English audiences, add: _"Generate all slide text and chart labels in [language]."_

## Common failure modes

- **Agent pastes images instead of creating native charts.** This is the most common failure. Verify that charts are editable by opening the .pptx and clicking on a chart — you should be able to edit the data. If the agent falls back to images, explicitly instruct it to use `mschart` (R) or chart objects in `python-pptx` (Python).
- **Agent uses an R/Python → HTML → PPTX conversion pipeline.** This produces image-based slides, not native charts. The agent should create the .pptx directly using the PowerPoint package.
- **Chart types not supported.** Some chart types (e.g., heatmaps) may not be available as native PowerPoint charts. The agent should substitute with a table or the closest available chart type.
- **Slide layout is cluttered.** Review the deck for readability. Each slide should convey one main point. If a slide has too much content, ask the agent to split it.
- **Agent does not handle missing metrics gracefully.** If `Copilot_Assisted_Hours` or other optional metrics are not in the data, the agent should skip those slides rather than error.

<div class="prompt-nav">
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-roi/">← Previous: ROI Estimation</a>
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-agent-usage/">Next: Agent Usage Analysis →</a>
</div>
