---
layout: page
title: "Prompt — Copilot Causal Toolkit"
permalink: /frontier-analytics-prompt-causal-toolkit/
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

# Copilot Causal Toolkit — Run & Interpret

[← Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)

## Purpose

Help users run a causal inference analysis using the [Copilot Causal Toolkit]({{ site.baseurl }}/copilot-causal-toolkit/) with their own data, and then interpret the results in a non-technical, business-ready format. This page contains **two prompts**:

1. **Prompt 1 — Run the Analysis:** Guides an agent to help the user select the right notebook, configure parameters, and execute the analysis successfully.
2. **Prompt 2 — Interpret the Results:** Guides an agent to read the analysis outputs and produce a clear, executive-friendly interpretation.

## Audience

HR analytics leads, people analytics practitioners who want to evaluate the causal impact of Copilot on business outcomes (e.g., seller productivity, employee wellbeing, engagement).

## When to use

- You have exported a Person Query or Super Users Report CSV with Copilot metrics and collaboration data.
- You want to understand whether Copilot usage **causes** changes in outcomes like external collaboration hours, after-hours work, or employee engagement — beyond just correlation.
- You have the [Copilot Causal Toolkit](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/copilot-causal-toolkit) cloned or downloaded locally.

## Required inputs

- Person Query CSV **or** Super Users Report CSV export with at least 12 weeks of data
- The Copilot Causal Toolkit repository cloned locally
- Python 3.8+ with Jupyter support (VS Code or JupyterLab)

## Assumptions

- The analysis is Python-only (R is not used in this toolkit)
- The treatment variable is `Total_Copilot_actions_taken`
- The user may not be a data scientist — the agent should explain decisions clearly
- The toolkit notebooks and helper modules are already available in the repository

---

## Prompt 1 — Run the Analysis

```
You are a people analytics engineer helping a user run a causal inference analysis using the Copilot Causal Toolkit. The toolkit documentation and notebooks are available at: https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit/

The toolkit contains Jupyter notebooks that use Double Machine Learning (DML) from the `econml` package to estimate the causal effect of Copilot usage on business outcomes.

Your job is to help the user select the correct notebook, configure it for their data, and run it successfully. Follow these steps:

STEP 1: ENVIRONMENT CHECK
1. Verify that Python 3.8+ is available.
2. Check that all required packages are installed: numpy, pandas, matplotlib, scipy, scikit-learn, econml, vivainsights. If any are missing, install them: pip install numpy pandas matplotlib scipy scikit-learn econml vivainsights
3. Verify Jupyter notebook support is available (either via JupyterLab or VS Code with the Jupyter extension).

STEP 2: DATA INSPECTION
1. Ask the user where their CSV data file is located.
2. Load the data using `import_query()` from the `vivainsights` package: import vivainsights as vi data = vi.import_query("path/to/data.csv")
3. Print the shape, date range (MetricDate or Date column), and number of unique PersonIds.
4. Run `vi.extract_hr(data)` to identify available HR/organizational attribute columns.
5. List all columns containing "Copilot" to confirm Copilot metrics are present.
6. Check whether `Total_Copilot_actions_taken` exists in the data.
7. Determine whether the data is a Person Query (has `MetricDate`) or a Super Users Report (has `Date` instead).

STEP 3: NOTEBOOK SELECTION Based on the user's data and goals, recommend the appropriate notebook. The available notebooks are:

  For Person Query (PQ) data:
  - CI-DML_ExtCollabHours_PQ.ipynb → Seller Productivity scenario (Outcome: External_collaboration_hours)
  - CI-DML_AftCollabHours_PQ.ipynb → Burnout Prevention scenario (Outcome: After_hours_collaboration_hours)
  - CI-DML_Engagement_PQ.ipynb → Employee Engagement scenario (Outcome: ordinal survey metric, e.g. eSat from Glint)

  For Super Users Report (SUR) data:
  - CI-DML_ExtCollabHours_SUR.ipynb → Seller Productivity scenario
  - CI-DML_AftCollabHours_SUR.ipynb → Burnout Prevention scenario

Ask the user which business question they want to answer:
  a) "Does Copilot usage increase time spent with external customers/partners?" → ExtCollabHours
  b) "Does Copilot usage reduce after-hours work and burnout risk?" → AftCollabHours
  c) "Does Copilot usage improve employee engagement survey scores?" → Engagement (PQ only)

If the user is unsure, recommend starting with the Seller Productivity scenario (ExtCollabHours) as it typically has the clearest business interpretation.

STEP 4: CONFIGURE PARAMETERS Open the selected notebook and update the following configuration sections:

1. FILE PATHS: Update `data_file_path` to point to the user's CSV file. Update `output_base_dir` to include the user's organization name.

2. ORGANIZATIONAL ATTRIBUTES: Update `SUBGROUP_VARS` to match the HR attribute columns identified by `extract_hr()` in Step 2. Include 2-4 key attributes (e.g., Organization, FunctionType, LevelDesignation, IsManager). Only include attributes that actually exist in the data.

3. NETWORK AND COLLABORATION VARIABLES: Verify that `NETWORK_VARS` and `COLLABORATION_VARS` match columns in the data. Remove any that do not exist.

4. ANALYSIS DIRECTION (AftCollabHours notebooks only): Set `FIND_NEGATIVE_EFFECTS`:
   - True = find subgroups where Copilot reduces after-hours work (typical use case)
   - False = find subgroups where Copilot increases after-hours work

5. ENGAGEMENT-SPECIFIC (Engagement notebook only): Update the outcome variable name and scale parameters (`OUTCOME_SCALE_MIN`, `OUTCOME_SCALE_MAX`) to match the survey metric.

Print a summary of all configured parameters for the user to review before proceeding.

STEP 5: RUN THE NOTEBOOK
1. Recommend running cell-by-cell for first-time users.
2. After each major section, pause and verify the output looks correct:
   - Data loading: correct shape, no missing key columns
   - Preprocessing: no unexpected dropped rows
   - DML estimation: model converges without errors
   - Subgroup analysis: results are generated for each subgroup variable
3. If errors occur, diagnose and fix them:
   - FileNotFoundError → check file path
   - KeyError → column name mismatch, update variable lists
   - ValueError → data type issues, check for non-numeric values
   - MemoryError → reduce data size or close other applications

STEP 6: VERIFY OUTPUTS
1. Check the output/ directory for generated files:
   - significant_subgroups_[timestamp].csv
   - sensitivity_analysis_results_[timestamp].json
   - Subgroup-specific folders with ATE plots, results CSVs, and transition matrices
2. Confirm that at least one subgroup folder was created with results.
3. Print a summary of what was generated and where the files are located.

IMPORTANT NOTES
- Do NOT modify the helper modules in script/modules/ — they are shared across notebooks.
- The treatment variable is always `Total_Copilot_actions_taken`. Do not change this.
- If the user's data has fewer than 12 weeks, warn them that results may be less reliable.
- Keep explanations accessible — the user may not be a data scientist.
```

## Adaptation notes — Prompt 1

- **Multiple scenarios:** If the user wants to run more than one scenario (e.g., both seller productivity and burnout prevention), run them sequentially — each notebook generates its own output folder.
- **Custom outcome variables:** For the Engagement notebook, the user needs to specify which survey metric to use. If they are unsure, suggest starting with `eSat` if it exists in the data.
- **Super Users Report limitations:** SUR data may lack some metrics available in Person Query. If critical columns are missing, recommend re-exporting from a Person Query instead.

---

## Prompt 2 — Interpret the Results

```
You are a senior people analytics advisor helping an HR analytics leader understand the results of a causal inference analysis. The analysis was run using the Copilot Causal Toolkit, which uses Double Machine Learning (DML) to estimate whether Copilot usage causes changes in a business outcome.

The analysis outputs are located in the output/ directory of the toolkit. Your job is to read these outputs and produce a clear, non-technical interpretation suitable for sharing with senior leadership.

STEP 1: LOCATE AND READ OUTPUTS
1. Scan the output/ directory for the most recent analysis results.
2. Read the key output files:
   - significant_subgroups_[timestamp].csv → which employee subgroups show statistically significant effects
   - sensitivity_analysis_results_[timestamp].json → how robust the findings are to potential hidden biases
   - Inside each subgroup folder:
     - ate_results_[treatment]_[timestamp].csv → the estimated causal effect (ATE)
     - definition.txt → how the subgroup is defined
     - transition_matrix_[treatment]_[timestamp].csv → how users move between usage levels
3. Also check for any ATE plot images (ate_plot_[timestamp].png) that visualize the effects.

STEP 2: PRODUCE AN EXECUTIVE INTERPRETATION Write a clear, non-technical summary structured as follows:

### 1. Headline Finding (2-3 sentences) State the main result in plain language. For example:
- "Copilot usage is associated with a statistically significant [increase/decrease] in [outcome] of [X hours/points] per person per week, after controlling for other factors."
- Or: "No statistically significant overall effect was found, but specific subgroups showed meaningful effects."

### 2. What This Analysis Does (3-4 sentences) Explain in accessible language:
- This is a causal inference analysis, not just a correlation. It uses a method called Double Machine Learning to isolate the effect of Copilot usage from other factors that might influence the outcome.
- Explain what the treatment variable is (Copilot actions) and what the outcome variable is.
- Note that this accounts for confounding factors such as collaboration patterns, network size, and focus time.

### 3. Key Results Table Present a summary table with:
- Subgroup name and definition
- Estimated effect size (Average Treatment Effect) and direction
- Statistical significance (p-value or confidence interval, explained in plain terms)
- Practical significance (is the effect large enough to matter?)

### 4. Subgroup Insights (3-5 bullets) For each significant subgroup, explain:
- Which group of employees is this? (e.g., "Managers in Engineering")
- What is the estimated effect? (e.g., "1.2 fewer after-hours collaboration hours per week")
- How confident are we? Reference the sensitivity analysis — higher E-values mean the finding is more robust to hidden biases.
- What does this mean practically? (e.g., "This suggests that Copilot helps managers in Engineering reduce after-hours work by roughly 1 hour per week.")

### 5. Robustness Assessment Interpret the sensitivity analysis results:
- E-values above 2.0 suggest the finding is reasonably robust — an unmeasured confounder would need to be at least twice as strong as the measured ones to explain away the result.
- E-values below 1.5 suggest the finding is more fragile and should be interpreted cautiously.
- Note any subgroups where effects were NOT statistically significant — absence of evidence is not evidence of absence.

### 6. Recommendations (2-4 bullets) Based on the results, suggest actionable next steps:
- If positive effects found: recommend expanding Copilot enablement in high-impact subgroups
- If negative or null effects found: suggest investigating barriers to effective Copilot use
- Recommend repeating the analysis with more data (longer time period) if effects are borderline
- Suggest combining with qualitative feedback (interviews, surveys) for a fuller picture

### 7. Caveats Include standard caveats:
- Causal inference from observational data has limitations — DML reduces but does not eliminate confounding risk.
- Results apply to the specific time period and population analyzed.
- The sensitivity analysis indicates robustness but cannot guarantee no unmeasured confounders.
- Subgroup results with small sample sizes should be interpreted cautiously.

FORMATTING
- Use clear section headers.
- Avoid statistical jargon — translate p-values, confidence intervals, and E-values into plain language (e.g., "We are 95% confident the true effect is between X and Y").
- Use bold text for key numbers and findings.
- Target length: 2-4 pages.
- Save as "copilot_causal_analysis_interpretation_YYYYMMDD.md".

IMPORTANT NOTES
- Do NOT overstate findings. If effects are small or borderline significant, say so clearly.
- Do NOT claim definitive causation — use language like "the analysis suggests" or "the estimated causal effect is."
- Frame everything in business terms the audience cares about (time saved, wellbeing impact, engagement improvement), not statistical terms.
- If no significant effects are found, this is still a valid and useful result — frame it constructively (e.g., "The data does not yet show a measurable effect, which may indicate the need for a longer observation period or targeted enablement strategies").
```

## Adaptation notes — Prompt 2

- **Audience customization:** If the audience is more technical (e.g., data science peers), you can ask the agent to include more statistical detail (confidence intervals, model diagnostics, covariate balance checks).
- **Multiple scenarios:** If the user ran multiple notebooks (e.g., both ExtCollabHours and AftCollabHours), ask the agent to synthesize findings across all scenarios into a single interpretation document.
- **Visualization requests:** You can ask the agent to generate additional charts from the output CSVs (e.g., a forest plot of subgroup effects, or a bar chart comparing effect sizes across organizational segments).

## Common failure modes

- **Agent cannot find the output files.** Ensure the agent is looking in the correct `output/` subdirectory. The exact folder name depends on what was set for `output_base_dir` during configuration.
- **Agent misinterprets the ATE direction.** A negative ATE for after-hours collaboration is a *positive* outcome (less burnout). Make sure the agent correctly contextualizes the direction of the effect relative to the business question.
- **Agent overstates statistical significance.** A p-value of 0.04 is borderline — the agent should present this with appropriate hedging rather than claiming a definitive causal effect.
- **Agent ignores sensitivity analysis.** The E-values are critical for assessing robustness. If the agent skips this section, prompt it: _"Also interpret the sensitivity analysis results from the JSON file."_
- **No significant results found.** This is a valid outcome. The agent should not treat it as a failure, but rather frame it constructively with suggestions for next steps (more data, different outcome, qualitative complement).

<div class="prompt-nav">
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-audit-parsing/">← Previous: Audit Log Parsing</a>
  <span class="nav-disabled">Next →</span>
</div>
