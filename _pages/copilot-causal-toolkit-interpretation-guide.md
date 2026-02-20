---
layout: default
title: "Copilot Causal Toolkit - Interpretation Guide"
permalink: /copilot-causal-toolkit-interpretation-guide/
---

# Copilot Causal Toolkit: Interpretation Guide

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide default Minima navigation to prevent duplicates */
.site-header .site-nav,
.site-header .trigger,
.site-header .page-link {
  display: none !important;
}
</style>

Now that you have successfully run the Copilot Causal Toolkit notebooks, you should see results appear in your Jupyter notebook and in your output directory (and with no errors). This document provides detailed guidance on how to interpret every output the toolkit produces, and what actions you should consider as a result.

For instructions on setting up and running the toolkit, see the main [Copilot Causal Toolkit](/copilot-causal-toolkit/) page.

---

## What you should see in the outputs

After completing a notebook run, the toolkit produces two categories of outputs:

1. **In-notebook outputs** — plots, tables, and summary statistics displayed inline as you run each cell
2. **Saved files** — CSV, PNG, JSON, and TXT files written to the `output/` directory

### Output directory structure

Your `output/` directory will follow this structure:

```
output/
└── Subgroup Analysis - [YOUR COMPANY]/
    ├── significant_subgroups_[timestamp].csv
    ├── all_significant_subgroups_effects_[timestamp].png
    ├── sensitivity_analysis_[timestamp].png
    ├── sensitivity_analysis_results_[timestamp].json
    │
    ├── Subgroup_1_[Name]/
    │   ├── ate_plot_Total_Copilot_actions_taken_[timestamp].png
    │   ├── ate_results_Total_Copilot_actions_taken_[timestamp].csv
    │   ├── subgroup_definition.txt
    │   └── transition_matrix_Total_Copilot_actions_taken_[timestamp].csv
    │
    ├── Subgroup_2_[Name]/
    │   └── (same files as above)
    │
    ├── Subgroup_3_[Name]/
    ├── Subgroup_4_[Name]/
    └── Subgroup_5_[Name]/
```

Each `[timestamp]` follows the format `YYYYMMDD_HHMM` (e.g., `20260116_1137`) and corresponds to when you ran the analysis, so you can track multiple runs.

---

## Interpreting results from the notebook

As you run the notebook cell by cell, several key outputs appear inline. This section walks through what to look for at each stage.

### Data loading and filtering summary

Early cells print a summary of your data after loading and filtering. Look for:

- **Shape** — number of rows (person-weeks) and columns in the dataset
- **Unique individuals** — the number of distinct people in the data
- **Date range** — confirms the time period you're analyzing
- **Copilot users** — the count and percentage of person-weeks with `Total_Copilot_actions_taken > 0`
- **Winsorization** — the 95th percentile cap applied to the treatment variable to reduce the influence of extreme outliers

**What to check:** Verify the number of unique individuals is reasonable for your organization. If the Copilot user percentage is very low (e.g., <5%), the analysis may have limited statistical power.

### Exploratory plots

The notebook generates three exploratory visualizations before the causal analysis:

1. **Bar chart of the outcome variable by Organization** — shows which organizational units have the highest/lowest outcome values (e.g., after-hours collaboration hours)
2. **Time trend of the outcome variable** — shows how the outcome has changed week-over-week across the analysis period
3. **Time trend of the treatment variable** — shows how Copilot adoption has changed over time

**What to check:** Look for obvious data quality issues such as sudden spikes or drops, missing weeks, or unreasonably large or small values. If the outcome or treatment variable is flat over time, the analysis may have limited variation to work with.

### Data aggregation

The notebook aggregates the longitudinal (person-week) data into a cross-sectional (person-level) snapshot by averaging all numeric metrics across weeks for each person. The output confirms:

- The shape of the aggregated dataset (one row per person)
- The grouping variables used (e.g., `PersonId`, `Organization`, `FunctionType`)
- The metrics averaged (collaboration, network, treatment, and outcome variables)

**What to check:** Ensure all expected variables are present and no critical variables are listed as missing.

### Subgroup combinations

The toolkit creates all pairwise combinations of the organizational attributes you specified in `SUBGROUP_VARS` (e.g., Organization × FunctionType, Organization × IsManager, etc.) and shows:

- How many combinations were generated
- How many meet the minimum sample size threshold (default: 50 observations)
- The 10 largest subgroups by sample size

**What to check:** If very few subgroups meet the minimum size threshold, consider reducing `min_group_size` or using fewer `SUBGROUP_VARS`.

### CATE model results

The CausalForestDML model estimates an **individual treatment effect** for each person. The notebook displays:

- **Treatment effects range** — the minimum and maximum individual effects
- **Mean treatment effect** — the average effect across all individuals (the overall ATE)

These individual effects are then used to identify which subgroups show the strongest effects.

### Top 5 subgroups

The notebook identifies and prints the top 5 subgroups with the most significant treatment effects. For each subgroup, it shows:

- **Name** — the combination of organizational attributes defining the subgroup (e.g., `Organization_Sales__and__FunctionType_IC`)
- **Effect** — the mean treatment effect in hours (with a ↓ or ↑ indicator)
- **p-value** — the statistical significance of the effect

The direction of "top" depends on the `FIND_NEGATIVE_EFFECTS` toggle:
- **True** (default for after-hours analysis): selects subgroups where Copilot most *reduces* the outcome
- **False** (default for external collaboration analysis): selects subgroups where Copilot most *increases* the outcome

---

## Interpreting significant subgroups (.csv)

**File:** `significant_subgroups_[timestamp].csv`

This is the most important summary file. It lists **all statistically significant subgroups** (p < 0.05), sorted by effect size.

### Column descriptions

| Column | Description |
|--------|------------|
| `name` | The subgroup identifier, combining two organizational attributes (e.g., `Organization__Aggregated__Sales__and__FunctionType__Aggregated__IC`) |
| `var1` | The first grouping variable name |
| `val1` | The value of the first grouping variable for this subgroup |
| `var2` | The second grouping variable name |
| `val2` | The value of the second grouping variable for this subgroup |
| `mean_effect` | The **average treatment effect** (ATE) in hours — the estimated change in the outcome when moving from 0 to the average Copilot usage level |
| `std_effect` | Standard deviation of individual treatment effects within the subgroup |
| `p_value` | Statistical significance (values < 0.05 are considered significant) |
| `n_observations` | Number of person-level observations in this subgroup |
| `n_users` | Number of unique individuals (typically equals `n_observations` since data is aggregated per-person) |
| `significant` | Boolean flag — `True` if p_value < 0.05 |

### How to read `mean_effect`

The `mean_effect` represents the **estimated causal change** in the outcome variable (in hours per week) when an individual moves from zero Copilot usage to the dataset's average usage level.

- **Negative values** (e.g., `-0.224`) mean Copilot usage is associated with a **reduction** in the outcome. For after-hours collaboration, this is the desirable direction (less burnout risk).
- **Positive values** (e.g., `+0.101`) mean Copilot usage is associated with an **increase** in the outcome. For external collaboration, this is the desirable direction (more customer engagement).

**Example from a real output:**

| Subgroup | mean_effect | p_value | n_users | Interpretation |
|----------|------------|---------|---------|----------------|
| Global Delivery | -0.2236 | 0.0000 | 652 | Each user's after-hours collaboration reduced by ~0.22 hours/week (~13 min/week) |
| Commerce | -0.1752 | 0.0042 | 59 | ~0.18 hours/week reduction (~11 min/week) |
| Testing | +0.0583 | 0.0477 | 113 | ~0.06 hours/week increase (~3.5 min/week) |

### What to look for

1. **Effect direction consistency**: Are most subgroups showing effects in the same direction, or is it mixed? Mixed results may indicate heterogeneity worth investigating.
2. **Effect magnitude**: Consider whether the effect sizes are practically meaningful. A statistically significant reduction of 0.05 hours/week (3 minutes) may not warrant action, while 0.5 hours/week (30 minutes) is substantial.
3. **Sample sizes**: Larger subgroups (higher `n_users`) produce more reliable estimates. Be cautious with subgroups that barely meet the minimum threshold.
4. **p-values**: Lower p-values indicate stronger statistical evidence. Effects with p < 0.001 are particularly strong.
5. **Opposing effects**: Pay close attention to subgroups that show effects in the *opposite* direction to the majority. These may represent populations where Copilot has unintended consequences, or they may reflect confounding.

---

## Interpreting the subgroup comparison bar chart (.png)

**File:** `all_significant_subgroups_effects_[timestamp].png`

<!-- TODO: Add screenshot here -->

This horizontal bar chart visualizes the `mean_effect` for **all statistically significant subgroups** side by side. It provides the quickest visual summary of your results.

### Reading the chart

- **Each bar** represents one significant subgroup
- **Bar length** represents the magnitude of the treatment effect (in hours)
- **Bar color** distinguishes direction:
  - **Green bars** = negative effects (reductions in the outcome)
  - **Blue bars** = positive effects (increases in the outcome)
- **Value labels** on each bar show the exact `mean_effect` rounded to 4 decimal places
- **p-value annotations** appear for highly significant effects (p < 0.01)

### Summary statistics box

In the top-left corner, a text box shows:
- **Total** — number of significant subgroups displayed
- **Positive** — count of subgroups with positive effects
- **Negative** — count of subgroups with negative effects
- **Avg Effect** — the average effect across all significant subgroups

### What to look for

1. **Distribution of effects**: Are most bars on one side of zero, or evenly split? A lopsided distribution suggests a general trend, while an even split suggests heterogeneity.
2. **Outlier subgroups**: Any bars that are substantially longer than the others deserve attention — these are the populations most affected by Copilot.
3. **Contrast with non-plotted subgroups**: Remember that this chart only shows *statistically significant* subgroups. Non-significant subgroups (which are not plotted) may simply lack sufficient sample size.

---

## Interpreting each subgroup's detailed results

For each of the top 5 subgroups, the toolkit generates a dedicated folder with four files. This section explains how to read each one.

### Subgroup definition (`subgroup_definition.txt`)

**Example content:**
```
Organization__Aggregated_=Global Delivery, FunctionType__Aggregated_=#ERROR
```

This text file records which two organizational attribute values define this subgroup. It is deliberately simple so you can quickly reference which population was analyzed.

**How to use:** When presenting findings to stakeholders, translate the machine-readable format into business language. For example:
- `Organization__Aggregated_=Global Delivery` → "Employees in the Global Delivery organization"
- `FunctionType__Aggregated_=Sales` → "Employees in the Sales function"

If a value appears as `#ERROR` or looks unexpected, this typically means the attribute had missing or null values in the source data. You may want to verify the data quality for that field.

---

### ATE results table (`ate_results_Total_Copilot_actions_taken_[timestamp].csv`)

This file contains the detailed **dose-response estimates** — how the outcome changes at each level of Copilot usage (0, 1, 2, 3, ... actions per week).

#### Column descriptions

| Column | Description |
|--------|------------|
| `Treatment` | The Copilot usage level (actions per week). Ranges from 0 to the maximum observed in the subgroup. |
| `ATE_Featurized` | Estimated treatment effect at this usage level from the **featurized model** (uses spline transformation to capture non-linear dose-response patterns). |
| `ATE_Baseline` | Estimated treatment effect at this usage level from the **baseline model** (assumes a simple linear effect). |
| `CI_Lower_Featurized` | Lower bound of the 95% confidence interval for the featurized model. |
| `CI_Upper_Featurized` | Upper bound of the 95% confidence interval for the featurized model. |
| `CI_Lower_Baseline` | Lower bound of the 95% confidence interval for the baseline model. |
| `CI_Upper_Baseline` | Upper bound of the 95% confidence interval for the baseline model. |
| `P_Value_Featurized` | Statistical significance at this treatment level for the featurized model. |
| `P_Value_Baseline` | Statistical significance at this treatment level for the baseline model. |

#### Understanding the two models

The toolkit fits **two DML models** for each subgroup, and for good reason:

1. **Featurized model** (SplineTransformer with 4 knots, degree 3): This model can capture non-linear dose-response patterns — for example, diminishing returns at high usage levels, or a threshold effect where benefits only kick in above a certain usage level. The spline transformation allows the effect curve to bend and flex.

2. **Baseline model** (no featurization): This model assumes a constant, linear effect — each additional Copilot action has the same impact regardless of the starting level. It serves as a simpler benchmark.

**When the two models agree** (their curves are close together), the effect is likely linear and straightforward. **When they diverge**, the relationship is non-linear, and the featurized model is usually more informative.

#### How to read the table

Each row represents a treatment level. The `ATE_Featurized` and `ATE_Baseline` columns tell you the **estimated change in the outcome** (in hours) when a person moves from 0 actions/week to that treatment level.

**Example interpretation:**

| Treatment | ATE_Featurized | ATE_Baseline | Interpretation |
|-----------|---------------|-------------|----------------|
| 0 | 0.000 | 0.000 | Baseline (no Copilot usage) — effect is always 0 at Treatment=0 |
| 5 | 0.387 | -0.099 | Featurized model estimates +0.39 hours change; baseline estimates -0.10 hours |
| 10 | 0.376 | -0.198 | Featurized effect has plateaued; baseline continues linearly |
| 15 | -0.067 | -0.297 | Featurized effect turns negative at higher usage levels |
| 20 | -0.558 | -0.397 | Both models now show negative effects |

In this example, the featurized model reveals a **non-linear pattern**: moderate Copilot usage (5-10 actions/week) has a positive effect, but very high usage (15+ actions) reverses the effect. The baseline model, constrained to be linear, misses this nuance entirely.

#### Confidence intervals and significance

- **Narrow confidence intervals** (CI_Lower and CI_Upper close together) indicate precise estimates.
- **Wide confidence intervals** indicate high uncertainty — typically at extreme treatment levels where few observations exist.
- **P_Value < 0.05** indicates the effect at that specific treatment level is statistically significant.
- **P_Value close to 1.0** (especially at Treatment=0) is expected and correct — the effect at zero treatment is always zero by definition.

#### What to look for

1. **Where is the effect strongest?** Identify the treatment level(s) where `ATE_Featurized` is largest in absolute terms.
2. **Are there diminishing returns?** If the effect flattens or reverses at higher treatment levels, there may be an optimal "sweet spot" for Copilot usage.
3. **Do the two models agree?** If the featurized and baseline models diverge substantially, the non-linear pattern is important.
4. **Where are confidence intervals narrowest?** This is where the estimate is most reliable, typically at treatment levels near the population median.

---

### ATE dose-response plot (`ate_plot_Total_Copilot_actions_taken_[timestamp].png`)

<!-- TODO: Add screenshot here -->

This plot is a visual representation of the ATE results table. It is the single most important visual output for each subgroup.

#### Reading the plot

- **X-axis**: Copilot usage level (actions per week), ranging from 0 to the maximum observed in the subgroup
- **Y-axis**: Average Treatment Effect on the outcome variable (hours per week)
- **Blue solid line** (`ATE (Featurized)`): The non-linear dose-response curve from the featurized model
- **Blue shaded region**: 95% confidence interval for the featurized model
- **Red dashed line** (`ATE (Baseline)`): The linear dose-response curve from the baseline model
- **Red shaded region**: 95% confidence interval for the baseline model
- **Black horizontal line at y=0**: The "no effect" reference line
- **Title**: Shows the subgroup name, total observations, and number of unique users

#### Interpreting common curve shapes

**1. Steadily declining curve (negative slope throughout)**

The outcome decreases at all Copilot usage levels. For after-hours collaboration, this means Copilot consistently reduces evening/weekend work. The stronger the slope, the stronger the reduction per additional action.

**2. Steadily rising curve (positive slope throughout)**

The outcome increases at all usage levels. For external collaboration, this means Copilot consistently increases time with customers/partners.

**3. Inverted-U shape (rises then falls)**

The outcome increases at low-to-moderate usage levels, then decreases at high usage. This suggests an optimal usage range. Users below the peak may benefit from increased adoption; users above the peak might be experiencing diminishing or negative returns.

**4. U shape (falls then rises)**

The outcome decreases at low usage levels, then increases at high levels. This is less common but may indicate that heavy Copilot users are in demanding roles that naturally involve more of the outcome behavior.

**5. Flat curve (near zero throughout)**

No meaningful effect at any usage level. If the confidence band is narrow and covers zero, you can be fairly confident there is no effect. If the confidence band is wide, the sample may be too small to detect an effect.

#### Confidence band interpretation

- **Narrow band (featurized CI stays tight)**: The estimate is precise; you can trust the shape of the curve.
- **Wide band near the edges**: Fewer people have extreme usage levels, so estimates at the very low and very high ends are less reliable.
- **Band crosses zero**: At that usage level, you cannot rule out no effect. Only where the entire band is above or below zero can you claim a statistically significant effect.
- **Featurized band much wider than baseline band**: The additional flexibility of the spline model comes at the cost of wider confidence intervals, which is expected. If this happens, the simpler baseline model may be more appropriate.

---

### Transition matrix (`transition_matrix_Total_Copilot_actions_taken_[timestamp].csv`)

This file compares **raw outcome differences** between groups of people at different Copilot usage levels. It serves as a simple, descriptive cross-check on the causal model results.

#### Column descriptions

| Column | Description |
|--------|------------|
| `Bucket_i` | The lower treatment intensity bucket (e.g., "Bottom 25%") |
| `Bucket_i_T_Range` | The range of Copilot usage values in bucket i (e.g., "(2.0, 7.5]") |
| `Users_in_Bucket_i` | Number of users in the lower bucket |
| `Bucket_j` | The higher treatment intensity bucket (e.g., "Top 10%") |
| `Bucket_j_T_Range` | The range of Copilot usage values in bucket j |
| `Users_in_Bucket_j` | Number of users in the higher bucket |
| `dYij` | The raw mean outcome difference between bucket j and bucket i (in hours). Positive = higher-usage group has a higher outcome; Negative = higher-usage group has a lower outcome. |
| `P_Value` | P-value from a two-sample t-test comparing outcomes between the two buckets |

#### Treatment buckets

The toolkit divides people into five treatment intensity buckets based on their average Copilot usage:

| Bucket | Definition |
|--------|-----------|
| Bottom 25% | 0th–25th percentile of usage |
| 25–50% | 25th–50th percentile |
| 50–75% | 50th–75th percentile |
| 75–90% | 75th–90th percentile |
| Top 10% | 90th–100th percentile |

Each row in the table compares two buckets (all 10 possible pairwise comparisons).

#### How to interpret

**Example row:**

| Bucket_i | Bucket_i_T_Range | Users_in_Bucket_i | Bucket_j | Bucket_j_T_Range | Users_in_Bucket_j | dYij | P_Value |
|----------|-----------------|-------------------|----------|-----------------|-------------------|------|---------|
| Bottom 25% | (2.0, 7.5] | 163 | Top 10% | (28.1, 53.0] | 66 | 0.096 | 0.867 |

This means: People in the Top 10% of Copilot usage (28-53 actions/week) have 0.096 hours/week *more* after-hours collaboration than people in the Bottom 25% (2-7.5 actions/week), but this difference is **not statistically significant** (p = 0.867).

#### Important caveats

- The transition matrix is a **descriptive** (not causal) analysis. It does not adjust for confounders the way the DML model does. It is provided as a sanity check and intuition builder.
- **Non-significant p-values** (p > 0.05) are common, especially with small sample sizes. This does not contradict the DML results, which use the full dataset and more sophisticated modeling.
- **Large dYij with large p-value:** The difference is large but unreliable — probably driven by a small sample. Be cautious about interpreting it.
- **Small dYij with small p-value:** The difference is small but precisely estimated — statistically significant but possibly not practically important.

#### What to look for

1. **Monotonic trends in dYij**: If dYij consistently increases or decreases as you compare increasingly different buckets (e.g., Bottom 25% vs. 25-50%, vs. 50-75%, etc.), this supports a dose-response relationship.
2. **Consistency with the DML results**: Do the raw differences point in the same direction as the causal estimates? If not, confounders may be playing a significant role (which is exactly why we use DML).
3. **Bucket sizes**: Verify that each bucket has a reasonable number of users. Very small buckets (e.g., <20 users) produce unreliable comparisons.

---

## Interpreting the sensitivity analysis

The sensitivity analysis assesses the **robustness** of results to potential unmeasured confounders — factors that might affect both Copilot usage and the outcome but were not included in the model. This is critical because no observational study can guarantee that all confounders have been measured.

### Sensitivity analysis results file (`sensitivity_analysis_results_[timestamp].json`)

This JSON file contains the complete sensitivity analysis output. It has three main sections:

#### 1. Overall analysis (`overall_analysis`)

Results for the entire population (not subgroup-specific):

| Field | Description |
|-------|------------|
| `sample_size` | Total number of individuals analyzed |
| `mean_effect` | Overall average treatment effect |
| `standard_error` | Standard error of the mean effect |
| `evalue_point` | E-value for the point estimate |
| `evalue_ci` | E-value for the confidence interval (more conservative) |
| `rosenbaum_critical_gamma` | Critical Gamma from Rosenbaum bounds (null if beyond tested range) |
| `original_p_value` | P-value from the Wilcoxon signed-rank test |

#### 2. Subgroup analysis (`subgroup_analysis`)

An array of results for each of the top 5 subgroups:

| Field | Description |
|-------|------------|
| `subgroup_name` | Name of the subgroup |
| `rank` | Rank (1 = strongest effect) |
| `mean_effect` | Mean treatment effect for this subgroup |
| `p_value` | Statistical significance |
| `sample_size` | Number of individuals |
| `evalue_point` | E-value for the point estimate |
| `evalue_ci` | E-value for the confidence interval |
| `rosenbaum_gamma` | Critical Gamma (null if beyond tested range) |
| `robustness_score` | Combined robustness measure: the minimum of `evalue_ci` and `rosenbaum_gamma` |

#### 3. Robustness categories (`robustness_categories`)

Each subgroup is classified into one of three categories:

| Category | Criteria | Meaning |
|----------|---------|---------|
| **Very Robust** | robustness_score ≥ 2.0 | An unmeasured confounder would need very strong associations with both treatment and outcome to explain away the result |
| **Moderately Robust** | robustness_score 1.5–2.0 | Moderate confounding could partially explain the result |
| **Potentially Fragile** | robustness_score < 1.5 | Relatively weak confounding could explain the result — interpret with caution |

### Sensitivity analysis plot (`sensitivity_analysis_[timestamp].png`)

<!-- TODO: Add screenshot here -->

This figure contains two side-by-side plots:

#### Left panel: Treatment Effect vs E-value

- **X-axis**: Mean treatment effect (hours)
- **Y-axis**: E-value (confidence interval)
- **Each point**: One subgroup, labeled by rank number
- **Red dashed line at E-value = 2.0**: Threshold for "strong robustness"

**Interpretation**: Points above the red line are robust to moderate unmeasured confounding. Points below are more sensitive. Ideally, subgroups with the largest effects should also have the highest E-values.

#### Right panel: Rosenbaum Gamma by subgroup

- **X-axis**: Subgroup rank
- **Y-axis**: Critical Gamma (Γ)
- **Bar color**:
  - **Green** (Γ ≥ 2.0): Strong robustness
  - **Orange** (Γ 1.5–2.0): Moderate robustness
  - **Red** (Γ < 1.5): Potentially fragile
- **Dashed lines**: Thresholds at Γ = 2.0 (red) and Γ = 1.5 (orange)

**Interpretation**: Taller green bars are better. A subgroup whose bar reaches above the red dashed line has results that would survive even substantial hidden bias in treatment assignment.

---

### Understanding E-values

The **E-value** (VanderWeele & Ding, 2017) answers the question:

> *"How strong would an unmeasured confounder need to be to completely explain away this effect?"*

Specifically, the E-value is the **minimum risk ratio** that an unmeasured confounder would need to have with **both** the treatment (Copilot usage) **and** the outcome (e.g., after-hours collaboration) to fully explain away the observed effect, conditional on all the covariates already in the model.

#### E-value thresholds

| E-value | Robustness level | What it means |
|---------|-----------------|--------------|
| < 1.5 | Potentially fragile | A relatively weak confounder (e.g., a factor that increases both Copilot usage and after-hours work by 50%) could explain the result |
| 1.5 – 2.0 | Moderate | Would require a moderately strong confounder |
| 2.0 – 3.0 | Strong | Would require a confounder with strong associations to both treatment and outcome |
| > 3.0 | Very strong | Highly unlikely that any single unmeasured confounder could explain the result |

#### Two E-values are reported

- **E-value (point estimate)**: How strong a confounder would need to be to reduce the point estimate to zero. This is the less conservative measure.
- **E-value (confidence interval)**: How strong a confounder would need to be to shift the entire confidence interval to include zero. This is the more conservative and more important measure.

Always focus on the **CI E-value** when assessing robustness.

#### Contextualizing E-values

To give E-values practical meaning, compare them against plausible confounders in your data:

- **Job complexity**: Could high-complexity roles both increase Copilot adoption (because these users have more tasks to automate) and increase after-hours work (because the work is demanding)? If the risk ratio of this association is less than the E-value, the result survives.
- **Manager expectations**: Could managers who push for tool adoption also push for longer hours? Again, compare the plausible strength of this confounder against the E-value.
- **Intrinsic motivation**: Could highly motivated employees both adopt Copilot faster and work more hours? Consider whether this confounder's strength would exceed the E-value.

---

### Understanding Rosenbaum bounds (Gamma)

The **Rosenbaum bounds** test (Rosenbaum, 2002) answers a slightly different question:

> *"How much hidden bias in treatment assignment could exist before the result becomes statistically non-significant?"*

The **Critical Gamma (Γ)** is the maximum factor by which two otherwise identical individuals could differ in their odds of treatment (due to unobserved factors) while the result still retains statistical significance at p < 0.05.

#### Gamma thresholds

| Γ value | Robustness level | What it means |
|---------|-----------------|--------------|
| < 1.5 | Fragile | Even modest differences in treatment assignment odds could eliminate statistical significance |
| 1.5 – 2.0 | Moderate | Tolerates moderate hidden bias |
| 2.0 – 3.0 | Strong | Results survive substantial hidden bias |
| > 3.0 (or null) | Very robust | Either beyond the tested range (Γ > 5.0) or the result is extremely robust |

#### When Gamma is null

A `null` value for `rosenbaum_gamma` can mean:
- The critical Gamma exceeds the tested range (Γ > 5.0) — the result is very robust
- The sample was too small for the analysis (n < 10)
- The original result was already non-significant

Check the `original_p_value` field to distinguish: if p is very small (e.g., < 0.001), a null Gamma usually means the result is very robust.

#### E-value vs Rosenbaum Gamma: Key distinction

| Metric | What it tests | Focus |
|--------|--------------|-------|
| E-value | Could confounding explain away the **magnitude** of the effect? | Effect size |
| Rosenbaum Γ | Could hidden bias eliminate **statistical significance**? | P-value |

Both matter. A result can have a low E-value but high Gamma (significant but potentially confounded) or a high E-value but low Gamma (robust in magnitude but marginally significant). Ideally, both should be high.

---

### Interpreting the robustness score

The `robustness_score` is a single summary metric: the **minimum** of the CI E-value and the Critical Gamma. It represents the "weakest link" — whichever sensitivity measure is most concerning.

| Robustness score | Verdict |
|-----------------|---------|
| ≥ 2.0 | Confident — results are robust |
| 1.5 – 2.0 | Cautiously optimistic — acknowledge limitations |
| < 1.5 | Exercise caution — results may be sensitive to unmeasured confounding |

---

### How to report sensitivity analysis results

**Template for robust results (score ≥ 2.0):**

> "The sensitivity analysis suggests these findings are robust to unmeasured confounding. An unmeasured confounder would need to be associated with both Copilot usage and the outcome by a risk ratio of at least [E-value] to explain away the observed effect. Additionally, hidden bias in treatment assignment would need to exceed Γ = [value] to overturn the statistical significance."

**Template for moderate results (score 1.5–2.0):**

> "The results show moderate robustness. While statistical significance is maintained under moderate hidden bias assumptions (Γ = [value]), the effect magnitude could potentially be explained by a moderately strong unmeasured confounder (E-value = [value]). Plausible confounders such as [job complexity, manager expectations] should be considered when interpreting these findings."

**Template for fragile results (score < 1.5):**

> "These findings should be interpreted with caution. The sensitivity analysis indicates that relatively modest unmeasured confounding (E-value = [value]) could potentially explain the observed effect. While the direction of the effect is suggestive, these results are best treated as hypothesis-generating rather than confirmatory. We recommend validating with alternative study designs."

---

## Putting it all together: A structured review process

For a systematic review of your results, we recommend the following workflow:

### Step 1: Start with the summary

Open `significant_subgroups_[timestamp].csv` and answer:
- How many subgroups showed significant effects?
- Are effects predominantly in one direction, or mixed?
- Which subgroups have the largest effects and the largest sample sizes?

### Step 2: Examine the bar chart

Open `all_significant_subgroups_effects_[timestamp].png` and look for:
- Overall pattern — mostly positive or mostly negative?
- Any outlier subgroups with especially strong or unexpected effects?
- Is the average effect practically meaningful?

### Step 3: Dive into the top subgroups

For each of the top 5 subgroups, review:
1. **`subgroup_definition.txt`** — Who is in this group?
2. **`ate_plot_[...].png`** — What does the dose-response curve look like? Linear or non-linear? Where is the effect strongest?
3. **`ate_results_[...].csv`** — At what treatment level is the effect statistically significant? What's the precise estimate at key usage levels (e.g., 5, 10, 15 actions/week)?
4. **`transition_matrix_[...].csv`** — Do the raw outcome differences support the causal findings?

### Step 4: Check robustness

Open `sensitivity_analysis_results_[timestamp].json` and `sensitivity_analysis_[timestamp].png`:
- Which subgroups are classified as "Very Robust," "Moderately Robust," or "Potentially Fragile"?
- Are the subgroups with the largest effects also the most robust?
- Would plausible unmeasured confounders exceed the E-value thresholds?

### Step 5: Synthesize and communicate

Prepare your findings for stakeholders:

1. **Lead with the business question** — "Does Copilot usage affect [outcome]?"
2. **State the overall finding** — "Across [N] employees, we found [direction] effects on [outcome] associated with increased Copilot usage."
3. **Highlight key subgroups** — "The strongest effects were observed in [subgroup], where [describe the dose-response pattern]."
4. **Acknowledge limitations** — "Sensitivity analysis classified [N] of 5 top subgroups as [robustness level]. Plausible unmeasured confounders include [examples]."
5. **Recommend next steps** — Based on the findings, suggest targeted interventions, additional data collection, or follow-up analyses.

---

## Glossary

| Term | Definition |
|------|-----------|
| **ATE** | Average Treatment Effect — the estimated causal effect of treatment on the outcome, averaged across all individuals |
| **CATE** | Conditional Average Treatment Effect — the treatment effect conditional on individual characteristics (allows for heterogeneity) |
| **CausalForestDML** | A machine learning-based causal inference estimator that discovers heterogeneous treatment effects across subgroups |
| **Confidence Interval (CI)** | A range of values within which the true effect is likely to fall (95% of the time at the 95% confidence level) |
| **Confounder** | A variable that affects both the treatment (Copilot usage) and the outcome, which must be controlled for to avoid bias |
| **DML** | Double Machine Learning — a framework that uses ML to flexibly model confounding while maintaining valid causal inference |
| **Dose-response curve** | A curve showing how the outcome changes across different treatment levels (rather than just comparing treated vs. untreated) |
| **E-value** | The minimum strength of confounding (on the risk ratio scale) needed to explain away the observed effect |
| **Featurized model** | A DML model that uses spline transformations to capture non-linear dose-response patterns |
| **LinearDML** | A DML estimator that models treatment effects as a linear function after residualizing confounders |
| **Rosenbaum Gamma (Γ)** | The maximum odds ratio of treatment assignment due to unobserved factors that still preserves statistical significance |
| **Robustness score** | The minimum of the E-value (CI) and Rosenbaum Gamma — represents overall sensitivity to unmeasured confounding |
| **SplineTransformer** | A feature transformation that models non-linear relationships using piecewise polynomials with smooth joins |
| **Transition matrix** | A descriptive comparison of raw outcome differences across treatment intensity buckets |
| **Winsorization** | Capping extreme values at a percentile threshold (e.g., 95th) to reduce the influence of outliers |
