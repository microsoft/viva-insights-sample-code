---
layout: default
title: "Causal Toolkit — How It Works (Methodology)"
eyebrow: "Causal inference · Toolkit · Reference"
description: "The methodology behind the Copilot Causal Toolkit — double machine learning, causal forests, data aggregation, assumptions, model specifications, and limitations."
permalink: /copilot-causal-toolkit-methodology/
css: "/assets/css/causal-toolkit.css"
---

<nav class="ct-series-nav" aria-label="Toolkit steps">
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit/">Overview</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/"><span class="ct-chip-step">1</span>Set up</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-data/"><span class="ct-chip-step">2</span>Data</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/"><span class="ct-chip-step">3</span>Configure</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-run/"><span class="ct-chip-step">4</span>Run</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/"><span class="ct-chip-step">5</span>Interpret</a>
  <a class="ct-chip is-current" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">How it works</a>
</nav>

# How it works

The toolkit uses **Double Machine Learning (DML)** with **two-way fixed-effects residualization** to estimate the causal effect of Copilot usage on workplace outcomes. This page is reference material for readers who want to understand the technical approach — you don't need to read it to run the analysis.

## What is causal inference?

Traditional correlation analysis can tell us that Copilot users have different collaboration patterns, but not whether Copilot *causes* those differences. People who adopt Copilot early might already differ from non-adopters in ways that affect their work.

Causal inference methods estimate what would happen if we *increased* someone's Copilot usage, accounting for:

- Pre-existing differences between people (some are naturally more productive)
- Time trends (work patterns change over time for everyone)
- Confounding factors (people with more meetings might also use Copilot more)

For a broader introduction, see the conceptual [Causal Inference in Copilot Analytics]({{ site.baseurl }}/causal-inference/) guide.

## The Double Machine Learning framework

The toolkit uses two primary DML estimators.

### LinearDML — Average Treatment Effects (ATE)

**Purpose:** estimate the average dose-response relationship across all individuals.

**How it works:**

1. Uses machine learning to model both how Copilot usage depends on covariates (propensity modeling) and how outcomes depend on covariates (outcome modeling).
2. Removes the predictable parts using these models (residualization).
3. Estimates the causal effect from the remaining "unexplained" variation.

**Key features:**

- **Spline featurization** — models non-linear dose-response curves (e.g. diminishing returns at high usage).
- **Cross-fitting** — sample splitting to avoid overfitting and ensure valid confidence intervals.
- **Doubly robust** — estimates stay valid even if one of the two models (propensity or outcome) is misspecified.

**Output:** a dose-response curve showing how outcomes change at different usage levels (e.g. 5, 10, 15, 20 actions/week).

### CausalForestDML — Heterogeneous Treatment Effects (CATE)

**Purpose:** identify which subgroups experience different treatment effects (heterogeneity analysis).

**How it works:**

1. Applies the same residualization as LinearDML.
2. Uses a random forest to discover which combinations of attributes predict larger/smaller effects.
3. Identifies subgroups automatically without pre-specifying them.

**Key features:**

- **Adaptive subgroup discovery** — finds meaningful subgroups from the data.
- **Individual Treatment Effect (ITE) estimation** — person-specific effect estimates.
- **Tree-based interpretation** — results can be visualized as decision trees.

**Output:** rankings of subgroups by treatment-effect magnitude, with confidence intervals for each.

## Data aggregation approach

Before running DML, the notebooks **aggregate the longitudinal data by person**.

**Why this matters:**

- **Simplifies analysis** — converts panel data (person-weeks) to cross-sectional data (person-level averages).
- **Reduces noise** — smooths out week-to-week fluctuations.
- **Person-level interpretation** — effects represent differences between individuals rather than within-person changes.

**The approach** — for each person: take the mean of all numeric variables across observed weeks, keep time-invariant demographic attributes, and produce one row per person.

<div class="ct-callout is-important" markdown="1">
<span class="ct-callout-label">What this means for interpretation</span>
Estimates compare people with different **average** Copilot usage levels. They show **cross-sectional associations** adjusted for observed covariates, and cannot distinguish effects of *adopting* Copilot from pre-existing differences. This is more descriptive than purely causal, but DML's double robustness provides some protection against confounding.
</div>

## Key assumptions

1. **Unconfoundedness** — we observe all important factors affecting both Copilot usage and outcomes (addressed by including comprehensive behavioral and demographic covariates).
2. **Positivity (overlap)** — every individual has some probability of any usage level (checked via treatment-distribution plots).
3. **SUTVA** — one person's Copilot usage doesn't affect another's outcomes (a potential concern with team-level spillovers).

## Model specifications used

**ML models for nuisance functions:**

- **Treatment model (T):** Random Forest Regressor with 100 trees.
- **Outcome model (Y):** Random Forest Regressor with 100 trees.
- **Final stage:** Linear regression (LinearDML) or Random Forest (CausalForestDML).

**Treatment featurization:**

- **Spline transformation:** 4 knots, degree-3 polynomial (smooth non-linear curves).
- **Alternative:** polynomial features up to degree 2 (for comparison).

**Control variables:**

- Time-varying: collaboration hours, meeting hours, email hours, network size, focus time.
- Time-invariant (via fixed effects): person-specific baselines, week-specific trends.
- Demographic: organization, function, level, manager status (for heterogeneity analysis).

## Interpreting the results

**The ATE tells you** the average effect of increasing Copilot usage by one action per week, how effects vary across the usage spectrum, and whether there are diminishing returns at high usage.

**The CATE tells you** which subgroups benefit most (or least), whether effects are positive for some groups and negative for others, and where to target adoption efforts.

**Confidence intervals** — 95% intervals are provided for all estimates; width reflects uncertainty, and non-overlap with zero indicates statistical significance at the 5% level.

## Robustness and sensitivity

The analysis includes several robustness checks:

1. **Alternative model specifications** — re-run with different ML algorithms.
2. **Different control variables** — test sensitivity to covariate selection.
3. **Varying featurization** — compare spline vs. polynomial treatment modeling.
4. **Subsampling** — check whether results are driven by outliers or specific periods.

Results are stored in `sensitivity_analysis_results_[timestamp].json`; the [Interpretation Guide]({{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/#interpreting-the-sensitivity-analysis) explains how to read them, including E-values and Rosenbaum bounds.

## Limitations and caveats

<div class="ct-callout is-warning" markdown="1">
<span class="ct-callout-label">What this analysis can and cannot tell you</span>

**Can tell you:** whether Copilot usage increases or decreases outcomes on average; which usage levels show the strongest effects; which subgroups experience different effects.

**Cannot tell you:** whether effects persist beyond the study period (external validity); the *mechanisms* explaining why Copilot has these effects (needs mediation analysis); long-term effects beyond 6 months (data-limited); team-level or organizational spillover effects (this is an individual-level analysis).
</div>

**Recommended interpretation:**

- Treat results as **descriptive of the observed period** rather than universal laws.
- Consider whether identified subgroups reflect genuine heterogeneity or chance variation.
- Validate findings by re-running on new data as it becomes available.
- Use results to inform hypotheses for further investigation, not as definitive proof.

<nav class="ct-pager" aria-label="Toolkit pagination">
  <a class="ct-pager-link" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/">
    <span class="ct-pager-dir">← Back</span>
    <span class="ct-pager-title">5 · Interpreting the outputs</span>
  </a>
  <a class="ct-pager-link is-next" href="{{ site.baseurl }}/copilot-causal-toolkit/">
    <span class="ct-pager-dir">Up →</span>
    <span class="ct-pager-title">Toolkit overview</span>
  </a>
</nav>
