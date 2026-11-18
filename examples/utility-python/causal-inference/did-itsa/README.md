# Difference-in-Differences and Interrupted Time Series Examples

A teaching notebook for difference-in-differences (DiD) and interrupted time series analysis (ITSA) on synthetic weekly Viva Insights style data. It shows how the estimators work and how to read their output. It is a learning resource and is not intended to produce the main estimate in a real analysis.

## What the notebook demonstrates

`DiD_ITSA_Analysis.ipynb` uses two helper modules in this folder, `did_analysis.py` and `itsa_analysis.py`. It covers:

- A 2x2 DiD, comparing treated and control groups before and after the intervention.
- Covariate-adjusted DiD variants, using the two-tier control strategy set out in the notebook.
- A group-mean event-time plot for the single licensing cohort, aligned on `LicensingWeek` and referenced to the calendar week immediately before licensing. It uses no regression and no fixed effects.
- ITSA with a control series, and ITSA on a single series.

## Expected data

The notebook reads `../data/longitudinal_data.csv`, a synthetic person-week panel. The configuration cell sets these columns for the notebook's own summaries and plots. The helper modules hard-code `External Collaboration Hours`, `Licensed`, `Post`, `LicensedXPost` and `Week` in their regression formulas, so rename your columns to match before using them:

| Setting | Default column | Meaning |
|---|---|---|
| `PERSON_ID_VAR` | `PersonID` | Unique person identifier |
| `TIME_VAR` | `Week` | Calendar week index (0 to 15 in the bundled data) |
| `OUTCOME_VARIABLE` | `External Collaboration Hours` | Outcome measure |
| `TREATMENT_INDICATOR` | `Licensed` | 1 for treated, 0 for control |
| `TREATMENT_WEEK_VAR` | `LicensingWeek` | Week the treatment started |
| `POST_TREATMENT_VAR` | `Post` | 1 for weeks after treatment |
| `BALANCE_VARIABLES`, `ADDITIONAL_CONTROLS` | Various | Pre-treatment and behavioural controls |

## How to run

1. Install Python 3.9 or later and the packages the notebook imports, including `pandas`, `numpy`, `statsmodels`, `scipy`, `scikit-learn`, `matplotlib` and `jupyter`.
2. Open `DiD_ITSA_Analysis.ipynb` from this folder, so that the `did_analysis` and `itsa_analysis` imports and the relative data path resolve.
3. Edit the configuration cell if you use your own data, then run all cells.

## Limitations

- **Not a regression event study, and not staggered-robust.** The event-study plot compares group means week by week, and the DiD models are pooled OLS without person or week fixed effects. When units adopt at different times, these comparisons can mix early and late adopters in ways that give misleading estimates, particularly when effects change over time or differ between adoption cohorts.
- **Default OLS standard errors.** The models do not cluster standard errors by person or by team, so confidence intervals are likely to be too narrow for repeated weekly observations. The event-study plot is descriptive and has no inferential intervals or significance claims. It does not account for the shared baseline or repeated observations by person.
- **No adjustment for function-led rollouts.** The estimators do not condition on function or other group-level timing differences, so they are not suited to producing the main estimate where rollout followed function or team lines.
- **Synthetic data.** Results on the bundled data show how the methods behave. They say nothing about any real organisation.
- **Association by default.** DiD and ITSA estimates are consistent with a causal reading only if the design assumptions hold, such as parallel trends and no concurrent changes. Check these before drawing conclusions.

## Where to go next

For choosing a design under staggered, function-led rollouts, see the [causal decision guide](https://microsoft.github.io/viva-insights-sample-code/articles/causal-questions-ai-usage/#route-staggered-did). For staggered-robust event studies, use [`event-study-did.Rmd`](../../../utility-r/event-study-did.Rmd) in R or [`event-study-did.py`](../../event-study-did.py) in Python.
