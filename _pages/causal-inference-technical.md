---
layout: default
title: "Causal Inference Technical Implementation"
permalink: /causal-inference-technical/
---

# Causal Inference Technical Implementation Guide

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

## Overview

This technical guide provides detailed implementation instructions for applying causal inference methods to Copilot analytics data. It includes code examples, methodological details, and practical considerations for data scientists and analysts.

For a conceptual introduction to causal inference in Copilot analytics, see our [main guide]({{ site.baseurl }}/causal-inference/).

---

## Pre-Analysis Setup

### Data Requirements and Structure

Before implementing any causal inference method, ensure your data meets these requirements:

```python
import pandas as pd
import numpy as np

# Required data structure
required_columns = {
    'user_id': 'str',           # Unique identifier for each user
    'time_period': 'datetime',   # Time dimension (weekly/monthly)
    'treatment': 'int',          # Copilot usage (0/1 or continuous)
    'outcome': 'float',          # Productivity metric
    'confounders': 'various'     # Job role, tenure, team size, etc.
}

# Example data validation
def validate_causal_data(df):
    checks = {
        'temporal_ordering': df['time_period'].is_monotonic,
        'treatment_variation': df['treatment'].nunique() > 1,
        'outcome_completeness': df['outcome'].notna().mean() > 0.8,
        'sufficient_sample': len(df) > 100
    }
    return checks
```

### Variable Definition and Measurement

```python
# Treatment variable definitions
def define_treatment_binary(df, threshold=1):
    """Binary treatment: Used Copilot (1) vs Not Used (0)"""
    return (df['copilot_actions'] >= threshold).astype(int)

def define_treatment_continuous(df):
    """Continuous treatment: Number of Copilot actions per week"""
    return df['copilot_actions_per_week']

def define_treatment_categorical(df):
    """Categorical treatment: Usage intensity levels"""
    return pd.cut(df['copilot_actions'], 
                  bins=[0, 1, 10, 50, np.inf], 
                  labels=['None', 'Light', 'Moderate', 'Heavy'])

# Outcome variable examples
def productivity_metrics(df):
    """Various productivity outcome measures"""
    outcomes = {
        'tickets_resolved': df['tickets_closed'] / df['time_period_weeks'],
        'case_resolution_time': df['avg_case_hours'],
        'customer_satisfaction': df['csat_score'],
        'revenue_per_user': df['deals_closed'] * df['avg_deal_size']
    }
    return outcomes
```

---

## Method 1: Regression Adjustment

### Basic Implementation

```python
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

def regression_adjustment(df, outcome_col, treatment_col, confounder_cols):
    """
    Simple regression adjustment for causal inference
    """
    # Prepare formula
    confounders_str = ' + '.join(confounder_cols)
    formula = f'{outcome_col} ~ {treatment_col} + {confounders_str}'
    
    # Fit model
    model = smf.ols(formula, data=df).fit()
    
    # Extract treatment effect
    treatment_effect = model.params[treatment_col]
    confidence_interval = model.conf_int().loc[treatment_col]
    
    return {
        'ate': treatment_effect,
        'ci_lower': confidence_interval[0],
        'ci_upper': confidence_interval[1],
        'p_value': model.pvalues[treatment_col],
        'model': model
    }

# Example usage
confounders = ['tenure_months', 'job_level', 'team_size', 'manager_span']
result = regression_adjustment(df, 'tickets_per_week', 'copilot_usage', confounders)
print(f"Treatment Effect: {result['ate']:.2f} ({result['ci_lower']:.2f}, {result['ci_upper']:.2f})")
```

### Advanced Regression with Interactions

```python
def regression_with_interactions(df, outcome_col, treatment_col, confounder_cols, interaction_vars=None):
    """
    Regression adjustment with interaction terms for heterogeneous effects
    """
    base_formula = f'{outcome_col} ~ {treatment_col} + {" + ".join(confounder_cols)}'
    
    # Add interaction terms
    if interaction_vars:
        interactions = [f'{treatment_col}:{var}' for var in interaction_vars]
        formula = base_formula + ' + ' + ' + '.join(interactions)
    else:
        formula = base_formula
    
    model = smf.ols(formula, data=df).fit()
    
    # Calculate marginal effects for different groups
    effects = {}
    if interaction_vars:
        for var in interaction_vars:
            for level in df[var].unique():
                subset = df[df[var] == level]
                marginal_effect = model.params[treatment_col]
                if f'{treatment_col}:{var}' in model.params:
                    marginal_effect += model.params[f'{treatment_col}:{var}'] * level
                effects[f'{var}_{level}'] = marginal_effect
    
    return model, effects

# Example: Different effects by job level
model, effects = regression_with_interactions(
    df, 'tickets_per_week', 'copilot_usage', 
    confounders, interaction_vars=['job_level']
)
```

---

## Method 2: Propensity Score Methods

### Propensity Score Estimation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def estimate_propensity_scores(df, treatment_col, confounder_cols, method='logistic'):
    """
    Estimate propensity scores using various methods
    """
    X = df[confounder_cols]
    y = df[treatment_col]
    
    if method == 'logistic':
        model = LogisticRegression(random_state=42)
    elif method == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Method must be 'logistic' or 'random_forest'")
    
    # Fit model and predict probabilities
    model.fit(X, y)
    propensity_scores = model.predict_proba(X)[:, 1]
    
    # Check overlap assumption
    overlap_check = check_overlap(propensity_scores, y)
    
    return propensity_scores, model, overlap_check

def check_overlap(ps, treatment):
    """Check overlap assumption for propensity scores"""
    treated_ps = ps[treatment == 1]
    control_ps = ps[treatment == 0]
    
    overlap_stats = {
        'treated_range': (treated_ps.min(), treated_ps.max()),
        'control_range': (control_ps.min(), control_ps.max()),
        'common_support': (max(treated_ps.min(), control_ps.min()), 
                          min(treated_ps.max(), control_ps.max())),
        'overlap_quality': len(ps[(ps > 0.1) & (ps < 0.9)]) / len(ps)
    }
    
    return overlap_stats
```

### Propensity Score Matching

```python
from scipy.spatial.distance import cdist
import warnings

def propensity_score_matching(df, ps_col, treatment_col, outcome_col, ratio=1, caliper=0.1):
    """
    Nearest neighbor propensity score matching
    """
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    matched_pairs = []
    
    for _, treated_unit in treated.iterrows():
        # Calculate distances to all control units
        distances = np.abs(control[ps_col] - treated_unit[ps_col])
        
        # Apply caliper constraint
        valid_matches = distances <= caliper
        if not valid_matches.any():
            continue
            
        # Find closest matches
        closest_indices = distances[valid_matches].nsmallest(ratio).index
        
        for idx in closest_indices:
            matched_pairs.append({
                'treated_id': treated_unit.name,
                'control_id': idx,
                'treated_outcome': treated_unit[outcome_col],
                'control_outcome': control.loc[idx, outcome_col],
                'ps_distance': distances[idx]
            })
    
    # Calculate treatment effect
    matched_df = pd.DataFrame(matched_pairs)
    ate = (matched_df['treated_outcome'] - matched_df['control_outcome']).mean()
    
    return ate, matched_df

# Example usage
ps_scores, ps_model, overlap = estimate_propensity_scores(
    df, 'copilot_usage', confounders
)
df['propensity_score'] = ps_scores

ate, matches = propensity_score_matching(
    df, 'propensity_score', 'copilot_usage', 'tickets_per_week'
)
print(f"Matched ATE: {ate:.2f}")
```

### Inverse Probability Weighting

```python
def inverse_probability_weighting(df, ps_col, treatment_col, outcome_col):
    """
    Inverse probability of treatment weighting (IPTW)
    """
    # Calculate weights
    weights = np.where(
        df[treatment_col] == 1,
        1 / df[ps_col],  # Weight for treated units
        1 / (1 - df[ps_col])  # Weight for control units
    )
    
    # Stabilize weights
    weights = weights / weights.mean()
    
    # Calculate weighted outcomes
    treated_outcome = np.average(
        df[df[treatment_col] == 1][outcome_col],
        weights=weights[df[treatment_col] == 1]
    )
    
    control_outcome = np.average(
        df[df[treatment_col] == 0][outcome_col],
        weights=weights[df[treatment_col] == 0]
    )
    
    ate = treated_outcome - control_outcome
    
    return ate, weights

# Example usage
ate_iptw, weights = inverse_probability_weighting(
    df, 'propensity_score', 'copilot_usage', 'tickets_per_week'
)
print(f"IPTW ATE: {ate_iptw:.2f}")
```

---

## Method 3: Difference-in-Differences

### Standard DiD Implementation

```python
import pandas as pd
from linearmodels import PanelOLS

def difference_in_differences(df, unit_col, time_col, treatment_col, outcome_col, post_col):
    """
    Standard difference-in-differences estimation
    """
    # Create interaction term
    df['treatment_post'] = df[treatment_col] * df[post_col]
    
    # Prepare data for panel regression
    df_panel = df.set_index([unit_col, time_col])
    
    # Run DiD regression: Y = α + β₁T + β₂Post + β₃(T×Post) + ε
    formula = f'{outcome_col} ~ {treatment_col} + {post_col} + treatment_post + EntityEffects + TimeEffects'
    
    model = PanelOLS.from_formula(formula, df_panel).fit(cov_type='clustered', cluster_entity=True)
    
    # Treatment effect is the coefficient on interaction term
    ate = model.params['treatment_post']
    
    return {
        'ate': ate,
        'std_error': model.std_errors['treatment_post'],
        'p_value': model.pvalues['treatment_post'],
        'confidence_interval': model.conf_int().loc['treatment_post'],
        'model': model
    }

# Example usage for phased Copilot rollout
df['post_rollout'] = (df['date'] >= '2024-01-01').astype(int)
df['treated_department'] = (df['department'].isin(['Sales', 'Support'])).astype(int)

did_result = difference_in_differences(
    df, 'user_id', 'date', 'treated_department', 'tickets_per_week', 'post_rollout'
)
print(f"DiD ATE: {did_result['ate']:.2f} ± {1.96 * did_result['std_error']:.2f}")
```

### Event Study Design

```python
def event_study_analysis(df, unit_col, time_col, treatment_col, outcome_col, event_time_col):
    """
    Event study design to test parallel trends and estimate dynamic effects
    """
    # Create relative time indicators
    relative_times = []
    for t in range(-6, 7):  # 6 periods before to 6 periods after
        indicator = f'rel_time_{t}'
        df[indicator] = ((df[event_time_col] == t) & (df[treatment_col] == 1)).astype(int)
        if t != -1:  # Omit t=-1 as reference period
            relative_times.append(indicator)
    
    # Prepare panel data
    df_panel = df.set_index([unit_col, time_col])
    
    # Event study regression
    formula = f'{outcome_col} ~ {" + ".join(relative_times)} + EntityEffects + TimeEffects'
    model = PanelOLS.from_formula(formula, df_panel).fit(cov_type='clustered', cluster_entity=True)
    
    # Extract coefficients and plot
    coefficients = {}
    for var in relative_times:
        time_point = int(var.split('_')[-1])
        coefficients[time_point] = {
            'estimate': model.params[var],
            'std_error': model.std_errors[var]
        }
    
    return coefficients, model

# Plotting function for event study
def plot_event_study(coefficients):
    import matplotlib.pyplot as plt
    
    times = sorted(coefficients.keys())
    estimates = [coefficients[t]['estimate'] for t in times]
    std_errors = [coefficients[t]['std_error'] for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(times, estimates, yerr=[1.96*se for se in std_errors], 
                 marker='o', capsize=5)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.5)
    plt.xlabel('Relative Time (Periods)')
    plt.ylabel('Treatment Effect')
    plt.title('Event Study: Dynamic Treatment Effects')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Method 4: Instrumental Variables

### Two-Stage Least Squares

```python
from linearmodels.iv import IV2SLS

def instrumental_variables_analysis(df, outcome_col, treatment_col, instrument_col, confounder_cols):
    """
    Two-stage least squares with instrumental variables
    """
    # Prepare formula: outcome ~ confounders + [treatment ~ instrument + confounders]
    confounders_str = ' + '.join(confounder_cols)
    formula = f'{outcome_col} ~ {confounders_str} + [{treatment_col} ~ {instrument_col} + {confounders_str}]'
    
    # First stage: Check instrument strength
    first_stage_formula = f'{treatment_col} ~ {instrument_col} + {confounders_str}'
    first_stage = smf.ols(first_stage_formula, data=df).fit()
    f_stat = first_stage.fvalue
    
    # Second stage: IV estimation
    iv_model = IV2SLS.from_formula(formula, df).fit()
    
    # Weak instrument test
    weak_instrument = f_stat < 10  # Rule of thumb: F-stat should be > 10
    
    return {
        'ate': iv_model.params[treatment_col],
        'std_error': iv_model.std_errors[treatment_col],
        'confidence_interval': iv_model.conf_int().loc[treatment_col],
        'first_stage_f_stat': f_stat,
        'weak_instrument_warning': weak_instrument,
        'model': iv_model
    }

# Example: Using random license assignment as instrument
# (This requires that licenses were randomly assigned initially)
iv_result = instrumental_variables_analysis(
    df, 'tickets_per_week', 'copilot_usage', 'license_assigned', confounders
)

if iv_result['weak_instrument_warning']:
    print("Warning: Weak instrument detected!")
print(f"IV ATE: {iv_result['ate']:.2f}")
```

---

## Method 5: Doubly Robust Methods

### AIPW (Augmented Inverse Probability Weighting)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

def doubly_robust_estimation(df, outcome_col, treatment_col, confounder_cols):
    """
    Augmented Inverse Probability Weighting (AIPW) - Doubly Robust method
    """
    X = df[confounder_cols]
    T = df[treatment_col]
    Y = df[outcome_col]
    
    # Step 1: Estimate propensity scores
    ps_model = LogisticRegression(random_state=42)
    propensity_scores = cross_val_predict(ps_model, X, T, method='predict_proba')[:, 1]
    
    # Step 2: Estimate outcome regression for both treatment levels
    # Outcome model for treated units
    treated_idx = T == 1
    outcome_model_1 = RandomForestRegressor(random_state=42)
    outcome_model_1.fit(X[treated_idx], Y[treated_idx])
    mu_1 = outcome_model_1.predict(X)
    
    # Outcome model for control units
    control_idx = T == 0
    outcome_model_0 = RandomForestRegressor(random_state=42)
    outcome_model_0.fit(X[control_idx], Y[control_idx])
    mu_0 = outcome_model_0.predict(X)
    
    # Step 3: AIPW estimation
    # Treated units component
    treated_component = (T * (Y - mu_1) / propensity_scores) + mu_1
    
    # Control units component  
    control_component = ((1 - T) * (Y - mu_0) / (1 - propensity_scores)) + mu_0
    
    # Average treatment effect
    ate = np.mean(treated_component - control_component)
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_ates = []
    
    for _ in range(n_bootstrap):
        bootstrap_idx = np.random.choice(len(df), size=len(df), replace=True)
        bootstrap_ate = np.mean(
            treated_component[bootstrap_idx] - control_component[bootstrap_idx]
        )
        bootstrap_ates.append(bootstrap_ate)
    
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)
    
    return {
        'ate': ate,
        'confidence_interval': (ci_lower, ci_upper),
        'propensity_scores': propensity_scores,
        'outcome_predictions': {'mu_0': mu_0, 'mu_1': mu_1}
    }

# Example usage
dr_result = doubly_robust_estimation(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
print(f"Doubly Robust ATE: {dr_result['ate']:.2f} "
      f"({dr_result['confidence_interval'][0]:.2f}, {dr_result['confidence_interval'][1]:.2f})")
```

---

## Sensitivity Analysis and Robustness Checks

### Sensitivity to Unobserved Confounding

```python
def sensitivity_analysis_unobserved_confounding(base_estimate, confounder_strength_range):
    """
    Assess sensitivity to potential unobserved confounders
    """
    sensitivity_results = []
    
    for gamma in confounder_strength_range:
        # Rosenbaum bounds calculation
        # This is a simplified version - full implementation requires more complex calculations
        lower_bound = base_estimate / (1 + gamma)
        upper_bound = base_estimate / (1 - gamma) if gamma < 1 else np.inf
        
        sensitivity_results.append({
            'confounder_strength': gamma,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'significant': lower_bound > 0 or upper_bound < 0
        })
    
    return sensitivity_results

# Example usage
gamma_range = np.arange(0, 0.5, 0.05)
sensitivity = sensitivity_analysis_unobserved_confounding(ate, gamma_range)
```

### Placebo Tests

```python
def placebo_tests(df, outcome_col, treatment_col, confounder_cols, method='regression'):
    """
    Conduct placebo tests using pre-treatment periods or alternative outcomes
    """
    placebo_results = {}
    
    # Temporal placebo: Use pre-treatment period
    pre_treatment = df[df['time_period'] < df['treatment_start_date']]
    if len(pre_treatment) > 0:
        if method == 'regression':
            placebo_result = regression_adjustment(
                pre_treatment, outcome_col, treatment_col, confounder_cols
            )
            placebo_results['temporal'] = placebo_result
    
    # Outcome placebo: Use unrelated outcomes
    unrelated_outcomes = ['vacation_days', 'sick_days', 'training_hours']
    for outcome in unrelated_outcomes:
        if outcome in df.columns:
            placebo_result = regression_adjustment(
                df, outcome, treatment_col, confounder_cols
            )
            placebo_results[f'outcome_{outcome}'] = placebo_result
    
    return placebo_results

# Example usage
placebo_results = placebo_tests(df, 'tickets_per_week', 'copilot_usage', confounders)
```

---

## Results Interpretation and Reporting

### Effect Size Calculation and Business Translation

```python
def interpret_causal_results(ate, baseline_mean, confidence_interval, outcome_unit='tickets'):
    """
    Translate causal estimates into business-relevant terms
    """
    # Calculate effect sizes
    absolute_effect = ate
    percentage_effect = (ate / baseline_mean) * 100
    
    # Statistical significance
    is_significant = not (confidence_interval[0] <= 0 <= confidence_interval[1])
    
    # Business translation examples
    if outcome_unit == 'tickets':
        monthly_effect = absolute_effect * 4.33  # weeks to months
        annual_effect = absolute_effect * 52     # weeks to year
        
        business_translation = {
            'weekly_increase': f"{absolute_effect:.1f} additional tickets per week",
            'monthly_increase': f"{monthly_effect:.1f} additional tickets per month", 
            'annual_increase': f"{annual_effect:.0f} additional tickets per year",
            'percentage_improvement': f"{percentage_effect:.1f}% improvement"
        }
    
    return {
        'absolute_effect': absolute_effect,
        'percentage_effect': percentage_effect,
        'is_significant': is_significant,
        'business_translation': business_translation,
        'confidence_interval': confidence_interval
    }

# Example usage
baseline_productivity = df[df['copilot_usage'] == 0]['tickets_per_week'].mean()
interpretation = interpret_causal_results(
    ate=2.5, 
    baseline_mean=baseline_productivity,
    confidence_interval=(1.8, 3.2),
    outcome_unit='tickets'
)

print(f"Copilot causes {interpretation['business_translation']['percentage_improvement']}")
print(f"That's {interpretation['business_translation']['annual_increase']}")
```

### Heterogeneous Treatment Effects Analysis

```python
def analyze_heterogeneous_effects(df, outcome_col, treatment_col, confounder_cols, subgroup_vars):
    """
    Analyze how treatment effects vary across different subgroups
    """
    subgroup_effects = {}
    
    for var in subgroup_vars:
        subgroup_effects[var] = {}
        
        for level in df[var].unique():
            subset = df[df[var] == level]
            if len(subset) > 50:  # Ensure sufficient sample size
                effect = regression_adjustment(
                    subset, outcome_col, treatment_col, confounder_cols
                )
                subgroup_effects[var][level] = effect
    
    return subgroup_effects

# Example usage
subgroup_vars = ['job_level', 'department', 'tenure_category']
heterogeneous_effects = analyze_heterogeneous_effects(
    df, 'tickets_per_week', 'copilot_usage', confounders, subgroup_vars
)

# Print results
for var, levels in heterogeneous_effects.items():
    print(f"\nHeterogeneous effects by {var}:")
    for level, effect in levels.items():
        print(f"  {level}: {effect['ate']:.2f} ({effect['ci_lower']:.2f}, {effect['ci_upper']:.2f})")
```

---

## Best Practices and Common Pitfalls

### Data Quality Checks

```python
def causal_inference_data_quality_check(df, outcome_col, treatment_col, confounder_cols):
    """
    Comprehensive data quality assessment for causal inference
    """
    quality_report = {}
    
    # 1. Missing data assessment
    missing_rates = df[confounder_cols + [outcome_col, treatment_col]].isnull().mean()
    quality_report['missing_data'] = missing_rates[missing_rates > 0.05]
    
    # 2. Treatment variation
    treatment_balance = df[treatment_col].value_counts(normalize=True)
    quality_report['treatment_balance'] = treatment_balance
    
    # 3. Outcome distribution
    quality_report['outcome_stats'] = df[outcome_col].describe()
    
    # 4. Temporal coverage
    if 'time_period' in df.columns:
        quality_report['temporal_coverage'] = {
            'start_date': df['time_period'].min(),
            'end_date': df['time_period'].max(),
            'n_periods': df['time_period'].nunique()
        }
    
    # 5. Sample size by subgroup
    if len(confounder_cols) > 0:
        sample_sizes = df.groupby(confounder_cols[0]).size()
        quality_report['min_subgroup_size'] = sample_sizes.min()
    
    return quality_report

# Example usage
quality_check = causal_inference_data_quality_check(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
```

### Model Diagnostics

```python
def model_diagnostics(model, df, residual_threshold=3):
    """
    Run diagnostics on causal inference models
    """
    diagnostics = {}
    
    # Residual analysis
    residuals = model.resid
    diagnostics['residual_stats'] = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'outliers': np.sum(np.abs(residuals) > residual_threshold)
    }
    
    # Homoscedasticity test (Breusch-Pagan)
    from statsmodels.stats.diagnostic import het_breuschpagan
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, model.model.exog)
    diagnostics['homoscedasticity'] = {
        'test_statistic': lm,
        'p_value': lm_pvalue,
        'homoscedastic': lm_pvalue > 0.05
    }
    
    # Normality test
    from scipy.stats import shapiro
    shapiro_stat, shapiro_p = shapiro(residuals)
    diagnostics['normality'] = {
        'test_statistic': shapiro_stat,
        'p_value': shapiro_p,
        'normal': shapiro_p > 0.05
    }
    
    return diagnostics
```

---

## Resources and Further Reading

### Recommended Libraries

```python
# Essential libraries for causal inference in Python
libraries = {
    'causal_inference': [
        'dowhy',           # Microsoft's causal inference library
        'causalml',        # Uber's causal ML library  
        'econml',          # Microsoft's econometrics + ML
        'causallib',       # IBM's causal inference library
    ],
    'statistical_models': [
        'statsmodels',     # Statistical modeling
        'linearmodels',    # Panel data and IV models
        'scikit-learn',    # Machine learning
    ],
    'data_manipulation': [
        'pandas',          # Data manipulation
        'numpy',           # Numerical computing
    ],
    'visualization': [
        'matplotlib',      # Basic plotting
        'seaborn',         # Statistical visualization
        'plotly',          # Interactive plots
    ]
}

# Installation command
install_command = """
pip install dowhy causalml econml causallib statsmodels linearmodels 
pip install scikit-learn pandas numpy matplotlib seaborn plotly
"""
```

### Example Analysis Workflow

```python
def complete_causal_analysis_workflow(df, outcome_col, treatment_col, confounder_cols):
    """
    Complete workflow for causal analysis of Copilot data
    """
    results = {}
    
    # Step 1: Data quality check
    quality_check = causal_inference_data_quality_check(df, outcome_col, treatment_col, confounder_cols)
    results['data_quality'] = quality_check
    
    # Step 2: Multiple estimation methods
    methods = {
        'regression': lambda: regression_adjustment(df, outcome_col, treatment_col, confounder_cols),
        'propensity_matching': lambda: propensity_score_analysis(df, outcome_col, treatment_col, confounder_cols),
        'doubly_robust': lambda: doubly_robust_estimation(df, outcome_col, treatment_col, confounder_cols)
    }
    
    estimates = {}
    for method_name, method_func in methods.items():
        try:
            estimates[method_name] = method_func()
        except Exception as e:
            print(f"Warning: {method_name} failed with error: {e}")
    
    results['estimates'] = estimates
    
    # Step 3: Sensitivity analysis
    if 'regression' in estimates:
        base_ate = estimates['regression']['ate']
        sensitivity = sensitivity_analysis_unobserved_confounding(base_ate, np.arange(0, 0.3, 0.05))
        results['sensitivity'] = sensitivity
    
    # Step 4: Business interpretation
    baseline_mean = df[df[treatment_col] == 0][outcome_col].mean()
    for method, estimate in estimates.items():
        if 'ate' in estimate:
            interpretation = interpret_causal_results(
                estimate['ate'], baseline_mean, 
                estimate.get('confidence_interval', (np.nan, np.nan))
            )
            results[f'{method}_interpretation'] = interpretation
    
    return results

# Run complete analysis
complete_results = complete_causal_analysis_workflow(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
```

---

*This technical guide provides the foundation for implementing rigorous causal inference methods with Copilot data. Always validate assumptions, check robustness, and interpret results in business context.*
