---
layout: default
title: "Causal Inference: Difference-in-Differences"
permalink: /causal-inference-did/
---

# Method 3: Difference-in-Differences (DiD)

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

Difference-in-Differences (DiD) exploits variation in the timing of treatment implementation across groups or regions. By comparing changes over time between treated and control groups, DiD controls for time-invariant confounders that affect both groups equally.

**Navigation:**
- [← Back to Propensity Score Methods]({{ site.baseurl }}/causal-inference-propensity/)
- [Next: Instrumental Variables →]({{ site.baseurl }}/causal-inference-iv/)

---

## When to Use Difference-in-Differences

**Key Advantages:**
- **Controls for unobserved confounders** that are constant over time
- **Natural experiment** design when treatment timing varies
- **Policy evaluation** strength for interventions with staggered rollout
- **Intuitive interpretation** as "difference of differences"

**Best for:**
- Interventions rolled out across teams/regions at different times
- Pre/post comparisons with control groups
- Policy changes with clear implementation dates
- When treatment assignment is correlated with time-invariant factors

**Key Assumptions:**
- **Parallel trends**: Control and treatment groups would follow similar trends without treatment
- **No anticipation effects**: Behavior doesn't change before actual treatment
- **Stable composition**: Group composition remains stable over time

---

## Data Structure and Preparation

DiD requires panel data with observations before and after treatment for both treated and control groups.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.formula.api as smf
from scipy import stats

def prepare_did_data(df, unit_col, time_col, treatment_col, outcome_col, treatment_date):
    """
    Prepare data for Difference-in-Differences analysis
    
    DiD Setup:
    - Panel data: Multiple observations per unit over time
    - Treatment timing: Clear before/after periods
    - Control group: Never treated or treated later
    
    Parameters:
    - unit_col: Identifier for units (teams, individuals, regions)
    - time_col: Time variable (date, period, month)
    - treatment_col: Binary treatment indicator
    - outcome_col: Outcome variable
    - treatment_date: When treatment began
    """
    df_did = df.copy()
    
    # Convert time column if needed
    if df_did[time_col].dtype == 'object':
        df_did[time_col] = pd.to_datetime(df_did[time_col])
    
    # Create before/after treatment indicator
    df_did['post_treatment'] = (df_did[time_col] >= treatment_date).astype(int)
    
    # Create treatment-time interaction
    df_did['treated_post'] = df_did[treatment_col] * df_did['post_treatment']
    
    print(f"=== DIFFERENCE-IN-DIFFERENCES DATA SETUP ===")
    print(f"Units: {df_did[unit_col].nunique()}")
    print(f"Time periods: {df_did[time_col].nunique()}")
    print(f"Treatment date: {treatment_date}")
    
    # Sample composition
    treated_units = df_did[df_did[treatment_col] == 1][unit_col].nunique()
    control_units = df_did[df_did[treatment_col] == 0][unit_col].nunique()
    
    pre_periods = (df_did[time_col] < treatment_date).sum()
    post_periods = (df_did[time_col] >= treatment_date).sum()
    
    print(f"Treated units: {treated_units}")
    print(f"Control units: {control_units}")
    print(f"Pre-treatment observations: {pre_periods}")
    print(f"Post-treatment observations: {post_periods}")
    
    # Check for balanced panel
    expected_obs = df_did[unit_col].nunique() * df_did[time_col].nunique()
    actual_obs = len(df_did)
    panel_balance = actual_obs / expected_obs
    
    print(f"Panel balance: {panel_balance:.1%} of expected observations")
    if panel_balance < 0.9:
        print("⚠️  Warning: Unbalanced panel - consider investigating missing data")
    
    # Data quality checks
    quality_issues = []
    
    # Check for units that change treatment status
    treatment_switches = df_did.groupby(unit_col)[treatment_col].nunique()
    switchers = (treatment_switches > 1).sum()
    if switchers > 0:
        quality_issues.append(f"{switchers} units change treatment status")
    
    # Check for missing outcomes
    missing_outcomes = df_did[outcome_col].isnull().sum()
    if missing_outcomes > 0:
        quality_issues.append(f"{missing_outcomes} missing outcome values")
    
    if quality_issues:
        print("Data quality issues:")
        for issue in quality_issues:
            print(f"  ⚠️  {issue}")
    
    return df_did

# Example data preparation
# This assumes you have panel data with team_id, date, copilot_usage, tickets_per_week
treatment_start = pd.to_datetime('2024-03-01')
df_did = prepare_did_data(
    df, 'team_id', 'date', 'copilot_usage', 'tickets_per_week', treatment_start
)
```

---

## Pre-Treatment Trends Analysis

The key assumption in DiD is **parallel trends** - that treated and control groups would have followed similar trends in the absence of treatment.

```python
def test_parallel_trends(df_did, unit_col, time_col, treatment_col, outcome_col, 
                        treatment_date, periods_before=6):
    """
    Test the parallel trends assumption
    
    Parallel trends is the identifying assumption of DiD:
    - Control group shows what would have happened to treatment group
    - Requires similar pre-treatment trends between groups
    - Tested using pre-treatment data only
    """
    # Focus on pre-treatment period
    pre_treatment = df_did[df_did[time_col] < treatment_date].copy()
    
    if len(pre_treatment) == 0:
        print("❌ No pre-treatment data available for trends test")
        return None
    
    # Limit to recent pre-treatment periods for relevance
    periods = sorted(pre_treatment[time_col].unique())[-periods_before:]
    pre_treatment = pre_treatment[pre_treatment[time_col].isin(periods)]
    
    print(f"=== PARALLEL TRENDS ANALYSIS ===")
    print(f"Pre-treatment periods analyzed: {len(periods)}")
    print(f"Date range: {min(periods)} to {max(periods)}")
    
    # Calculate group means by period
    trends_data = pre_treatment.groupby([time_col, treatment_col])[outcome_col].mean().reset_index()
    trends_pivot = trends_data.pivot(index=time_col, columns=treatment_col, values=outcome_col)
    trends_pivot.columns = ['Control', 'Treatment']
    
    # Visual trends test
    plt.figure(figsize=(12, 6))
    plt.plot(trends_pivot.index, trends_pivot['Control'], 'o-', label='Control Group', linewidth=2)
    plt.plot(trends_pivot.index, trends_pivot['Treatment'], 's-', label='Treatment Group', linewidth=2)
    plt.axvline(x=treatment_date, color='red', linestyle='--', alpha=0.7, label='Treatment Start')
    plt.xlabel('Date')
    plt.ylabel(outcome_col.replace('_', ' ').title())
    plt.title('Pre-Treatment Trends: Parallel Trends Test')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Statistical test for parallel trends
    # Create time trend variable (periods from first observation)
    pre_treatment['time_trend'] = (pre_treatment[time_col] - pre_treatment[time_col].min()).dt.days
    
    # Test if treatment group has different time trend
    trend_formula = f'{outcome_col} ~ time_trend * {treatment_col}'
    trend_model = smf.ols(trend_formula, data=pre_treatment).fit()
    
    interaction_coef = trend_model.params.get(f'time_trend:{treatment_col}', 0)
    interaction_pvalue = trend_model.pvalues.get(f'time_trend:{treatment_col}', 1)
    
    print(f"\n=== PARALLEL TRENDS TEST ===")
    print(f"Time trend interaction coefficient: {interaction_coef:.4f}")
    print(f"P-value: {interaction_pvalue:.4f}")
    
    if interaction_pvalue > 0.05:
        print("✅ PASS: No evidence against parallel trends (p > 0.05)")
        trends_valid = True
    else:
        print("⚠️  CONCERN: Evidence of different pre-trends (p ≤ 0.05)")
        print("   Consider: longer pre-period, group-specific trends, or alternative methods")
        trends_valid = False
    
    # Calculate trend differences
    control_trend = trends_pivot['Control'].pct_change().mean()
    treatment_trend = trends_pivot['Treatment'].pct_change().mean()
    trend_difference = treatment_trend - control_trend
    
    print(f"\nPre-treatment growth rates:")
    print(f"Control group: {control_trend:.1%} per period")
    print(f"Treatment group: {treatment_trend:.1%} per period")
    print(f"Difference: {trend_difference:.1%} per period")
    
    return {
        'trends_valid': trends_valid,
        'interaction_coef': interaction_coef,
        'p_value': interaction_pvalue,
        'trend_difference': trend_difference,
        'trends_data': trends_pivot
    }

# Test parallel trends assumption
trends_test = test_parallel_trends(
    df_did, 'team_id', 'date', 'copilot_usage', 'tickets_per_week', treatment_start
)
```

---

## Basic Difference-in-Differences Estimation

The core DiD specification compares the change in outcomes between treatment and control groups.

```python
def estimate_basic_did(df_did, treatment_col, outcome_col, unit_col='team_id'):
    """
    Basic two-period Difference-in-Differences estimation
    
    DiD Formula: (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
    
    Interpretation:
    - First difference: Change in treatment group
    - Second difference: Change in control group (counterfactual)
    - DiD effect: Treatment impact above natural trends
    """
    
    # Calculate group means by period
    group_means = df_did.groupby([treatment_col, 'post_treatment'])[outcome_col].mean().reset_index()
    means_pivot = group_means.pivot(index='post_treatment', columns=treatment_col, values=outcome_col)
    means_pivot.columns = ['Control', 'Treatment']
    means_pivot.index = ['Pre-treatment', 'Post-treatment']
    
    print(f"=== BASIC DIFFERENCE-IN-DIFFERENCES ===")
    print("\nGroup means by period:")
    print(means_pivot.round(3))
    
    # Calculate differences
    control_change = means_pivot.loc['Post-treatment', 'Control'] - means_pivot.loc['Pre-treatment', 'Control']
    treatment_change = means_pivot.loc['Post-treatment', 'Treatment'] - means_pivot.loc['Pre-treatment', 'Treatment']
    did_effect = treatment_change - control_change
    
    print(f"\n=== DIFFERENCE CALCULATIONS ===")
    print(f"Control group change: {control_change:.3f}")
    print(f"Treatment group change: {treatment_change:.3f}")
    print(f"Difference-in-Differences: {did_effect:.3f}")
    
    # Regression-based DiD (more flexible for standard errors)
    did_formula = f'{outcome_col} ~ {treatment_col} + post_treatment + treated_post'
    did_model = smf.ols(did_formula, data=df_did).fit(cov_type='cluster', cov_kwds={'groups': df_did[unit_col]})
    
    # Extract results
    did_coef = did_model.params['treated_post']
    did_se = did_model.bse['treated_post']
    did_ci = did_model.conf_int().loc['treated_post']
    did_pvalue = did_model.pvalues['treated_post']
    
    print(f"\n=== REGRESSION-BASED DiD ===")
    print(f"Treatment effect: {did_coef:.3f}")
    print(f"Standard error: {did_se:.3f}")
    print(f"95% CI: [{did_ci[0]:.3f}, {did_ci[1]:.3f}]")
    print(f"P-value: {did_pvalue:.4f}")
    
    # Effect size interpretation
    outcome_std = df_did[outcome_col].std()
    effect_size = did_coef / outcome_std
    
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    
    if abs(effect_size) < 0.2:
        magnitude = "small"
    elif abs(effect_size) < 0.5:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    print(f"Effect magnitude: {magnitude}")
    
    # Statistical significance
    if did_pvalue < 0.001:
        significance = "highly significant (p < 0.001)"
    elif did_pvalue < 0.01:
        significance = "very significant (p < 0.01)"
    elif did_pvalue < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p ≥ 0.05)"
    
    print(f"Statistical significance: {significance}")
    
    return {
        'did_effect': did_coef,
        'se': did_se,
        'ci': [did_ci[0], did_ci[1]],
        'p_value': did_pvalue,
        'effect_size': effect_size,
        'model': did_model,
        'group_means': means_pivot
    }

# Estimate basic DiD effect
basic_did_results = estimate_basic_did(df_did, 'copilot_usage', 'tickets_per_week')
```

---

## Event Study Analysis

Event studies examine treatment effects period-by-period around the intervention, providing detailed insights into effect dynamics.

```python
def event_study_analysis(df_did, time_col, treatment_col, outcome_col, unit_col, 
                        treatment_date, periods_before=6, periods_after=6):
    """
    Event study analysis for dynamic treatment effects
    
    Event studies show:
    - Pre-treatment effects (should be zero if parallel trends hold)
    - Treatment effect evolution over time
    - Whether effects persist, grow, or fade
    """
    
    df_event = df_did.copy()
    
    # Create event time variable (periods relative to treatment)
    if df_event[time_col].dtype == 'datetime64[ns]':
        # For datetime, use months
        df_event['event_time'] = ((df_event[time_col] - treatment_date).dt.days / 30).round().astype(int)
    else:
        # For numeric time
        treatment_period = df_event[df_event[time_col] >= treatment_date][time_col].min()
        df_event['event_time'] = df_event[time_col] - treatment_period
    
    # Focus on relevant event window
    event_window = (df_event['event_time'] >= -periods_before) & (df_event['event_time'] <= periods_after)
    df_event = df_event[event_window].copy()
    
    # Create period-specific treatment indicators
    # Omit t=-1 as reference period
    for t in range(-periods_before, periods_after + 1):
        if t != -1:  # Reference period
            df_event[f'treat_t{t}'] = ((df_event['event_time'] == t) & 
                                      (df_event[treatment_col] == 1)).astype(int)
    
    # Event study regression
    treatment_vars = [f'treat_t{t}' for t in range(-periods_before, periods_after + 1) if t != -1]
    formula = f'{outcome_col} ~ {" + ".join(treatment_vars)} + C({unit_col}) + C(event_time)'
    
    print(f"=== EVENT STUDY ANALYSIS ===")
    print(f"Event window: t-{periods_before} to t+{periods_after}")
    print(f"Reference period: t-1")
    print(f"Observations in window: {len(df_event)}")
    
    event_model = smf.ols(formula, data=df_event).fit(cov_type='cluster', 
                                                     cov_kwds={'groups': df_event[unit_col]})
    
    # Extract coefficients and confidence intervals
    event_results = []
    for t in range(-periods_before, periods_after + 1):
        if t == -1:
            # Reference period
            coef, se, ci_low, ci_high = 0, 0, 0, 0
        else:
            var_name = f'treat_t{t}'
            if var_name in event_model.params:
                coef = event_model.params[var_name]
                se = event_model.bse[var_name]
                ci = event_model.conf_int().loc[var_name]
                ci_low, ci_high = ci[0], ci[1]
            else:
                coef, se, ci_low, ci_high = 0, 0, 0, 0
        
        event_results.append({
            'event_time': t,
            'coefficient': coef,
            'se': se,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
    
    event_df = pd.DataFrame(event_results)
    
    # Plot event study results
    plt.figure(figsize=(12, 6))
    plt.plot(event_df['event_time'], event_df['coefficient'], 'o-', linewidth=2, markersize=6)
    plt.fill_between(event_df['event_time'], event_df['ci_low'], event_df['ci_high'], 
                     alpha=0.3, label='95% CI')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Treatment Start')
    plt.xlabel('Periods Relative to Treatment')
    plt.ylabel(f'Effect on {outcome_col.replace("_", " ").title()}')
    plt.title('Event Study: Dynamic Treatment Effects')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Pre-treatment effects test (should be zero)
    pre_treatment_effects = event_df[event_df['event_time'] < 0]['coefficient']
    pre_treatment_test = stats.ttest_1samp(pre_treatment_effects, 0)
    
    print(f"\n=== PRE-TREATMENT EFFECTS TEST ===")
    print(f"Mean pre-treatment effect: {pre_treatment_effects.mean():.4f}")
    print(f"T-statistic: {pre_treatment_test.statistic:.3f}")
    print(f"P-value: {pre_treatment_test.pvalue:.4f}")
    
    if pre_treatment_test.pvalue > 0.05:
        print("✅ PASS: No significant pre-treatment effects")
    else:
        print("⚠️  CONCERN: Significant pre-treatment effects detected")
        print("   May indicate parallel trends violation")
    
    # Post-treatment effects summary
    post_effects = event_df[event_df['event_time'] >= 0]['coefficient']
    
    print(f"\n=== POST-TREATMENT EFFECTS ===")
    print(f"Immediate effect (t=0): {event_df[event_df['event_time']==0]['coefficient'].iloc[0]:.3f}")
    print(f"Average post-treatment effect: {post_effects.mean():.3f}")
    print(f"Maximum effect: {post_effects.max():.3f}")
    print(f"Effect persistence: {len(post_effects[post_effects > 0])} of {len(post_effects)} periods positive")
    
    return {
        'event_results': event_df,
        'model': event_model,
        'pre_treatment_test': pre_treatment_test,
        'immediate_effect': event_df[event_df['event_time']==0]['coefficient'].iloc[0],
        'average_post_effect': post_effects.mean()
    }

# Perform event study analysis
event_study_results = event_study_analysis(
    df_did, 'date', 'copilot_usage', 'tickets_per_week', 'team_id', treatment_start
)
```

---

## Robustness Checks and Diagnostics

### Multiple Robustness Tests

```python
def did_robustness_checks(df_did, treatment_col, outcome_col, unit_col, time_col):
    """
    Comprehensive robustness checks for DiD analysis
    """
    robustness_results = {}
    
    print("=== DIFFERENCE-IN-DIFFERENCES ROBUSTNESS CHECKS ===")
    
    # 1. Alternative time periods
    print("\n1. ALTERNATIVE TIME WINDOWS")
    
    # Different post-treatment windows
    windows = [1, 3, 6]  # months
    for window in windows:
        df_window = df_did[df_did['post_treatment'] == 0].copy()
        post_data = df_did[(df_did['post_treatment'] == 1) & 
                          (df_did[time_col] <= df_did[df_did['post_treatment'] == 1][time_col].min() + 
                           pd.DateOffset(months=window))]
        df_window = pd.concat([df_window, post_data])
        
        if len(df_window) > 0:
            did_formula = f'{outcome_col} ~ {treatment_col} + post_treatment + treated_post'
            model = smf.ols(did_formula, data=df_window).fit(cov_type='cluster', 
                                                           cov_kwds={'groups': df_window[unit_col]})
            effect = model.params.get('treated_post', np.nan)
            print(f"  {window}-month window: {effect:.3f}")
            robustness_results[f'window_{window}m'] = effect
    
    # 2. Excluding boundary periods
    print("\n2. EXCLUDING BOUNDARY PERIODS")
    
    # Exclude first/last month to check sensitivity
    df_trimmed = df_did.copy()
    first_date = df_trimmed[time_col].min()
    last_date = df_trimmed[time_col].max()
    
    df_trimmed = df_trimmed[
        (df_trimmed[time_col] > first_date) & 
        (df_trimmed[time_col] < last_date)
    ]
    
    if len(df_trimmed) > 0:
        did_formula = f'{outcome_col} ~ {treatment_col} + post_treatment + treated_post'
        model_trimmed = smf.ols(did_formula, data=df_trimmed).fit(cov_type='cluster', 
                                                                 cov_kwds={'groups': df_trimmed[unit_col]})
        effect_trimmed = model_trimmed.params.get('treated_post', np.nan)
        print(f"  Excluding boundary periods: {effect_trimmed:.3f}")
        robustness_results['trimmed_boundaries'] = effect_trimmed
    
    # 3. Alternative control groups
    print("\n3. ALTERNATIVE CONTROL GROUP DEFINITIONS")
    
    # If you have multiple potential control groups, test each
    # This example assumes binary treatment, but could be extended
    
    # 4. Placebo tests
    print("\n4. PLACEBO TESTS")
    
    # Fake treatment date (earlier than actual)
    fake_treatment_date = df_did[time_col].min() + (df_did[time_col].max() - df_did[time_col].min()) / 3
    
    df_placebo = df_did[df_did[time_col] < df_did[df_did['post_treatment'] == 1][time_col].min()].copy()
    if len(df_placebo) > 0:
        df_placebo['fake_post'] = (df_placebo[time_col] >= fake_treatment_date).astype(int)
        df_placebo['fake_treated_post'] = df_placebo[treatment_col] * df_placebo['fake_post']
        
        placebo_formula = f'{outcome_col} ~ {treatment_col} + fake_post + fake_treated_post'
        placebo_model = smf.ols(placebo_formula, data=df_placebo).fit(cov_type='cluster', 
                                                                     cov_kwds={'groups': df_placebo[unit_col]})
        placebo_effect = placebo_model.params.get('fake_treated_post', np.nan)
        placebo_p = placebo_model.pvalues.get('fake_treated_post', np.nan)
        
        print(f"  Placebo test effect: {placebo_effect:.3f} (p={placebo_p:.3f})")
        robustness_results['placebo_effect'] = placebo_effect
        robustness_results['placebo_p_value'] = placebo_p
        
        if placebo_p > 0.05:
            print("  ✅ PASS: No placebo effect detected")
        else:
            print("  ⚠️  CONCERN: Significant placebo effect")
    
    # 5. Alternative outcome transformations
    print("\n5. OUTCOME TRANSFORMATIONS")
    
    if (df_did[outcome_col] > 0).all():
        # Log transformation
        df_log = df_did.copy()
        df_log[f'log_{outcome_col}'] = np.log(df_log[outcome_col])
        
        log_formula = f'log_{outcome_col} ~ {treatment_col} + post_treatment + treated_post'
        log_model = smf.ols(log_formula, data=df_log).fit(cov_type='cluster', 
                                                         cov_kwds={'groups': df_log[unit_col]})
        log_effect = log_model.params.get('treated_post', np.nan)
        print(f"  Log-transformed effect: {log_effect:.3f} ({np.exp(log_effect)-1:.1%} change)")
        robustness_results['log_effect'] = log_effect
    
    return robustness_results

# Perform robustness checks
robustness_results = did_robustness_checks(
    df_did, 'copilot_usage', 'tickets_per_week', 'team_id', 'date'
)
```

### Sensitivity Analysis

```python
def did_sensitivity_analysis(df_did, treatment_col, outcome_col, unit_col):
    """
    Sensitivity analysis for potential confounders
    """
    print("=== SENSITIVITY ANALYSIS ===")
    
    # Test sensitivity to different assumptions about unobserved confounders
    # Following Imbens (2003) approach
    
    baseline_effect = basic_did_results['did_effect']
    baseline_se = basic_did_results['se']
    
    print(f"Baseline DiD effect: {baseline_effect:.3f} ± {baseline_se:.3f}")
    
    # Simulate different levels of selection bias
    selection_bias_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    print(f"\nSensitivity to unobserved confounders:")
    print(f"{'Bias Level':<12} {'Adjusted Effect':<15} {'Still Significant?'}")
    print("-" * 40)
    
    for bias in selection_bias_levels:
        adjusted_effect = baseline_effect - bias
        z_stat = adjusted_effect / baseline_se
        still_significant = abs(z_stat) > 1.96
        
        sig_status = "Yes" if still_significant else "No"
        print(f"{bias:<12.1f} {adjusted_effect:<15.3f} {sig_status}")
    
    # Calculate minimum bias needed to eliminate significance
    min_bias_to_null = baseline_effect - 1.96 * baseline_se
    
    print(f"\nMinimum confounding bias to eliminate significance: {min_bias_to_null:.3f}")
    
    if min_bias_to_null > 0.2:
        print("✅ Result appears robust to moderate confounding")
    elif min_bias_to_null > 0.1:
        print("⚠️  Result moderately sensitive to confounding")
    else:
        print("❌ Result highly sensitive to confounding")
    
    return {
        'baseline_effect': baseline_effect,
        'min_bias_to_null': min_bias_to_null,
        'sensitivity_levels': selection_bias_levels
    }

# Perform sensitivity analysis
sensitivity_results = did_sensitivity_analysis(
    df_did, 'copilot_usage', 'tickets_per_week', 'team_id'
)
```

---

## Advanced DiD Specifications

### Two-Way Fixed Effects

```python
def two_way_fixed_effects_did(df_did, treatment_col, outcome_col, unit_col, time_col):
    """
    Two-way fixed effects DiD specification
    
    Includes:
    - Unit fixed effects (control for time-invariant unit characteristics)
    - Time fixed effects (control for common time trends)
    """
    
    # Create time period identifiers for fixed effects
    df_twfe = df_did.copy()
    df_twfe['time_period'] = df_twfe[time_col].dt.to_period('M')
    
    # Two-way fixed effects regression
    twfe_formula = f'{outcome_col} ~ treated_post + C({unit_col}) + C(time_period)'
    twfe_model = smf.ols(twfe_formula, data=df_twfe).fit(cov_type='cluster', 
                                                        cov_kwds={'groups': df_twfe[unit_col]})
    
    twfe_effect = twfe_model.params['treated_post']
    twfe_se = twfe_model.bse['treated_post']
    twfe_ci = twfe_model.conf_int().loc['treated_post']
    
    print(f"=== TWO-WAY FIXED EFFECTS DiD ===")
    print(f"Treatment effect: {twfe_effect:.3f}")
    print(f"Standard error: {twfe_se:.3f}")
    print(f"95% CI: [{twfe_ci[0]:.3f}, {twfe_ci[1]:.3f}]")
    print(f"R-squared: {twfe_model.rsquared:.3f}")
    
    # Compare with basic DiD
    basic_effect = basic_did_results['did_effect']
    difference = twfe_effect - basic_effect
    
    print(f"\nComparison with basic DiD:")
    print(f"Basic DiD effect: {basic_effect:.3f}")
    print(f"TWFE effect: {twfe_effect:.3f}")
    print(f"Difference: {difference:.3f}")
    
    if abs(difference) < 0.1:
        print("✅ Results consistent between specifications")
    else:
        print("⚠️  Notable difference between specifications")
        print("   Fixed effects may be capturing important variation")
    
    return {
        'twfe_effect': twfe_effect,
        'twfe_se': twfe_se,
        'model': twfe_model,
        'comparison_with_basic': difference
    }

# Estimate two-way fixed effects model
twfe_results = two_way_fixed_effects_did(
    df_did, 'copilot_usage', 'tickets_per_week', 'team_id', 'date'
)
```

---

## Business Translation Framework

### Converting DiD Results to Business Insights

```python
def translate_did_results_to_business(did_results, outcome_col, treatment_description, 
                                     outcome_units="units", time_period="month"):
    """
    Translate DiD statistical results into business language
    """
    effect = did_results['did_effect']
    se = did_results['se']
    ci = did_results['ci']
    p_value = did_results['p_value']
    effect_size = did_results['effect_size']
    
    print(f"=== BUSINESS IMPACT TRANSLATION ===")
    print(f"Intervention: {treatment_description}")
    print(f"Outcome: {outcome_col.replace('_', ' ').title()}")
    
    # Effect direction and magnitude
    if effect > 0:
        direction = "increase"
        impact_verb = "improved"
    else:
        direction = "decrease"
        impact_verb = "reduced"
        effect = abs(effect)
    
    print(f"\n📊 HEADLINE FINDING:")
    print(f"{treatment_description} led to a {effect:.1f} {outcome_units} {direction}")
    print(f"in {outcome_col.replace('_', ' ')} per {time_period}")
    
    # Confidence interval interpretation
    print(f"\n📈 CONFIDENCE RANGE:")
    if ci[0] > 0 or ci[1] < 0:  # CI doesn't include zero
        print(f"We can be 95% confident the true effect is between")
        print(f"{ci[0]:.1f} and {ci[1]:.1f} {outcome_units} per {time_period}")
    else:
        print(f"The effect could range from {ci[0]:.1f} to {ci[1]:.1f} {outcome_units}")
        print(f"(includes possibility of no effect)")
    
    # Statistical confidence
    print(f"\n🎯 STATISTICAL CONFIDENCE:")
    if p_value < 0.001:
        confidence_level = "very high confidence (p < 0.001)"
    elif p_value < 0.01:
        confidence_level = "high confidence (p < 0.01)"
    elif p_value < 0.05:
        confidence_level = "moderate confidence (p < 0.05)"
    else:
        confidence_level = "low confidence (p ≥ 0.05)"
    
    print(f"Statistical significance: {confidence_level}")
    
    # Effect size interpretation
    print(f"\n📏 PRACTICAL SIGNIFICANCE:")
    if abs(effect_size) < 0.2:
        practical_significance = "small practical impact"
    elif abs(effect_size) < 0.5:
        practical_significance = "moderate practical impact"
    else:
        practical_significance = "large practical impact"
    
    print(f"Effect size: {practical_significance}")
    
    # Business implications
    print(f"\n💼 BUSINESS IMPLICATIONS:")
    
    if p_value < 0.05 and abs(effect_size) >= 0.2:
        print(f"✅ RECOMMENDATION: Scale {treatment_description}")
        print(f"   - Statistically significant with meaningful impact")
        print(f"   - Expected {direction} of {effect:.1f} {outcome_units} per {time_period}")
        
        # ROI considerations
        print(f"\n💰 ROI CONSIDERATIONS:")
        print(f"   - Calculate implementation costs vs. {effect:.1f} {outcome_units} benefit")
        print(f"   - Consider scalability across {df_did[unit_col].nunique()} units")
        
    elif p_value < 0.05 and abs(effect_size) < 0.2:
        print(f"⚠️  RECOMMENDATION: Consider cost-benefit analysis")
        print(f"   - Statistically significant but small practical impact")
        print(f"   - Evaluate if {effect:.1f} {outcome_units} improvement justifies costs")
        
    else:
        print(f"❌ RECOMMENDATION: Do not scale based on current evidence")
        print(f"   - Insufficient evidence of meaningful impact")
        print(f"   - Consider longer evaluation period or alternative approaches")
    
    # Implementation guidance
    print(f"\n🚀 IMPLEMENTATION GUIDANCE:")
    if effect > 0 and p_value < 0.05:
        print(f"   - Pilot expansion to similar units")
        print(f"   - Monitor key metrics during rollout")
        print(f"   - Expected benefit: {effect:.1f} {outcome_units} per unit per {time_period}")
    
    return {
        'headline_effect': f"{effect:.1f} {outcome_units} {direction}",
        'confidence_level': confidence_level,
        'practical_significance': practical_significance,
        'recommendation': 'scale' if (p_value < 0.05 and abs(effect_size) >= 0.2) else 'evaluate' if p_value < 0.05 else 'do_not_scale'
    }

# Translate results to business language
business_translation = translate_did_results_to_business(
    basic_did_results, 
    'tickets_per_week', 
    'GitHub Copilot implementation',
    'tickets',
    'week'
)
```

---

## When DiD May Not Work

### Diagnostic Checks

```python
def did_diagnostic_summary(trends_test, event_study_results, robustness_results):
    """
    Comprehensive diagnostic summary for DiD validity
    """
    print("=== DIFFERENCE-IN-DIFFERENCES DIAGNOSTIC SUMMARY ===")
    
    issues = []
    warnings = []
    
    # 1. Parallel trends
    if not trends_test['trends_valid']:
        issues.append("Parallel trends assumption violated")
    
    # 2. Pre-treatment effects in event study
    if event_study_results['pre_treatment_test'].pvalue < 0.05:
        issues.append("Significant pre-treatment effects detected")
    
    # 3. Placebo test
    if 'placebo_p_value' in robustness_results and robustness_results['placebo_p_value'] < 0.05:
        issues.append("Placebo test shows significant effect")
    
    # 4. Sensitivity to specification
    window_effects = [robustness_results.get(f'window_{w}m') for w in [1, 3, 6]]
    window_effects = [e for e in window_effects if not pd.isna(e)]
    
    if len(window_effects) > 1:
        window_range = max(window_effects) - min(window_effects)
        if window_range > abs(np.mean(window_effects)) * 0.5:
            warnings.append("Results sensitive to time window specification")
    
    # Summary
    print(f"\n🔍 DIAGNOSTIC RESULTS:")
    
    if not issues and not warnings:
        print("✅ ALL DIAGNOSTICS PASSED")
        print("   DiD assumptions appear satisfied")
        validity_score = "HIGH"
        
    elif issues:
        print("❌ MAJOR ISSUES DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        validity_score = "LOW"
        
        print("\n⚠️  RECOMMENDATIONS:")
        print("   - Consider alternative identification strategies")
        print("   - Instrumental variables if available")
        print("   - Synthetic control methods")
        print("   - Regression discontinuity if applicable")
        
    else:
        print("⚠️  MINOR CONCERNS:")
        for warning in warnings:
            print(f"   - {warning}")
        validity_score = "MODERATE"
        
        print("\n📋 RECOMMENDATIONS:")
        print("   - Report sensitivity analyses")
        print("   - Consider robustness checks")
        print("   - Triangulate with other methods if possible")
    
    print(f"\n📊 OVERALL VALIDITY: {validity_score}")
    
    return {
        'validity_score': validity_score,
        'issues': issues,
        'warnings': warnings
    }

# Generate comprehensive diagnostic summary
if 'trends_test' in locals() and 'event_study_results' in locals():
    diagnostic_summary = did_diagnostic_summary(trends_test, event_study_results, robustness_results)
```

**Navigation:**
- [← Back to Propensity Score Methods]({{ site.baseurl }}/causal-inference-propensity/)
- [Next: Instrumental Variables →]({{ site.baseurl }}/causal-inference-iv/)

---

*Difference-in-Differences leverages natural experiments and timing variation to identify causal effects. Always validate the parallel trends assumption and conduct comprehensive robustness checks before making business recommendations.*
