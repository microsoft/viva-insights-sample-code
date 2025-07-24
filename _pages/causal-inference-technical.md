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

**What you'll learn:**
- How to structure and validate data for causal analysis
- Step-by-step implementation of five major causal inference methods
- How to interpret statistical outputs and translate them into business insights
- Best practices for robust analysis and common pitfall avoidance

For a conceptual introduction to causal inference in Copilot analytics, see our [main guide]({{ site.baseurl }}/causal-inference/).

---

## Method-Specific Implementation Guides

Each causal inference method has its own detailed implementation guide with code examples, diagnostics, and interpretation guidance:

### [📊 Data Preparation & Setup]({{ site.baseurl }}/causal-inference-data-prep/)
Essential data validation, variable definition, and quality checks before running any causal analysis.

### [📈 Method 1: Regression Adjustment]({{ site.baseurl }}/causal-inference-regression/)
Linear regression-based causal inference with interaction effects and heterogeneous treatment analysis.

### [🎯 Method 2: Propensity Score Methods]({{ site.baseurl }}/causal-inference-propensity/)
Propensity score estimation, matching, and inverse probability weighting for balanced comparisons.

### [📅 Method 3: Difference-in-Differences]({{ site.baseurl }}/causal-inference-did/)
Temporal analysis using before/after comparisons with parallel trends testing.

### [🔧 Method 4: Instrumental Variables]({{ site.baseurl }}/causal-inference-iv/)
Two-stage least squares and instrumental variable analysis for unobserved confounding.

### [🤖 Method 5: Doubly Robust Methods]({{ site.baseurl }}/causal-inference-doubly-robust/)
Machine learning enhanced causal inference with protection against model misspecification.

### [✅ Validation & Robustness Testing]({{ site.baseurl }}/causal-inference-validation/)
Sensitivity analysis, placebo tests, and robustness checks for causal estimates.

---

## Quick Method Selection Guide

| **Method** | **Best For** | **Key Assumption** | **Data Requirements** |
|------------|--------------|-------------------|---------------------|
| **Regression Adjustment** | Few confounders, linear relationships | No unmeasured confounders | Cross-sectional or panel |
| **Propensity Scores** | Many confounders, binary treatment | No unmeasured confounders | Cross-sectional or panel |
| **Difference-in-Differences** | Policy interventions, natural experiments | Parallel trends | Panel data with pre/post periods |
| **Instrumental Variables** | Unobserved confounding suspected | Valid instrument exists | Cross-sectional or panel |
| **Doubly Robust** | Complex relationships, many variables | Either outcome or PS model correct | Large sample size |

---

## Understanding the Causal Analysis Workflow

Before diving into specific methods, it's important to understand the general workflow for causal analysis:

1. **Problem Definition**: Clearly articulate your causal question
2. **Data Preparation**: Structure and validate your dataset
3. **Method Selection**: Choose the appropriate causal inference technique
4. **Assumption Checking**: Verify that method assumptions are met
5. **Implementation**: Run the analysis with proper validation
6. **Results Interpretation**: Translate statistical outputs into business insights
7. **Robustness Testing**: Validate findings through sensitivity analysis

Each method we'll cover follows this general pattern, but with different technical requirements and assumptions.

---

## Getting Started

1. **Start with [Data Preparation]({{ site.baseurl }}/causal-inference-data-prep/)** to ensure your dataset meets causal analysis requirements
2. **Choose your method** based on your research question and data characteristics
3. **Follow the method-specific guide** for detailed implementation
4. **Validate your results** using robustness testing techniques
5. **Interpret and communicate** findings using business translation frameworks

---

## Resources and Libraries

### Essential Python Libraries
```python
# Core causal inference libraries
pip install dowhy causalml econml causallib

# Statistical analysis
pip install statsmodels linearmodels

# Machine learning and data manipulation
pip install scikit-learn pandas numpy matplotlib seaborn plotly
```

### R Libraries
```r
# Core causal inference packages
install.packages(c("causalTree", "grf", "MatchIt", "WeightIt"))

# Panel data and econometrics
install.packages(c("plm", "fixest", "did"))

# Data manipulation and visualization
install.packages(c("dplyr", "ggplot2", "tidyr"))
```

---

*This guide provides the foundation for implementing rigorous causal inference methods with Copilot data. Always validate assumptions, check robustness, and interpret results in business context.*

## Pre-Analysis Setup

### Understanding Your Data Requirements

Successful causal inference starts with properly structured data. Unlike simple correlation analysis, causal methods require specific data characteristics and measurement approaches.

**The Gold Standard: Panel Data Structure**
The ideal dataset for causal analysis is panel data - observations of the same individuals over multiple time periods. This structure allows us to observe changes within individuals and control for time-invariant characteristics.

Before implementing any causal inference method, ensure your data meets these requirements:

```python
import pandas as pd
import numpy as np

# Required data structure for causal analysis
required_columns = {
    'user_id': 'str',           # Unique identifier for each user
    'time_period': 'datetime',   # Time dimension (weekly/monthly)
    'treatment': 'int',          # Copilot usage (0/1 or continuous)
    'outcome': 'float',          # Productivity metric
    'confounders': 'various'     # Job role, tenure, team size, etc.
}

# Example data validation function
def validate_causal_data(df):
    """
    Comprehensive validation of data quality for causal inference
    Returns a dictionary of checks and recommendations
    """
    checks = {
        'temporal_ordering': df['time_period'].is_monotonic_increasing,
        'treatment_variation': df['treatment'].nunique() > 1,
        'outcome_completeness': df['outcome'].notna().mean() > 0.8,
        'sufficient_sample': len(df) > 100,
        'panel_balance': len(df['user_id'].unique()) * len(df['time_period'].unique()) == len(df)
    }
    
    # Generate recommendations based on failed checks
    recommendations = []
    if not checks['temporal_ordering']:
        recommendations.append("Sort data by time_period to ensure proper temporal ordering")
    if not checks['treatment_variation']:
        recommendations.append("Treatment variable shows no variation - check data quality")
    if not checks['outcome_completeness']:
        recommendations.append("High missing data in outcome - consider imputation or data cleaning")
    if not checks['sufficient_sample']:
        recommendations.append("Sample size may be too small for reliable causal inference")
    if not checks['panel_balance']:
        recommendations.append("Unbalanced panel - some users missing observations in some periods")
    
    return {'checks': checks, 'recommendations': recommendations}

# Run validation
validation_results = validate_causal_data(df)
print("Data Quality Checks:", validation_results['checks'])
if validation_results['recommendations']:
    print("Recommendations:", validation_results['recommendations'])
```

**Understanding the Output:**
- `temporal_ordering`: Ensures time flows correctly (critical for causal ordering)
- `treatment_variation`: Confirms we have both treated and untreated observations
- `outcome_completeness`: High missing data can bias results
- `sufficient_sample`: Small samples lead to unreliable estimates
- `panel_balance`: Balanced panels enable stronger causal identification
```

### Defining Treatment and Outcome Variables

**Treatment Variable Design**
How you define your treatment variable fundamentally affects your causal interpretation. Consider these different approaches:

```python
# Treatment variable definitions
def define_treatment_binary(df, threshold=1):
    """
    Binary treatment: Used Copilot (1) vs Not Used (0)
    
    This is the simplest approach - users either used Copilot or they didn't.
    Good for: Initial adoption analysis, policy evaluation
    Interpretation: "The effect of any Copilot usage vs none"
    """
    return (df['copilot_actions'] >= threshold).astype(int)

def define_treatment_continuous(df):
    """
    Continuous treatment: Number of Copilot actions per week
    
    Captures intensity of usage, not just adoption.
    Good for: Understanding dose-response relationships
    Interpretation: "The effect of each additional Copilot action per week"
    """
    return df['copilot_actions_per_week']

def define_treatment_categorical(df):
    """
    Categorical treatment: Usage intensity levels
    
    Balances simplicity with intensity measurement.
    Good for: Identifying thresholds, communicating to stakeholders
    Interpretation: "The effect of different usage intensities"
    """
    return pd.cut(df['copilot_actions'], 
                  bins=[0, 1, 10, 50, np.inf], 
                  labels=['None', 'Light', 'Moderate', 'Heavy'])

# Example of how choice affects interpretation
df['treatment_binary'] = define_treatment_binary(df)
df['treatment_continuous'] = define_treatment_continuous(df)
df['treatment_categorical'] = define_treatment_categorical(df)

print("Treatment variable distributions:")
print("Binary:", df['treatment_binary'].value_counts())
print("Continuous:", df['treatment_continuous'].describe())
print("Categorical:", df['treatment_categorical'].value_counts())
```

**Outcome Variable Selection**
Your outcome choice determines what causal effect you're measuring:

```python
def productivity_metrics(df):
    """
    Various productivity outcome measures with different interpretations
    """
    outcomes = {
        # Volume metrics
        'tickets_resolved': df['tickets_closed'] / df['time_period_weeks'],
        'emails_sent': df['emails_sent'] / df['time_period_weeks'],
        
        # Quality metrics
        'case_resolution_time': df['avg_case_hours'],
        'customer_satisfaction': df['csat_score'],
        
        # Business metrics
        'revenue_per_user': df['deals_closed'] * df['avg_deal_size'],
        'cost_savings': df['hours_saved'] * df['hourly_wage']
    }
    
    # Add interpretation guide
    interpretations = {
        'tickets_resolved': 'Additional tickets resolved per week due to Copilot',
        'case_resolution_time': 'Hours saved per case due to Copilot',
        'customer_satisfaction': 'CSAT point improvement due to Copilot',
        'revenue_per_user': 'Additional revenue per user per period due to Copilot'
    }
    
    return outcomes, interpretations

# Choose your outcome based on business priorities
outcomes, interpretations = productivity_metrics(df)
chosen_outcome = 'tickets_resolved'
print(f"Analyzing: {interpretations[chosen_outcome]}")
```

---

## Method 1: Regression Adjustment

### Conceptual Foundation

Regression adjustment is the most straightforward causal inference method. It estimates the treatment effect by controlling for confounding variables through linear regression. The key insight is that if you can measure and control for all variables that influence both treatment and outcome, the remaining correlation between treatment and outcome represents the causal effect.

**When to use:** When you have a good understanding of confounders and they're mostly captured in your data.
**Assumption:** No unmeasured confounders (strong assumption!)

### Basic Implementation

```python
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

def regression_adjustment(df, outcome_col, treatment_col, confounder_cols):
    """
    Simple regression adjustment for causal inference
    
    This function implements the fundamental equation:
    Y = α + β*Treatment + γ₁*X₁ + γ₂*X₂ + ... + ε
    
    The coefficient β represents our causal estimate of treatment effect.
    """
    # Prepare formula
    confounders_str = ' + '.join(confounder_cols)
    formula = f'{outcome_col} ~ {treatment_col} + {confounders_str}'
    
    print(f"Running regression: {formula}")
    
    # Fit model
    model = smf.ols(formula, data=df).fit()
    
    # Extract treatment effect
    treatment_effect = model.params[treatment_col]
    confidence_interval = model.conf_int().loc[treatment_col]
    
    results = {
        'ate': treatment_effect,
        'ci_lower': confidence_interval[0],
        'ci_upper': confidence_interval[1],
        'p_value': model.pvalues[treatment_col],
        'r_squared': model.rsquared,
        'model': model
    }
    
    return results

# Example usage with interpretation
confounders = ['tenure_months', 'job_level', 'team_size', 'manager_span']
result = regression_adjustment(df, 'tickets_per_week', 'copilot_usage', confounders)

print(f"\n=== CAUSAL EFFECT ESTIMATE ===")
print(f"Treatment Effect: {result['ate']:.2f} tickets per week")
print(f"95% Confidence Interval: ({result['ci_lower']:.2f}, {result['ci_upper']:.2f})")
print(f"P-value: {result['p_value']:.4f}")
print(f"Model R²: {result['r_squared']:.3f}")

# Interpret the results
if result['p_value'] < 0.05:
    print(f"\n✅ SIGNIFICANT EFFECT DETECTED")
    if result['ate'] > 0:
        print(f"Copilot usage INCREASES productivity by {result['ate']:.2f} tickets/week")
    else:
        print(f"Copilot usage DECREASES productivity by {abs(result['ate']):.2f} tickets/week")
else:
    print(f"\n❌ NO SIGNIFICANT EFFECT DETECTED")
    print(f"Cannot conclude that Copilot has a causal effect on productivity")
```

**Understanding the Output:**
- **Treatment Effect (ATE)**: The average causal effect across all users
- **Confidence Interval**: Range of plausible true effects (95% confidence)
- **P-value**: Probability of seeing this effect if true effect were zero
- **R²**: How much variation the model explains (model fit quality)
```

### Advanced Regression with Interaction Effects

Sometimes the treatment effect differs across subgroups. Interaction terms help us identify these heterogeneous effects - crucial for targeted deployment strategies.

```python
def regression_with_interactions(df, outcome_col, treatment_col, confounder_cols, interaction_vars=None):
    """
    Regression adjustment with interaction terms for heterogeneous effects
    
    Interaction terms allow the treatment effect to vary by subgroup:
    Y = α + β₁*Treatment + β₂*Group + β₃*(Treatment × Group) + controls
    
    The interaction coefficient β₃ tells us how the treatment effect
    differs between groups.
    """
    base_formula = f'{outcome_col} ~ {treatment_col} + {" + ".join(confounder_cols)}'
    
    # Add interaction terms
    if interaction_vars:
        interactions = [f'{treatment_col}:{var}' for var in interaction_vars]
        formula = base_formula + ' + ' + ' + '.join(interactions)
    else:
        formula = base_formula
    
    print(f"Running interaction model: {formula}")
    
    model = smf.ols(formula, data=df).fit()
    
    # Calculate marginal effects for different groups
    effects = {}
    main_effect = model.params[treatment_col]
    
    if interaction_vars:
        for var in interaction_vars:
            interaction_term = f'{treatment_col}:{var}'
            if interaction_term in model.params:
                interaction_coeff = model.params[interaction_term]
                
                # Calculate effects for each level of the interaction variable
                for level in df[var].unique():
                    if pd.api.types.is_numeric_dtype(df[var]):
                        # For continuous variables, show effect at different values
                        marginal_effect = main_effect + (interaction_coeff * level)
                        effects[f'{var}_{level}'] = marginal_effect
                    else:
                        # For categorical variables
                        if level == df[var].mode().iloc[0]:  # Reference category
                            effects[f'{var}_{level}'] = main_effect
                        else:
                            # This is simplified - proper categorical interactions need dummy encoding
                            effects[f'{var}_{level}'] = main_effect + interaction_coeff
    
    return model, effects

# Example: Different effects by job level
model, effects = regression_with_interactions(
    df, 'tickets_per_week', 'copilot_usage', 
    confounders, interaction_vars=['job_level']
)

print(f"\n=== HETEROGENEOUS TREATMENT EFFECTS ===")
for group, effect in effects.items():
    print(f"{group}: {effect:.2f} tickets/week")

# Interpret interaction effects
print(f"\n=== INTERACTION INTERPRETATION ===")
print("These results show how Copilot's effect varies by job level.")
print("Use this to target deployment to groups with highest returns.")
```

**Business Value of Interaction Analysis:**
- **Targeted Deployment**: Focus Copilot rollout on highest-impact groups
- **Training Customization**: Adjust training programs for different user types  
- **ROI Optimization**: Allocate licenses where they generate most value
```

---

## Method 2: Propensity Score Methods

Propensity score methods address selection bias by balancing groups on observed characteristics. Instead of controlling for confounders directly in the outcome model, we match treated and control units with similar probabilities of receiving treatment.

**Key Advantages:**
- **Dimension Reduction**: Summarizes many confounders into a single score
- **Transparency**: Shows which units are truly comparable
- **Assumption Testing**: Forces explicit consideration of overlap

**When to Use Propensity Scores:**
- High-dimensional confounding (many covariates)
- Need to demonstrate comparable groups to stakeholders
- Concerns about model specification in outcome analysis

### Propensity Score Estimation

The propensity score e(x) = P(Treatment = 1 | X = x) is the probability of treatment given observed characteristics. A well-estimated propensity score should achieve balance and have sufficient overlap between groups.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def estimate_propensity_scores(df, treatment_col, confounder_cols, method='logistic'):
    """
    Estimate propensity scores with comprehensive diagnostics
    
    The propensity score summarizes the probability of treatment assignment.
    Good propensity score models should:
    1. Achieve covariate balance between groups
    2. Have sufficient overlap (common support)
    3. Be well-calibrated
    """
    X = df[confounder_cols]
    y = df[treatment_col]
    
    print(f"=== PROPENSITY SCORE ESTIMATION ===")
    print(f"Method: {method}")
    print(f"Sample size: {len(df)} ({y.sum()} treated, {len(y) - y.sum()} control)")
    print(f"Covariates: {len(confounder_cols)}")
    
    if method == 'logistic':
        # Standardize features for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        propensity_scores = model.predict_proba(X_scaled)[:, 1]
        
        # Model performance
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        print(f"Cross-validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
    elif method == 'random_forest':
        # Random forest for complex interactions
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        propensity_scores = model.predict_proba(X)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': confounder_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"Top predictors: {', '.join(feature_importance.head(3)['feature'].tolist())}")
    
    # Add scores to dataframe
    df_with_ps = df.copy()
    df_with_ps['propensity_score'] = propensity_scores
    
    # Comprehensive overlap diagnostics
    overlap_diagnostics = check_overlap_detailed(propensity_scores, y)
    
    return df_with_ps, model, overlap_diagnostics

def check_overlap_detailed(ps, treatment):
    """
    Comprehensive overlap diagnostics for propensity scores
    
    Poor overlap indicates limited comparability between groups.
    """
    treated_ps = ps[treatment == 1]
    control_ps = ps[treatment == 0]
    
    print(f"\n=== OVERLAP DIAGNOSTICS ===")
    print(f"Treated group: min={treated_ps.min():.3f}, max={treated_ps.max():.3f}, mean={treated_ps.mean():.3f}")
    print(f"Control group: min={control_ps.min():.3f}, max={control_ps.max():.3f}, mean={control_ps.mean():.3f}")
    
    # Common support region
    common_min = max(treated_ps.min(), control_ps.min())
    common_max = min(treated_ps.max(), control_ps.max())
    print(f"Common support: [{common_min:.3f}, {common_max:.3f}]")
    
    # Overlap quality metrics
    in_common_support = (ps >= common_min) & (ps <= common_max)
    overlap_rate = in_common_support.mean()
    print(f"Proportion in common support: {overlap_rate:.1%}")
    
    # Extreme propensity scores (problematic regions)
    extreme_high = (ps > 0.9).sum()
    extreme_low = (ps < 0.1).sum()
    print(f"Extreme scores: {extreme_high} above 0.9, {extreme_low} below 0.1")
    
    # Balance assessment
    overlap_stats = {
        'treated_range': (treated_ps.min(), treated_ps.max()),
        'control_range': (control_ps.min(), control_ps.max()),
        'common_support': (common_min, common_max),
        'overlap_rate': overlap_rate,
        'extreme_count': extreme_high + extreme_low
    }
    
    # Warnings for poor overlap
    if overlap_rate < 0.8:
        print("⚠️  Warning: Poor overlap - less than 80% of units in common support")
    if extreme_high + extreme_low > len(ps) * 0.05:
        print("⚠️  Warning: Many extreme propensity scores suggest poor model fit")
    if common_max - common_min < 0.6:
        print("⚠️  Warning: Narrow common support region")
    
    return overlap_stats

# Estimate propensity scores with diagnostics
df_with_ps, ps_model, overlap_stats = estimate_propensity_scores(
    df, 'copilot_usage', confounders, method='logistic'
)
```

**Interpreting Propensity Score Quality:**
- **AUC 0.6-0.8**: Good discrimination, sufficient variation in treatment probability
- **AUC > 0.9**: Excellent prediction but may indicate poor overlap
- **Common Support**: Should include most observations (>80%)
- **Extreme Scores**: Values near 0 or 1 suggest deterministic assignment
### Propensity Score Matching

Once we have good propensity scores, we match treated units to similar control units. This creates a balanced dataset where treatment assignment appears "as good as random" within matched pairs.

```python
from scipy.spatial.distance import cdist
import warnings

def propensity_score_matching(df, ps_col, treatment_col, outcome_col, ratio=1, caliper=0.1):
    """
    Nearest neighbor propensity score matching with comprehensive analysis
    
    Matching Process:
    1. For each treated unit, find nearest control unit(s)
    2. Apply caliper constraint to ensure good matches
    3. Calculate treatment effect from matched pairs
    
    Parameters:
    - ratio: Number of control units to match to each treated unit
    - caliper: Maximum allowed difference in propensity scores
    """
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    print(f"=== PROPENSITY SCORE MATCHING ===")
    print(f"Treated units: {len(treated)}")
    print(f"Control units: {len(control)}")
    print(f"Matching ratio: 1:{ratio}")
    print(f"Caliper: {caliper}")
    
    matched_pairs = []
    unmatched_treated = []
    used_controls = set()
    
    for _, treated_unit in treated.iterrows():
        # Calculate distances to all unused control units
        available_controls = control[~control.index.isin(used_controls)]
        
        if len(available_controls) == 0:
            unmatched_treated.append(treated_unit.name)
            continue
            
        distances = np.abs(available_controls[ps_col] - treated_unit[ps_col])
        
        # Apply caliper constraint
        valid_matches = distances <= caliper
        if not valid_matches.any():
            unmatched_treated.append(treated_unit.name)
            continue
            
        # Find closest matches
        n_matches = min(ratio, valid_matches.sum())
        closest_indices = distances[valid_matches].nsmallest(n_matches).index
        
        for idx in closest_indices:
            matched_pairs.append({
                'treated_id': treated_unit.name,
                'control_id': idx,
                'treated_outcome': treated_unit[outcome_col],
                'control_outcome': control.loc[idx, outcome_col],
                'treated_ps': treated_unit[ps_col],
                'control_ps': control.loc[idx, ps_col],
                'ps_distance': distances[idx]
            })
            used_controls.add(idx)
    
    # Create matched dataset analysis
    matched_df = pd.DataFrame(matched_pairs)
    
    if len(matched_df) == 0:
        print("❌ No valid matches found - consider relaxing caliper or improving propensity score model")
        return None
    
    # Matching quality assessment
    print(f"\n=== MATCHING RESULTS ===")
    print(f"Successful matches: {len(matched_df)}")
    print(f"Unmatched treated units: {len(unmatched_treated)}")
    print(f"Match rate: {len(matched_df)/len(treated):.1%}")
    print(f"Average PS distance: {matched_df['ps_distance'].mean():.4f}")
    print(f"Max PS distance: {matched_df['ps_distance'].max():.4f}")
    
    # Calculate treatment effect
    individual_effects = matched_df['treated_outcome'] - matched_df['control_outcome']
    ate = individual_effects.mean()
    ate_se = individual_effects.std() / np.sqrt(len(individual_effects))
    
    print(f"\n=== TREATMENT EFFECT ===")
    print(f"Average Treatment Effect: {ate:.3f}")
    print(f"Standard Error: {ate_se:.3f}")
    print(f"95% CI: [{ate - 1.96*ate_se:.3f}, {ate + 1.96*ate_se:.3f}]")
    
    # Effect size interpretation
    outcome_std = df[outcome_col].std()
    cohens_d = ate / outcome_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    if abs(cohens_d) < 0.2:
        effect_size = "small"
    elif abs(cohens_d) < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect magnitude: {effect_size}")
    
    # Create matched dataset for further analysis
    matched_treated = df.loc[matched_df['treated_id']]
    matched_control = df.loc[matched_df['control_id']]
    matched_dataset = pd.concat([matched_treated, matched_control])
    
    results = {
        'ate': ate,
        'se': ate_se,
        'matched_pairs': matched_df,
        'matched_dataset': matched_dataset,
        'unmatched_treated': unmatched_treated,
        'match_quality': {
            'match_rate': len(matched_df)/len(treated),
            'avg_distance': matched_df['ps_distance'].mean(),
            'max_distance': matched_df['ps_distance'].max()
        }
    }
    
    return results

# Perform matching analysis
matching_results = propensity_score_matching(
    df_with_ps, 'propensity_score', 'copilot_usage', 'tickets_per_week', 
    ratio=1, caliper=0.05
)

if matching_results:
    print(f"\n=== BUSINESS INTERPRETATION ===")
    effect = matching_results['ate']
    if effect > 0:
        print(f"Copilot users complete {effect:.1f} more tickets per week")
        print(f"Monthly impact: ~{effect * 4:.1f} additional tickets per user")
    else:
        print(f"Copilot users complete {abs(effect):.1f} fewer tickets per week")
        print("Consider investigating implementation or training issues")
```

**Key Matching Diagnostics:**
- **Match Rate**: Percentage of treated units successfully matched (aim for >80%)
- **Caliper Choice**: Balance between match quality and sample size
- **PS Distance**: Average distance should be small (< 0.05)
- **Effect Size**: Cohen's d helps interpret practical significance

### Propensity Score Weighting (IPW)

Instead of matching, we can weight observations by their inverse propensity scores. This uses all data while balancing the groups - treated units get weight 1/e(x) and controls get weight 1/(1-e(x)).

```python
def inverse_propensity_weighting(df, ps_col, treatment_col, outcome_col, trim_weights=True):
    """
    Inverse Propensity Weighting (IPW) for causal inference
    
    IPW reweights the sample to create balance between groups:
    - Treated units: weight = 1 / propensity_score  
    - Control units: weight = 1 / (1 - propensity_score)
    
    This creates a "pseudo-population" where treatment assignment
    is independent of confounders.
    """
    df_ipw = df.copy()
    
    # Calculate IPW weights
    weights = np.where(
        df[treatment_col] == 1,
        1 / df[ps_col],  # Treated units
        1 / (1 - df[ps_col])  # Control units
    )
    
    df_ipw['ipw_weight'] = weights
    
    print(f"=== INVERSE PROPENSITY WEIGHTING ===")
    print(f"Sample size: {len(df_ipw)}")
    print(f"Weight statistics:")
    print(f"  Mean: {weights.mean():.2f}")
    print(f"  Std: {weights.std():.2f}")
    print(f"  Min: {weights.min():.2f}")
    print(f"  Max: {weights.max():.2f}")
    
    # Check for extreme weights
    extreme_threshold = 10
    extreme_weights = (weights > extreme_threshold).sum()
    if extreme_weights > 0:
        print(f"⚠️  {extreme_weights} observations have weights > {extreme_threshold}")
        print("   Consider trimming or improving propensity score model")
    
    # Optional weight trimming
    if trim_weights:
        # Trim weights at 1st and 99th percentiles
        lower_bound = np.percentile(weights, 1)
        upper_bound = np.percentile(weights, 99)
        
        weights_trimmed = np.clip(weights, lower_bound, upper_bound)
        df_ipw['ipw_weight_trimmed'] = weights_trimmed
        
        print(f"\nWeight trimming:")
        print(f"  Trimmed at [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  {(weights != weights_trimmed).sum()} weights modified")
        
        # Use trimmed weights for analysis
        analysis_weights = weights_trimmed
        weight_col = 'ipw_weight_trimmed'
    else:
        analysis_weights = weights
        weight_col = 'ipw_weight'
    
    # Calculate weighted treatment effect
    treated_outcome = df_ipw[df_ipw[treatment_col] == 1][outcome_col]
    control_outcome = df_ipw[df_ipw[treatment_col] == 0][outcome_col]
    
    treated_weights = df_ipw[df_ipw[treatment_col] == 1][weight_col]
    control_weights = df_ipw[df_ipw[treatment_col] == 0][weight_col]
    
    # Weighted means
    weighted_treated_mean = np.average(treated_outcome, weights=treated_weights)
    weighted_control_mean = np.average(control_outcome, weights=control_weights)
    
    ate_ipw = weighted_treated_mean - weighted_control_mean
    
    # Robust standard errors (simplified)
    # For production use, consider sandwich estimators
    treated_var = np.average((treated_outcome - weighted_treated_mean)**2, weights=treated_weights)
    control_var = np.average((control_outcome - weighted_control_mean)**2, weights=control_weights)
    
    se_treated = np.sqrt(treated_var / len(treated_outcome))
    se_control = np.sqrt(control_var / len(control_outcome))
    se_ate = np.sqrt(se_treated**2 + se_control**2)
    
    print(f"\n=== IPW TREATMENT EFFECT ===")
    print(f"Weighted treated mean: {weighted_treated_mean:.3f}")
    print(f"Weighted control mean: {weighted_control_mean:.3f}")
    print(f"Average Treatment Effect: {ate_ipw:.3f}")
    print(f"Standard Error: {se_ate:.3f}")
    print(f"95% CI: [{ate_ipw - 1.96*se_ate:.3f}, {ate_ipw + 1.96*se_ate:.3f}]")
    
    return {
        'ate': ate_ipw,
        'se': se_ate,
        'treated_mean': weighted_treated_mean,
        'control_mean': weighted_control_mean,
        'weights': df_ipw[weight_col],
        'effective_n': len(df_ipw)
    }

# Perform IPW analysis
ipw_results = inverse_propensity_weighting(
    df_with_ps, 'propensity_score', 'copilot_usage', 'tickets_per_week', 
    trim_weights=True
)
```

**IPW Advantages vs. Matching:**
- **Efficiency**: Uses entire sample, more statistical power
- **Transparency**: Clear weighting interpretation  
- **Flexibility**: Easier to incorporate into complex models

**IPW Diagnostics to Monitor:**
- **Weight Distribution**: Extreme weights indicate poor overlap
- **Effective Sample Size**: Heavy weighting reduces effective N
- **Balance Check**: Weighted covariates should be balanced

---

## Method 3: Difference-in-Differences

Difference-in-differences (DiD) exploits temporal variation to identify causal effects. It compares changes over time between treatment and control groups, controlling for time-invariant confounders through the "double difference."

**DiD Formula**: 
`Effect = (Treatment_After - Treatment_Before) - (Control_After - Control_Before)`

**Key Assumptions:**
1. **Parallel Trends**: Without treatment, both groups would follow similar trends
2. **No Spillovers**: Treatment doesn't affect control group
3. **Stable Unit Treatment**: Treatment effect is constant over time

**When to Use DiD:**
- Natural experiments with clear before/after periods
- Policy rollouts with staggered implementation  
- A/B tests where you can observe pre-treatment periods

### Basic Difference-in-Differences

```python
def difference_in_differences(df, outcome_col, treatment_col, time_col, unit_col):
    """
    Canonical difference-in-differences estimation
    
    This method compares the change in outcomes over time between
    treatment and control groups. The key insight is that time-invariant
    confounders are differenced out.
    
    Model: Y_it = α + β*Treatment_i + γ*Post_t + δ*(Treatment_i × Post_t) + ε_it
    
    Where δ is the DiD treatment effect.
    """
    
    print(f"=== DIFFERENCE-IN-DIFFERENCES ANALYSIS ===")
    
    # Create post-treatment indicator
    pre_period = df[df[time_col] == 'pre'].copy()
    post_period = df[df[time_col] == 'post'].copy()
    
    print(f"Pre-period observations: {len(pre_period)}")
    print(f"Post-period observations: {len(post_period)}")
    
    # Calculate group means by period
    group_means = df.groupby([treatment_col, time_col])[outcome_col].agg(['mean', 'count', 'std']).round(3)
    print(f"\n=== GROUP MEANS BY PERIOD ===")
    print(group_means)
    
    # Extract the four key quantities for DiD
    try:
        control_pre = group_means.loc[(0, 'pre'), 'mean']
        control_post = group_means.loc[(0, 'post'), 'mean']
        treated_pre = group_means.loc[(1, 'pre'), 'mean']
        treated_post = group_means.loc[(1, 'post'), 'mean']
        
        # Calculate differences
        control_change = control_post - control_pre
        treated_change = treated_post - treated_pre
        did_effect = treated_change - control_change
        
        print(f"\n=== DIFFERENCE-IN-DIFFERENCES CALCULATION ===")
        print(f"Control group change: {control_change:.3f}")
        print(f"Treatment group change: {treated_change:.3f}")
        print(f"DiD Effect: {did_effect:.3f}")
        
        # Interpret the effect
        if did_effect > 0:
            print(f"\nInterpretation: Treatment caused a {did_effect:.3f} unit increase")
            print(f"This represents the effect above what would have happened anyway")
        else:
            print(f"\nInterpretation: Treatment caused a {abs(did_effect):.3f} unit decrease")
        
    except KeyError as e:
        print(f"Error: Missing group in data - {e}")
        return None
    
    # Regression-based DiD for standard errors
    df_reg = df.copy()
    df_reg['post'] = (df_reg[time_col] == 'post').astype(int)
    df_reg['interaction'] = df_reg[treatment_col] * df_reg['post']
    
    # Run DiD regression: Y = α + β*Treatment + γ*Post + δ*(Treatment×Post)
    did_formula = f"{outcome_col} ~ {treatment_col} + post + interaction"
    did_model = smf.ols(did_formula, data=df_reg).fit()
    
    print(f"\n=== REGRESSION RESULTS ===")
    print(f"DiD Coefficient: {did_model.params['interaction']:.3f}")
    print(f"Standard Error: {did_model.bse['interaction']:.3f}")
    print(f"P-value: {did_model.pvalues['interaction']:.3f}")
    print(f"95% CI: [{did_model.conf_int().loc['interaction', 0]:.3f}, "
          f"{did_model.conf_int().loc['interaction', 1]:.3f}]")
    
    # Effect size
    outcome_std = df[outcome_col].std()
    cohens_d = did_effect / outcome_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    results = {
        'did_effect': did_effect,
        'model': did_model,
        'group_means': group_means,
        'control_change': control_change,
        'treated_change': treated_change
    }
    
    return results

# Example: Copilot rollout analysis
# Assumes we have data from before and after Copilot deployment
did_results = difference_in_differences(
    df_longitudinal, 'tickets_per_week', 'copilot_group', 'period', 'employee_id'
)
```

**Business Value of DiD:**
- **Causal Clarity**: Clear before/after comparison removes many confounders
- **Policy Evaluation**: Perfect for evaluating intervention impacts
- **Trend Analysis**: Shows whether effects persist over time

### Testing Parallel Trends Assumption

The parallel trends assumption is crucial for DiD validity. We test this by examining whether treatment and control groups had similar trends before the intervention.

```python
def test_parallel_trends(df, outcome_col, treatment_col, time_col, pre_periods):
    """
    Test parallel trends assumption using pre-treatment data
    
    Strategy:
    1. Use only pre-treatment periods
    2. Test for differential time trends between groups
    3. Visualize trends and test statistically
    """
    
    # Filter to pre-treatment periods only
    pre_data = df[df[time_col].isin(pre_periods)].copy()
    
    print(f"=== PARALLEL TRENDS TEST ===")
    print(f"Pre-treatment periods: {pre_periods}")
    print(f"Pre-treatment observations: {len(pre_data)}")
    
    # Create time trend variable (assuming ordered periods)
    time_mapping = {period: i for i, period in enumerate(sorted(pre_periods))}
    pre_data['time_trend'] = pre_data[time_col].map(time_mapping)
    
    # Test for differential trends: Y = α + β*Treatment + γ*Time + δ*(Treatment×Time)
    pre_data['treat_time_interaction'] = pre_data[treatment_col] * pre_data['time_trend']
    
    trend_formula = f"{outcome_col} ~ {treatment_col} + time_trend + treat_time_interaction"
    trend_model = smf.ols(trend_formula, data=pre_data).fit()
    
    # The interaction coefficient tests parallel trends
    interaction_coeff = trend_model.params['treat_time_interaction']
    interaction_pvalue = trend_model.pvalues['treat_time_interaction']
    
    print(f"\n=== STATISTICAL TEST ===")
    print(f"Differential trend coefficient: {interaction_coeff:.4f}")
    print(f"P-value: {interaction_pvalue:.4f}")
    
    if interaction_pvalue < 0.05:
        print("❌ VIOLATION: Significant differential pre-trends detected")
        print("   DiD estimates may be biased. Consider:")
        print("   - Group-specific time trends")
        print("   - Alternative identification strategies")
    else:
        print("✅ PASSED: No evidence against parallel trends assumption")
        print("   DiD identification appears valid")
    
    # Calculate group-specific trends
    group_trends = pre_data.groupby([treatment_col, time_col])[outcome_col].mean().reset_index()
    
    # Visual trend analysis
    print(f"\n=== TREND ANALYSIS ===")
    for group in [0, 1]:
        group_data = group_trends[group_trends[treatment_col] == group]
        trend_slope = np.polyfit(group_data['time_trend'] if 'time_trend' in group_data.columns 
                               else range(len(group_data)), group_data[outcome_col], 1)[0]
        group_name = "Treatment" if group == 1 else "Control"
        print(f"{group_name} group trend: {trend_slope:.3f} per period")
    
    return {
        'model': trend_model,
        'interaction_coeff': interaction_coeff,
        'p_value': interaction_pvalue,
        'parallel_trends_valid': interaction_pvalue >= 0.05,
        'group_trends': group_trends
    }

# Test parallel trends assumption
# Assumes multiple pre-treatment periods
parallel_trends_test = test_parallel_trends(
    df_longitudinal, 'tickets_per_week', 'copilot_group', 'period',
    pre_periods=['2023-Q1', '2023-Q2', '2023-Q3']  # Pre-rollout quarters
)

if not parallel_trends_test['parallel_trends_valid']:
    print("\n⚠️  Consider robustness checks:")
    print("- Synthetic control methods")
    print("- Regression discontinuity (if applicable)")
    print("- Alternative control groups")
```

**Parallel Trends Interpretation:**
- **Non-significant interaction**: Groups had similar pre-trends (good for DiD)
- **Significant interaction**: Differential pre-trends suggest bias
- **Visual inspection**: Plot trends to understand patterns

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
