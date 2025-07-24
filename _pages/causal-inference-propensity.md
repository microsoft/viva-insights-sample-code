---
layout: default
title: "Causal Inference: Propensity Score Methods"
permalink: /causal-inference-propensity/
---

# Method 2: Propensity Score Methods

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

Propensity score methods address selection bias by balancing groups on observed characteristics. Instead of controlling for confounders directly in the outcome model, we match treated and control units with similar probabilities of receiving treatment.

**Navigation:**
- [← Back to Regression Adjustment]({{ site.baseurl }}/causal-inference-regression/)
- [Next: Difference-in-Differences →]({{ site.baseurl }}/causal-inference-did/)

---

## When to Use Propensity Score Methods

**Key Advantages:**
- **Dimension Reduction**: Summarizes many confounders into a single score
- **Transparency**: Shows which units are truly comparable
- **Assumption Testing**: Forces explicit consideration of overlap

**Best for:**
- High-dimensional confounding (many covariates)
- Binary treatment variables
- Need to demonstrate comparable groups to stakeholders
- Concerns about model specification in outcome analysis

**Key Assumption:**
- **No unmeasured confounders** (same as regression adjustment)
- **Overlap/Common support**: Treated and control units exist across propensity score range

---

## Propensity Score Estimation

The propensity score e(x) = P(Treatment = 1 | X = x) is the probability of treatment given observed characteristics. A well-estimated propensity score should achieve balance and have sufficient overlap between groups.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

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
confounders = ['tenure_months', 'job_level', 'team_size', 'manager_span']
df_with_ps, ps_model, overlap_stats = estimate_propensity_scores(
    df, 'copilot_usage', confounders, method='logistic'
)
```

**Interpreting Propensity Score Quality:**
- **AUC 0.6-0.8**: Good discrimination, sufficient variation in treatment probability
- **AUC > 0.9**: Excellent prediction but may indicate poor overlap
- **Common Support**: Should include most observations (>80%)
- **Extreme Scores**: Values near 0 or 1 suggest deterministic assignment

---

## Propensity Score Matching

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

---

## Balance Assessment

### Checking Covariate Balance After Matching

```python
def assess_covariate_balance(df_original, df_matched, confounders, treatment_col):
    """
    Assess covariate balance before and after propensity score matching
    """
    from scipy.stats import ttest_ind
    
    balance_results = []
    
    for var in confounders:
        if var not in df_original.columns:
            continue
            
        # Before matching
        treated_before = df_original[df_original[treatment_col] == 1][var]
        control_before = df_original[df_original[treatment_col] == 0][var]
        
        if treated_before.dtype in ['int64', 'float64']:
            # Continuous variables
            diff_before = treated_before.mean() - control_before.mean()
            pooled_std_before = np.sqrt(((treated_before.var() + control_before.var()) / 2))
            std_diff_before = diff_before / pooled_std_before if pooled_std_before > 0 else 0
            
            # After matching
            treated_after = df_matched[df_matched[treatment_col] == 1][var]
            control_after = df_matched[df_matched[treatment_col] == 0][var]
            
            diff_after = treated_after.mean() - control_after.mean()
            pooled_std_after = np.sqrt(((treated_after.var() + control_after.var()) / 2))
            std_diff_after = diff_after / pooled_std_after if pooled_std_after > 0 else 0
            
            balance_results.append({
                'variable': var,
                'type': 'continuous',
                'std_diff_before': std_diff_before,
                'std_diff_after': std_diff_after,
                'improvement': abs(std_diff_before) - abs(std_diff_after)
            })
        else:
            # Categorical variables - use proportion differences
            treated_props_before = treated_before.value_counts(normalize=True)
            control_props_before = control_before.value_counts(normalize=True)
            max_diff_before = max(abs(treated_props_before - control_props_before).fillna(0))
            
            treated_props_after = df_matched[df_matched[treatment_col] == 1][var].value_counts(normalize=True)
            control_props_after = df_matched[df_matched[treatment_col] == 0][var].value_counts(normalize=True)
            max_diff_after = max(abs(treated_props_after - control_props_after).fillna(0))
            
            balance_results.append({
                'variable': var,
                'type': 'categorical',
                'max_diff_before': max_diff_before,
                'max_diff_after': max_diff_after,
                'improvement': max_diff_before - max_diff_after
            })
    
    # Create balance summary
    balance_df = pd.DataFrame(balance_results)
    
    print("=== COVARIATE BALANCE ASSESSMENT ===")
    print("Standard differences (continuous) / Max proportion differences (categorical)")
    print("Good balance: |standardized difference| < 0.1")
    
    for _, row in balance_df.iterrows():
        if row['type'] == 'continuous':
            before_val = row['std_diff_before']
            after_val = row['std_diff_after']
            threshold = 0.1
        else:
            before_val = row['max_diff_before']
            after_val = row['max_diff_after']
            threshold = 0.1
        
        status = "✅" if abs(after_val) < threshold else "⚠️"
        print(f"{status} {row['variable']}: {before_val:.3f} → {after_val:.3f} "
              f"(improvement: {row['improvement']:.3f})")
    
    # Overall balance assessment
    if row['type'] == 'continuous':
        poor_balance = (balance_df['std_diff_after'].abs() > 0.1).sum()
    else:
        poor_balance = (balance_df['max_diff_after'] > 0.1).sum()
    
    if poor_balance == 0:
        print(f"\n✅ EXCELLENT BALANCE: All covariates well-balanced")
    elif poor_balance <= len(balance_df) * 0.2:
        print(f"\n✅ GOOD BALANCE: {poor_balance}/{len(balance_df)} covariates with poor balance")
    else:
        print(f"\n⚠️  POOR BALANCE: {poor_balance}/{len(balance_df)} covariates with poor balance")
        print("   Consider: different caliper, more flexible PS model, or alternative methods")
    
    return balance_df

# Assess balance after matching
if matching_results:
    balance_assessment = assess_covariate_balance(
        df_with_ps, matching_results['matched_dataset'], 
        confounders, 'copilot_usage'
    )
```

---

## Inverse Propensity Weighting (IPW)

Instead of matching, we can weight observations by their inverse propensity scores. This uses all data while balancing the groups.

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

## Advanced Propensity Score Methods

### Stratification by Propensity Score

```python
def propensity_score_stratification(df, ps_col, treatment_col, outcome_col, n_strata=5):
    """
    Stratification approach to propensity score analysis
    """
    df_strat = df.copy()
    
    # Create propensity score strata
    df_strat['ps_stratum'] = pd.qcut(df_strat[ps_col], q=n_strata, labels=False)
    
    stratum_effects = []
    stratum_weights = []
    
    print(f"=== PROPENSITY SCORE STRATIFICATION ===")
    print(f"Number of strata: {n_strata}")
    
    for stratum in range(n_strata):
        stratum_data = df_strat[df_strat['ps_stratum'] == stratum]
        
        if stratum_data[treatment_col].nunique() < 2:
            print(f"Stratum {stratum}: Insufficient variation in treatment")
            continue
        
        # Calculate stratum-specific effect
        treated_mean = stratum_data[stratum_data[treatment_col] == 1][outcome_col].mean()
        control_mean = stratum_data[stratum_data[treatment_col] == 0][outcome_col].mean()
        stratum_effect = treated_mean - control_mean
        
        # Weight by stratum size
        stratum_weight = len(stratum_data) / len(df_strat)
        
        stratum_effects.append(stratum_effect)
        stratum_weights.append(stratum_weight)
        
        print(f"Stratum {stratum}: Effect = {stratum_effect:.3f}, Weight = {stratum_weight:.3f}, N = {len(stratum_data)}")
    
    # Calculate weighted average treatment effect
    ate_stratified = np.average(stratum_effects, weights=stratum_weights)
    
    print(f"\n=== STRATIFIED TREATMENT EFFECT ===")
    print(f"Weighted ATE: {ate_stratified:.3f}")
    
    return {
        'ate': ate_stratified,
        'stratum_effects': stratum_effects,
        'stratum_weights': stratum_weights
    }

# Perform stratification analysis
stratification_results = propensity_score_stratification(
    df_with_ps, 'propensity_score', 'copilot_usage', 'tickets_per_week'
)
```

### Propensity Score Regression

```python
def propensity_score_regression(df, ps_col, treatment_col, outcome_col, confounders):
    """
    Combine propensity scores with regression adjustment
    """
    import statsmodels.formula.api as smf
    
    # Include propensity score as a control variable
    enhanced_confounders = confounders + [ps_col]
    formula = f'{outcome_col} ~ {treatment_col} + {" + ".join(enhanced_confounders)}'
    
    model = smf.ols(formula, data=df).fit()
    
    ate = model.params[treatment_col]
    se = model.bse[treatment_col]
    ci = model.conf_int().loc[treatment_col]
    
    print(f"=== PROPENSITY SCORE REGRESSION ===")
    print(f"Treatment Effect: {ate:.3f}")
    print(f"Standard Error: {se:.3f}")
    print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    return {
        'ate': ate,
        'se': se,
        'model': model
    }

# Perform PS regression
ps_regression_results = propensity_score_regression(
    df_with_ps, 'propensity_score', 'copilot_usage', 'tickets_per_week', confounders
)
```

---

## Comparing Propensity Score Methods

### Method Comparison Framework

```python
def compare_ps_methods(df_with_ps, ps_col, treatment_col, outcome_col, confounders):
    """
    Compare different propensity score methods
    """
    results_comparison = {}
    
    # 1. Matching (if results exist)
    if matching_results:
        results_comparison['Matching'] = {
            'ate': matching_results['ate'],
            'se': matching_results['se'],
            'sample_size': len(matching_results['matched_dataset']),
            'method_specific': f"Match rate: {matching_results['match_quality']['match_rate']:.1%}"
        }
    
    # 2. IPW
    results_comparison['IPW'] = {
        'ate': ipw_results['ate'],
        'se': ipw_results['se'],
        'sample_size': ipw_results['effective_n'],
        'method_specific': f"Effective N: {ipw_results['effective_n']}"
    }
    
    # 3. Stratification
    results_comparison['Stratification'] = {
        'ate': stratification_results['ate'],
        'se': np.nan,  # Not calculated in simple version
        'sample_size': len(df_with_ps),
        'method_specific': f"Strata: {len(stratification_results['stratum_effects'])}"
    }
    
    # 4. PS Regression
    results_comparison['PS Regression'] = {
        'ate': ps_regression_results['ate'],
        'se': ps_regression_results['se'],
        'sample_size': len(df_with_ps),
        'method_specific': f"R²: {ps_regression_results['model'].rsquared:.3f}"
    }
    
    # Create comparison table
    comparison_df = pd.DataFrame(results_comparison).T
    
    print("=== PROPENSITY SCORE METHODS COMPARISON ===")
    print(comparison_df[['ate', 'se', 'sample_size', 'method_specific']])
    
    # Calculate range of estimates
    ates = [result['ate'] for result in results_comparison.values()]
    ate_range = max(ates) - min(ates)
    ate_mean = np.mean(ates)
    
    print(f"\nATE Range: {min(ates):.3f} to {max(ates):.3f}")
    print(f"Range as % of mean: {(ate_range/abs(ate_mean)*100):.1f}%")
    
    if ate_range / abs(ate_mean) < 0.2:
        print("✅ Results are consistent across methods")
    else:
        print("⚠️  Results vary significantly across methods")
        print("   Consider robustness checks and sensitivity analysis")
    
    return comparison_df

# Compare all PS methods
if 'matching_results' in locals() and 'ipw_results' in locals():
    method_comparison = compare_ps_methods(
        df_with_ps, 'propensity_score', 'copilot_usage', 'tickets_per_week', confounders
    )
```

---

## Best Practices and Troubleshooting

### Common Issues and Solutions

```python
def diagnose_ps_issues(df, ps_col, treatment_col, overlap_stats):
    """
    Diagnose common propensity score issues and provide solutions
    """
    issues = []
    solutions = []
    
    # 1. Poor overlap
    if overlap_stats['overlap_rate'] < 0.8:
        issues.append("Poor overlap between treatment and control groups")
        solutions.append("Consider: trimming extreme PS values, different PS model, or collect more data")
    
    # 2. Extreme propensity scores
    if overlap_stats['extreme_count'] > len(df) * 0.05:
        issues.append("Many extreme propensity scores (near 0 or 1)")
        solutions.append("Consider: more flexible PS model (random forest), interaction terms, or polynomial terms")
    
    # 3. Perfect prediction
    ps_values = df[ps_col]
    if (ps_values == 0).any() or (ps_values == 1).any():
        issues.append("Perfect prediction in propensity score model")
        solutions.append("Consider: removing perfectly predictive variables or using penalized regression")
    
    # 4. Low variation in propensity scores
    ps_std = ps_values.std()
    if ps_std < 0.1:
        issues.append("Low variation in propensity scores")
        solutions.append("Consider: adding more relevant confounders or checking treatment assignment mechanism")
    
    print("=== PROPENSITY SCORE DIAGNOSTICS ===")
    if issues:
        print("Issues found:")
        for i, (issue, solution) in enumerate(zip(issues, solutions), 1):
            print(f"{i}. {issue}")
            print(f"   Solution: {solution}")
    else:
        print("✅ No major issues detected")
    
    return {'issues': issues, 'solutions': solutions}

# Diagnose potential issues
ps_diagnostics = diagnose_ps_issues(df_with_ps, 'propensity_score', 'copilot_usage', overlap_stats)
```

### Implementation Checklist

Before using propensity score results:

- ✅ **Check overlap**: Ensure adequate common support (>80% of observations)
- ✅ **Assess balance**: Verify covariate balance after matching/weighting
- ✅ **Compare methods**: Test multiple PS approaches for consistency
- ✅ **Validate assumptions**: No unmeasured confounders assumption still applies
- ✅ **Sensitivity analysis**: Test robustness to PS model specification
- ✅ **Business translation**: Convert results to actionable insights

### When to Move to Other Methods

Consider alternative methods if:
- Poor overlap persists despite PS model improvements
- Results are highly sensitive to PS model specification
- Strong suspicion of unmeasured confounders
- Need to handle time-varying treatments
- Complex treatment timing or dose-response relationships

**Next Method:** [Difference-in-Differences →]({{ site.baseurl }}/causal-inference-did/)

---

*Propensity score methods provide powerful tools for creating balanced comparisons when randomization isn't possible. Always validate overlap and balance assumptions before interpreting results.*
