---
layout: default
title: "Causal Inference: Regression Adjustment"
permalink: /causal-inference-regression/
---

# Method 1: Regression Adjustment

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

Regression adjustment is the most straightforward causal inference method. It estimates the treatment effect by controlling for confounding variables through linear regression. The key insight is that if you can measure and control for all variables that influence both treatment and outcome, the remaining correlation between treatment and outcome represents the causal effect.

**Navigation:**
- [← Back to Data Preparation]({{ site.baseurl }}/causal-inference-data-prep/)
- [Next: Propensity Score Methods →]({{ site.baseurl }}/causal-inference-propensity/)

---

## When to Use Regression Adjustment

**Best for:**
- Few, well-understood confounders
- Linear relationships between variables
- When you have domain knowledge about confounding structure
- Initial exploratory causal analysis

**Key Assumption:** 
- **No unmeasured confounders** (Conditional Ignorability)
- All variables that influence both treatment and outcome are measured and included

**Strengths:**
- Simple and interpretable
- Fast to implement and compute
- Transparent modeling assumptions
- Easy to communicate to stakeholders

**Limitations:**
- Strong assumption about unmeasured confounders
- Sensitive to model misspecification
- May not handle complex interactions well

---

## Basic Implementation

### Simple Regression Adjustment

```python
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

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
        print(f"Monthly impact: ~{result['ate'] * 4:.1f} additional tickets per user")
    else:
        print(f"Copilot usage DECREASES productivity by {abs(result['ate']):.2f} tickets/week")
        print("Consider investigating implementation or training issues")
else:
    print(f"\n❌ NO SIGNIFICANT EFFECT DETECTED")
    print(f"Cannot conclude that Copilot has a causal effect on productivity")
    print(f"Consider: larger sample size, different model specification, or other methods")
```

**Understanding the Output:**
- **Treatment Effect (ATE)**: The average causal effect across all users
- **Confidence Interval**: Range of plausible true effects (95% confidence)
- **P-value**: Probability of seeing this effect if true effect were zero
- **R²**: How much variation the model explains (model fit quality)

---

## Model Diagnostics and Validation

### Comprehensive Model Diagnostics

```python
def regression_diagnostics(model, df, outcome_col, treatment_col):
    """
    Comprehensive diagnostic testing for regression adjustment
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    diagnostics = {}
    
    # 1. Residual analysis
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    # Normality test
    _, normality_p = stats.jarque_bera(residuals)
    diagnostics['normality_test'] = {
        'statistic': _,
        'p_value': normality_p,
        'normal': normality_p > 0.05
    }
    
    # Homoscedasticity test
    _, het_p, _, _ = stats.diagnostic.het_breuschpagan(residuals, model.model.exog)
    diagnostics['homoscedasticity_test'] = {
        'p_value': het_p,
        'homoscedastic': het_p > 0.05
    }
    
    # 2. Influential observations
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    
    # Identify outliers
    outliers = np.where(cooks_d > 4/len(df))[0]
    high_leverage = np.where(leverage > 2 * len(model.params) / len(df))[0]
    
    diagnostics['outliers'] = {
        'cooks_d_outliers': len(outliers),
        'high_leverage': len(high_leverage),
        'outlier_indices': outliers[:10]  # First 10 outliers
    }
    
    # 3. Model specification tests
    # Ramsey RESET test for functional form
    from statsmodels.stats.diagnostic import linear_reset
    reset_stat, reset_p = linear_reset(model, power=2)
    diagnostics['specification_test'] = {
        'reset_p_value': reset_p,
        'correct_specification': reset_p > 0.05
    }
    
    # 4. Multicollinearity check
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = model.model.exog_names[1:]  # Exclude intercept
    vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) 
                       for i in range(1, model.model.exog.shape[1])]
    
    diagnostics['multicollinearity'] = {
        'max_vif': vif_data['VIF'].max(),
        'high_vif_vars': vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
    }
    
    # Print diagnostic summary
    print("=== REGRESSION DIAGNOSTICS ===")
    print(f"Normality test p-value: {diagnostics['normality_test']['p_value']:.4f}")
    print(f"Homoscedasticity test p-value: {diagnostics['homoscedasticity_test']['p_value']:.4f}")
    print(f"Specification test p-value: {diagnostics['specification_test']['reset_p_value']:.4f}")
    print(f"Max VIF: {diagnostics['multicollinearity']['max_vif']:.2f}")
    print(f"Outliers detected: {diagnostics['outliers']['cooks_d_outliers']}")
    
    # Warnings
    warnings = []
    if not diagnostics['normality_test']['normal']:
        warnings.append("Non-normal residuals - consider robust standard errors")
    if not diagnostics['homoscedasticity_test']['homoscedastic']:
        warnings.append("Heteroscedasticity detected - use robust standard errors")
    if not diagnostics['specification_test']['correct_specification']:
        warnings.append("Model specification issues - consider interactions or polynomials")
    if diagnostics['multicollinearity']['max_vif'] > 5:
        warnings.append(f"High multicollinearity in: {diagnostics['multicollinearity']['high_vif_vars']}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("\n✅ All diagnostic tests passed")
    
    return diagnostics

# Run diagnostics
diagnostic_results = regression_diagnostics(
    result['model'], df, 'tickets_per_week', 'copilot_usage'
)
```

### Robust Standard Errors

```python
def regression_with_robust_se(df, outcome_col, treatment_col, confounder_cols):
    """
    Regression adjustment with robust standard errors
    """
    # Prepare formula
    confounders_str = ' + '.join(confounder_cols)
    formula = f'{outcome_col} ~ {treatment_col} + {confounders_str}'
    
    # Fit model with robust standard errors
    model = smf.ols(formula, data=df).fit(cov_type='HC3')  # HC3 robust standard errors
    
    # Extract results
    treatment_effect = model.params[treatment_col]
    robust_se = model.bse[treatment_col]
    robust_ci = model.conf_int().loc[treatment_col]
    
    results = {
        'ate': treatment_effect,
        'robust_se': robust_se,
        'ci_lower': robust_ci[0],
        'ci_upper': robust_ci[1],
        'p_value': model.pvalues[treatment_col],
        'model': model
    }
    
    print(f"=== ROBUST REGRESSION RESULTS ===")
    print(f"Treatment Effect: {results['ate']:.3f}")
    print(f"Robust Standard Error: {results['robust_se']:.3f}")
    print(f"95% CI: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")
    
    return results

# Run with robust standard errors
robust_result = regression_with_robust_se(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
```

---

## Advanced Regression with Interaction Effects

### Heterogeneous Treatment Effects

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

### Marginal Effects Analysis

```python
def calculate_marginal_effects(model, df, treatment_col, continuous_vars):
    """
    Calculate marginal effects for continuous interaction variables
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    marginal_effects = {}
    
    for var in continuous_vars:
        interaction_term = f'{treatment_col}:{var}'
        if interaction_term in model.params:
            # Calculate marginal effect at different percentiles
            percentiles = [10, 25, 50, 75, 90]
            var_percentiles = np.percentile(df[var], percentiles)
            
            main_effect = model.params[treatment_col]
            interaction_coeff = model.params[interaction_term]
            
            effects_at_percentiles = {}
            for p, value in zip(percentiles, var_percentiles):
                marginal_effect = main_effect + (interaction_coeff * value)
                effects_at_percentiles[f'p{p}'] = marginal_effect
            
            marginal_effects[var] = effects_at_percentiles
    
    return marginal_effects

# Calculate marginal effects for continuous variables
if 'tenure_months' in confounders:
    marginal_effects = calculate_marginal_effects(
        model, df, 'copilot_usage', ['tenure_months']
    )
    
    print("\n=== MARGINAL EFFECTS BY TENURE ===")
    for var, effects in marginal_effects.items():
        print(f"{var}:")
        for percentile, effect in effects.items():
            print(f"  {percentile}: {effect:.3f}")
```

---

## Polynomial and Non-linear Specifications

### Testing Non-linear Relationships

```python
def test_nonlinear_specifications(df, outcome_col, treatment_col, confounder_cols):
    """
    Test different functional forms to improve model specification
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    
    specifications = {}
    
    # 1. Linear specification (baseline)
    linear_formula = f'{outcome_col} ~ {treatment_col} + {" + ".join(confounder_cols)}'
    linear_model = smf.ols(linear_formula, data=df).fit()
    specifications['linear'] = {
        'model': linear_model,
        'r_squared': linear_model.rsquared,
        'aic': linear_model.aic,
        'ate': linear_model.params[treatment_col]
    }
    
    # 2. Quadratic treatment effect
    df_quad = df.copy()
    df_quad['treatment_squared'] = df_quad[treatment_col] ** 2
    quad_formula = f'{outcome_col} ~ {treatment_col} + treatment_squared + {" + ".join(confounder_cols)}'
    quad_model = smf.ols(quad_formula, data=df_quad).fit()
    specifications['quadratic'] = {
        'model': quad_model,
        'r_squared': quad_model.rsquared,
        'aic': quad_model.aic,
        'linear_coeff': quad_model.params[treatment_col],
        'quad_coeff': quad_model.params['treatment_squared']
    }
    
    # 3. Log specification (if outcome is positive)
    if (df[outcome_col] > 0).all():
        df_log = df.copy()
        df_log['log_outcome'] = np.log(df_log[outcome_col])
        log_formula = f'log_outcome ~ {treatment_col} + {" + ".join(confounder_cols)}'
        log_model = smf.ols(log_formula, data=df_log).fit()
        specifications['log'] = {
            'model': log_model,
            'r_squared': log_model.rsquared,
            'aic': log_model.aic,
            'ate_percent': log_model.params[treatment_col] * 100  # Percentage effect
        }
    
    # Compare specifications
    print("=== MODEL SPECIFICATION COMPARISON ===")
    for spec_name, spec_results in specifications.items():
        print(f"{spec_name.upper()}:")
        print(f"  R²: {spec_results['r_squared']:.4f}")
        print(f"  AIC: {spec_results['aic']:.2f}")
        if 'ate' in spec_results:
            print(f"  Treatment Effect: {spec_results['ate']:.3f}")
        elif 'ate_percent' in spec_results:
            print(f"  Treatment Effect: {spec_results['ate_percent']:.1f}%")
    
    # Recommend best specification
    best_spec = min(specifications.items(), key=lambda x: x[1]['aic'])
    print(f"\n✅ RECOMMENDED SPECIFICATION: {best_spec[0].upper()} (lowest AIC)")
    
    return specifications

# Test different specifications
specifications = test_nonlinear_specifications(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
```

---

## Sensitivity Analysis

### Testing Robustness to Model Specification

```python
def sensitivity_analysis_specification(df, outcome_col, treatment_col, all_confounders):
    """
    Test sensitivity of results to different confounder specifications
    """
    from itertools import combinations
    
    # Generate different confounder combinations
    results = []
    
    # Minimal specification
    minimal_confounders = all_confounders[:2]
    minimal_result = regression_adjustment(df, outcome_col, treatment_col, minimal_confounders)
    results.append({
        'specification': 'minimal',
        'confounders': minimal_confounders,
        'ate': minimal_result['ate'],
        'ci_lower': minimal_result['ci_lower'],
        'ci_upper': minimal_result['ci_upper'],
        'n_confounders': len(minimal_confounders)
    })
    
    # Medium specification
    medium_confounders = all_confounders[:len(all_confounders)//2]
    medium_result = regression_adjustment(df, outcome_col, treatment_col, medium_confounders)
    results.append({
        'specification': 'medium',
        'confounders': medium_confounders,
        'ate': medium_result['ate'],
        'ci_lower': medium_result['ci_lower'],
        'ci_upper': medium_result['ci_upper'],
        'n_confounders': len(medium_confounders)
    })
    
    # Full specification
    full_result = regression_adjustment(df, outcome_col, treatment_col, all_confounders)
    results.append({
        'specification': 'full',
        'confounders': all_confounders,
        'ate': full_result['ate'],
        'ci_lower': full_result['ci_lower'],
        'ci_upper': full_result['ci_upper'],
        'n_confounders': len(all_confounders)
    })
    
    # Create sensitivity plot
    sensitivity_df = pd.DataFrame(results)
    
    print("=== SENSITIVITY TO CONFOUNDER SPECIFICATION ===")
    for _, row in sensitivity_df.iterrows():
        print(f"{row['specification'].upper()}: ATE = {row['ate']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
              f"({row['n_confounders']} confounders)")
    
    # Check if confidence intervals overlap
    ate_range = (sensitivity_df['ate'].min(), sensitivity_df['ate'].max())
    ate_stability = (ate_range[1] - ate_range[0]) / abs(sensitivity_df['ate'].mean())
    
    print(f"\nATE range: [{ate_range[0]:.3f}, {ate_range[1]:.3f}]")
    print(f"Relative stability: {ate_stability:.1%}")
    
    if ate_stability < 0.2:
        print("✅ Results are stable across specifications")
    else:
        print("⚠️  Results vary significantly across specifications")
        print("   Consider robustness checks or alternative methods")
    
    return sensitivity_df

# Run sensitivity analysis
sensitivity_results = sensitivity_analysis_specification(
    df, 'tickets_per_week', 'copilot_usage', confounders
)
```

---

## Business Value Interpretation

### Converting Statistical Results to Business Impact

```python
def interpret_business_impact(ate, baseline_mean, confidence_interval, cost_per_license=50):
    """
    Convert regression results to business-relevant metrics
    """
    # Calculate percentage improvement
    percentage_improvement = (ate / baseline_mean) * 100
    
    # Calculate business value
    weekly_improvement = ate
    monthly_improvement = ate * 4.33  # Average weeks per month
    annual_improvement = ate * 52
    
    # ROI calculation (simplified)
    annual_license_cost = cost_per_license * 12
    # Assume each ticket represents $20 in value
    ticket_value = 20
    annual_value = annual_improvement * ticket_value
    roi = ((annual_value - annual_license_cost) / annual_license_cost) * 100
    
    # Effect size interpretation
    if abs(ate) < 0.5:
        effect_magnitude = "small"
    elif abs(ate) < 2:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"
    
    interpretation = {
        'statistical': {
            'ate': ate,
            'percentage_improvement': percentage_improvement,
            'confidence_interval': confidence_interval,
            'effect_magnitude': effect_magnitude
        },
        'business': {
            'weekly_improvement': weekly_improvement,
            'monthly_improvement': monthly_improvement,
            'annual_improvement': annual_improvement,
            'annual_value': annual_value,
            'roi_percent': roi
        }
    }
    
    # Create business summary
    print("=== BUSINESS IMPACT SUMMARY ===")
    print(f"Copilot Effect: {ate:.2f} additional tickets per week per user")
    print(f"Percentage Improvement: {percentage_improvement:.1f}%")
    print(f"Monthly Impact: {monthly_improvement:.1f} additional tickets per user")
    print(f"Annual Value: ${annual_value:.0f} per user")
    print(f"ROI: {roi:.0f}%")
    print(f"Effect Size: {effect_magnitude}")
    
    # Uncertainty interpretation
    ci_lower, ci_upper = confidence_interval
    print(f"\nUncertainty Range:")
    print(f"Best case: {ci_upper:.2f} tickets/week ({(ci_upper/baseline_mean)*100:.1f}% improvement)")
    print(f"Worst case: {ci_lower:.2f} tickets/week ({(ci_lower/baseline_mean)*100:.1f}% improvement)")
    
    return interpretation

# Interpret business impact
baseline_productivity = df[df['copilot_usage'] == 0]['tickets_per_week'].mean()
business_impact = interpret_business_impact(
    ate=result['ate'],
    baseline_mean=baseline_productivity,
    confidence_interval=(result['ci_lower'], result['ci_upper'])
)
```

---

## Best Practices and Recommendations

### Implementation Checklist

Before using regression adjustment results:

- ✅ **Check assumptions**: No unmeasured confounders assumption is critical
- ✅ **Run diagnostics**: Test for normality, homoscedasticity, specification
- ✅ **Use robust SEs**: Protect against heteroscedasticity
- ✅ **Test interactions**: Check for heterogeneous treatment effects
- ✅ **Sensitivity analysis**: Test robustness to model specification
- ✅ **Business translation**: Convert statistical results to actionable insights

### When to Move to Other Methods

Consider alternative methods if:

- Diagnostic tests fail consistently
- Results are sensitive to model specification
- You suspect unmeasured confounders
- Treatment assignment appears non-random
- Need to handle complex treatment timing

**Next Method:** [Propensity Score Methods →]({{ site.baseurl }}/causal-inference-propensity/)

---

*Regression adjustment provides a solid foundation for causal analysis when assumptions are met. Always validate results through diagnostics and sensitivity analysis before making business decisions.*
