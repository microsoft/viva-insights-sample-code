---
layout: default
title: "Causal Inference: Instrumental Variables"
permalink: /causal-inference-iv/
---

# Method 4: Instrumental Variables (IV)

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

Instrumental Variables (IV) address endogeneity by using variation that affects treatment assignment but only influences outcomes through the treatment itself. This powerful technique can recover causal effects even with unmeasured confounders.

**Navigation:**
- [← Back to Difference-in-Differences]({{ site.baseurl }}/causal-inference-did/)
- [Next: Doubly Robust Methods →]({{ site.baseurl }}/causal-inference-doubly-robust/)

---

## When to Use Instrumental Variables

**Key Advantages:**
- **Handles unmeasured confounders**: Controls for unobserved factors that affect both treatment and outcome
- **Natural experiments**: Exploits random or quasi-random variation
- **Policy evaluation**: Useful when treatment assignment has random component

**Best for:**
- Unmeasured confounding suspected
- Natural experiments available (lottery systems, policy changes, geographic variation)
- Treatment assignment partly random but not fully controlled
- Need to address reverse causality

**Key Requirements:**
1. **Relevance**: Instrument strongly predicts treatment assignment
2. **Exclusion restriction**: Instrument only affects outcome through treatment
3. **Independence**: Instrument is as-good-as randomly assigned

---

## Understanding Instrumental Variables Logic

The IV approach uses two-stage least squares (2SLS):
1. **First stage**: Predict treatment using instrument
2. **Second stage**: Use predicted treatment to estimate causal effect

```python
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def iv_conceptual_explanation():
    """
    Explain the IV logic conceptually
    """
    print("=== INSTRUMENTAL VARIABLES CONCEPTUAL FRAMEWORK ===")
    print("\nIV Logic:")
    print("1. Find variable (instrument) that:")
    print("   - Affects treatment assignment (RELEVANCE)")
    print("   - Only affects outcome through treatment (EXCLUSION)")
    print("   - Is as-good-as random (INDEPENDENCE)")
    print("\n2. Use instrument to create 'random' variation in treatment")
    print("3. Compare outcomes for this 'random' variation")
    
    print("\nExample Applications:")
    print("- Random assignment to managers → different treatment adoption")
    print("- Geographic distance → access to resources")
    print("- Calendar timing → administrative decisions")
    print("- System randomization → feature exposure")
    
    print("\nCausal Chain:")
    print("Instrument → Treatment → Outcome")
    print("     ↓           ↓")
    print("   Random    Causal Effect")
    print("  Variation   (what we want)")

iv_conceptual_explanation()
```

---

## First Stage Analysis

The first stage examines how well the instrument predicts treatment assignment. A strong first stage is crucial for valid IV estimation.

```python
def first_stage_analysis(df, instrument_col, treatment_col, confounders=None):
    """
    Comprehensive first stage analysis
    
    The first stage regression shows how the instrument affects treatment:
    Treatment = α + β₁ * Instrument + β₂ * Controls + ε
    
    Strong instruments should have:
    - High F-statistic (> 10, preferably > 20)
    - Significant coefficient on instrument
    - Sufficient variation explained
    """
    
    print(f"=== FIRST STAGE ANALYSIS ===")
    print(f"Instrument: {instrument_col}")
    print(f"Treatment: {treatment_col}")
    
    # Basic first stage regression
    if confounders:
        controls = " + ".join(confounders)
        first_stage_formula = f'{treatment_col} ~ {instrument_col} + {controls}'
    else:
        first_stage_formula = f'{treatment_col} ~ {instrument_col}'
    
    first_stage_model = smf.ols(first_stage_formula, data=df).fit()
    
    # Extract key statistics
    instrument_coef = first_stage_model.params[instrument_col]
    instrument_se = first_stage_model.bse[instrument_col]
    instrument_tstat = first_stage_model.tvalues[instrument_col]
    instrument_pvalue = first_stage_model.pvalues[instrument_col]
    f_statistic = first_stage_model.fvalue
    r_squared = first_stage_model.rsquared
    
    print(f"\n=== FIRST STAGE RESULTS ===")
    print(f"Instrument coefficient: {instrument_coef:.4f}")
    print(f"Standard error: {instrument_se:.4f}")
    print(f"T-statistic: {instrument_tstat:.2f}")
    print(f"P-value: {instrument_pvalue:.4f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"F-statistic: {f_statistic:.2f}")
    
    # Strength assessment
    print(f"\n=== INSTRUMENT STRENGTH ASSESSMENT ===")
    
    if f_statistic > 20:
        strength = "STRONG"
        color = "✅"
    elif f_statistic > 10:
        strength = "ADEQUATE"
        color = "⚠️"
    else:
        strength = "WEAK"
        color = "❌"
    
    print(f"{color} Instrument strength: {strength} (F = {f_statistic:.2f})")
    
    if f_statistic < 10:
        print("   Warning: Weak instrument may lead to:")
        print("   - Biased estimates")
        print("   - Invalid inference")
        print("   - Large standard errors")
    
    # Relevance test
    if instrument_pvalue < 0.001:
        relevance = "HIGHLY SIGNIFICANT"
    elif instrument_pvalue < 0.01:
        relevance = "VERY SIGNIFICANT"
    elif instrument_pvalue < 0.05:
        relevance = "SIGNIFICANT"
    else:
        relevance = "NOT SIGNIFICANT"
        print("❌ Instrument fails relevance condition")
    
    print(f"Relevance: {relevance} (p = {instrument_pvalue:.4f})")
    
    # Partial correlation (instrument-treatment correlation controlling for other factors)
    if confounders:
        # Residualize both variables
        instrument_resid_model = smf.ols(f'{instrument_col} ~ {controls}', data=df).fit()
        treatment_resid_model = smf.ols(f'{treatment_col} ~ {controls}', data=df).fit()
        
        partial_corr = np.corrcoef(instrument_resid_model.resid, treatment_resid_model.resid)[0, 1]
        print(f"Partial correlation (controlling for confounders): {partial_corr:.3f}")
    else:
        simple_corr = df[instrument_col].corr(df[treatment_col])
        print(f"Simple correlation: {simple_corr:.3f}")
    
    # Visual analysis
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Instrument vs Treatment
    plt.subplot(1, 2, 1)
    plt.scatter(df[instrument_col], df[treatment_col], alpha=0.5)
    plt.xlabel(instrument_col.replace('_', ' ').title())
    plt.ylabel(treatment_col.replace('_', ' ').title())
    plt.title('First Stage: Instrument vs Treatment')
    
    # Add regression line
    x_range = np.linspace(df[instrument_col].min(), df[instrument_col].max(), 100)
    y_pred = first_stage_model.params['Intercept'] + first_stage_model.params[instrument_col] * x_range
    plt.plot(x_range, y_pred, 'r-', linewidth=2)
    
    # Plot 2: First stage residuals
    plt.subplot(1, 2, 2)
    plt.scatter(first_stage_model.fittedvalues, first_stage_model.resid, alpha=0.5)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('First Stage Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': first_stage_model,
        'f_statistic': f_statistic,
        'instrument_coef': instrument_coef,
        'r_squared': r_squared,
        'strength': strength,
        'relevance_pvalue': instrument_pvalue
    }

# Example first stage analysis
# This assumes you have an instrument variable like 'manager_tenure' or 'team_assignment_date'
confounders = ['tenure_months', 'job_level', 'team_size']

# Example: Using manager assignment as instrument for Copilot adoption
first_stage_results = first_stage_analysis(
    df, 'manager_supports_ai', 'copilot_usage', confounders
)
```

---

## Two-Stage Least Squares (2SLS) Estimation

The core IV estimation uses two-stage least squares to isolate the causal effect.

```python
def two_stage_least_squares(df, instrument_col, treatment_col, outcome_col, confounders=None):
    """
    Two-Stage Least Squares estimation
    
    Stage 1: Treatment = α + β₁ * Instrument + β₂ * Controls + ε₁
    Stage 2: Outcome = γ + δ * Predicted_Treatment + γ₂ * Controls + ε₂
    
    The coefficient δ is the causal effect of treatment on outcome
    """
    
    print(f"=== TWO-STAGE LEAST SQUARES ESTIMATION ===")
    
    # Prepare data
    df_iv = df.dropna(subset=[instrument_col, treatment_col, outcome_col])
    
    if confounders:
        df_iv = df_iv.dropna(subset=confounders)
        controls = " + ".join(confounders)
        control_vars = confounders
    else:
        controls = ""
        control_vars = []
    
    n_obs = len(df_iv)
    print(f"Sample size: {n_obs}")
    
    # Stage 1: Predict treatment using instrument
    print(f"\n=== STAGE 1: PREDICT TREATMENT ===")
    
    if controls:
        stage1_formula = f'{treatment_col} ~ {instrument_col} + {controls}'
    else:
        stage1_formula = f'{treatment_col} ~ {instrument_col}'
    
    stage1_model = smf.ols(stage1_formula, data=df_iv).fit()
    predicted_treatment = stage1_model.fittedvalues
    
    # Add predicted treatment to dataset
    df_iv = df_iv.copy()
    df_iv['predicted_treatment'] = predicted_treatment
    
    print(f"R-squared: {stage1_model.rsquared:.3f}")
    print(f"F-statistic: {stage1_model.fvalue:.2f}")
    
    # Check instrument strength again
    f_stat = stage1_model.fvalue
    if f_stat < 10:
        print("⚠️  Warning: Weak instrument may lead to biased estimates")
    
    # Stage 2: Use predicted treatment to estimate causal effect
    print(f"\n=== STAGE 2: CAUSAL EFFECT ESTIMATION ===")
    
    if controls:
        stage2_formula = f'{outcome_col} ~ predicted_treatment + {controls}'
    else:
        stage2_formula = f'{outcome_col} ~ predicted_treatment'
    
    stage2_model = smf.ols(stage2_formula, data=df_iv).fit()
    
    # Extract IV estimate
    iv_effect = stage2_model.params['predicted_treatment']
    iv_se_naive = stage2_model.bse['predicted_treatment']  # This is incorrect - need IV standard errors
    
    # Proper IV standard errors using instrumental variables regression
    # Using statsmodels IV2SLS for correct standard errors
    from statsmodels.sandbox.regression.gmm import IV2SLS
    
    # Prepare variables for IV2SLS
    y = df_iv[outcome_col].values
    X_endog = df_iv[treatment_col].values.reshape(-1, 1)  # Endogenous variable
    X_instruments = df_iv[instrument_col].values.reshape(-1, 1)  # Instrument
    
    if control_vars:
        X_exog = df_iv[control_vars].values  # Exogenous controls
        X_exog = np.column_stack([np.ones(len(X_exog)), X_exog])  # Add constant
    else:
        X_exog = np.ones((len(y), 1))  # Just constant
    
    # IV estimation with correct standard errors
    iv_model = IV2SLS(y, X_exog, X_endog, X_instruments).fit()
    
    iv_effect_correct = iv_model.params[-1]  # Treatment effect (last parameter)
    iv_se_correct = iv_model.bse[-1]  # Correct standard error
    
    # Confidence interval
    ci_lower = iv_effect_correct - 1.96 * iv_se_correct
    ci_upper = iv_effect_correct + 1.96 * iv_se_correct
    
    print(f"IV effect estimate: {iv_effect_correct:.3f}")
    print(f"Standard error: {iv_se_correct:.3f}")
    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # T-statistic and p-value
    t_stat = iv_effect_correct / iv_se_correct
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Compare with OLS (biased) estimate
    print(f"\n=== COMPARISON WITH OLS ===")
    
    if controls:
        ols_formula = f'{outcome_col} ~ {treatment_col} + {controls}'
    else:
        ols_formula = f'{outcome_col} ~ {treatment_col}'
    
    ols_model = smf.ols(ols_formula, data=df_iv).fit()
    ols_effect = ols_model.params[treatment_col]
    ols_se = ols_model.bse[treatment_col]
    
    print(f"OLS estimate: {ols_effect:.3f} ± {ols_se:.3f}")
    print(f"IV estimate: {iv_effect_correct:.3f} ± {iv_se_correct:.3f}")
    print(f"Difference: {iv_effect_correct - ols_effect:.3f}")
    
    # Interpret the difference
    if abs(iv_effect_correct - ols_effect) > 0.1:
        print("⚠️  Large difference suggests significant endogeneity bias")
    else:
        print("✅ Similar estimates suggest minimal endogeneity bias")
    
    return {
        'iv_effect': iv_effect_correct,
        'iv_se': iv_se_correct,
        'ci': [ci_lower, ci_upper],
        'p_value': p_value,
        'ols_effect': ols_effect,
        'stage1_model': stage1_model,
        'stage2_model': stage2_model,
        'iv_model': iv_model
    }

# Perform 2SLS estimation
iv_results = two_stage_least_squares(
    df, 'manager_supports_ai', 'copilot_usage', 'tickets_per_week', confounders
)
```

---

## Instrument Validity Tests

Critical tests to validate that the instrument satisfies IV assumptions.

```python
def test_instrument_validity(df, instrument_col, treatment_col, outcome_col, confounders=None):
    """
    Comprehensive instrument validity testing
    
    Tests:
    1. Relevance: Instrument predicts treatment
    2. Balance: Instrument independent of confounders  
    3. Reduced form: Instrument affects outcome (through treatment)
    4. Overidentification: If multiple instruments available
    """
    
    print(f"=== INSTRUMENT VALIDITY TESTS ===")
    
    # 1. RELEVANCE TEST (already done in first stage, but summarize)
    if confounders:
        controls = " + ".join(confounders)
        first_stage_formula = f'{treatment_col} ~ {instrument_col} + {controls}'
    else:
        first_stage_formula = f'{treatment_col} ~ {instrument_col}'
    
    first_stage_model = smf.ols(first_stage_formula, data=df).fit()
    f_stat = first_stage_model.fvalue
    instrument_pvalue = first_stage_model.pvalues[instrument_col]
    
    print(f"\n1. RELEVANCE TEST")
    print(f"   F-statistic: {f_stat:.2f}")
    print(f"   P-value: {instrument_pvalue:.4f}")
    
    if f_stat > 10 and instrument_pvalue < 0.05:
        print("   ✅ PASS: Instrument is relevant")
        relevance_pass = True
    else:
        print("   ❌ FAIL: Weak or irrelevant instrument")
        relevance_pass = False
    
    # 2. BALANCE TEST (instrument should be independent of confounders)
    print(f"\n2. BALANCE TEST (Independence)")
    
    if confounders:
        balance_results = []
        
        for confounder in confounders:
            # Test if instrument is correlated with confounder
            balance_model = smf.ols(f'{confounder} ~ {instrument_col}', data=df).fit()
            balance_coef = balance_model.params[instrument_col]
            balance_pvalue = balance_model.pvalues[instrument_col]
            
            balance_results.append({
                'confounder': confounder,
                'coefficient': balance_coef,
                'p_value': balance_pvalue,
                'balanced': balance_pvalue > 0.05
            })
            
            status = "✅" if balance_pvalue > 0.05 else "⚠️"
            print(f"   {status} {confounder}: p = {balance_pvalue:.3f}")
        
        balanced_count = sum([r['balanced'] for r in balance_results])
        total_count = len(balance_results)
        
        if balanced_count == total_count:
            print("   ✅ PASS: Instrument balanced on all confounders")
            balance_pass = True
        elif balanced_count / total_count > 0.8:
            print("   ⚠️  CONCERN: Some imbalance detected")
            balance_pass = False
        else:
            print("   ❌ FAIL: Systematic imbalance with confounders")
            balance_pass = False
    else:
        print("   No confounders specified for balance test")
        balance_pass = True
    
    # 3. REDUCED FORM TEST (instrument should affect outcome)
    print(f"\n3. REDUCED FORM TEST")
    
    if confounders:
        reduced_form_formula = f'{outcome_col} ~ {instrument_col} + {controls}'
    else:
        reduced_form_formula = f'{outcome_col} ~ {instrument_col}'
    
    reduced_form_model = smf.ols(reduced_form_formula, data=df).fit()
    reduced_form_coef = reduced_form_model.params[instrument_col]
    reduced_form_pvalue = reduced_form_model.pvalues[instrument_col]
    
    print(f"   Coefficient: {reduced_form_coef:.4f}")
    print(f"   P-value: {reduced_form_pvalue:.4f}")
    
    if reduced_form_pvalue < 0.05:
        print("   ✅ PASS: Instrument affects outcome")
        reduced_form_pass = True
    else:
        print("   ❌ CONCERN: No direct effect of instrument on outcome")
        print("   May indicate weak instrument or violated exclusion restriction")
        reduced_form_pass = False
    
    # 4. EXCLUSION RESTRICTION (cannot be directly tested, but provide guidance)
    print(f"\n4. EXCLUSION RESTRICTION (Untestable)")
    print("   Assumption: Instrument only affects outcome through treatment")
    print("   Requires domain knowledge and logical reasoning")
    print("   Consider:")
    print("   - Are there plausible alternative channels?")
    print("   - Does the instrument make economic/business sense?")
    print("   - Are there unmodeled pathways?")
    
    # Overall validity assessment
    print(f"\n=== OVERALL VALIDITY ASSESSMENT ===")
    
    tests_passed = sum([relevance_pass, balance_pass, reduced_form_pass])
    
    if tests_passed == 3:
        validity = "HIGH"
        color = "✅"
        recommendation = "Proceed with IV analysis"
    elif tests_passed == 2:
        validity = "MODERATE"
        color = "⚠️"
        recommendation = "Use IV with caution, report sensitivity tests"
    else:
        validity = "LOW"
        color = "❌"
        recommendation = "Do not use this instrument"
    
    print(f"{color} Instrument validity: {validity}")
    print(f"   Tests passed: {tests_passed}/3")
    print(f"   Recommendation: {recommendation}")
    
    return {
        'relevance_pass': relevance_pass,
        'balance_pass': balance_pass,
        'reduced_form_pass': reduced_form_pass,
        'validity_score': validity,
        'tests_passed': tests_passed,
        'f_statistic': f_stat
    }

# Test instrument validity
validity_results = test_instrument_validity(
    df, 'manager_supports_ai', 'copilot_usage', 'tickets_per_week', confounders
)
```

---

## Instrument Strength and Weak IV Diagnostics

### Comprehensive Weak Instrument Diagnostics

```python
def weak_instrument_diagnostics(first_stage_results, iv_results, confidence_level=0.95):
    """
    Advanced diagnostics for weak instrument problems
    
    Weak instruments can cause:
    - Biased estimates
    - Poor finite-sample properties  
    - Invalid confidence intervals
    """
    
    print(f"=== WEAK INSTRUMENT DIAGNOSTICS ===")
    
    f_stat = first_stage_results['f_statistic']
    iv_effect = iv_results['iv_effect']
    iv_se = iv_results['iv_se']
    
    # 1. Stock-Yogo critical values for weak instrument test
    print(f"\n1. STOCK-YOGO WEAK INSTRUMENT TEST")
    print(f"   First-stage F-statistic: {f_stat:.2f}")
    
    # Critical values for 10% maximal IV size (approximate)
    if f_stat > 16.38:
        print("   ✅ STRONG: Reject weak instrument hypothesis (10% max size)")
    elif f_stat > 6.66:
        print("   ⚠️  MODERATE: Weak instrument concerns (size distortion possible)")
    else:
        print("   ❌ WEAK: Strong evidence of weak instrument problem")
    
    # 2. Effective F-statistic (Olea and Pflueger 2013)
    # Simplified version - in practice, use specialized software
    print(f"\n2. EFFECTIVE F-STATISTIC")
    print(f"   F-statistic: {f_stat:.2f}")
    
    if f_stat > 104:
        print("   ✅ Inference robust to weak instruments")
    elif f_stat > 37:
        print("   ⚠️  Moderate robustness")
    else:
        print("   ❌ Inference not robust to weak instruments")
        print("   Consider: Anderson-Rubin confidence intervals")
    
    # 3. Concentration parameter and bias assessment
    print(f"\n3. BIAS ASSESSMENT")
    
    # Rule of thumb: bias ≈ k/F where k is number of endogenous variables (1 here)
    approximate_bias = 1 / f_stat if f_stat > 0 else float('inf')
    
    print(f"   Approximate relative bias: {approximate_bias:.1%}")
    
    if approximate_bias < 0.05:
        print("   ✅ Bias likely < 5%")
    elif approximate_bias < 0.10:
        print("   ⚠️  Bias may be 5-10%")
    else:
        print("   ❌ Bias likely > 10%")
    
    # 4. Confidence interval diagnostics
    print(f"\n4. CONFIDENCE INTERVAL DIAGNOSTICS")
    
    iv_ci = iv_results['ci']
    ci_width = iv_ci[1] - iv_ci[0]
    
    print(f"   95% CI: [{iv_ci[0]:.3f}, {iv_ci[1]:.3f}]")
    print(f"   CI width: {ci_width:.3f}")
    
    # Compare CI width to OLS
    ols_effect = iv_results['ols_effect']
    ols_se = iv_results.get('ols_se', iv_se)  # Fallback if not available
    ols_ci_width = 2 * 1.96 * ols_se
    
    width_ratio = ci_width / ols_ci_width if ols_ci_width > 0 else float('inf')
    print(f"   IV CI width / OLS CI width: {width_ratio:.1f}")
    
    if width_ratio < 2:
        print("   ✅ Reasonable precision loss")
    elif width_ratio < 5:
        print("   ⚠️  Moderate precision loss")
    else:
        print("   ❌ Severe precision loss")
    
    # 5. Recommendations based on diagnostics
    print(f"\n=== RECOMMENDATIONS ===")
    
    if f_stat > 20:
        print("✅ STRONG INSTRUMENT:")
        print("   - Standard IV inference valid")
        print("   - Minimal weak instrument bias")
        print("   - Proceed with confidence")
        
    elif f_stat > 10:
        print("⚠️  ADEQUATE INSTRUMENT:")
        print("   - IV estimates likely valid")
        print("   - Consider reporting weak-IV robust tests")
        print("   - Monitor precision")
        
    else:
        print("❌ WEAK INSTRUMENT:")
        print("   - Standard IV inference invalid")
        print("   - Consider alternative approaches:")
        print("     * Find stronger instruments")
        print("     * Use weak-IV robust methods")
        print("     * Anderson-Rubin confidence intervals")
        print("     * Limited information maximum likelihood")
    
    return {
        'f_statistic': f_stat,
        'strength_category': 'strong' if f_stat > 20 else 'adequate' if f_stat > 10 else 'weak',
        'approximate_bias': approximate_bias,
        'ci_width_ratio': width_ratio
    }

# Perform weak instrument diagnostics
if 'first_stage_results' in locals() and 'iv_results' in locals():
    weak_iv_diagnostics = weak_instrument_diagnostics(first_stage_results, iv_results)
```

---

## Multiple Instruments and Overidentification Tests

When multiple instruments are available, we can test the overidentifying restrictions.

```python
def multiple_instruments_analysis(df, instruments, treatment_col, outcome_col, confounders=None):
    """
    Analysis with multiple instruments and overidentification tests
    
    Multiple instruments allow:
    - Stronger first stage
    - Overidentification tests
    - Robustness checks
    """
    
    print(f"=== MULTIPLE INSTRUMENTS ANALYSIS ===")
    print(f"Instruments: {instruments}")
    print(f"Number of instruments: {len(instruments)}")
    
    if len(instruments) < 2:
        print("Need at least 2 instruments for overidentification tests")
        return None
    
    # Prepare data
    all_vars = instruments + [treatment_col, outcome_col]
    if confounders:
        all_vars.extend(confounders)
    
    df_multi = df.dropna(subset=all_vars)
    
    # 1. Joint first stage with all instruments
    print(f"\n=== JOINT FIRST STAGE ===")
    
    if confounders:
        controls = " + ".join(confounders)
        instruments_str = " + ".join(instruments)
        first_stage_formula = f'{treatment_col} ~ {instruments_str} + {controls}'
    else:
        instruments_str = " + ".join(instruments)
        first_stage_formula = f'{treatment_col} ~ {instruments_str}'
    
    joint_first_stage = smf.ols(first_stage_formula, data=df_multi).fit()
    
    print(f"Joint F-statistic: {joint_first_stage.fvalue:.2f}")
    print(f"R-squared: {joint_first_stage.rsquared:.3f}")
    
    # Test joint significance of instruments
    # F-test for instruments jointly equal to zero
    restricted_formula = first_stage_formula.replace(f' + {instruments_str}', '')
    restricted_model = smf.ols(restricted_formula, data=df_multi).fit()
    
    # F-test calculation
    rss_restricted = restricted_model.ssr
    rss_unrestricted = joint_first_stage.ssr
    df_num = len(instruments)
    df_denom = len(df_multi) - joint_first_stage.df_model - 1
    
    f_joint = ((rss_restricted - rss_unrestricted) / df_num) / (rss_unrestricted / df_denom)
    f_pvalue = 1 - stats.f.cdf(f_joint, df_num, df_denom)
    
    print(f"Joint instrument F-test: {f_joint:.2f} (p = {f_pvalue:.4f})")
    
    if f_joint > 10:
        print("✅ Strong joint instrument strength")
    elif f_joint > 5:
        print("⚠️  Moderate joint instrument strength")
    else:
        print("❌ Weak instruments jointly")
    
    # 2. Individual instrument strength
    print(f"\n=== INDIVIDUAL INSTRUMENT STRENGTH ===")
    
    individual_results = {}
    for instrument in instruments:
        if confounders:
            single_formula = f'{treatment_col} ~ {instrument} + {controls}'
        else:
            single_formula = f'{treatment_col} ~ {instrument}'
        
        single_model = smf.ols(single_formula, data=df_multi).fit()
        single_f = single_model.fvalue
        single_coef = single_model.params[instrument]
        single_pvalue = single_model.pvalues[instrument]
        
        individual_results[instrument] = {
            'f_stat': single_f,
            'coefficient': single_coef,
            'p_value': single_pvalue
        }
        
        strength = "Strong" if single_f > 10 else "Moderate" if single_f > 5 else "Weak"
        print(f"   {instrument}: F = {single_f:.2f}, {strength}")
    
    # 3. Two-Stage Least Squares with multiple instruments
    print(f"\n=== 2SLS WITH MULTIPLE INSTRUMENTS ===")
    
    from statsmodels.sandbox.regression.gmm import IV2SLS
    
    # Prepare variables
    y = df_multi[outcome_col].values
    X_endog = df_multi[treatment_col].values.reshape(-1, 1)
    X_instruments = df_multi[instruments].values
    
    if confounders:
        X_exog = df_multi[confounders].values
        X_exog = np.column_stack([np.ones(len(X_exog)), X_exog])
    else:
        X_exog = np.ones((len(y), 1))
    
    # IV estimation
    iv_multi_model = IV2SLS(y, X_exog, X_endog, X_instruments).fit()
    
    iv_effect = iv_multi_model.params[-1]
    iv_se = iv_multi_model.bse[-1]
    
    print(f"IV effect: {iv_effect:.3f}")
    print(f"Standard error: {iv_se:.3f}")
    
    # 4. Overidentification test (Hansen J-test)
    print(f"\n=== OVERIDENTIFICATION TEST ===")
    
    if len(instruments) > 1:  # Need overidentification for the test
        # Simplified J-test calculation
        # In practice, use specialized econometric software
        
        # Get second-stage residuals
        predicted_treatment = joint_first_stage.fittedvalues
        df_stage2 = df_multi.copy()
        df_stage2['predicted_treatment'] = predicted_treatment
        
        if confounders:
            stage2_formula = f'{outcome_col} ~ predicted_treatment + {controls}'
        else:
            stage2_formula = f'{outcome_col} ~ predicted_treatment'
        
        stage2_model = smf.ols(stage2_formula, data=df_stage2).fit()
        residuals = stage2_model.resid
        
        # Project residuals on instruments (simplified)
        if confounders:
            aux_formula = f'residuals ~ {instruments_str} + {controls}'
        else:
            aux_formula = f'residuals ~ {instruments_str}'
        
        df_aux = df_multi.copy()
        df_aux['residuals'] = residuals
        aux_model = smf.ols(aux_formula, data=df_aux).fit()
        
        # J-statistic approximation
        j_stat = len(df_multi) * aux_model.rsquared
        degrees_freedom = len(instruments) - 1  # Number of overidentifying restrictions
        j_pvalue = 1 - stats.chi2.cdf(j_stat, degrees_freedom)
        
        print(f"Hansen J-statistic: {j_stat:.2f}")
        print(f"Degrees of freedom: {degrees_freedom}")
        print(f"P-value: {j_pvalue:.4f}")
        
        if j_pvalue > 0.05:
            print("✅ PASS: Overidentifying restrictions not rejected")
            print("   Instruments appear valid")
        else:
            print("❌ FAIL: Overidentifying restrictions rejected")
            print("   Some instruments may be invalid")
            print("   Consider:")
            print("   - Examining each instrument individually")
            print("   - Testing exclusion restrictions")
            print("   - Alternative instrument sets")
    else:
        print("Need multiple instruments for overidentification test")
        j_pvalue = None
    
    return {
        'joint_f_statistic': f_joint,
        'individual_results': individual_results,
        'iv_effect': iv_effect,
        'iv_se': iv_se,
        'overid_test_pvalue': j_pvalue,
        'model': iv_multi_model
    }

# Example with multiple instruments
# instruments = ['manager_supports_ai', 'team_assignment_random', 'onboarding_period']
# multi_iv_results = multiple_instruments_analysis(
#     df, instruments, 'copilot_usage', 'tickets_per_week', confounders
# )
```

---

## Business Application Examples

### Common IV Applications in Business Analytics

```python
def iv_business_examples():
    """
    Common instrumental variable applications in business settings
    """
    
    print("=== INSTRUMENTAL VARIABLES: BUSINESS APPLICATIONS ===")
    
    examples = [
        {
            'context': 'Technology Adoption',
            'treatment': 'AI tool usage',
            'outcome': 'Productivity metrics',
            'instrument': 'Random manager assignment (some managers encourage AI)',
            'logic': 'Manager attitude affects adoption but not directly productivity'
        },
        {
            'context': 'Training Programs',
            'treatment': 'Training completion',
            'outcome': 'Performance scores',
            'instrument': 'Training slot availability due to scheduling',
            'logic': 'Random scheduling affects training access, not direct performance'
        },
        {
            'context': 'Remote Work',
            'treatment': 'Work-from-home days',
            'outcome': 'Job satisfaction',
            'instrument': 'Distance from office',
            'logic': 'Distance affects WFH choice but not direct job satisfaction'
        },
        {
            'context': 'Software Features',
            'treatment': 'Feature usage',
            'outcome': 'User engagement',
            'instrument': 'Gradual rollout timing',
            'logic': 'Rollout timing affects access but not underlying engagement'
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['context'].upper()}")
        print(f"   Treatment: {example['treatment']}")
        print(f"   Outcome: {example['outcome']}")
        print(f"   Instrument: {example['instrument']}")
        print(f"   Logic: {example['logic']}")
    
    print(f"\n=== KEY SUCCESS FACTORS ===")
    print("1. Strong relevance: Instrument strongly predicts treatment")
    print("2. Exclusion restriction: No direct effect on outcome")
    print("3. Independence: Instrument assignment is quasi-random")
    print("4. Business logic: Mechanism makes intuitive sense")

iv_business_examples()
```

### Interpreting IV Results for Business Decisions

```python
def interpret_iv_for_business(iv_results, treatment_description, outcome_description):
    """
    Translate IV results into business language and recommendations
    """
    
    iv_effect = iv_results['iv_effect']
    iv_se = iv_results['iv_se']
    ci = iv_results['ci']
    ols_effect = iv_results['ols_effect']
    
    print(f"=== BUSINESS INTERPRETATION: IV RESULTS ===")
    
    # 1. Main finding
    print(f"\n📊 CAUSAL EFFECT ESTIMATE:")
    print(f"Treatment: {treatment_description}")
    print(f"Outcome: {outcome_description}")
    
    if iv_effect > 0:
        direction = "increases"
        change_type = "improvement"
    else:
        direction = "decreases"
        change_type = "reduction"
        iv_effect = abs(iv_effect)
    
    print(f"\nCausal Effect: {treatment_description} {direction} {outcome_description}")
    print(f"by {iv_effect:.2f} units on average")
    
    # 2. Confidence and uncertainty
    print(f"\n🎯 CONFIDENCE INTERVAL:")
    if ci[0] > 0 or ci[1] < 0:
        print(f"95% confident the true effect is between {ci[0]:.2f} and {ci[1]:.2f}")
        print("✅ Effect is statistically distinguishable from zero")
    else:
        print(f"Effect could range from {ci[0]:.2f} to {ci[1]:.2f}")
        print("⚠️  Cannot rule out zero effect")
    
    # 3. Comparison with naive analysis
    print(f"\n🔍 ENDOGENEITY CORRECTION:")
    print(f"Naive correlation analysis: {ols_effect:.2f}")
    print(f"Causal effect (IV): {iv_effect:.2f}")
    
    bias_magnitude = abs(iv_effect - ols_effect)
    if bias_magnitude > 0.1:
        print(f"⚠️  Large bias correction: {bias_magnitude:.2f}")
        print("   Simple correlation would be misleading")
        print("   IV analysis reveals true causal impact")
    else:
        print("✅ Minimal bias detected")
        print("   Correlation close to causal effect")
    
    # 4. Business implications
    print(f"\n💼 BUSINESS IMPLICATIONS:")
    
    if ci[0] > 0 or ci[1] < 0:  # Significant effect
        if iv_effect > 0.2:  # Substantial effect
            print("✅ STRONG BUSINESS CASE:")
            print(f"   - Clear causal evidence of {change_type}")
            print(f"   - Effect size: {iv_effect:.2f} units per treatment")
            print("   - Recommendation: Scale intervention")
        else:
            print("⚠️  MODEST BUSINESS CASE:")
            print(f"   - Statistically significant but small effect")
            print("   - Consider cost-benefit analysis")
            print("   - May warrant targeted implementation")
    else:
        print("❌ INSUFFICIENT EVIDENCE:")
        print("   - Cannot establish causal effect")
        print("   - Do not scale based on current evidence")
        print("   - Consider longer study or alternative approaches")
    
    # 5. Methodological notes
    print(f"\n🔬 METHODOLOGICAL NOTES:")
    print("✅ Advantages of IV analysis:")
    print("   - Controls for unmeasured confounders")
    print("   - Identifies causal (not just correlational) effects")
    print("   - Robust to selection bias")
    
    print("\n⚠️  Limitations to consider:")
    print("   - Larger uncertainty than simple correlation")
    print("   - Relies on instrument validity assumptions")
    print("   - Effect may be local to specific population")
    
    return {
        'causal_effect': iv_effect,
        'direction': direction,
        'significant': ci[0] > 0 or ci[1] < 0,
        'bias_corrected': bias_magnitude,
        'recommendation': 'scale' if (ci[0] > 0 or ci[1] < 0) and iv_effect > 0.2 else 'evaluate'
    }

# Interpret results for business
if 'iv_results' in locals():
    business_interpretation = interpret_iv_for_business(
        iv_results,
        'GitHub Copilot adoption',
        'weekly ticket completion'
    )
```

---

## Troubleshooting Common IV Problems

### Diagnostic Framework

```python
def iv_troubleshooting_guide(validity_results, weak_iv_diagnostics, iv_results):
    """
    Comprehensive troubleshooting guide for IV problems
    """
    
    print("=== INSTRUMENTAL VARIABLES TROUBLESHOOTING ===")
    
    issues = []
    solutions = []
    
    # 1. Weak instrument problems
    if weak_iv_diagnostics['strength_category'] == 'weak':
        issues.append("Weak instrument (F < 10)")
        solutions.extend([
            "Find stronger instruments with more variation",
            "Combine multiple related instruments",
            "Use weak-IV robust inference methods",
            "Consider Anderson-Rubin confidence intervals"
        ])
    
    # 2. Failed relevance
    if not validity_results['relevance_pass']:
        issues.append("Instrument fails relevance condition")
        solutions.extend([
            "Check instrument measurement quality",
            "Verify sufficient variation in instrument",
            "Consider nonlinear relationships",
            "Transform or interact instrument variables"
        ])
    
    # 3. Balance concerns
    if not validity_results['balance_pass']:
        issues.append("Instrument correlated with confounders")
        solutions.extend([
            "Add more control variables",
            "Consider instrument × confounder interactions",
            "Test robustness to different control sets",
            "Examine instrument assignment mechanism"
        ])
    
    # 4. Large standard errors
    iv_se = iv_results['iv_se']
    ols_se = iv_results.get('ols_se', iv_se)
    if iv_se / ols_se > 3:
        issues.append("Very large standard errors")
        solutions.extend([
            "Increase sample size if possible",
            "Find stronger instruments",
            "Combine multiple studies (meta-analysis)",
            "Consider alternative identification strategies"
        ])
    
    # 5. Implausible effect sizes
    iv_effect = abs(iv_results['iv_effect'])
    ols_effect = abs(iv_results['ols_effect'])
    if iv_effect > 3 * ols_effect:
        issues.append("Implausibly large IV effect")
        solutions.extend([
            "Check for outliers in instrument",
            "Verify exclusion restriction",
            "Consider measurement error in treatment",
            "Examine heterogeneous treatment effects"
        ])
    
    # Report findings
    print(f"\n🔍 ISSUES DETECTED: {len(issues)}")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   No major issues detected")
    
    print(f"\n💡 SOLUTIONS TO CONSIDER:")
    if solutions:
        unique_solutions = list(set(solutions))
        for i, solution in enumerate(unique_solutions, 1):
            print(f"   {i}. {solution}")
    else:
        print("   Analysis appears robust")
    
    # Overall recommendation
    print(f"\n📋 OVERALL RECOMMENDATION:")
    
    severity_score = len(issues)
    
    if severity_score == 0:
        print("✅ PROCEED: IV analysis appears valid")
        print("   Report results with confidence")
        
    elif severity_score <= 2:
        print("⚠️  PROCEED WITH CAUTION:")
        print("   Address identified issues")
        print("   Report robustness checks")
        print("   Consider sensitivity analysis")
        
    else:
        print("❌ DO NOT PROCEED:")
        print("   Too many validity concerns")
        print("   Find alternative instruments or methods")
        print("   Consider other identification strategies")
    
    return {
        'issues_count': len(issues),
        'severity': 'low' if severity_score <= 1 else 'medium' if severity_score <= 3 else 'high',
        'recommendation': 'proceed' if severity_score <= 1 else 'caution' if severity_score <= 3 else 'stop'
    }

# Generate troubleshooting report
if all(var in locals() for var in ['validity_results', 'weak_iv_diagnostics', 'iv_results']):
    troubleshooting_report = iv_troubleshooting_guide(
        validity_results, weak_iv_diagnostics, iv_results
    )
```

**Navigation:**
- [← Back to Difference-in-Differences]({{ site.baseurl }}/causal-inference-did/)
- [Next: Doubly Robust Methods →]({{ site.baseurl }}/causal-inference-doubly-robust/)

---

*Instrumental Variables provide powerful tools for causal inference when unmeasured confounding is suspected. Always validate instrument strength and exclusion restrictions before interpreting results.*
