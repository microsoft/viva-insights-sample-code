---
layout: default
title: "Causal Inference: Validation & Testing"
permalink: /causal-inference-validation/
---

# Validation & Testing Framework

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

This comprehensive validation framework helps ensure your causal inference results are robust, reliable, and ready for business decision-making. We cover assumption testing, robustness checks, sensitivity analysis, and comprehensive reporting standards.

**Navigation:**
- [← Back to Doubly Robust Methods]({{ site.baseurl }}/causal-inference-doubly-robust/)
- [Back to Technical Overview]({{ site.baseurl }}/causal-inference-technical/)

---

## Assumption Testing Framework

Every causal inference method relies on key assumptions. Systematic testing helps validate these assumptions and assess result reliability.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings

def comprehensive_assumption_testing(df, treatment_col, outcome_col, confounders, method='regression'):
    """
    Comprehensive assumption testing framework for causal inference
    
    Tests vary by method but generally include:
    - Overlap/common support
    - No unmeasured confounding (indirect tests)
    - Correct functional form
    - Stable unit treatment value assumption (SUTVA)
    """
    
    print(f"=== COMPREHENSIVE ASSUMPTION TESTING ===")
    print(f"Method: {method}")
    print(f"Sample size: {len(df)}")
    
    test_results = {}
    
    # 1. OVERLAP AND COMMON SUPPORT
    print(f"\n1. OVERLAP AND COMMON SUPPORT TESTING")
    
    if method in ['propensity_score', 'doubly_robust']:
        # Estimate propensity scores for overlap testing
        from sklearn.linear_model import LogisticRegression
        
        X = df[confounders]
        treatment = df[treatment_col]
        
        ps_model = LogisticRegression(random_state=42, max_iter=1000)
        ps_model.fit(X, treatment)
        ps_scores = ps_model.predict_proba(X)[:, 1]
        
        # Overlap diagnostics
        treated_ps = ps_scores[treatment == 1]
        control_ps = ps_scores[treatment == 0]
        
        overlap_min = max(treated_ps.min(), control_ps.min())
        overlap_max = min(treated_ps.max(), control_ps.max())
        
        in_overlap = (ps_scores >= overlap_min) & (ps_scores <= overlap_max)
        overlap_rate = in_overlap.mean()
        
        extreme_ps = ((ps_scores < 0.1) | (ps_scores > 0.9)).sum()
        
        print(f"   Overlap rate: {overlap_rate:.1%}")
        print(f"   Extreme PS (< 0.1 or > 0.9): {extreme_ps}")
        
        overlap_pass = overlap_rate > 0.8 and extreme_ps < len(df) * 0.05
        test_results['overlap'] = {
            'pass': overlap_pass,
            'overlap_rate': overlap_rate,
            'extreme_count': extreme_ps
        }
        
        if overlap_pass:
            print("   ✅ PASS: Good overlap")
        else:
            print("   ❌ FAIL: Poor overlap")
    
    else:
        print("   Overlap testing not applicable for this method")
        test_results['overlap'] = {'pass': True, 'note': 'Not applicable'}
    
    # 2. BALANCE TESTING (for covariates)
    print(f"\n2. COVARIATE BALANCE TESTING")
    
    balance_results = []
    
    for var in confounders:
        if var not in df.columns:
            continue
            
        treated_values = df[df[treatment_col] == 1][var]
        control_values = df[df[treatment_col] == 0][var]
        
        if treated_values.dtype in ['int64', 'float64']:
            # Continuous variables - standardized difference
            pooled_std = np.sqrt((treated_values.var() + control_values.var()) / 2)
            std_diff = (treated_values.mean() - control_values.mean()) / pooled_std if pooled_std > 0 else 0
            
            balance_test = abs(std_diff) < 0.25  # Common threshold
            balance_results.append({
                'variable': var,
                'std_diff': std_diff,
                'pass': balance_test
            })
            
            status = "✅" if balance_test else "❌"
            print(f"   {status} {var}: std diff = {std_diff:.3f}")
            
        else:
            # Categorical variables - chi-square test
            contingency_table = pd.crosstab(df[var], df[treatment_col])
            chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
            
            balance_test = p_value > 0.05
            balance_results.append({
                'variable': var,
                'chi2_p': p_value,
                'pass': balance_test
            })
            
            status = "✅" if balance_test else "❌"
            print(f"   {status} {var}: chi2 p = {p_value:.3f}")
    
    balance_pass_rate = np.mean([r['pass'] for r in balance_results])
    test_results['balance'] = {
        'pass': balance_pass_rate > 0.8,
        'pass_rate': balance_pass_rate,
        'details': balance_results
    }
    
    if balance_pass_rate > 0.8:
        print(f"   ✅ OVERALL: {balance_pass_rate:.1%} of variables balanced")
    else:
        print(f"   ❌ OVERALL: Only {balance_pass_rate:.1%} of variables balanced")
    
    # 3. FUNCTIONAL FORM TESTING
    print(f"\n3. FUNCTIONAL FORM TESTING")
    
    # Test for nonlinear relationships using RESET test approach
    X = df[confounders]
    y = df[outcome_col]
    
    # Linear model
    from sklearn.linear_model import LinearRegression
    linear_model = LinearRegression()
    linear_score = cross_val_score(linear_model, X, y, cv=5, scoring='r2').mean()
    
    # Add polynomial terms and test improvement
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    poly_score = cross_val_score(linear_model, X_poly, y, cv=5, scoring='r2').mean()
    
    # Random forest as flexible benchmark
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_score = cross_val_score(rf_model, X, y, cv=5, scoring='r2').mean()
    
    print(f"   Linear R²: {linear_score:.3f}")
    print(f"   Polynomial R²: {poly_score:.3f}")
    print(f"   Random Forest R²: {rf_score:.3f}")
    
    # If flexible models substantially outperform linear, suggest nonlinearity
    linear_adequate = (rf_score - linear_score) < 0.05
    
    test_results['functional_form'] = {
        'pass': linear_adequate,
        'linear_r2': linear_score,
        'rf_r2': rf_score,
        'improvement': rf_score - linear_score
    }
    
    if linear_adequate:
        print("   ✅ PASS: Linear form appears adequate")
    else:
        print("   ⚠️  CONCERN: Nonlinear relationships detected")
        print("   Consider: polynomial terms, interactions, or flexible methods")
    
    # 4. PLACEBO TESTING (where applicable)
    print(f"\n4. PLACEBO TESTING")
    
    if method in ['difference_in_differences', 'event_study']:
        # Test with fake treatment date
        if 'date' in df.columns:
            # Create fake treatment date earlier in time
            fake_treatment_date = df['date'].min() + (df['date'].max() - df['date'].min()) / 3
            
            pre_treatment_data = df[df['date'] < df[df['post_treatment'] == 1]['date'].min()]
            
            if len(pre_treatment_data) > 0:
                pre_treatment_data = pre_treatment_data.copy()
                pre_treatment_data['fake_post'] = (pre_treatment_data['date'] >= fake_treatment_date).astype(int)
                pre_treatment_data['fake_interaction'] = (pre_treatment_data[treatment_col] * 
                                                        pre_treatment_data['fake_post'])
                
                # Run placebo regression
                import statsmodels.formula.api as smf
                placebo_formula = f'{outcome_col} ~ {treatment_col} + fake_post + fake_interaction'
                
                try:
                    placebo_model = smf.ols(placebo_formula, data=pre_treatment_data).fit()
                    placebo_effect = placebo_model.params.get('fake_interaction', 0)
                    placebo_p = placebo_model.pvalues.get('fake_interaction', 1)
                    
                    placebo_pass = placebo_p > 0.05
                    
                    print(f"   Placebo effect: {placebo_effect:.3f} (p = {placebo_p:.3f})")
                    
                    test_results['placebo'] = {
                        'pass': placebo_pass,
                        'effect': placebo_effect,
                        'p_value': placebo_p
                    }
                    
                    if placebo_pass:
                        print("   ✅ PASS: No placebo effect")
                    else:
                        print("   ❌ FAIL: Significant placebo effect")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not run placebo test: {e}")
                    test_results['placebo'] = {'pass': None, 'error': str(e)}
            else:
                print("   Insufficient pre-treatment data for placebo test")
                test_results['placebo'] = {'pass': None, 'note': 'Insufficient data'}
        else:
            print("   No date variable available for placebo test")
            test_results['placebo'] = {'pass': None, 'note': 'No date variable'}
    else:
        print("   Placebo testing not applicable for this method")
        test_results['placebo'] = {'pass': True, 'note': 'Not applicable'}
    
    # 5. SUTVA TESTING (Stable Unit Treatment Value Assumption)
    print(f"\n5. SUTVA (No Interference) TESTING")
    
    # Indirect test: look for clustering in treatment assignment
    if 'team_id' in df.columns or 'department' in df.columns:
        cluster_var = 'team_id' if 'team_id' in df.columns else 'department'
        
        # Calculate intra-cluster correlation of treatment
        cluster_treatment = df.groupby(cluster_var)[treatment_col].agg(['mean', 'count'])
        
        # Variance between vs within clusters
        overall_treatment_rate = df[treatment_col].mean()
        between_cluster_var = ((cluster_treatment['mean'] - overall_treatment_rate) ** 2 * 
                              cluster_treatment['count']).sum() / df.shape[0]
        within_cluster_var = overall_treatment_rate * (1 - overall_treatment_rate)
        
        icc = between_cluster_var / (between_cluster_var + within_cluster_var)
        
        print(f"   Intracluster correlation: {icc:.3f}")
        
        sutva_concern = icc > 0.1  # Threshold for concern
        
        test_results['sutva'] = {
            'pass': not sutva_concern,
            'icc': icc,
            'concern_level': 'high' if icc > 0.2 else 'moderate' if icc > 0.1 else 'low'
        }
        
        if sutva_concern:
            print("   ⚠️  CONCERN: High clustering suggests potential interference")
            print("   Consider: cluster-robust standard errors or network analysis")
        else:
            print("   ✅ PASS: Low clustering, SUTVA likely satisfied")
    else:
        print("   No cluster variables available for SUTVA testing")
        test_results['sutva'] = {'pass': True, 'note': 'No cluster data'}
    
    # OVERALL ASSESSMENT
    print(f"\n=== OVERALL ASSUMPTION TESTING SUMMARY ===")
    
    passed_tests = sum([1 for test in test_results.values() 
                       if test.get('pass') is True])
    total_tests = sum([1 for test in test_results.values() 
                      if test.get('pass') is not None])
    
    if total_tests > 0:
        pass_rate = passed_tests / total_tests
        print(f"Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1%})")
        
        if pass_rate >= 0.8:
            overall_validity = "HIGH"
            print("✅ STRONG: Assumptions appear well-satisfied")
        elif pass_rate >= 0.6:
            overall_validity = "MODERATE"
            print("⚠️  MODERATE: Some assumptions concerns")
        else:
            overall_validity = "LOW"
            print("❌ WEAK: Multiple assumption violations")
    else:
        overall_validity = "UNKNOWN"
        print("⚠️  UNKNOWN: Could not assess assumptions")
    
    test_results['overall'] = {
        'validity': overall_validity,
        'pass_rate': pass_rate if total_tests > 0 else None
    }
    
    return test_results

# Example usage
confounders = ['tenure_months', 'job_level', 'team_size', 'manager_span']
assumption_tests = comprehensive_assumption_testing(
    df, 'copilot_usage', 'tickets_per_week', confounders, method='regression'
)
```

---

## Robustness Testing Framework

Robustness tests examine how sensitive results are to different specifications and assumptions.

```python
def comprehensive_robustness_testing(df, treatment_col, outcome_col, confounders):
    """
    Comprehensive robustness testing across multiple dimensions
    """
    
    print("=== COMPREHENSIVE ROBUSTNESS TESTING ===")
    
    robustness_results = {}
    
    # 1. MODEL SPECIFICATION ROBUSTNESS
    print("\n1. MODEL SPECIFICATION ROBUSTNESS")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import statsmodels.formula.api as smf
    
    specifications = [
        {'name': 'Linear', 'controls': confounders},
        {'name': 'Linear + Interactions', 'controls': confounders + ['tenure_months*job_level']},
        {'name': 'Polynomial', 'type': 'poly'},
        {'name': 'Random Forest', 'type': 'rf'}
    ]
    
    spec_results = {}
    
    for spec in specifications:
        try:
            if spec.get('type') == 'rf':
                # Random Forest
                X = df[confounders + [treatment_col]]
                y = df[outcome_col]
                
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X, y)
                
                # Feature importance for treatment
                feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
                treatment_importance = feature_importance.get(treatment_col, 0)
                
                # Approximate effect using partial dependence
                treated_X = X.copy()
                treated_X[treatment_col] = 1
                control_X = X.copy()
                control_X[treatment_col] = 0
                
                effect_estimate = rf_model.predict(treated_X).mean() - rf_model.predict(control_X).mean()
                
                spec_results[spec['name']] = {
                    'estimate': effect_estimate,
                    'method': 'random_forest',
                    'treatment_importance': treatment_importance
                }
                
            elif spec.get('type') == 'poly':
                # Polynomial features
                from sklearn.preprocessing import PolynomialFeatures
                
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_base = df[confounders]
                X_poly = poly.fit_transform(X_base)
                
                # Add treatment
                X_final = np.column_stack([X_poly, df[treatment_col]])
                
                poly_model = LinearRegression()
                poly_model.fit(X_final, df[outcome_col])
                
                effect_estimate = poly_model.coef_[-1]  # Treatment coefficient
                
                spec_results[spec['name']] = {
                    'estimate': effect_estimate,
                    'method': 'polynomial'
                }
                
            else:
                # Linear regression
                controls_str = ' + '.join(spec['controls'])
                formula = f"{outcome_col} ~ {treatment_col} + {controls_str}"
                
                model = smf.ols(formula, data=df).fit()
                effect_estimate = model.params[treatment_col]
                se = model.bse[treatment_col]
                
                spec_results[spec['name']] = {
                    'estimate': effect_estimate,
                    'se': se,
                    'method': 'linear'
                }
                
            print(f"   {spec['name']}: {effect_estimate:.3f}")
            
        except Exception as e:
            print(f"   {spec['name']}: Failed ({e})")
            spec_results[spec['name']] = {'estimate': np.nan, 'error': str(e)}
    
    # Assess specification sensitivity
    valid_estimates = [r['estimate'] for r in spec_results.values() 
                      if not np.isnan(r['estimate'])]
    
    if len(valid_estimates) > 1:
        estimate_range = max(valid_estimates) - min(valid_estimates)
        estimate_mean = np.mean(valid_estimates)
        
        relative_range = estimate_range / abs(estimate_mean) if estimate_mean != 0 else float('inf')
        
        print(f"\n   Range: {min(valid_estimates):.3f} to {max(valid_estimates):.3f}")
        print(f"   Relative range: {relative_range:.1%}")
        
        if relative_range < 0.2:
            print("   ✅ ROBUST: Results consistent across specifications")
            spec_robustness = "HIGH"
        elif relative_range < 0.5:
            print("   ⚠️  MODERATE: Some sensitivity to specification")
            spec_robustness = "MODERATE"
        else:
            print("   ❌ SENSITIVE: High sensitivity to specification")
            spec_robustness = "LOW"
    else:
        spec_robustness = "UNKNOWN"
    
    robustness_results['specification'] = {
        'robustness': spec_robustness,
        'results': spec_results,
        'relative_range': relative_range if len(valid_estimates) > 1 else None
    }
    
    # 2. SAMPLE COMPOSITION ROBUSTNESS
    print("\n2. SAMPLE COMPOSITION ROBUSTNESS")
    
    sample_tests = [
        {'name': 'Exclude 5% outliers', 'type': 'outlier_exclusion', 'pct': 0.05},
        {'name': 'Exclude 10% outliers', 'type': 'outlier_exclusion', 'pct': 0.10},
        {'name': 'Bootstrap resample', 'type': 'bootstrap', 'n_samples': 100}
    ]
    
    sample_results = {}
    baseline_formula = f"{outcome_col} ~ {treatment_col} + {' + '.join(confounders)}"
    baseline_model = smf.ols(baseline_formula, data=df).fit()
    baseline_estimate = baseline_model.params[treatment_col]
    
    for test in sample_tests:
        try:
            if test['type'] == 'outlier_exclusion':
                # Remove outliers based on outcome variable
                Q1 = df[outcome_col].quantile(test['pct']/2)
                Q3 = df[outcome_col].quantile(1 - test['pct']/2)
                
                df_trimmed = df[(df[outcome_col] >= Q1) & (df[outcome_col] <= Q3)]
                
                model = smf.ols(baseline_formula, data=df_trimmed).fit()
                effect_estimate = model.params[treatment_col]
                
                sample_results[test['name']] = {
                    'estimate': effect_estimate,
                    'n_obs': len(df_trimmed)
                }
                
            elif test['type'] == 'bootstrap':
                # Bootstrap resampling
                bootstrap_estimates = []
                
                for _ in range(test['n_samples']):
                    df_boot = df.sample(n=len(df), replace=True, random_state=np.random.randint(10000))
                    
                    try:
                        boot_model = smf.ols(baseline_formula, data=df_boot).fit()
                        bootstrap_estimates.append(boot_model.params[treatment_col])
                    except:
                        continue
                
                if bootstrap_estimates:
                    effect_estimate = np.mean(bootstrap_estimates)
                    effect_std = np.std(bootstrap_estimates)
                    
                    sample_results[test['name']] = {
                        'estimate': effect_estimate,
                        'std': effect_std,
                        'ci': [np.percentile(bootstrap_estimates, 2.5), 
                              np.percentile(bootstrap_estimates, 97.5)]
                    }
                else:
                    sample_results[test['name']] = {'estimate': np.nan, 'error': 'Bootstrap failed'}
            
            print(f"   {test['name']}: {effect_estimate:.3f}")
            
        except Exception as e:
            print(f"   {test['name']}: Failed ({e})")
            sample_results[test['name']] = {'estimate': np.nan, 'error': str(e)}
    
    # Assess sample sensitivity
    sample_estimates = [baseline_estimate] + [r['estimate'] for r in sample_results.values() 
                                            if not np.isnan(r['estimate'])]
    
    if len(sample_estimates) > 1:
        sample_range = max(sample_estimates) - min(sample_estimates)
        sample_mean = np.mean(sample_estimates)
        sample_relative_range = sample_range / abs(sample_mean) if sample_mean != 0 else float('inf')
        
        if sample_relative_range < 0.15:
            sample_robustness = "HIGH"
            print("   ✅ ROBUST: Results stable across samples")
        elif sample_relative_range < 0.3:
            sample_robustness = "MODERATE"
            print("   ⚠️  MODERATE: Some sample sensitivity")
        else:
            sample_robustness = "LOW"
            print("   ❌ SENSITIVE: High sample sensitivity")
    else:
        sample_robustness = "UNKNOWN"
    
    robustness_results['sample'] = {
        'robustness': sample_robustness,
        'results': sample_results,
        'baseline': baseline_estimate
    }
    
    # 3. SUBGROUP ROBUSTNESS
    print("\n3. SUBGROUP ROBUSTNESS")
    
    subgroup_results = {}
    
    # Test on different subgroups if relevant variables exist
    subgroup_vars = ['job_level', 'tenure_months', 'team_size']
    
    for var in subgroup_vars:
        if var in df.columns:
            try:
                if df[var].dtype in ['object', 'category']:
                    # Categorical variable
                    for category in df[var].unique()[:3]:  # Limit to top 3 categories
                        subgroup_df = df[df[var] == category]
                        
                        if len(subgroup_df) > 30:  # Minimum sample size
                            model = smf.ols(baseline_formula, data=subgroup_df).fit()
                            effect = model.params[treatment_col]
                            
                            subgroup_results[f"{var}_{category}"] = {
                                'estimate': effect,
                                'n_obs': len(subgroup_df)
                            }
                            
                else:
                    # Continuous variable - split at median
                    median_val = df[var].median()
                    
                    high_df = df[df[var] >= median_val]
                    low_df = df[df[var] < median_val]
                    
                    for name, subgroup_df in [('high', high_df), ('low', low_df)]:
                        if len(subgroup_df) > 30:
                            model = smf.ols(baseline_formula, data=subgroup_df).fit()
                            effect = model.params[treatment_col]
                            
                            subgroup_results[f"{var}_{name}"] = {
                                'estimate': effect,
                                'n_obs': len(subgroup_df)
                            }
                            
            except Exception as e:
                print(f"   {var}: Failed ({e})")
    
    # Display subgroup results
    if subgroup_results:
        for subgroup, result in subgroup_results.items():
            print(f"   {subgroup}: {result['estimate']:.3f} (n={result['n_obs']})")
        
        # Check for heterogeneity
        subgroup_estimates = [r['estimate'] for r in subgroup_results.values()]
        subgroup_range = max(subgroup_estimates) - min(subgroup_estimates)
        
        if subgroup_range > abs(baseline_estimate):
            print("   ⚠️  HETEROGENEITY: Large variation across subgroups")
            subgroup_robustness = "LOW"
        else:
            print("   ✅ CONSISTENT: Similar effects across subgroups")
            subgroup_robustness = "HIGH"
    else:
        subgroup_robustness = "UNKNOWN"
        print("   No subgroup analysis possible")
    
    robustness_results['subgroup'] = {
        'robustness': subgroup_robustness,
        'results': subgroup_results
    }
    
    # OVERALL ROBUSTNESS ASSESSMENT
    print(f"\n=== OVERALL ROBUSTNESS ASSESSMENT ===")
    
    robustness_scores = [
        robustness_results['specification']['robustness'],
        robustness_results['sample']['robustness'],
        robustness_results['subgroup']['robustness']
    ]
    
    high_count = robustness_scores.count('HIGH')
    moderate_count = robustness_scores.count('MODERATE')
    low_count = robustness_scores.count('LOW')
    
    if high_count >= 2:
        overall_robustness = "HIGH"
        print("✅ HIGHLY ROBUST: Results consistent across multiple tests")
    elif high_count + moderate_count >= 2:
        overall_robustness = "MODERATE"
        print("⚠️  MODERATELY ROBUST: Generally consistent with some concerns")
    else:
        overall_robustness = "LOW"
        print("❌ LOW ROBUSTNESS: Results sensitive to specification/sampling")
    
    robustness_results['overall'] = {
        'robustness': overall_robustness,
        'component_scores': {
            'specification': robustness_results['specification']['robustness'],
            'sample': robustness_results['sample']['robustness'],
            'subgroup': robustness_results['subgroup']['robustness']
        }
    }
    
    return robustness_results

# Perform robustness testing
robustness_tests = comprehensive_robustness_testing(
    df, 'copilot_usage', 'tickets_per_week', confounders
)
```

---

## Sensitivity Analysis Framework

Sensitivity analysis examines how results change under different assumptions about unmeasured confounding and model parameters.

```python
def sensitivity_analysis_framework(df, treatment_col, outcome_col, confounders, baseline_estimate):
    """
    Comprehensive sensitivity analysis for unmeasured confounding
    """
    
    print("=== SENSITIVITY ANALYSIS FRAMEWORK ===")
    
    # 1. ROSENBAUM BOUNDS (for matched/observational studies)
    print("\n1. ROSENBAUM BOUNDS ANALYSIS")
    
    def rosenbaum_bounds(gamma_values, baseline_effect, baseline_se):
        """
        Calculate how large unmeasured confounding needs to be to change conclusions
        """
        bounds_results = []
        
        for gamma in gamma_values:
            # Under Rosenbaum model, true odds ratio could be gamma times observed
            # This affects the effective sample size and power
            
            # Simplified approximation - in practice use specialized software
            adjusted_se = baseline_se * np.sqrt(gamma)
            z_stat = baseline_effect / adjusted_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            bounds_results.append({
                'gamma': gamma,
                'adjusted_se': adjusted_se,
                'z_stat': z_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        return bounds_results
    
    # Estimate baseline standard error (simplified)
    import statsmodels.formula.api as smf
    
    formula = f"{outcome_col} ~ {treatment_col} + {' + '.join(confounders)}"
    baseline_model = smf.ols(formula, data=df).fit()
    baseline_se = baseline_model.bse[treatment_col]
    
    gamma_values = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    bounds_results = rosenbaum_bounds(gamma_values, baseline_estimate, baseline_se)
    
    print("   Gamma (confounding strength) | P-value | Significant?")
    print("   " + "-" * 45)
    
    critical_gamma = None
    for result in bounds_results:
        sig_status = "Yes" if result['significant'] else "No"
        print(f"   {result['gamma']:>6.1f}                    | {result['p_value']:>7.3f} | {sig_status:>10}")
        
        if result['significant'] and critical_gamma is None:
            continue
        elif not result['significant'] and critical_gamma is None:
            critical_gamma = result['gamma']
    
    if critical_gamma:
        print(f"\n   Critical Gamma: {critical_gamma}")
        print(f"   Interpretation: Unmeasured confounding would need odds ratio > {critical_gamma}")
        print(f"   to eliminate statistical significance")
        
        if critical_gamma > 2.0:
            print("   ✅ ROBUST: Large confounding needed to change conclusion")
        elif critical_gamma > 1.5:
            print("   ⚠️  MODERATE: Moderate confounding could change conclusion")
        else:
            print("   ❌ SENSITIVE: Small confounding could change conclusion")
    else:
        print("   ✅ ROBUST: Remains significant even with strong confounding")
    
    # 2. E-VALUE ANALYSIS
    print("\n2. E-VALUE ANALYSIS")
    
    def calculate_e_value(effect_estimate, ci_lower=None):
        """
        Calculate E-value: minimum strength of unmeasured confounding
        to explain away the observed effect
        """
        
        # Convert to risk ratio scale (approximation for small effects)
        rr = np.exp(effect_estimate) if abs(effect_estimate) < 1 else 1 + effect_estimate
        
        if rr >= 1:
            e_value = rr + np.sqrt(rr * (rr - 1))
        else:
            rr_inv = 1 / rr
            e_value = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))
        
        return e_value
    
    e_value_point = calculate_e_value(baseline_estimate)
    
    # E-value for confidence interval (if available)
    ci = baseline_model.conf_int().loc[treatment_col]
    ci_lower = ci[0] if baseline_estimate > 0 else ci[1]
    
    if ci_lower * baseline_estimate > 0:  # Same sign
        e_value_ci = calculate_e_value(ci_lower)
    else:
        e_value_ci = 1.0  # CI includes null
    
    print(f"   Point estimate E-value: {e_value_point:.2f}")
    print(f"   Confidence interval E-value: {e_value_ci:.2f}")
    
    print(f"\n   Interpretation:")
    print(f"   - Unmeasured confounder would need to be associated with both")
    print(f"     treatment and outcome by risk ratio of {e_value_point:.1f}")
    print(f"   - To explain away the lower confidence limit: {e_value_ci:.1f}")
    
    if e_value_ci > 2.0:
        print("   ✅ ROBUST: Strong confounding needed to explain results")
    elif e_value_ci > 1.5:
        print("   ⚠️  MODERATE: Moderate confounding could explain results")
    else:
        print("   ❌ SENSITIVE: Weak confounding could explain results")
    
    # 3. SIMULATION-BASED SENSITIVITY
    print("\n3. SIMULATION-BASED SENSITIVITY ANALYSIS")
    
    def simulate_unmeasured_confounder(df, treatment_col, outcome_col, confounders,
                                     u_treatment_corr, u_outcome_effect, n_sims=1000):
        """
        Simulate unmeasured confounder and see how it affects estimates
        """
        
        results = []
        
        for sim in range(n_sims):
            # Generate unmeasured confounder
            n = len(df)
            u = np.random.normal(0, 1, n)
            
            # Create df copy for this simulation
            df_sim = df.copy()
            
            # Add unmeasured confounder effect to treatment (selection bias)
            treatment_probs = stats.logistic.cdf(u_treatment_corr * u)
            df_sim[f'{treatment_col}_biased'] = np.random.binomial(1, treatment_probs)
            
            # Add unmeasured confounder effect to outcome
            df_sim[f'{outcome_col}_biased'] = (df_sim[outcome_col] + 
                                              u_outcome_effect * u + 
                                              np.random.normal(0, 0.1, n))
            
            # Estimate treatment effect ignoring U
            try:
                formula = f"{outcome_col}_biased ~ {treatment_col}_biased + {' + '.join(confounders)}"
                model = smf.ols(formula, data=df_sim).fit()
                biased_estimate = model.params[f'{treatment_col}_biased']
                results.append(biased_estimate)
            except:
                continue
        
        return results
    
    # Test different confounding scenarios
    confounding_scenarios = [
        {'u_treatment': 0.5, 'u_outcome': 0.2, 'name': 'Weak confounding'},
        {'u_treatment': 1.0, 'u_outcome': 0.5, 'name': 'Moderate confounding'},
        {'u_treatment': 2.0, 'u_outcome': 1.0, 'name': 'Strong confounding'}
    ]
    
    simulation_results = {}
    
    for scenario in confounding_scenarios:
        try:
            sim_estimates = simulate_unmeasured_confounder(
                df, treatment_col, outcome_col, confounders,
                scenario['u_treatment'], scenario['u_outcome'], n_sims=100
            )
            
            if sim_estimates:
                mean_bias = np.mean(sim_estimates) - baseline_estimate
                std_bias = np.std(sim_estimates)
                
                simulation_results[scenario['name']] = {
                    'mean_bias': mean_bias,
                    'std_bias': std_bias,
                    'estimates': sim_estimates
                }
                
                print(f"   {scenario['name']}: Mean bias = {mean_bias:.3f} ± {std_bias:.3f}")
                
        except Exception as e:
            print(f"   {scenario['name']}: Simulation failed ({e})")
    
    # OVERALL SENSITIVITY ASSESSMENT
    print(f"\n=== OVERALL SENSITIVITY ASSESSMENT ===")
    
    sensitivity_indicators = []
    
    # Rosenbaum bounds
    if critical_gamma and critical_gamma > 1.5:
        sensitivity_indicators.append("Robust to Rosenbaum bounds")
    else:
        sensitivity_indicators.append("Sensitive to Rosenbaum bounds")
    
    # E-values
    if e_value_ci > 2.0:
        sensitivity_indicators.append("Robust to E-value analysis")
    else:
        sensitivity_indicators.append("Sensitive to E-value analysis")
    
    # Simulation
    if simulation_results:
        max_bias = max([abs(r['mean_bias']) for r in simulation_results.values()])
        if max_bias < abs(baseline_estimate) * 0.5:
            sensitivity_indicators.append("Robust to simulation analysis")
        else:
            sensitivity_indicators.append("Sensitive to simulation analysis")
    
    robust_count = sum([1 for s in sensitivity_indicators if "Robust" in s])
    
    if robust_count >= 2:
        overall_sensitivity = "ROBUST"
        print("✅ ROBUST TO UNMEASURED CONFOUNDING")
        print("   Results unlikely to be explained by unobserved factors")
    elif robust_count >= 1:
        overall_sensitivity = "MODERATE"
        print("⚠️  MODERATE SENSITIVITY TO UNMEASURED CONFOUNDING")
        print("   Results could be affected by unobserved factors")
    else:
        overall_sensitivity = "SENSITIVE"
        print("❌ SENSITIVE TO UNMEASURED CONFOUNDING")
        print("   Results could easily be explained by unobserved factors")
    
    return {
        'overall_sensitivity': overall_sensitivity,
        'rosenbaum_critical_gamma': critical_gamma,
        'e_value_point': e_value_point,
        'e_value_ci': e_value_ci,
        'simulation_results': simulation_results
    }

# Perform sensitivity analysis
sensitivity_results = sensitivity_analysis_framework(
    df, 'copilot_usage', 'tickets_per_week', confounders, 
    baseline_estimate=0.25  # Replace with actual baseline estimate
)
```

---

## Comprehensive Reporting Framework

### Executive Summary Generator

```python
def generate_executive_summary(method_results, assumption_tests, robustness_tests, 
                              sensitivity_results, treatment_name, outcome_name):
    """
    Generate comprehensive executive summary for causal inference analysis
    """
    
    print("=" * 80)
    print("CAUSAL INFERENCE ANALYSIS: EXECUTIVE SUMMARY")
    print("=" * 80)
    
    # HEADLINE FINDING
    effect_estimate = method_results.get('effect', 0)
    effect_se = method_results.get('se', 0)
    effect_ci = method_results.get('ci', [0, 0])
    
    print(f"\n🎯 HEADLINE FINDING")
    print(f"Treatment: {treatment_name}")
    print(f"Outcome: {outcome_name}")
    
    if effect_estimate > 0:
        direction = "increases"
        magnitude = effect_estimate
    else:
        direction = "decreases"
        magnitude = abs(effect_estimate)
    
    print(f"\nResult: {treatment_name} {direction} {outcome_name} by {magnitude:.2f} units")
    
    # Statistical significance
    if effect_ci[0] > 0 or effect_ci[1] < 0:
        significance = "statistically significant"
        significance_icon = "✅"
    else:
        significance = "not statistically significant"
        significance_icon = "⚠️"
    
    print(f"{significance_icon} Statistical significance: {significance}")
    print(f"   95% Confidence Interval: [{effect_ci[0]:.2f}, {effect_ci[1]:.2f}]")
    
    # METHODOLOGICAL QUALITY
    print(f"\n🔬 METHODOLOGICAL QUALITY")
    
    # Assumption testing
    assumption_validity = assumption_tests.get('overall', {}).get('validity', 'UNKNOWN')
    if assumption_validity == 'HIGH':
        assumption_icon = "✅"
        assumption_text = "Strong"
    elif assumption_validity == 'MODERATE':
        assumption_icon = "⚠️"
        assumption_text = "Moderate"
    else:
        assumption_icon = "❌"
        assumption_text = "Weak"
    
    print(f"{assumption_icon} Assumption validity: {assumption_text}")
    
    # Robustness
    overall_robustness = robustness_tests.get('overall', {}).get('robustness', 'UNKNOWN')
    if overall_robustness == 'HIGH':
        robustness_icon = "✅"
        robustness_text = "Highly robust"
    elif overall_robustness == 'MODERATE':
        robustness_icon = "⚠️"
        robustness_text = "Moderately robust"
    else:
        robustness_icon = "❌"
        robustness_text = "Not robust"
    
    print(f"{robustness_icon} Result robustness: {robustness_text}")
    
    # Sensitivity to unmeasured confounding
    sensitivity_level = sensitivity_results.get('overall_sensitivity', 'UNKNOWN')
    if sensitivity_level == 'ROBUST':
        sensitivity_icon = "✅"
        sensitivity_text = "Robust to unmeasured confounding"
    elif sensitivity_level == 'MODERATE':
        sensitivity_icon = "⚠️"
        sensitivity_text = "Moderately sensitive to unmeasured confounding"
    else:
        sensitivity_icon = "❌"
        sensitivity_text = "Sensitive to unmeasured confounding"
    
    print(f"{sensitivity_icon} Confounding sensitivity: {sensitivity_text}")
    
    # OVERALL CONFIDENCE LEVEL
    print(f"\n📊 OVERALL CONFIDENCE ASSESSMENT")
    
    quality_scores = [
        1 if assumption_validity == 'HIGH' else 0.5 if assumption_validity == 'MODERATE' else 0,
        1 if overall_robustness == 'HIGH' else 0.5 if overall_robustness == 'MODERATE' else 0,
        1 if sensitivity_level == 'ROBUST' else 0.5 if sensitivity_level == 'MODERATE' else 0,
        1 if significance == 'statistically significant' else 0
    ]
    
    overall_score = sum(quality_scores) / len(quality_scores)
    
    if overall_score >= 0.8:
        confidence_level = "HIGH"
        confidence_icon = "✅"
        confidence_text = "High confidence in causal conclusion"
    elif overall_score >= 0.5:
        confidence_level = "MODERATE"
        confidence_icon = "⚠️"
        confidence_text = "Moderate confidence in causal conclusion"
    else:
        confidence_level = "LOW"
        confidence_icon = "❌"
        confidence_text = "Low confidence in causal conclusion"
    
    print(f"{confidence_icon} Overall confidence: {confidence_level}")
    print(f"   {confidence_text}")
    
    # BUSINESS RECOMMENDATION
    print(f"\n💼 BUSINESS RECOMMENDATION")
    
    if confidence_level == "HIGH" and significance == 'statistically significant' and magnitude > 0.1:
        recommendation = "SCALE"
        recommendation_icon = "🚀"
        recommendation_text = f"Recommend scaling {treatment_name}"
        recommendation_detail = f"Strong evidence of {magnitude:.1f} unit {direction} in {outcome_name}"
        
    elif confidence_level in ["HIGH", "MODERATE"] and significance == 'statistically significant':
        recommendation = "PILOT"
        recommendation_icon = "🧪"
        recommendation_text = f"Recommend pilot expansion of {treatment_name}"
        recommendation_detail = "Evidence supports cautious scaling with monitoring"
        
    elif confidence_level == "MODERATE":
        recommendation = "INVESTIGATE"
        recommendation_icon = "🔍"
        recommendation_text = "Recommend further investigation"
        recommendation_detail = "Mixed evidence requires additional validation"
        
    else:
        recommendation = "DO NOT SCALE"
        recommendation_icon = "⛔"
        recommendation_text = f"Do not scale {treatment_name} based on current evidence"
        recommendation_detail = "Insufficient evidence for causal effect"
    
    print(f"{recommendation_icon} Recommendation: {recommendation}")
    print(f"   {recommendation_text}")
    print(f"   Rationale: {recommendation_detail}")
    
    # KEY LIMITATIONS
    print(f"\n⚠️  KEY LIMITATIONS")
    
    limitations = []
    
    if assumption_validity != 'HIGH':
        limitations.append("Some methodological assumptions not fully satisfied")
    
    if overall_robustness != 'HIGH':
        limitations.append("Results show some sensitivity to model specification")
    
    if sensitivity_level != 'ROBUST':
        limitations.append("Potential sensitivity to unmeasured confounding")
    
    if effect_ci[0] <= 0 and effect_ci[1] >= 0:
        limitations.append("Confidence interval includes possibility of no effect")
    
    if not limitations:
        limitations.append("No major limitations identified")
    
    for i, limitation in enumerate(limitations, 1):
        print(f"   {i}. {limitation}")
    
    # NEXT STEPS
    print(f"\n📋 RECOMMENDED NEXT STEPS")
    
    if recommendation in ["SCALE", "PILOT"]:
        next_steps = [
            f"Design implementation plan for {treatment_name}",
            f"Establish monitoring framework for {outcome_name}",
            "Create rollout timeline with success metrics",
            "Plan for ongoing evaluation and adjustment"
        ]
    elif recommendation == "INVESTIGATE":
        next_steps = [
            "Collect additional data to address limitations",
            "Test alternative identification strategies",
            "Conduct subgroup analyses for heterogeneity",
            "Validate findings with external data sources"
        ]
    else:
        next_steps = [
            "Investigate alternative interventions",
            "Reassess measurement and data quality",
            "Consider randomized controlled trial",
            "Explore different outcome measures"
        ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")
    
    # TECHNICAL APPENDIX REFERENCE
    print(f"\n📚 TECHNICAL DETAILS")
    print("   See full technical appendix for:")
    print("   - Detailed methodology and assumptions")
    print("   - Complete diagnostic test results")
    print("   - Robustness and sensitivity analyses")
    print("   - Model specifications and code")
    
    print("\n" + "=" * 80)
    
    return {
        'effect_estimate': effect_estimate,
        'confidence_level': confidence_level,
        'recommendation': recommendation,
        'overall_score': overall_score,
        'limitations': limitations,
        'next_steps': next_steps
    }

# Generate executive summary
# This would use results from your chosen causal inference method
method_results = {
    'effect': 0.25,  # Replace with actual results
    'se': 0.08,
    'ci': [0.09, 0.41]
}

executive_summary = generate_executive_summary(
    method_results, assumption_tests, robustness_tests, 
    sensitivity_results, 'GitHub Copilot adoption', 'weekly ticket completion'
)
```

---

## Quality Assurance Checklist

### Pre-Publication Checklist

```python
def causal_inference_qa_checklist():
    """
    Quality assurance checklist for causal inference studies
    """
    
    print("=" * 80)
    print("CAUSAL INFERENCE QUALITY ASSURANCE CHECKLIST")
    print("=" * 80)
    
    checklist_sections = [
        {
            'section': 'DATA QUALITY',
            'items': [
                'Missing data patterns analyzed and addressed',
                'Outliers identified and appropriately handled',
                'Variable definitions clearly documented',
                'Sample size adequate for analysis',
                'Treatment and outcome variables properly measured',
                'Confounders identified through domain knowledge',
                'Data preprocessing steps documented and justified'
            ]
        },
        {
            'section': 'METHOD SELECTION',
            'items': [
                'Identification strategy clearly articulated',
                'Method choice justified given data and context',
                'Key assumptions explicitly stated',
                'Alternative methods considered',
                'Limitations of chosen method acknowledged',
                'Estimand (what parameter is being estimated) defined'
            ]
        },
        {
            'section': 'ASSUMPTION TESTING',
            'items': [
                'All key assumptions tested where possible',
                'Overlap/common support assessed',
                'Balance tests conducted and reported',
                'Functional form assumptions examined',
                'Placebo tests performed where applicable',
                'SUTVA (no interference) assumption considered'
            ]
        },
        {
            'section': 'ROBUSTNESS ANALYSIS',
            'items': [
                'Multiple model specifications tested',
                'Sample composition robustness checked',
                'Subgroup analyses conducted',
                'Cross-validation performed where appropriate',
                'Bootstrap or other resampling methods used',
                'Results stable across reasonable variations'
            ]
        },
        {
            'section': 'SENSITIVITY ANALYSIS',
            'items': [
                'Sensitivity to unmeasured confounding assessed',
                'E-values or similar metrics calculated',
                'Bounds analysis performed where applicable',
                'Simulation studies conducted',
                'Critical confounding levels identified',
                'Plausibility of required confounding discussed'
            ]
        },
        {
            'section': 'STATISTICAL INFERENCE',
            'items': [
                'Appropriate standard errors calculated',
                'Clustering accounted for if relevant',
                'Multiple testing corrections applied if needed',
                'Confidence intervals reported',
                'Statistical vs practical significance discussed',
                'Effect sizes interpreted appropriately'
            ]
        },
        {
            'section': 'INTERPRETATION & REPORTING',
            'items': [
                'Causal language used appropriately',
                'Effect sizes translated to business metrics',
                'Confidence levels clearly communicated',
                'Limitations explicitly acknowledged',
                'External validity discussed',
                'Business implications clearly stated'
            ]
        },
        {
            'section': 'REPRODUCIBILITY',
            'items': [
                'Analysis code documented and available',
                'Random seeds set for reproducible results',
                'Software versions and packages documented',
                'Data preprocessing steps fully specified',
                'Model specifications completely described',
                'Results can be independently verified'
            ]
        }
    ]
    
    for section in checklist_sections:
        print(f"\n{section['section']}")
        print("-" * len(section['section']))
        
        for item in section['items']:
            print(f"☐ {item}")
    
    print(f"\n{'FINAL SIGN-OFF'}")
    print("-" * len('FINAL SIGN-OFF'))
    print("☐ All checklist items reviewed and addressed")
    print("☐ Results ready for business decision-making")
    print("☐ Appropriate stakeholders have reviewed findings")
    print("☐ Implementation plan developed if applicable")
    
    print("\n" + "=" * 80)

causal_inference_qa_checklist()
```

---

## Final Integration and Next Steps

### Choosing the Right Method

```python
def method_selection_guide():
    """
    Guide for selecting the appropriate causal inference method
    """
    
    print("=== CAUSAL INFERENCE METHOD SELECTION GUIDE ===")
    
    decision_tree = [
        {
            'condition': 'Randomized controlled trial possible?',
            'yes': 'Use experimental design',
            'no': 'Continue to observational methods'
        },
        {
            'condition': 'Strong instrumental variable available?',
            'yes': 'Consider Instrumental Variables (Method 4)',
            'no': 'Continue assessment'
        },
        {
            'condition': 'Natural experiment with timing variation?',
            'yes': 'Consider Difference-in-Differences (Method 3)',
            'no': 'Continue assessment'
        },
        {
            'condition': 'High-dimensional confounding with ML?',
            'yes': 'Consider Doubly Robust/DML (Method 5)',
            'no': 'Continue assessment'
        },
        {
            'condition': 'Good propensity score overlap?',
            'yes': 'Consider Propensity Score Methods (Method 2)',
            'no': 'Consider Regression Adjustment (Method 1)'
        }
    ]
    
    print("\nDecision Framework:")
    for i, decision in enumerate(decision_tree, 1):
        print(f"{i}. {decision['condition']}")
        print(f"   Yes → {decision['yes']}")
        print(f"   No → {decision['no']}")
    
    print(f"\nMethod Comparison Summary:")
    
    methods = [
        {
            'name': 'Regression Adjustment',
            'strengths': 'Simple, interpretable, efficient',
            'weaknesses': 'Parametric assumptions, confounding bias',
            'best_for': 'Linear relationships, well-measured confounders'
        },
        {
            'name': 'Propensity Score',
            'strengths': 'Dimension reduction, transparency',
            'weaknesses': 'Overlap requirements, same assumptions',
            'best_for': 'High-dimensional confounding, binary treatment'
        },
        {
            'name': 'Difference-in-Differences',
            'strengths': 'Time-invariant confounders, policy evaluation',
            'weaknesses': 'Parallel trends assumption, timing required',
            'best_for': 'Staggered rollouts, natural experiments'
        },
        {
            'name': 'Instrumental Variables',
            'strengths': 'Unmeasured confounding, strong identification',
            'weaknesses': 'Strong assumptions, weak instrument problems',
            'best_for': 'Endogeneity concerns, natural randomization'
        },
        {
            'name': 'Doubly Robust',
            'strengths': 'Double protection, ML-friendly, efficiency',
            'weaknesses': 'Complexity, still requires assumptions',
            'best_for': 'Complex relationships, model uncertainty'
        }
    ]
    
    for method in methods:
        print(f"\n{method['name']}:")
        print(f"  Strengths: {method['strengths']}")
        print(f"  Weaknesses: {method['weaknesses']}")
        print(f"  Best for: {method['best_for']}")

method_selection_guide()
```

### Implementation Roadmap

```python
def implementation_roadmap():
    """
    Roadmap for implementing causal inference in business analytics
    """
    
    print("=== CAUSAL INFERENCE IMPLEMENTATION ROADMAP ===")
    
    phases = [
        {
            'phase': 'Phase 1: Foundation (Weeks 1-4)',
            'objectives': [
                'Establish causal inference team and governance',
                'Inventory existing data and potential use cases',
                'Set up technical infrastructure and tools',
                'Train team on causal inference fundamentals'
            ],
            'deliverables': [
                'Team charter and roles defined',
                'Data inventory and gap analysis',
                'Technical setup complete',
                'Training materials and documentation'
            ]
        },
        {
            'phase': 'Phase 2: Pilot Projects (Weeks 5-12)',
            'objectives': [
                'Select 2-3 high-impact pilot use cases',
                'Apply causal inference methods to pilot cases',
                'Develop standard analysis templates',
                'Create validation and QA processes'
            ],
            'deliverables': [
                'Pilot project results and learnings',
                'Standardized analysis templates',
                'Quality assurance checklist',
                'Best practices documentation'
            ]
        },
        {
            'phase': 'Phase 3: Scale and Standardize (Weeks 13-24)',
            'objectives': [
                'Scale to additional business units',
                'Integrate with existing analytics workflows',
                'Develop self-service capabilities',
                'Create stakeholder training programs'
            ],
            'deliverables': [
                'Scaled implementation across units',
                'Integrated analytics pipeline',
                'Self-service tools and documentation',
                'Stakeholder training completed'
            ]
        },
        {
            'phase': 'Phase 4: Advanced Applications (Weeks 25-52)',
            'objectives': [
                'Implement advanced methods (DML, TMLE)',
                'Develop real-time causal inference',
                'Create automated insight generation',
                'Establish center of excellence'
            ],
            'deliverables': [
                'Advanced methodology implementation',
                'Real-time inference capabilities',
                'Automated insight platform',
                'Center of excellence established'
            ]
        }
    ]
    
    for phase in phases:
        print(f"\n{phase['phase']}")
        print("Objectives:")
        for obj in phase['objectives']:
            print(f"  • {obj}")
        print("Deliverables:")
        for deliv in phase['deliverables']:
            print(f"  ✓ {deliv}")
    
    print(f"\n=== SUCCESS METRICS ===")
    
    metrics = [
        'Number of causal analyses completed',
        'Business decisions influenced by causal evidence',
        'Reduction in A/B test requirements',
        'Stakeholder satisfaction with insights quality',
        'Time from question to actionable insight',
        'ROI from causal inference investments'
    ]
    
    for metric in metrics:
        print(f"  📊 {metric}")

implementation_roadmap()
```

**Navigation:**
- [← Back to Doubly Robust Methods]({{ site.baseurl }}/causal-inference-doubly-robust/)
- [Back to Technical Overview]({{ site.baseurl }}/causal-inference-technical/)

---

*This validation framework ensures your causal inference results are robust, reliable, and ready for business decision-making. Always conduct comprehensive testing before making recommendations that could impact business operations or strategy.*
