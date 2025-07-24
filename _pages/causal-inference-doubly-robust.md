---
layout: default
title: "Causal Inference: Doubly Robust Methods"
permalink: /causal-inference-doubly-robust/
---

# Method 5: Doubly Robust Methods & Double Machine Learning

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

Doubly Robust methods combine regression adjustment and propensity score methods to provide protection against model misspecification. These methods remain consistent if either the outcome model OR the propensity score model is correctly specified, but not necessarily both.

**Navigation:**
- [← Back to Instrumental Variables]({{ site.baseurl }}/causal-inference-iv/)
- [Next: Validation & Testing →]({{ site.baseurl }}/causal-inference-validation/)

---

## When to Use Doubly Robust Methods

**Key Advantages:**
- **Double protection**: Consistent if either outcome or propensity model is correct
- **Machine learning friendly**: Can incorporate flexible ML models
- **Bias reduction**: Often performs better than single-model approaches
- **Transparent uncertainty**: Clear assessment of model dependence

**Best for:**
- Complex, high-dimensional confounding
- Uncertainty about functional form
- When both regression and propensity score approaches seem reasonable
- Machine learning applications in causal inference

**Key Insight:**
You only need to get ONE of the two models (outcome or propensity) approximately right, rather than both exactly right.

---

## Basic Doubly Robust Estimation

The core doubly robust estimator combines regression predictions with propensity score weighting.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def doubly_robust_estimation(df, treatment_col, outcome_col, confounders, 
                           ps_method='logistic', outcome_method='linear'):
    """
    Basic doubly robust estimation combining propensity scores and outcome regression
    
    DR Formula:
    τ_DR = E[μ₁(X) - μ₀(X)] + E[W/e(X) * (Y - μ₁(X))] - E[(1-W)/(1-e(X)) * (Y - μ₀(X))]
    
    Where:
    - μ₁(X), μ₀(X) = outcome models for treated/control
    - e(X) = propensity score
    - W = treatment indicator
    """
    
    print(f"=== DOUBLY ROBUST ESTIMATION ===")
    print(f"Sample size: {len(df)}")
    print(f"Treatment: {treatment_col}")
    print(f"Outcome: {outcome_col}")
    print(f"Confounders: {len(confounders)}")
    
    # Prepare data
    df_dr = df.dropna(subset=[treatment_col, outcome_col] + confounders).copy()
    X = df_dr[confounders]
    y = df_dr[outcome_col]
    treatment = df_dr[treatment_col]
    
    print(f"Analysis sample: {len(df_dr)} ({treatment.sum()} treated, {len(treatment) - treatment.sum()} control)")
    
    # Step 1: Estimate propensity scores
    print(f"\n=== STEP 1: PROPENSITY SCORE ESTIMATION ===")
    
    if ps_method == 'logistic':
        ps_model = LogisticRegression(random_state=42, max_iter=1000)
        # Use cross-validation to avoid overfitting
        ps_scores = cross_val_predict(ps_model, X, treatment, cv=5, method='predict_proba')[:, 1]
    elif ps_method == 'random_forest':
        ps_model = RandomForestClassifier(n_estimators=100, random_state=42)
        ps_scores = cross_val_predict(ps_model, X, treatment, cv=5, method='predict_proba')[:, 1]
    
    # Fit final model for interpretation
    ps_model.fit(X, treatment)
    
    print(f"Propensity score method: {ps_method}")
    print(f"Propensity score range: [{ps_scores.min():.3f}, {ps_scores.max():.3f}]")
    print(f"Mean treated PS: {ps_scores[treatment == 1].mean():.3f}")
    print(f"Mean control PS: {ps_scores[treatment == 0].mean():.3f}")
    
    # Check overlap
    overlap_concern = (ps_scores < 0.1).sum() + (ps_scores > 0.9).sum()
    if overlap_concern > len(df_dr) * 0.05:
        print(f"⚠️  Warning: {overlap_concern} observations with extreme propensity scores")
    
    # Step 2: Estimate outcome models
    print(f"\n=== STEP 2: OUTCOME MODEL ESTIMATION ===")
    
    # Separate models for treated and control groups
    X_treated = X[treatment == 1]
    y_treated = y[treatment == 1]
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]
    
    if outcome_method == 'linear':
        outcome_model_1 = LinearRegression()
        outcome_model_0 = LinearRegression()
    elif outcome_method == 'random_forest':
        outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
        outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit outcome models
    outcome_model_1.fit(X_treated, y_treated)
    outcome_model_0.fit(X_control, y_control)
    
    # Predict outcomes for all observations
    mu_1 = outcome_model_1.predict(X)  # Predicted outcome if treated
    mu_0 = outcome_model_0.predict(X)  # Predicted outcome if control
    
    print(f"Outcome model method: {outcome_method}")
    print(f"Treated outcome model R²: {outcome_model_1.score(X_treated, y_treated):.3f}")
    print(f"Control outcome model R²: {outcome_model_0.score(X_control, y_control):.3f}")
    
    # Step 3: Compute doubly robust estimate
    print(f"\n=== STEP 3: DOUBLY ROBUST COMBINATION ===")
    
    # Regression component (difference in predicted outcomes)
    regression_component = (mu_1 - mu_0).mean()
    
    # IPW bias correction terms
    ipw_treated_correction = ((treatment * (y - mu_1)) / ps_scores).mean()
    ipw_control_correction = (((1 - treatment) * (y - mu_0)) / (1 - ps_scores)).mean()
    
    # Final doubly robust estimate
    dr_estimate = regression_component + ipw_treated_correction - ipw_control_correction
    
    print(f"Regression component: {regression_component:.3f}")
    print(f"IPW treated correction: {ipw_treated_correction:.3f}")
    print(f"IPW control correction: {ipw_control_correction:.3f}")
    print(f"Doubly robust estimate: {dr_estimate:.3f}")
    
    # Influence function for standard errors (simplified)
    # Full implementation would use more sophisticated methods
    influence_function = (
        (mu_1 - mu_0) +
        (treatment * (y - mu_1)) / ps_scores -
        ((1 - treatment) * (y - mu_0)) / (1 - ps_scores) -
        dr_estimate
    )
    
    dr_se = influence_function.std() / np.sqrt(len(df_dr))
    dr_ci = [dr_estimate - 1.96 * dr_se, dr_estimate + 1.96 * dr_se]
    
    print(f"Standard error: {dr_se:.3f}")
    print(f"95% CI: [{dr_ci[0]:.3f}, {dr_ci[1]:.3f}]")
    
    # Compare with component methods
    print(f"\n=== COMPARISON WITH COMPONENT METHODS ===")
    
    # Pure regression estimate
    reg_only = regression_component
    
    # Pure IPW estimate
    ipw_treated_mean = ((treatment * y) / ps_scores).sum() / (treatment / ps_scores).sum()
    ipw_control_mean = (((1 - treatment) * y) / (1 - ps_scores)).sum() / ((1 - treatment) / (1 - ps_scores)).sum()
    ipw_only = ipw_treated_mean - ipw_control_mean
    
    print(f"Regression only: {reg_only:.3f}")
    print(f"IPW only: {ipw_only:.3f}")
    print(f"Doubly robust: {dr_estimate:.3f}")
    
    # Store results
    results = {
        'dr_estimate': dr_estimate,
        'dr_se': dr_se,
        'dr_ci': dr_ci,
        'regression_component': regression_component,
        'ipw_component': ipw_only,
        'ps_scores': ps_scores,
        'mu_1': mu_1,
        'mu_0': mu_0,
        'ps_model': ps_model,
        'outcome_model_1': outcome_model_1,
        'outcome_model_0': outcome_model_0,
        'influence_function': influence_function
    }
    
    return results

# Perform doubly robust estimation
confounders = ['tenure_months', 'job_level', 'team_size', 'manager_span']
dr_results = doubly_robust_estimation(
    df, 'copilot_usage', 'tickets_per_week', confounders,
    ps_method='random_forest', outcome_method='random_forest'
)
```

---

## Double Machine Learning (DML)

Double Machine Learning extends doubly robust methods with cross-fitting to avoid overfitting bias when using flexible ML models.

```python
def double_machine_learning(df, treatment_col, outcome_col, confounders, 
                           n_folds=5, ml_method='random_forest'):
    """
    Double Machine Learning (Chernozhukov et al. 2018)
    
    DML Process:
    1. Split data into K folds
    2. For each fold:
       - Train ML models on other folds
       - Predict on held-out fold
    3. Combine predictions to estimate treatment effect
    
    This avoids overfitting bias from using same data for model fitting and inference
    """
    
    print(f"=== DOUBLE MACHINE LEARNING ===")
    print(f"Method: {ml_method}")
    print(f"Cross-validation folds: {n_folds}")
    
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    # Prepare data
    df_dml = df.dropna(subset=[treatment_col, outcome_col] + confounders).copy()
    X = df_dml[confounders].values
    y = df_dml[outcome_col].values
    d = df_dml[treatment_col].values
    
    print(f"Sample size: {len(df_dml)}")
    
    # Initialize cross-fitting
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Storage for cross-fitted predictions
    y_hat = np.zeros(len(df_dml))  # Outcome predictions
    d_hat = np.zeros(len(df_dml))  # Treatment predictions
    theta_estimates = []  # Treatment effect estimates per fold
    
    print(f"\n=== CROSS-FITTING PROCEDURE ===")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"Fold {fold_idx + 1}/{n_folds}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        d_train, d_test = d[train_idx], d[test_idx]
        
        # Step 1: Predict outcomes (nuisance function l)
        if ml_method == 'random_forest':
            y_model = RandomForestRegressor(n_estimators=100, random_state=42)
            d_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif ml_method == 'linear':
            y_model = LinearRegression()
            d_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Fit models on training set
        y_model.fit(X_train, y_train)
        d_model.fit(X_train, d_train)
        
        # Predict on test set
        y_hat[test_idx] = y_model.predict(X_test)
        if ml_method == 'random_forest':
            d_hat[test_idx] = d_model.predict_proba(X_test)[:, 1]
        else:
            d_hat[test_idx] = d_model.predict_proba(X_test)[:, 1]
        
        # Step 2: Estimate treatment effect on residualized variables
        y_residual = y_test - y_hat[test_idx]
        d_residual = d_test - d_hat[test_idx]
        
        # Avoid division by very small numbers
        valid_idx = np.abs(d_residual) > 0.01
        if valid_idx.sum() > 0:
            theta_fold = np.sum(y_residual[valid_idx] * d_residual[valid_idx]) / np.sum(d_residual[valid_idx] ** 2)
            theta_estimates.append(theta_fold)
            print(f"  Fold {fold_idx + 1} estimate: {theta_fold:.3f}")
        else:
            print(f"  Fold {fold_idx + 1}: Insufficient variation")
    
    # Step 3: Final DML estimate
    dml_estimate = np.mean(theta_estimates)
    
    # Alternative: Compute on full cross-fitted predictions
    y_residual_full = y - y_hat
    d_residual_full = d - d_hat
    
    valid_mask = np.abs(d_residual_full) > 0.01
    if valid_mask.sum() > len(df_dml) * 0.8:  # Need sufficient valid observations
        dml_estimate_full = np.sum(y_residual_full[valid_mask] * d_residual_full[valid_mask]) / np.sum(d_residual_full[valid_mask] ** 2)
    else:
        dml_estimate_full = dml_estimate
        print("⚠️  Using fold-average due to insufficient variation in full sample")
    
    # Standard errors (simplified - full implementation more complex)
    influence_scores = y_residual_full * d_residual_full / np.mean(d_residual_full ** 2) - dml_estimate_full
    dml_se = np.std(influence_scores) / np.sqrt(len(df_dml))
    dml_ci = [dml_estimate_full - 1.96 * dml_se, dml_estimate_full + 1.96 * dml_se]
    
    print(f"\n=== DOUBLE MACHINE LEARNING RESULTS ===")
    print(f"Fold-average estimate: {dml_estimate:.3f}")
    print(f"Full-sample estimate: {dml_estimate_full:.3f}")
    print(f"Standard error: {dml_se:.3f}")
    print(f"95% CI: [{dml_ci[0]:.3f}, {dml_ci[1]:.3f}]")
    
    # Diagnostics
    print(f"\n=== DIAGNOSTIC CHECKS ===")
    
    # Check prediction quality
    y_pred_quality = 1 - np.var(y - y_hat) / np.var(y)
    d_pred_quality = 1 - np.var(d - d_hat) / np.var(d)
    
    print(f"Outcome prediction R²: {y_pred_quality:.3f}")
    print(f"Treatment prediction R²: {d_pred_quality:.3f}")
    
    if y_pred_quality < 0.1:
        print("⚠️  Poor outcome prediction - consider better models or features")
    if d_pred_quality < 0.1:
        print("⚠️  Poor treatment prediction - may have weak variation")
    
    # Check residual properties
    y_residual_mean = np.mean(y_residual_full)
    d_residual_mean = np.mean(d_residual_full)
    
    print(f"Mean outcome residual: {y_residual_mean:.4f} (should be ~0)")
    print(f"Mean treatment residual: {d_residual_mean:.4f} (should be ~0)")
    
    if abs(y_residual_mean) > 0.1 or abs(d_residual_mean) > 0.1:
        print("⚠️  Large residual means suggest poor model fit")
    
    return {
        'dml_estimate': dml_estimate_full,
        'dml_se': dml_se,
        'dml_ci': dml_ci,
        'fold_estimates': theta_estimates,
        'y_hat': y_hat,
        'd_hat': d_hat,
        'y_pred_quality': y_pred_quality,
        'd_pred_quality': d_pred_quality
    }

# Perform Double Machine Learning
dml_results = double_machine_learning(
    df, 'copilot_usage', 'tickets_per_week', confounders,
    n_folds=5, ml_method='random_forest'
)
```

---

## Model Selection and Robustness

### Testing Different Model Combinations

```python
def test_model_robustness(df, treatment_col, outcome_col, confounders):
    """
    Test robustness across different model specifications
    """
    
    print("=== DOUBLY ROBUST MODEL ROBUSTNESS TESTING ===")
    
    # Different model combinations
    model_combinations = [
        {'ps': 'logistic', 'outcome': 'linear', 'name': 'Linear-Linear'},
        {'ps': 'logistic', 'outcome': 'random_forest', 'name': 'Linear-RF'},
        {'ps': 'random_forest', 'outcome': 'linear', 'name': 'RF-Linear'},
        {'ps': 'random_forest', 'outcome': 'random_forest', 'name': 'RF-RF'}
    ]
    
    results_comparison = {}
    
    for combo in model_combinations:
        print(f"\n--- {combo['name']} ---")
        
        try:
            results = doubly_robust_estimation(
                df, treatment_col, outcome_col, confounders,
                ps_method=combo['ps'], outcome_method=combo['outcome']
            )
            
            results_comparison[combo['name']] = {
                'estimate': results['dr_estimate'],
                'se': results['dr_se'],
                'ci': results['dr_ci']
            }
            
            print(f"Estimate: {results['dr_estimate']:.3f} ± {results['dr_se']:.3f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            results_comparison[combo['name']] = None
    
    # Compare results
    print(f"\n=== ROBUSTNESS COMPARISON ===")
    
    valid_results = {k: v for k, v in results_comparison.items() if v is not None}
    
    if len(valid_results) > 1:
        estimates = [r['estimate'] for r in valid_results.values()]
        estimate_range = max(estimates) - min(estimates)
        estimate_mean = np.mean(estimates)
        
        print(f"Estimates range: {min(estimates):.3f} to {max(estimates):.3f}")
        print(f"Range as % of mean: {(estimate_range/abs(estimate_mean)*100):.1f}%")
        
        if estimate_range / abs(estimate_mean) < 0.2:
            print("✅ Results robust across model specifications")
        else:
            print("⚠️  Results sensitive to model choice")
            print("   Consider investigating model fit quality")
        
        # Display comparison table
        comparison_df = pd.DataFrame(valid_results).T
        print(f"\nComparison table:")
        print(comparison_df)
    
    return results_comparison

# Test model robustness
robustness_results = test_model_robustness(
    df, 'copilot_usage', 'tickets_per_week', confounders
)
```

### Cross-Validation for Model Assessment

```python
def cross_validate_dr_methods(df, treatment_col, outcome_col, confounders, cv_folds=5):
    """
    Cross-validation assessment of different DR methods
    """
    
    from sklearn.model_selection import KFold
    
    print(f"=== CROSS-VALIDATION OF DR METHODS ===")
    print(f"CV folds: {cv_folds}")
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    df_cv = df.dropna(subset=[treatment_col, outcome_col] + confounders)
    
    methods = ['DR-Linear', 'DR-RF', 'DML-RF']
    cv_results = {method: [] for method in methods}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(df_cv)):
        print(f"\nFold {fold_idx + 1}/{cv_folds}")
        
        df_train = df_cv.iloc[train_idx]
        df_test = df_cv.iloc[test_idx]
        
        # Doubly Robust - Linear
        try:
            dr_linear = doubly_robust_estimation(
                df_train, treatment_col, outcome_col, confounders,
                ps_method='logistic', outcome_method='linear'
            )
            cv_results['DR-Linear'].append(dr_linear['dr_estimate'])
        except:
            cv_results['DR-Linear'].append(np.nan)
        
        # Doubly Robust - Random Forest
        try:
            dr_rf = doubly_robust_estimation(
                df_train, treatment_col, outcome_col, confounders,
                ps_method='random_forest', outcome_method='random_forest'
            )
            cv_results['DR-RF'].append(dr_rf['dr_estimate'])
        except:
            cv_results['DR-RF'].append(np.nan)
        
        # Double Machine Learning
        try:
            dml = double_machine_learning(
                df_train, treatment_col, outcome_col, confounders,
                n_folds=3, ml_method='random_forest'
            )
            cv_results['DML-RF'].append(dml['dml_estimate'])
        except:
            cv_results['DML-RF'].append(np.nan)
    
    # Analyze cross-validation results
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    
    for method, estimates in cv_results.items():
        valid_estimates = [e for e in estimates if not np.isnan(e)]
        
        if len(valid_estimates) > 0:
            mean_est = np.mean(valid_estimates)
            std_est = np.std(valid_estimates)
            
            print(f"{method}: {mean_est:.3f} ± {std_est:.3f} (CV mean ± std)")
            print(f"  Valid folds: {len(valid_estimates)}/{cv_folds}")
            
            if std_est / abs(mean_est) < 0.2:
                stability = "Stable"
            elif std_est / abs(mean_est) < 0.5:
                stability = "Moderate"
            else:
                stability = "Unstable"
            
            print(f"  Stability: {stability}")
        else:
            print(f"{method}: Failed on all folds")
    
    return cv_results

# Perform cross-validation assessment
cv_assessment = cross_validate_dr_methods(
    df, 'copilot_usage', 'tickets_per_week', confounders
)
```

---

## Advanced Doubly Robust Methods

### Targeted Maximum Likelihood Estimation (TMLE)

```python
def targeted_maximum_likelihood(df, treatment_col, outcome_col, confounders):
    """
    Targeted Maximum Likelihood Estimation (TMLE)
    
    TMLE is a doubly robust method that:
    1. Estimates initial outcome regression
    2. Updates estimates using propensity scores
    3. Provides efficient, unbiased estimates
    """
    
    print("=== TARGETED MAXIMUM LIKELIHOOD ESTIMATION ===")
    
    # This is a simplified implementation
    # For production use, consider specialized packages like tmle3 (R) or tmle (Python)
    
    df_tmle = df.dropna(subset=[treatment_col, outcome_col] + confounders).copy()
    X = df_tmle[confounders]
    y = df_tmle[outcome_col]
    treatment = df_tmle[treatment_col]
    
    print(f"Sample size: {len(df_tmle)}")
    
    # Step 1: Initial outcome regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    
    # Fit separate models for treated and control
    outcome_model_1 = RandomForestRegressor(n_estimators=100, random_state=42)
    outcome_model_0 = RandomForestRegressor(n_estimators=100, random_state=42)
    
    treated_mask = treatment == 1
    control_mask = treatment == 0
    
    outcome_model_1.fit(X[treated_mask], y[treated_mask])
    outcome_model_0.fit(X[control_mask], y[control_mask])
    
    # Initial predictions
    Q1_n = outcome_model_1.predict(X)
    Q0_n = outcome_model_0.predict(X)
    
    # Step 2: Propensity score estimation
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X, treatment)
    g_n = ps_model.predict_proba(X)[:, 1]
    
    # Step 3: Clever covariate
    H1_n = treatment / g_n
    H0_n = (1 - treatment) / (1 - g_n)
    
    # Step 4: Targeting step (fluctuation)
    # This step updates the initial fit to reduce bias
    
    # For treated units
    epsilon_1 = 0  # Would be estimated via MLE fluctuation
    Q1_n_star = Q1_n + epsilon_1 * H1_n
    
    # For control units  
    epsilon_0 = 0  # Would be estimated via MLE fluctuation
    Q0_n_star = Q0_n + epsilon_0 * H0_n
    
    # Simplified: use the mean of residuals as fluctuation parameter
    residuals_1 = y[treated_mask] - Q1_n[treated_mask]
    residuals_0 = y[control_mask] - Q0_n[control_mask]
    
    epsilon_1 = residuals_1.mean() if len(residuals_1) > 0 else 0
    epsilon_0 = residuals_0.mean() if len(residuals_0) > 0 else 0
    
    Q1_n_star = Q1_n + epsilon_1
    Q0_n_star = Q0_n + epsilon_0
    
    # Step 5: Final TMLE estimate
    tmle_estimate = Q1_n_star.mean() - Q0_n_star.mean()
    
    # Influence curve for standard errors (simplified)
    IC = (H1_n * (y - Q1_n_star) - H0_n * (y - Q0_n_star) + 
          Q1_n_star - Q0_n_star - tmle_estimate)
    
    tmle_se = IC.std() / np.sqrt(len(df_tmle))
    tmle_ci = [tmle_estimate - 1.96 * tmle_se, tmle_estimate + 1.96 * tmle_se]
    
    print(f"Initial outcome regression estimate: {(Q1_n.mean() - Q0_n.mean()):.3f}")
    print(f"TMLE estimate: {tmle_estimate:.3f}")
    print(f"Standard error: {tmle_se:.3f}")
    print(f"95% CI: [{tmle_ci[0]:.3f}, {tmle_ci[1]:.3f}]")
    
    return {
        'tmle_estimate': tmle_estimate,
        'tmle_se': tmle_se,
        'tmle_ci': tmle_ci,
        'Q1_star': Q1_n_star,
        'Q0_star': Q0_n_star,
        'influence_curve': IC
    }

# Note: This is a simplified TMLE implementation
# For production use, consider specialized TMLE packages
# tmle_results = targeted_maximum_likelihood(
#     df, 'copilot_usage', 'tickets_per_week', confounders
# )
```

---

## Diagnostics and Model Assessment

### Comprehensive DR Diagnostics

```python
def dr_diagnostics_comprehensive(dr_results, dml_results, df, treatment_col, outcome_col):
    """
    Comprehensive diagnostics for doubly robust methods
    """
    
    print("=== DOUBLY ROBUST DIAGNOSTICS ===")
    
    # 1. Model fit quality
    print("\n1. MODEL FIT QUALITY")
    
    # Propensity score overlap
    ps_scores = dr_results['ps_scores']
    overlap_issues = (ps_scores < 0.05).sum() + (ps_scores > 0.95).sum()
    
    print(f"Extreme propensity scores: {overlap_issues}")
    if overlap_issues > len(ps_scores) * 0.05:
        print("⚠️  Poor overlap - consider trimming or alternative methods")
    else:
        print("✅ Good propensity score overlap")
    
    # Outcome model fit
    treatment = df[treatment_col]
    outcome = df[outcome_col]
    
    treated_r2 = dr_results['outcome_model_1'].score(
        df[treatment == 1][['tenure_months', 'job_level', 'team_size', 'manager_span']], 
        outcome[treatment == 1]
    )
    control_r2 = dr_results['outcome_model_0'].score(
        df[treatment == 0][['tenure_months', 'job_level', 'team_size', 'manager_span']], 
        outcome[treatment == 0]
    )
    
    print(f"Treated outcome model R²: {treated_r2:.3f}")
    print(f"Control outcome model R²: {control_r2:.3f}")
    
    if min(treated_r2, control_r2) < 0.1:
        print("⚠️  Poor outcome model fit")
    else:
        print("✅ Reasonable outcome model fit")
    
    # 2. Component comparison
    print("\n2. COMPONENT COMPARISON")
    
    dr_estimate = dr_results['dr_estimate']
    reg_component = dr_results['regression_component']
    ipw_component = dr_results['ipw_component']
    
    print(f"Regression component: {reg_component:.3f}")
    print(f"IPW component: {ipw_component:.3f}")
    print(f"Doubly robust: {dr_estimate:.3f}")
    
    # Check if components are similar (suggests both models are reasonable)
    component_diff = abs(reg_component - ipw_component)
    if component_diff < abs(dr_estimate) * 0.2:
        print("✅ Components agree - both models likely reasonable")
    else:
        print("⚠️  Components disagree - model misspecification likely")
    
    # 3. DML comparison
    if dml_results:
        print("\n3. DML COMPARISON")
        
        dml_estimate = dml_results['dml_estimate']
        dr_dml_diff = abs(dr_estimate - dml_estimate)
        
        print(f"Basic DR: {dr_estimate:.3f}")
        print(f"DML: {dml_estimate:.3f}")
        print(f"Difference: {dr_dml_diff:.3f}")
        
        if dr_dml_diff < abs(dr_estimate) * 0.1:
            print("✅ DR and DML agree - results robust")
        else:
            print("⚠️  DR and DML disagree - investigate further")
    
    # 4. Influence function diagnostics
    print("\n4. INFLUENCE FUNCTION DIAGNOSTICS")
    
    influence = dr_results['influence_function']
    
    print(f"Influence function mean: {influence.mean():.4f} (should be ~0)")
    print(f"Influence function std: {influence.std():.3f}")
    
    # Check for outliers in influence function
    influence_outliers = (np.abs(influence) > 3 * influence.std()).sum()
    print(f"Influence outliers: {influence_outliers} ({influence_outliers/len(influence):.1%})")
    
    if influence_outliers > len(influence) * 0.05:
        print("⚠️  Many influence function outliers")
    
    # 5. Overall assessment
    print("\n5. OVERALL ASSESSMENT")
    
    issues = []
    if overlap_issues > len(ps_scores) * 0.05:
        issues.append("Poor propensity score overlap")
    if min(treated_r2, control_r2) < 0.1:
        issues.append("Poor outcome model fit")
    if component_diff > abs(dr_estimate) * 0.2:
        issues.append("Component models disagree")
    if abs(influence.mean()) > 0.01:
        issues.append("Non-zero influence function mean")
    
    if not issues:
        print("✅ ALL DIAGNOSTICS PASSED")
        quality = "HIGH"
    elif len(issues) <= 2:
        print("⚠️  MINOR CONCERNS:")
        for issue in issues:
            print(f"   - {issue}")
        quality = "MODERATE"
    else:
        print("❌ MAJOR ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        quality = "LOW"
    
    print(f"\nOverall quality: {quality}")
    
    return {
        'quality': quality,
        'issues': issues,
        'overlap_problems': overlap_issues,
        'model_fit_quality': min(treated_r2, control_r2),
        'component_agreement': component_diff < abs(dr_estimate) * 0.2
    }

# Perform comprehensive diagnostics
if 'dr_results' in locals():
    dr_diagnostics = dr_diagnostics_comprehensive(
        dr_results, dml_results, df, 'copilot_usage', 'tickets_per_week'
    )
```

---

## Business Translation and Interpretation

### Converting DR Results to Business Insights

```python
def interpret_dr_results_for_business(dr_results, dml_results, treatment_name, outcome_name):
    """
    Translate doubly robust results into business language
    """
    
    print("=== BUSINESS INTERPRETATION: DOUBLY ROBUST ANALYSIS ===")
    
    dr_estimate = dr_results['dr_estimate']
    dr_se = dr_results['dr_se']
    dr_ci = dr_results['dr_ci']
    
    # Main finding
    print(f"\n📊 CAUSAL EFFECT ESTIMATE:")
    print(f"Treatment: {treatment_name}")
    print(f"Outcome: {outcome_name}")
    
    if dr_estimate > 0:
        direction = "increases"
        improvement = "improvement"
    else:
        direction = "decreases"
        improvement = "reduction"
        dr_estimate = abs(dr_estimate)
    
    print(f"\nResult: {treatment_name} {direction} {outcome_name} by {dr_estimate:.2f} units")
    
    # Confidence assessment
    print(f"\n🎯 CONFIDENCE ASSESSMENT:")
    print(f"95% Confidence Interval: [{dr_ci[0]:.2f}, {dr_ci[1]:.2f}]")
    
    if dr_ci[0] > 0 or dr_ci[1] < 0:
        print("✅ Statistically significant effect")
        confidence_level = "High"
    else:
        print("⚠️  Effect not statistically significant")
        confidence_level = "Low"
    
    # Method advantages
    print(f"\n🛡️  METHODOLOGICAL ADVANTAGES:")
    print("✅ Doubly robust protection:")
    print("   - Consistent if EITHER outcome OR propensity model is correct")
    print("   - More reliable than single-model approaches")
    print("   - Reduces model misspecification bias")
    
    # Compare with DML if available
    if dml_results:
        dml_estimate = dml_results['dml_estimate']
        print(f"\n🔍 ROBUSTNESS CHECK:")
        print(f"Standard DR estimate: {dr_results['dr_estimate']:.3f}")
        print(f"DML estimate: {dml_estimate:.3f}")
        
        difference = abs(dr_results['dr_estimate'] - dml_estimate)
        if difference < abs(dr_results['dr_estimate']) * 0.1:
            print("✅ Results consistent across methods")
            robustness = "High"
        else:
            print("⚠️  Some variation across methods")
            robustness = "Moderate"
    else:
        robustness = "Not assessed"
    
    # Business implications
    print(f"\n💼 BUSINESS IMPLICATIONS:")
    
    if confidence_level == "High" and dr_estimate > 0.1:
        print("✅ STRONG BUSINESS CASE:")
        print(f"   - Clear evidence of {improvement}")
        print(f"   - Effect size: {dr_estimate:.2f} units")
        print("   - Methodologically robust")
        print("   - Recommendation: Scale intervention")
        
        # ROI framework
        print(f"\n💰 ROI FRAMEWORK:")
        print(f"   - Quantify {dr_estimate:.2f} unit {improvement} per user")
        print("   - Calculate implementation costs")
        print("   - Estimate net benefit across organization")
        
    elif confidence_level == "High":
        print("⚠️  MODEST BUSINESS CASE:")
        print(f"   - Statistically significant but small effect")
        print("   - Consider cost-benefit analysis")
        print("   - May warrant targeted implementation")
        
    else:
        print("❌ INSUFFICIENT EVIDENCE:")
        print("   - Cannot establish clear causal effect")
        print("   - Do not scale based on current evidence")
        print("   - Consider longer study period or larger sample")
    
    # Implementation guidance
    print(f"\n🚀 IMPLEMENTATION GUIDANCE:")
    
    if confidence_level == "High":
        print("📋 Next steps:")
        print("   1. Validate findings with additional data")
        print("   2. Design pilot expansion program")
        print("   3. Monitor key metrics during rollout")
        print(f"   4. Track {outcome_name} improvements")
        
        print("\n📊 Monitoring framework:")
        print(f"   - Baseline: Current {outcome_name} levels")
        print(f"   - Target: {dr_estimate:.1f} unit improvement")
        print("   - Timeline: Quarterly measurement")
        print("   - Adjustment: Refine based on results")
    
    # Limitations and caveats
    print(f"\n⚠️  LIMITATIONS TO CONSIDER:")
    print("1. Assumes no unmeasured confounders")
    print("2. Effect may vary across subgroups")
    print("3. Based on historical data - may not generalize")
    print("4. Requires continued model validity")
    
    return {
        'effect_size': dr_estimate,
        'direction': direction,
        'confidence_level': confidence_level,
        'robustness': robustness,
        'business_recommendation': 'scale' if confidence_level == "High" and dr_estimate > 0.1 else 'evaluate'
    }

# Generate business interpretation
if 'dr_results' in locals():
    business_interpretation = interpret_dr_results_for_business(
        dr_results, dml_results,
        'GitHub Copilot adoption',
        'weekly ticket completion'
    )
```

---

## Best Practices and Guidelines

### Implementation Checklist

```python
def dr_implementation_checklist():
    """
    Best practices checklist for doubly robust methods
    """
    
    print("=== DOUBLY ROBUST METHODS: IMPLEMENTATION CHECKLIST ===")
    
    checklist_items = [
        {
            'category': 'Data Preparation',
            'items': [
                'Check for missing data patterns',
                'Validate treatment and outcome variables',
                'Ensure sufficient sample size for subgroups',
                'Remove or handle extreme outliers'
            ]
        },
        {
            'category': 'Model Selection',
            'items': [
                'Test multiple propensity score specifications',
                'Try different outcome model forms',
                'Consider machine learning for complex relationships',
                'Validate models on hold-out data'
            ]
        },
        {
            'category': 'Overlap Assessment',
            'items': [
                'Check propensity score distribution',
                'Identify regions of poor overlap',
                'Consider trimming extreme propensity scores',
                'Assess common support assumption'
            ]
        },
        {
            'category': 'Robustness Testing',
            'items': [
                'Compare DR with component methods',
                'Test sensitivity to model specifications',
                'Cross-validate results',
                'Check for influential observations'
            ]
        },
        {
            'category': 'Inference',
            'items': [
                'Use appropriate standard errors',
                'Consider clustering if relevant',
                'Report confidence intervals',
                'Assess statistical significance'
            ]
        },
        {
            'category': 'Interpretation',
            'items': [
                'Translate to business metrics',
                'Consider effect size practical significance',
                'Acknowledge limitations and assumptions',
                'Provide implementation guidance'
            ]
        }
    ]
    
    for category_info in checklist_items:
        print(f"\n{category_info['category'].upper()}:")
        for item in category_info['items']:
            print(f"   ☐ {item}")
    
    print("\n=== COMMON PITFALLS TO AVOID ===")
    pitfalls = [
        "Using same data for model selection and inference",
        "Ignoring overlap problems",
        "Over-interpreting small effect sizes",
        "Assuming both models are correctly specified",
        "Not testing robustness across specifications",
        "Forgetting about external validity"
    ]
    
    for i, pitfall in enumerate(pitfalls, 1):
        print(f"   {i}. {pitfall}")

dr_implementation_checklist()
```

**Navigation:**
- [← Back to Instrumental Variables]({{ site.baseurl }}/causal-inference-iv/)
- [Next: Validation & Testing →]({{ site.baseurl }}/causal-inference-validation/)

---

*Doubly Robust methods provide powerful protection against model misspecification by combining multiple approaches. Always test robustness across different model specifications and validate assumptions before making business decisions.*
