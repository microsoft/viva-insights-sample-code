"""
Subgroup analysis utilities for treatment effect estimation.

This module provides functions for:
- Creating subgroup definitions
- Analyzing treatment transitions within subgroups
- Running ATE analysis for specific subgroups
- Identifying top-performing subgroups based on treatment effects
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def create_subgroup_definition(var1: str, val1: str, var2: str, val2: str) -> str:
    """
    Create a human-readable subgroup definition string.
    
    Parameters
    ----------
    var1 : str
        First variable name
    val1 : str
        First variable value
    var2 : str
        Second variable name
    val2 : str
        Second variable value
        
    Returns
    -------
    str
        Formatted subgroup definition (e.g., "Organization=Sales, FunctionType=IC")
    """
    return f"{var1}={val1}, {var2}={val2}"


def create_transition_matrix(
    subgroup_data: pd.DataFrame, 
    treatment_var: str, 
    person_id_var: str
) -> pd.DataFrame:
    """
    Create transition matrix for treatment levels using quantile-based buckets.
    
    This function analyzes outcome differences between treatment intensity buckets
    by comparing mean outcomes across quantile-based treatment groups.
    
    The function expects OUTCOME_VAR to be defined globally in the notebook.
    
    Parameters
    ----------
    subgroup_data : pd.DataFrame
        Data for the specific subgroup
    treatment_var : str
        Name of the treatment variable column
    person_id_var : str
        Name of the person ID variable column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Bucket_i: Lower treatment bucket name
        - Bucket_i_T_Range: Treatment value range for bucket i
        - Users_in_Bucket_i: Number of users in bucket i
        - Bucket_j: Higher treatment bucket name
        - Bucket_j_T_Range: Treatment value range for bucket j
        - Users_in_Bucket_j: Number of users in bucket j
        - dYij: Difference in mean outcomes (bucket j - bucket i)
        - P_Value: Statistical significance of outcome difference
        
        Returns empty DataFrame if insufficient data.
    """
    from scipy.stats import ttest_ind
    
    # Try to get OUTCOME_VAR from global scope
    try:
        import sys
        frame = sys._getframe(1)
        global_vars = frame.f_globals
        OUTCOME_VAR = global_vars.get('OUTCOME_VAR', 'External_collaboration_hours')
    except:
        OUTCOME_VAR = 'External_collaboration_hours'  # Fallback default
    
    # Define treatment quantiles for buckets
    treatment_values = subgroup_data[treatment_var].dropna()
    
    if len(treatment_values) < 10:
        return pd.DataFrame()  # Skip if too few observations
    
    quantiles = [0, 0.25, 0.50, 0.75, 0.90, 1.0]
    buckets = pd.qcut(treatment_values, q=quantiles, labels=['Bottom 25%', '25-50%', '50-75%', '75-90%', 'Top 10%'], duplicates='drop')
    
    # Create bucket mapping
    bucket_ranges = {}
    for bucket_name in buckets.cat.categories:
        bucket_data = treatment_values[buckets == bucket_name]
        if len(bucket_data) > 0:
            bucket_ranges[bucket_name] = f"({bucket_data.min():.1f}, {bucket_data.max():.1f}]"
    
    # Calculate transitions between buckets (simplified version)
    transition_data = []
    bucket_names = list(buckets.cat.categories)
    
    for i, bucket_i in enumerate(bucket_names):
        users_i = len(treatment_values[buckets == bucket_i])
        for j, bucket_j in enumerate(bucket_names):
            if i < j:  # Only transitions to higher buckets
                users_j = len(treatment_values[buckets == bucket_j])
                
                # Simplified effect calculation (difference in means)
                mean_i = subgroup_data[subgroup_data[treatment_var].isin(treatment_values[buckets == bucket_i])][OUTCOME_VAR].mean()
                mean_j = subgroup_data[subgroup_data[treatment_var].isin(treatment_values[buckets == bucket_j])][OUTCOME_VAR].mean()
                
                if pd.isna(mean_i) or pd.isna(mean_j):
                    continue
                    
                dYij = mean_j - mean_i
                
                # Simplified p-value calculation (t-test)
                group_i_outcomes = subgroup_data[subgroup_data[treatment_var].isin(treatment_values[buckets == bucket_i])][OUTCOME_VAR].dropna()
                group_j_outcomes = subgroup_data[subgroup_data[treatment_var].isin(treatment_values[buckets == bucket_j])][OUTCOME_VAR].dropna()
                
                if len(group_i_outcomes) > 1 and len(group_j_outcomes) > 1:
                    _, p_value = ttest_ind(group_j_outcomes, group_i_outcomes)
                else:
                    p_value = 1.0
                
                transition_data.append({
                    'Bucket_i': bucket_i,
                    'Bucket_i_T_Range': bucket_ranges.get(bucket_i, ''),
                    'Users_in_Bucket_i': users_i,
                    'Bucket_j': bucket_j, 
                    'Bucket_j_T_Range': bucket_ranges.get(bucket_j, ''),
                    'Users_in_Bucket_j': users_j,
                    'dYij': round(dYij, 3),
                    'P_Value': round(p_value, 4)
                })
    
    return pd.DataFrame(transition_data)


def run_ate_for_subgroup(
    subgroup_data: pd.DataFrame,
    subgroup_info: Dict[str, Any],
    treatment_var: str = 'Total_Copilot_actions_taken',
    outcome_var: str = 'After_hours_collaboration_hours'
) -> Optional[Dict[str, Any]]:
    """
    Run sophisticated ATE analysis for a specific subgroup using Double Machine Learning.
    
    This function performs a comprehensive dose-response analysis by:
    1. Fitting LinearDML with spline featurization for non-linear effects
    2. Fitting baseline LinearDML without featurization for comparison
    3. Generating dose-response curves across treatment levels
    4. Computing confidence intervals and p-values
    
    The function expects these global variables to be defined:
    - NETWORK_VARS: List of network-related variables
    - DEMOGRAPHIC_VARS: List of demographic variables
    - COLLABORATION_VARS: List of collaboration metrics
    - PERSON_ID_VAR: Name of person identifier column
    
    Parameters
    ----------
    subgroup_data : pd.DataFrame
        Data for the specific subgroup (already filtered)
    subgroup_info : dict
        Dictionary containing subgroup metadata (var1, val1, var2, val2, name)
    treatment_var : str, default='Total_Copilot_actions_taken'
        Name of the treatment variable
    outcome_var : str, default='After_hours_collaboration_hours'
        Name of the outcome variable
        
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'ate_results': DataFrame with columns Treatment, ATE_Featurized, ATE_Baseline, 
                        CI_Lower_Featurized, CI_Upper_Featurized, CI_Lower_Baseline, 
                        CI_Upper_Baseline, P_Value_Featurized, P_Value_Baseline
        - 'subgroup_clean': Cleaned DataFrame for this subgroup
        - 'estimators': Dict with 'featurized' and 'baseline' LinearDML models
        
        Returns None if insufficient data for analysis.
    """
    from econml.dml import LinearDML
    from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
    from sklearn.ensemble import RandomForestRegressor
    from scipy import stats
    
    # These variables are expected to be defined globally in the notebook
    # We reference them here to maintain consistency with the original design
    try:
        # Try to access global variables - if not available, use reasonable defaults
        import sys
        frame = sys._getframe(1)
        global_vars = frame.f_globals
        
        NETWORK_VARS = global_vars.get('NETWORK_VARS', ['Internal_network_size', 'External_network_size'])
        DEMOGRAPHIC_VARS = global_vars.get('DEMOGRAPHIC_VARS', ['Organization', 'FunctionType', 'IsManager'])
        COLLABORATION_VARS = global_vars.get('COLLABORATION_VARS', ['Meeting_hours', 'Email_hours', 'Chat_hours'])
        PERSON_ID_VAR = global_vars.get('PERSON_ID_VAR', 'PersonId')
    except:
        # Fallback defaults if frame inspection fails
        NETWORK_VARS = ['Internal_network_size', 'External_network_size']
        DEMOGRAPHIC_VARS = ['Organization', 'FunctionType', 'IsManager']
        COLLABORATION_VARS = ['Meeting_hours', 'Email_hours', 'Chat_hours']
        PERSON_ID_VAR = 'PersonId'
    
    print(f"\n--- Analyzing subgroup: {subgroup_info['name']} ---")
    print(f"Sample size: {len(subgroup_data)} observations, {subgroup_data[PERSON_ID_VAR].nunique()} users")
    
    # Prepare basic variables for ATE
    T_sub = subgroup_data[treatment_var].fillna(0).values
    Y_sub = subgroup_data[outcome_var].fillna(0).values
    
    # Effect modifiers: Network characteristics (expect heterogeneous effects by network size/type)
    effect_modifier_vars = NETWORK_VARS[:2]  # Internal/External network size
    available_effect_modifiers = [var for var in effect_modifier_vars if var in subgroup_data.columns]
    
    # Confounders: Demographics + collaboration controls
    confounder_vars = DEMOGRAPHIC_VARS + COLLABORATION_VARS[:3]
    available_confounders = [var for var in confounder_vars if var in subgroup_data.columns]
    
    # Prepare X (effect modifiers) for featurized model
    if available_effect_modifiers:
        X_sub_df = pd.get_dummies(subgroup_data[available_effect_modifiers], drop_first=True)
        X_sub_df = X_sub_df.fillna(0).astype(float)
        X_sub = X_sub_df.values
        print(f"   • Using {len(available_effect_modifiers)} effect modifier variables (X): {available_effect_modifiers}")
        print(f"   • X matrix shape after encoding: {X_sub.shape}")
    else:
        X_sub = None
        print(f"   • No effect modifier variables available for this subgroup")
    
    # Prepare W (confounders) - used in both models
    if available_confounders:
        W_sub_df = pd.get_dummies(subgroup_data[available_confounders], drop_first=True)
        W_sub_df = W_sub_df.fillna(0).astype(float)
        W_sub = W_sub_df.values
        print(f"   • Using {len(available_confounders)} confounder variables (W): {available_confounders}")
        print(f"   • W matrix shape after encoding: {W_sub.shape}")
    else:
        W_sub = subgroup_data[COLLABORATION_VARS[:3]].fillna(0).values  # Fallback to basic controls
        print(f"   • Using basic collaboration controls as W: {COLLABORATION_VARS[:3]}")
    
    # Remove missing values
    missing_conditions = [pd.isna(T_sub), pd.isna(Y_sub)]
    
    if W_sub is not None and W_sub.ndim > 1:
        missing_conditions.append(pd.isna(W_sub).any(axis=1))
    elif W_sub is not None:
        missing_conditions.append(pd.isna(W_sub))
    
    if X_sub is not None:
        X_sub_df_clean = pd.DataFrame(X_sub)
        missing_conditions.append(X_sub_df_clean.isna().any(axis=1))
    
    valid_mask = ~np.logical_or.reduce(missing_conditions)
    
    # Apply mask to all arrays
    if X_sub is not None:
        X_sub = X_sub[valid_mask]
    if W_sub is not None:
        W_sub = W_sub[valid_mask]
    
    T_sub = T_sub[valid_mask]
    Y_sub = Y_sub[valid_mask]
    
    # Convert boolean mask to integer positions to avoid index alignment issues
    valid_positions = np.where(valid_mask)[0]
    subgroup_clean = subgroup_data.iloc[valid_positions].copy()
    
    if len(T_sub) < 50 or subgroup_data[PERSON_ID_VAR].nunique() < 10:
        print(f"⚠️ Skipping subgroup {subgroup_info['name']} - insufficient data ({len(T_sub)} observations, {subgroup_data[PERSON_ID_VAR].nunique()} users)")
        return None
    
    # Fit LinearDML for ATE with spline featurizer
    spline_featurizer = SplineTransformer(n_knots=4, degree=3, include_bias=False)
    
    print(f"   • Using featurizer: SplineTransformer with 4 knots")
    
    ate_estimator_featurized = LinearDML(
        model_t=RandomForestRegressor(n_estimators=50, random_state=123),
        model_y=RandomForestRegressor(n_estimators=50, random_state=123),
        treatment_featurizer=spline_featurizer,
        cv=3,
        random_state=123
    )
    
    ate_estimator_baseline = LinearDML(
        model_t=RandomForestRegressor(n_estimators=50, random_state=123),
        model_y=RandomForestRegressor(n_estimators=50, random_state=123),
        cv=3,
        random_state=123
    )
    
    # Ensure T_sub is properly shaped for featurizers
    if T_sub.ndim == 1:
        T_sub_2d = T_sub.reshape(-1, 1)
    else:
        T_sub_2d = T_sub
    
    print(f"   • Treatment array shape: {T_sub.shape} -> {T_sub_2d.shape}")
    
    # Fit models
    print(f"   • Fitting featurized model with effect modifiers...")
    if X_sub is not None and W_sub is not None:
        ate_estimator_featurized.fit(Y_sub, T_sub_2d, X=X_sub, W=W_sub)
    elif W_sub is not None:
        ate_estimator_featurized.fit(Y_sub, T_sub_2d, W=W_sub)
    else:
        ate_estimator_featurized.fit(Y_sub, T_sub_2d)
    
    print(f"   • Fitting baseline model (average effects)...")
    if W_sub is not None:
        ate_estimator_baseline.fit(Y_sub, T_sub, W=W_sub)
    else:
        ate_estimator_baseline.fit(Y_sub, T_sub)
    
    # Create treatment grid
    treatment_max = int(np.ceil(T_sub.max()))
    treatment_grid = np.arange(0, treatment_max + 1, 1).astype(float)
    print(f"   • Treatment grid: 0 to {treatment_max} with {len(treatment_grid)} integer points")
    
    # Estimate ATE for different treatment levels
    print(f"   • Estimating treatment effects across dose levels...")
    if X_sub is not None:
        # Use median network profile
        X_median = np.percentile(X_sub, 50, axis=0).reshape(1, -1)
        treatment_grid_2d = treatment_grid.reshape(-1, 1)
        ate_featurized = ate_estimator_featurized.effect(X=X_median, T0=0, T1=treatment_grid_2d).flatten()
    else:
        treatment_grid_2d = treatment_grid.reshape(-1, 1)
        ate_featurized = ate_estimator_featurized.effect(T0=0, T1=treatment_grid_2d).flatten()
    
    ate_baseline = ate_estimator_baseline.effect(T0=0, T1=treatment_grid).flatten()
    
    print(f"   • Featurized ATE range: {ate_featurized.min():.4f} to {ate_featurized.max():.4f}")
    print(f"   • Baseline ATE range: {ate_baseline.min():.4f} to {ate_baseline.max():.4f}")
    
    # Calculate confidence intervals
    try:
        if X_sub is not None:
            ate_featurized_ci = ate_estimator_featurized.effect_interval(X=X_median, T0=0, T1=treatment_grid_2d, alpha=0.05)
        else:
            ate_featurized_ci = ate_estimator_featurized.effect_interval(T0=0, T1=treatment_grid_2d, alpha=0.05)
    except (AssertionError, ValueError) as e:
        print(f"⚠️ Warning: Could not calculate featurized confidence intervals: {e}")
        ate_featurized_se = np.std(ate_featurized) * np.ones_like(ate_featurized)
        ate_featurized_ci = (ate_featurized - 1.96 * ate_featurized_se, 
                            ate_featurized + 1.96 * ate_featurized_se)
    
    try:
        ate_baseline_ci = ate_estimator_baseline.effect_interval(T0=0, T1=treatment_grid, alpha=0.05)
    except (AssertionError, ValueError) as e:
        print(f"⚠️ Warning: Could not calculate baseline confidence intervals: {e}")
        ate_baseline_se = np.std(ate_baseline) * np.ones_like(ate_baseline)
        ate_baseline_ci = (ate_baseline - 1.96 * ate_baseline_se,
                          ate_baseline + 1.96 * ate_baseline_se)
    
    # Calculate p-values
    ate_featurized_se = (ate_featurized_ci[1] - ate_featurized_ci[0]) / (2 * 1.96)
    ate_baseline_se = (ate_baseline_ci[1] - ate_baseline_ci[0]) / (2 * 1.96)
    
    ate_featurized_se = np.where(ate_featurized_se == 0, 1e-6, ate_featurized_se)
    ate_baseline_se = np.where(ate_baseline_se == 0, 1e-6, ate_baseline_se)
    
    ate_featurized_pval = 2 * (1 - stats.norm.cdf(np.abs(ate_featurized / ate_featurized_se)))
    ate_baseline_pval = 2 * (1 - stats.norm.cdf(np.abs(ate_baseline / ate_baseline_se)))
    
    # Create results dataframe
    ate_results = pd.DataFrame({
        'Treatment': treatment_grid,
        'ATE_Featurized': ate_featurized,
        'ATE_Baseline': ate_baseline,
        'CI_Lower_Featurized': ate_featurized_ci[0].flatten() if hasattr(ate_featurized_ci[0], 'flatten') else ate_featurized_ci[0],
        'CI_Upper_Featurized': ate_featurized_ci[1].flatten() if hasattr(ate_featurized_ci[1], 'flatten') else ate_featurized_ci[1],
        'CI_Lower_Baseline': ate_baseline_ci[0].flatten() if hasattr(ate_baseline_ci[0], 'flatten') else ate_baseline_ci[0],
        'CI_Upper_Baseline': ate_baseline_ci[1].flatten() if hasattr(ate_baseline_ci[1], 'flatten') else ate_baseline_ci[1],
        'P_Value_Featurized': ate_featurized_pval,
        'P_Value_Baseline': ate_baseline_pval
    })
    
    return {
        'ate_results': ate_results,
        'subgroup_clean': subgroup_clean,
        'estimators': {'featurized': ate_estimator_featurized, 'baseline': ate_estimator_baseline}
    }


def identify_top_subgroups(
    data: pd.DataFrame,
    subgroup_vars: List[str],
    treatment_effects_col: str = 'individual_treatment_effect',
    person_id_var: str = 'PersonId',
    min_group_size: int = 30,
    find_negative_effects: bool = True,
    top_n: int = 5
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Identify top subgroups with highest/lowest treatment effects.
    
    This function creates all pairwise combinations of subgroup variables,
    calculates mean treatment effects for each subgroup, and returns the
    top N subgroups based on effect size and statistical significance.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing individual treatment effects and subgroup variables
    subgroup_vars : list of str
        List of variable names to use for subgroup creation (will create pairwise combinations)
    treatment_effects_col : str, default='individual_treatment_effect'
        Name of column containing individual treatment effects
    person_id_var : str, default='PersonId'
        Name of person ID variable
    min_group_size : int, default=30
        Minimum number of observations required for a subgroup to be included
    find_negative_effects : bool, default=True
        If True, find subgroups with most negative effects (reductions)
        If False, find subgroups with most positive effects (increases)
    top_n : int, default=5
        Number of top subgroups to return
        
    Returns
    -------
    tuple of (pd.DataFrame, list of dict)
        - DataFrame with all subgroup effects sorted by effect size
        - List of dictionaries containing detailed subgroup information
    """
    # Generate all pairwise combinations of subgroup variables
    from itertools import combinations
    
    subgroup_combinations = []
    for var1, var2 in combinations(subgroup_vars, 2):
        # Get unique values for each variable
        var1_values = data[var1].dropna().unique()
        var2_values = data[var2].dropna().unique()
        
        # Create combinations
        for val1 in var1_values:
            for val2 in var2_values:
                subgroup_combinations.append({
                    'var1': var1,
                    'val1': val1,
                    'var2': var2,
                    'val2': val2,
                    'name': create_subgroup_definition(var1, val1, var2, val2)
                })
    
    logger.info(f"Generated {len(subgroup_combinations)} subgroup combinations")
    
    # Calculate treatment effects for each subgroup
    subgroup_effects = []
    
    for combo in subgroup_combinations:
        # Create mask for this subgroup
        mask = (
            (data[combo['var1']] == combo['val1']) & 
            (data[combo['var2']] == combo['val2'])
        )
        
        if mask.sum() < min_group_size:
            continue
        
        subgroup_data = data[mask]
        mean_effect = subgroup_data[treatment_effects_col].mean()
        std_effect = subgroup_data[treatment_effects_col].std()
        n_obs = len(subgroup_data)
        n_users = subgroup_data[person_id_var].nunique()
        
        # Calculate statistical significance (t-test against 0)
        if n_obs > 1 and std_effect > 0:
            t_stat = mean_effect / (std_effect / np.sqrt(n_obs))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - 1))
        else:
            p_value = 1.0
        
        subgroup_effects.append({
            'name': combo['name'],
            'var1': combo['var1'],
            'val1': combo['val1'],
            'var2': combo['var2'],
            'val2': combo['val2'],
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'p_value': p_value,
            'n_observations': n_obs,
            'n_users': n_users,
            'significant': p_value < 0.05
        })
    
    # Convert to DataFrame and sort
    subgroup_effects_df = pd.DataFrame(subgroup_effects)
    
    if len(subgroup_effects_df) == 0:
        logger.warning("No subgroups met the minimum size requirement")
        return subgroup_effects_df, []
    
    # Filter to significant subgroups and sort by effect size
    significant_subgroups = subgroup_effects_df[
        subgroup_effects_df['significant']
    ].sort_values(
        'mean_effect',
        ascending=find_negative_effects  # True = most negative first, False = most positive first
    )
    
    # Get top N subgroups
    top_subgroups = significant_subgroups.head(top_n)
    
    logger.info(
        f"Found {len(significant_subgroups)} significant subgroups. "
        f"Returning top {len(top_subgroups)} with {'most negative' if find_negative_effects else 'most positive'} effects."
    )
    
    # Convert to list of dictionaries for easy iteration
    top_subgroups_list = top_subgroups.to_dict('records')
    
    return top_subgroups, top_subgroups_list


def generate_subgroup_summary(
    top_subgroups: pd.DataFrame,
    effect_direction: str = "negative"
) -> str:
    """
    Generate a text summary of top subgroups.
    
    Parameters
    ----------
    top_subgroups : pd.DataFrame
        DataFrame containing top subgroup analysis results
    effect_direction : str, default="negative"
        Direction of effects to describe ("negative" or "positive")
        
    Returns
    -------
    str
        Formatted summary text
    """
    if len(top_subgroups) == 0:
        return "No significant subgroups identified."
    
    summary_lines = [
        f"\n{'='*60}",
        f"Top {len(top_subgroups)} Subgroups with Most {effect_direction.title()} Effects",
        f"{'='*60}\n"
    ]
    
    for idx, (_, row) in enumerate(top_subgroups.iterrows(), 1):
        summary_lines.extend([
            f"{idx}. {row['name']}",
            f"   • Mean effect: {row['mean_effect']:.4f} hours",
            f"   • p-value: {row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''}",
            f"   • Sample size: {row['n_observations']} observations ({row['n_users']} users)",
            ""
        ])
    
    return "\n".join(summary_lines)
