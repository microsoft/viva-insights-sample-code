---
layout: default
title: "Causal Inference: Data Preparation & Setup"
eyebrow: "Causal inference"
permalink: /causal-inference-data-prep/
---

# Data Preparation & Setup for Causal Inference
## Overview

Successful causal inference starts with properly structured data. Unlike simple correlation analysis, causal methods require specific data characteristics and measurement approaches. This guide covers essential data preparation steps before implementing any causal inference method.

**Navigation:**
- [← Back to Technical Overview]({{ site.baseurl }}/causal-inference-technical/)
- [Next: Regression Adjustment →]({{ site.baseurl }}/causal-inference-regression/)

---

## Understanding Your Data Requirements

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
    # Per-unit temporal ordering: panel data interleaves units, so a global
    # is_monotonic_increasing check spuriously fails. Sort within each unit and
    # require strictly increasing periods (no duplicate periods per unit).
    if 'user_id' in df.columns and 'time_period' in df.columns:
        temporal_ok = (
            df.sort_values(['user_id', 'time_period'])
              .groupby('user_id')['time_period']
              .apply(lambda s: s.is_monotonic_increasing and not s.duplicated().any())
              .all()
        )
    else:
        temporal_ok = df['time_period'].sort_values().is_monotonic_increasing

    checks = {
        'temporal_ordering': bool(temporal_ok),
        'treatment_variation': df['treatment'].nunique() > 1,
        'outcome_completeness': df['outcome'].notna().mean() > 0.8,
        'sufficient_sample': len(df) > 100,
    }

    # Panel structure: an *unbalanced* panel is still perfectly valid for most
    # methods, so we do not flag it as a failure. The genuine data-quality
    # problem is duplicate (user_id, time_period) rows; balance is reported as
    # information only.
    panel_info = {}
    if {'user_id', 'time_period'}.issubset(df.columns):
        dup_rows = int(df.duplicated(subset=['user_id', 'time_period']).sum())
        n_users = df['user_id'].nunique()
        n_periods = df['time_period'].nunique()
        checks['no_duplicate_user_periods'] = dup_rows == 0
        panel_info = {
            'balanced': bool(n_users * n_periods == len(df)) and dup_rows == 0,
            'n_users': n_users,
            'n_periods': n_periods,
            'duplicate_user_periods': dup_rows,
        }
    
    # Generate recommendations based on failed checks
    recommendations = []
    if not checks['temporal_ordering']:
        recommendations.append("Sort each user's records by time_period; some units are out of order or have duplicate periods")
    if not checks['treatment_variation']:
        recommendations.append("Treatment variable shows no variation - check data quality")
    if not checks['outcome_completeness']:
        recommendations.append("High missing data in outcome - consider imputation or data cleaning")
    if not checks['sufficient_sample']:
        recommendations.append("Sample size may be too small for reliable causal inference")
    if not checks.get('no_duplicate_user_periods', True):
        recommendations.append("Duplicate user-period rows found - aggregate or de-duplicate before analysis")
    if panel_info and not panel_info['balanced']:
        recommendations.append("Panel is unbalanced (this is acceptable); ensure your method handles missing user-periods")
    
    return {'checks': checks, 'panel_info': panel_info, 'recommendations': recommendations}

# Run validation
validation_results = validate_causal_data(df)
print("Data Quality Checks:", validation_results['checks'])
if validation_results['recommendations']:
    print("Recommendations:", validation_results['recommendations'])
```

**Understanding the Output:**
- `temporal_ordering`: Ensures each unit's periods are strictly increasing (critical for causal ordering)
- `treatment_variation`: Confirms we have both treated and untreated observations
- `outcome_completeness`: High missing data can bias results
- `sufficient_sample`: Small samples lead to unreliable estimates
- `no_duplicate_user_periods`: Flags duplicate (user, period) rows — the real panel data-quality issue
- `panel_info`: Reports whether the panel is balanced (informational — unbalanced panels are still valid)

---

## Defining Treatment and Outcome Variables

### Treatment Variable Design

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

### Outcome Variable Selection

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

## Identifying and Measuring Confounders

Confounders are variables that influence both treatment assignment and the outcome. Proper identification and measurement of confounders is crucial for causal inference.

### Common Confounders in Copilot Analytics

```python
def identify_potential_confounders():
    """
    Comprehensive list of potential confounders for Copilot analysis
    """
    confounders = {
        # Individual characteristics
        'demographic': [
            'tenure_months',
            'job_level', 
            'department',
            'role_type',
            'education_level'
        ],
        
        # Work characteristics
        'work_context': [
            'team_size',
            'manager_span',
            'workload_intensity',
            'meeting_hours_per_week',
            'project_complexity'
        ],
        
        # Historical performance
        'baseline_performance': [
            'tickets_resolved_baseline',
            'performance_rating_last_period',
            'training_completed',
            'tool_adoption_history'
        ],
        
        # Technology context
        'technology': [
            'system_access_level',
            'other_tools_used',
            'technical_skill_level',
            'device_type'
        ]
    }
    
    return confounders

# Create confounder dataset
def prepare_confounders(df):
    """
    Prepare and validate confounder variables
    """
    confounders = identify_potential_confounders()
    
    # Check availability of confounders in data
    available_confounders = []
    missing_confounders = []
    
    for category, variables in confounders.items():
        for var in variables:
            if var in df.columns:
                available_confounders.append(var)
            else:
                missing_confounders.append(var)
    
    print(f"Available confounders: {len(available_confounders)}")
    print(f"Missing confounders: {len(missing_confounders)}")
    
    if missing_confounders:
        print(f"Consider collecting: {missing_confounders[:5]}...")
    
    return available_confounders

# Prepare confounder list
available_confounders = prepare_confounders(df)
```

### Confounder Quality Assessment

```python
def assess_confounder_quality(df, confounders, treatment_col, outcome_col):
    """
    Assess the quality and relevance of potential confounders
    """
    assessments = {}
    
    for confounder in confounders:
        if confounder not in df.columns:
            continue
            
        # Missing data rate
        missing_rate = df[confounder].isnull().mean()
        
        # Correlation with treatment
        if df[confounder].dtype in ['int64', 'float64']:
            treatment_corr = df[confounder].corr(df[treatment_col])
            outcome_corr = df[confounder].corr(df[outcome_col])
        else:
            # For categorical variables, use association measures
            treatment_corr = "categorical"
            outcome_corr = "categorical"
        
        # Variance check
        if df[confounder].dtype in ['int64', 'float64']:
            has_variance = df[confounder].var() > 0
        else:
            has_variance = df[confounder].nunique() > 1
        
        assessments[confounder] = {
            'missing_rate': missing_rate,
            'treatment_correlation': treatment_corr,
            'outcome_correlation': outcome_corr,
            'has_variance': has_variance,
            'data_type': str(df[confounder].dtype)
        }
    
    return assessments

# Assess confounder quality
confounder_assessments = assess_confounder_quality(
    df, available_confounders, 'copilot_usage', 'tickets_per_week'
)

# Print assessment summary
for var, assessment in confounder_assessments.items():
    if assessment['missing_rate'] > 0.2:
        print(f"⚠️  {var}: High missing data ({assessment['missing_rate']:.1%})")
    elif not assessment['has_variance']:
        print(f"⚠️  {var}: No variance in data")
    else:
        print(f"✅ {var}: Good quality confounder")
```

---

## Data Quality Checks and Validation

### Comprehensive Data Quality Assessment

```python
def comprehensive_data_quality_check(df, outcome_col, treatment_col, confounder_cols):
    """
    Comprehensive data quality assessment for causal inference
    """
    quality_report = {
        'sample_size': len(df),
        'time_periods': df['time_period'].nunique() if 'time_period' in df.columns else 1,
        'unique_users': df['user_id'].nunique() if 'user_id' in df.columns else len(df),
        'treatment_balance': df[treatment_col].mean(),
        'outcome_distribution': {
            'mean': df[outcome_col].mean(),
            'std': df[outcome_col].std(),
            'skewness': df[outcome_col].skew(),
            'outliers': ((df[outcome_col] - df[outcome_col].mean()).abs() > 3 * df[outcome_col].std()).sum()
        },
        'missing_data': {
            'outcome': df[outcome_col].isnull().mean(),
            'treatment': df[treatment_col].isnull().mean(),
            'confounders': {col: df[col].isnull().mean() for col in confounder_cols if col in df.columns}
        }
    }
    
    # Data quality warnings
    warnings = []
    
    if quality_report['sample_size'] < 500:
        warnings.append("Small sample size may lead to unreliable estimates")
    
    if quality_report['treatment_balance'] < 0.1 or quality_report['treatment_balance'] > 0.9:
        warnings.append("Imbalanced treatment assignment - consider different specification")
    
    if quality_report['outcome_distribution']['outliers'] > len(df) * 0.05:
        warnings.append("Many outliers detected - consider winsorization or robust methods")
    
    if any(rate > 0.1 for rate in quality_report['missing_data']['confounders'].values()):
        warnings.append("High missing data in confounders - consider imputation")
    
    quality_report['warnings'] = warnings
    
    return quality_report

# Run comprehensive quality check
quality_report = comprehensive_data_quality_check(
    df, 'tickets_per_week', 'copilot_usage', available_confounders
)

# Print quality report
print("=== DATA QUALITY REPORT ===")
print(f"Sample size: {quality_report['sample_size']}")
print(f"Treatment balance: {quality_report['treatment_balance']:.1%}")
print(f"Outcome mean: {quality_report['outcome_distribution']['mean']:.2f}")

if quality_report['warnings']:
    print("\n⚠️  WARNINGS:")
    for warning in quality_report['warnings']:
        print(f"   - {warning}")
```

### Data Preprocessing for Causal Analysis

```python
def preprocess_for_causal_analysis(df, outcome_col, treatment_col, confounder_cols):
    """
    Standardized preprocessing pipeline for causal inference
    """
    processed_df = df.copy()
    
    # 1. Handle missing data
    # Simple imputation for demonstration - use more sophisticated methods in practice
    for col in confounder_cols:
        if col in processed_df.columns:
            if processed_df[col].dtype in ['int64', 'float64']:
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            else:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode().iloc[0])
    
    # 2. Outlier treatment (winsorization)
    from scipy.stats import mstats
    
    for col in [outcome_col] + confounder_cols:
        if col in processed_df.columns and processed_df[col].dtype in ['int64', 'float64']:
            processed_df[col] = mstats.winsorize(processed_df[col], limits=[0.01, 0.01])
    
    # 3. Create categorical variables from continuous ones if needed
    if 'tenure_months' in processed_df.columns:
        processed_df['tenure_category'] = pd.cut(
            processed_df['tenure_months'],
            bins=[0, 12, 36, 60, np.inf],
            labels=['Junior', 'Mid', 'Senior', 'Expert']
        )
    
    # 4. Validate final dataset
    final_validation = validate_causal_data(processed_df)
    
    print("=== PREPROCESSING COMPLETE ===")
    print(f"Final sample size: {len(processed_df)}")
    print(f"Data quality checks passed: {sum(final_validation['checks'].values())}/{len(final_validation['checks'])}")
    
    return processed_df, final_validation

# Preprocess data
processed_df, validation_results = preprocess_for_causal_analysis(
    df, 'tickets_per_week', 'copilot_usage', available_confounders
)
```

---

## Creating Analysis-Ready Datasets

### Standard Dataset Formats

```python
def create_analysis_datasets(df, outcome_col, treatment_col, confounder_cols,
                             intervention_date=None):
    """
    Create standardized datasets for different causal inference methods

    intervention_date : the real date/period when the intervention took effect.
        Required to build the before/after (DiD) dataset. There is no safe
        default — inferring it from the data (e.g. the midpoint) fabricates the
        treatment timing and invalidates the design.
    """
    datasets = {}
    
    # 1. Cross-sectional dataset (for regression, propensity scores)
    if 'time_period' in df.columns:
        # Take the latest period for each user
        datasets['cross_sectional'] = (
            df.sort_values('time_period')
            .groupby('user_id')
            .last()
            .reset_index()
        )
    else:
        datasets['cross_sectional'] = df.copy()
    
    # 2. Panel dataset (for DiD, fixed effects)
    if 'time_period' in df.columns and 'user_id' in df.columns:
        datasets['panel'] = df.copy()
        # Ensure balanced panel for DiD
        user_periods = df.groupby('user_id')['time_period'].nunique()
        complete_users = user_periods[user_periods == df['time_period'].nunique()].index
        datasets['panel_balanced'] = df[df['user_id'].isin(complete_users)].copy()
    
    # 3. Before/After dataset (for DiD with clear intervention point)
    if 'time_period' in df.columns:
        # The intervention date must be supplied explicitly. Inferring it from
        # the data midpoint would invent the treatment timing.
        if intervention_date is None:
            raise ValueError(
                "intervention_date is required to build a before/after (DiD) "
                "dataset. Pass the real intervention/rollout date rather than "
                "letting the function guess it from the data."
            )
        df_did = df.copy()
        df_did['post_treatment'] = (df_did['time_period'] >= intervention_date).astype(int)
        datasets['before_after'] = df_did
    
    # Print dataset summaries
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} observations, {dataset['user_id'].nunique() if 'user_id' in dataset.columns else 'N/A'} users")
    
    return datasets

# Create analysis-ready datasets
# Supply the real intervention/rollout date (replace with your own).
analysis_datasets = create_analysis_datasets(
    processed_df, 'tickets_per_week', 'copilot_usage', available_confounders,
    intervention_date='2024-01-15'
)
```

---

## Best Practices Summary

### Data Quality Checklist

Before proceeding with causal analysis, ensure:

- ✅ **Temporal ordering**: Treatment occurs before outcome measurement
- ✅ **Sufficient variation**: Both treated and untreated observations exist
- ✅ **Complete outcomes**: Low missing data in key variables (<10%)
- ✅ **Adequate sample size**: At least 100 observations, preferably 500+
- ✅ **Balanced treatment**: Treatment prevalence between 10-90%
- ✅ **Measured confounders**: Key confounding variables are captured
- ✅ **Data consistency**: Variables measured consistently across time/groups

### Common Pitfalls to Avoid

1. **Selection bias in data collection**: Ensure representative sampling
2. **Measurement error**: Validate that variables capture intended constructs
3. **Time-varying confounders**: Account for confounders that change over time
4. **Spillover effects**: Consider whether treatment affects control group
5. **Survivorship bias**: Account for users who drop out of analysis

---

## Next Steps

Once your data is prepared and validated:

1. **Choose your causal method** based on data structure and research question
2. **Start with simple methods** (regression adjustment) before moving to complex ones
3. **Always check assumptions** specific to your chosen method
4. **Validate results** through robustness testing and sensitivity analysis

**Continue to:** [Regression Adjustment Method →]({{ site.baseurl }}/causal-inference-regression/)

---

*Proper data preparation is the foundation of reliable causal inference. Take time to understand your data structure and validate quality before proceeding with analysis.*
