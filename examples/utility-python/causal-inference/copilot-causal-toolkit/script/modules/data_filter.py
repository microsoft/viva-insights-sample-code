"""
Data filtering and preprocessing utilities for causal inference analysis.

This module provides functions for:
- Date range filtering
- Treatment/outcome filtering (e.g., Copilot users only)
- Missing value handling
- Person-level aggregation of longitudinal data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def apply_date_filter(
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_column: str = 'MetricDate'
) -> pd.DataFrame:
    """
    Filter data to a specific date range.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with date column
    start_date : str, optional
        Start date in format 'YYYY-MM-DD' (e.g., '2025-01-01')
        If None, no lower bound is applied
    end_date : str, optional
        End date in format 'YYYY-MM-DD' (e.g., '2025-06-30')
        If None, no upper bound is applied
    date_column : str, default='MetricDate'
        Name of the date column to filter on
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
        
    Examples
    --------
    >>> filtered_data = apply_date_filter(data, '2025-01-01', '2025-06-30')
    >>> print(f"Kept {len(filtered_data)} of {len(data)} rows")
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    
    original_count = len(data)
    
    # Apply filters
    if start_date is None and end_date is None:
        logger.info("No date filter provided; keeping all data")
        return data
    
    mask = pd.Series(True, index=data.index)
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        mask &= (data[date_column] >= start_dt)
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        mask &= (data[date_column] <= end_dt)
    
    filtered_data = data[mask].copy()
    filtered_count = len(filtered_data)
    
    date_range_str = f"{start_date or 'beginning'} to {end_date or 'end'}"
    logger.info(
        f"Date filter applied: {date_range_str} "
        f"(kept {filtered_count:,}/{original_count:,}, {filtered_count/original_count:.1%})"
    )
    
    return filtered_data


def filter_by_treatment_threshold(
    data: pd.DataFrame,
    treatment_var: str,
    threshold: float = 0,
    operator: str = '>'
) -> pd.DataFrame:
    """
    Filter data based on treatment variable threshold.
    
    Common use case: Filter to only Copilot users (treatment > 0).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    treatment_var : str
        Name of treatment variable column
    threshold : float, default=0
        Threshold value for filtering
    operator : str, default='>'
        Comparison operator: '>', '>=', '<', '<=', '==', '!='
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
        
    Examples
    --------
    >>> # Keep only Copilot users
    >>> copilot_users = filter_by_treatment_threshold(
    ...     data, 'Total_Copilot_actions_taken', threshold=0, operator='>'
    ... )
    """
    original_count = len(data)
    
    # Apply filter based on operator
    if operator == '>':
        mask = data[treatment_var] > threshold
    elif operator == '>=':
        mask = data[treatment_var] >= threshold
    elif operator == '<':
        mask = data[treatment_var] < threshold
    elif operator == '<=':
        mask = data[treatment_var] <= threshold
    elif operator == '==':
        mask = data[treatment_var] == threshold
    elif operator == '!=':
        mask = data[treatment_var] != threshold
    else:
        raise ValueError(f"Invalid operator: {operator}")
    
    filtered_data = data[mask].copy()
    filtered_count = len(filtered_data)
    
    logger.info(
        f"Treatment filter applied: {treatment_var} {operator} {threshold} "
        f"(kept {filtered_count:,}/{original_count:,}, {filtered_count/original_count:.1%})"
    )
    
    return filtered_data


def filter_missing_values(
    data: pd.DataFrame,
    required_columns: List[str],
    strategy: str = 'drop'
) -> pd.DataFrame:
    """
    Handle missing values in required columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    required_columns : list of str
        Columns that must not have missing values
    strategy : str, default='drop'
        How to handle missing values:
        - 'drop': Remove rows with any missing values in required columns
        - 'warn': Just report missing values without dropping
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (if strategy='drop') or original (if strategy='warn')
    """
    # Check which columns actually exist
    existing_cols = [col for col in required_columns if col in data.columns]
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        logger.warning(f"Columns not found in data: {missing_cols}")
    
    if not existing_cols:
        logger.warning("No required columns found in data")
        return data
    
    # Check for missing values
    missing_counts = data[existing_cols].isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        logger.info("No missing values found in required columns")
        return data
    
    # Report missing values
    cols_with_missing = missing_counts[missing_counts > 0]
    logger.info(f"Missing values found in {len(cols_with_missing)} columns:")
    for col, count in cols_with_missing.items():
        pct = count / len(data) * 100
        logger.info(f"  • {col}: {count:,} ({pct:.1f}%)")
    
    if strategy == 'drop':
        original_count = len(data)
        clean_data = data.dropna(subset=existing_cols).copy()
        cleaned_count = len(clean_data)
        dropped_count = original_count - cleaned_count
        
        logger.info(
            f"Dropped {dropped_count:,} rows with missing values "
            f"(kept {cleaned_count:,}/{original_count:,}, {cleaned_count/original_count:.1%})"
        )
        return clean_data
    
    else:  # strategy == 'warn'
        return data


def aggregate_by_person(
    data: pd.DataFrame,
    person_id_var: str,
    outcome_var: str,
    treatment_var: str,
    numeric_controls: List[str],
    categorical_controls: List[str],
    date_column: str = 'MetricDate'
) -> pd.DataFrame:
    """
    Aggregate longitudinal data to person-level by taking means.
    
    This converts panel data (person-weeks) to cross-sectional data (person-level)
    by averaging all numeric variables over time for each person.
    
    Parameters
    ----------
    data : pd.DataFrame
        Longitudinal data with multiple rows per person
    person_id_var : str
        Column name for person identifier
    outcome_var : str
        Outcome variable to aggregate
    treatment_var : str
        Treatment variable to aggregate
    numeric_controls : list of str
        Numeric control variables to aggregate (mean)
    categorical_controls : list of str
        Categorical variables to keep (first value per person)
    date_column : str, default='MetricDate'
        Date column name (will be dropped after aggregation)
        
    Returns
    -------
    pd.DataFrame
        Person-level aggregated data with one row per person
        
    Notes
    -----
    - Numeric variables are averaged across all weeks
    - Categorical variables use the first observed value (assumed time-invariant)
    - Date column is dropped after aggregation
    - Resulting data has one row per unique person
    """
    # Validate required columns exist
    required_cols = [person_id_var, outcome_var, treatment_var]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Required columns not found: {missing}")
    
    # Build aggregation dictionary
    agg_dict = {
        treatment_var: 'mean',
        outcome_var: 'mean'
    }
    
    # Add numeric controls (take mean)
    for col in numeric_controls:
        if col in data.columns and col not in agg_dict:
            agg_dict[col] = 'mean'
    
    # Add categorical controls (take first value)
    for col in categorical_controls:
        if col in data.columns and col not in agg_dict:
            agg_dict[col] = 'first'
    
    # Group by person and aggregate
    logger.info(f"Aggregating {len(data):,} rows to person-level...")
    logger.info(f"  • Unique persons: {data[person_id_var].nunique():,}")
    logger.info(f"  • Aggregating {len(agg_dict)} variables")
    
    person_data = data.groupby(person_id_var, as_index=False).agg(agg_dict)
    
    logger.info(f"✓ Aggregated to {len(person_data):,} person-level observations")
    logger.info(f"  • Average weeks per person: {len(data)/len(person_data):.1f}")
    
    return person_data


def create_data_snapshot(
    data: pd.DataFrame,
    person_id_var: str = 'PersonId',
    outcome_var: str = 'After_hours_collaboration_hours',
    treatment_var: str = 'Total_Copilot_actions_taken',
    demographic_vars: Optional[List[str]] = None,
    network_vars: Optional[List[str]] = None,
    collaboration_vars: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filter_copilot_users: bool = True,
    date_column: str = 'MetricDate'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main orchestration function to create a clean, person-level data snapshot.
    
    This function applies all filtering and aggregation steps in sequence:
    1. Date filtering (optional)
    2. Copilot user filtering (optional)
    3. Missing value removal
    4. Person-level aggregation
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw longitudinal data
    person_id_var : str, default='PersonId'
        Person identifier column
    outcome_var : str, default='After_hours_collaboration_hours'
        Outcome variable
    treatment_var : str, default='Total_Copilot_actions_taken'
        Treatment variable
    demographic_vars : list of str, optional
        Demographic/organizational attributes
    network_vars : list of str, optional
        Network-related metrics
    collaboration_vars : list of str, optional
        Collaboration behavior metrics
    start_date : str, optional
        Start date for filtering (YYYY-MM-DD)
    end_date : str, optional
        End date for filtering (YYYY-MM-DD)
    filter_copilot_users : bool, default=True
        Whether to filter to only Copilot users (treatment > 0)
    date_column : str, default='MetricDate'
        Date column name
        
    Returns
    -------
    tuple of (pd.DataFrame, dict)
        - Person-level aggregated data
        - Metadata dictionary with filtering statistics
        
    Examples
    --------
    >>> data_snapshot, metadata = create_data_snapshot(
    ...     data=raw_data,
    ...     start_date='2025-01-01',
    ...     end_date='2025-06-30',
    ...     demographic_vars=['Organization', 'FunctionType'],
    ...     network_vars=['Internal_network_size'],
    ...     collaboration_vars=['Collaboration_hours']
    ... )
    >>> print(f"Created snapshot with {len(data_snapshot)} people")
    """
    logger.info("=== Creating Data Snapshot ===")
    
    # Initialize metadata
    metadata = {
        'original_rows': len(data),
        'original_persons': data[person_id_var].nunique(),
        'filters_applied': []
    }
    
    # Step 1: Date filtering
    if start_date or end_date:
        data = apply_date_filter(data, start_date, end_date, date_column)
        metadata['filters_applied'].append('date_range')
        metadata['date_range'] = {'start': start_date, 'end': end_date}
    
    # Step 2: Copilot user filtering
    if filter_copilot_users:
        data = filter_by_treatment_threshold(data, treatment_var, threshold=0, operator='>')
        metadata['filters_applied'].append('copilot_users')
    
    metadata['after_filtering_rows'] = len(data)
    metadata['after_filtering_persons'] = data[person_id_var].nunique()
    
    # Step 3: Prepare variable lists
    demographic_vars = demographic_vars or []
    network_vars = network_vars or []
    collaboration_vars = collaboration_vars or []
    
    numeric_controls = network_vars + collaboration_vars
    categorical_controls = demographic_vars
    
    # Remove duplicates and ensure columns exist
    numeric_controls = [col for col in list(set(numeric_controls)) if col in data.columns]
    categorical_controls = [col for col in list(set(categorical_controls)) if col in data.columns]
    
    # Step 4: Missing value handling
    required_cols = [outcome_var, treatment_var] + numeric_controls + categorical_controls
    data = filter_missing_values(data, required_cols, strategy='drop')
    
    metadata['after_dropping_missing_rows'] = len(data)
    metadata['after_dropping_missing_persons'] = data[person_id_var].nunique()
    
    # Step 5: Person-level aggregation
    person_data = aggregate_by_person(
        data=data,
        person_id_var=person_id_var,
        outcome_var=outcome_var,
        treatment_var=treatment_var,
        numeric_controls=numeric_controls,
        categorical_controls=categorical_controls,
        date_column=date_column
    )
    
    metadata['final_persons'] = len(person_data)
    metadata['variables_aggregated'] = len(numeric_controls) + len(categorical_controls) + 2  # +2 for treatment and outcome
    
    # Summary statistics
    logger.info("\n=== Data Snapshot Summary ===")
    logger.info(f"Original data: {metadata['original_rows']:,} rows, {metadata['original_persons']:,} persons")
    logger.info(f"After filtering: {metadata['after_filtering_rows']:,} rows, {metadata['after_filtering_persons']:,} persons")
    logger.info(f"After cleaning: {metadata['after_dropping_missing_rows']:,} rows, {metadata['after_dropping_missing_persons']:,} persons")
    logger.info(f"Final snapshot: {metadata['final_persons']:,} persons")
    logger.info(f"Filters applied: {', '.join(metadata['filters_applied']) if metadata['filters_applied'] else 'none'}")
    logger.info(f"Variables included: {metadata['variables_aggregated']}")
    
    return person_data, metadata


def get_data_summary_stats(
    data: pd.DataFrame,
    treatment_var: str,
    outcome_var: str
) -> Dict[str, Any]:
    """
    Calculate summary statistics for treatment and outcome variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Person-level data
    treatment_var : str
        Treatment variable name
    outcome_var : str
        Outcome variable name
        
    Returns
    -------
    dict
        Dictionary with summary statistics
    """
    summary = {}
    
    for var_name, var in [('treatment', treatment_var), ('outcome', outcome_var)]:
        if var in data.columns:
            values = data[var].dropna()
            summary[var_name] = {
                'variable': var,
                'n': len(values),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'q25': float(values.quantile(0.25)),
                'q50': float(values.quantile(0.50)),
                'q75': float(values.quantile(0.75)),
                'q95': float(values.quantile(0.95))
            }
    
    return summary
