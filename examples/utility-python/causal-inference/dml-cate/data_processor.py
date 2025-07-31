"""
Data processing utilities for treatment effect estimation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from pathlib import Path
import logging
from config import DataConfig

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class for treatment effect estimation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._data = None  # Store loaded data for auto-detection
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data_path = Path(self.config.data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            data = pd.read_csv(data_path)
            self._data = data  # Store for auto-detection
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate data contains required columns."""
        required_cols = [self.config.treatment_var, self.config.outcome_var]
        required_cols.extend(self.config.w_numeric_vars)
        required_cols.extend(self.config.x_numeric_vars)
        required_cols.extend(self.config.w_categorical_vars)
        required_cols.extend(self.config.x_categorical_vars)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
          # Check for missing values in critical columns
        critical_cols = [self.config.treatment_var, self.config.outcome_var]
        for col in critical_cols:
            if data[col].isnull().sum() > 0:
                logger.warning(f"Missing values found in {col}: {data[col].isnull().sum()}")
    
    def prepare_variables_ate(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare variables for ATE analysis (matching notebook before "Applied example").
        
        Returns:
            T: Treatment variable (n_samples, 1)
            Y: Outcome variable (n_samples, 1)
            W: Confounding variables (all variables except treatment, outcome, PersonID)
        """        # Extract treatment and outcome
        T = data[self.config.treatment_var].values.reshape(-1, 1)
        Y = data[self.config.outcome_var].values.ravel()  # Flatten Y for ATE analysis
        
        # For ATE: Use all variables except PersonID, treatment, outcome as confounders
        # This matches the original notebook approach
        exclude_cols = ["PersonID", "GitHub Copilot Usage", "M365 Copilot Usage", 
                       "Teams Copilot Usage", "Coding Productivity"]
        
        # Get all remaining columns
        remaining_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Separate categorical and numeric columns
        categorical_cols = ["LevelDesignation", "Region", "Function", "Org"]
        numeric_cols = [col for col in remaining_cols if col not in categorical_cols]
        
        # Prepare numeric variables
        W_numeric = data[numeric_cols] if numeric_cols else pd.DataFrame()
        
        # One-hot encode categorical variables
        W_categorical = pd.get_dummies(
            data[categorical_cols], 
            drop_first=False
        ) if categorical_cols else pd.DataFrame()
        
        # Combine all confounding variables
        W = pd.concat([W_numeric, W_categorical], axis=1).values
        
        logger.info(f"ATE variables prepared - T: {T.shape}, Y: {Y.shape}, W: {W.shape}")
        
        return T, Y, W
    
    def prepare_variables_cate(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare variables for CATE analysis (matching notebook after "Applied example").
        
        Returns:
            T: Treatment variable (n_samples, 1)
            Y: Outcome variable (n_samples, 1)
            W: Confounding variables (specific subset)
            X: Effect heterogeneity variables (specific subset)
        """
        # Extract treatment and outcome
        T = data[self.config.treatment_var].values.reshape(-1, 1)
        Y = data[self.config.outcome_var].values.ravel()  # Flatten Y for CATE analysis
        
        # Prepare confounding variables (W) - specific subset from notebook
        W_numeric = data[['External Network Size', 'Strong Ties', 'Diverse Ties',
                         'External Collaboration Hours', 'After-hours Meeting Hours',
                         'After-hours Email Hours', 'Available-to-focus Hours',
                         'Calendared Learning Time', 'Active Connected Hours',
                         'External 1:1 Meeting Hours', 'Weekend Collaboration Hours',
                         'Uninterrupted Hours']]
        
        # W categorical variables
        cols_to_encode_W = ["Region", "Org"]
        W_categorical = pd.get_dummies(data[cols_to_encode_W], drop_first=False)
        
        # Combine W variables
        W = pd.concat([W_numeric, W_categorical], axis=1).values
        
        # Prepare effect heterogeneity variables (X)
        X_numeric = data[["Internal Network Size"]]
        
        # X categorical variables
        cols_to_encode_X = ["LevelDesignation", "Function"]
        X_categorical = pd.get_dummies(data[cols_to_encode_X], drop_first=False)
        
        # Combine X variables
        X = pd.concat([X_numeric, X_categorical], axis=1).values
        
        logger.info(f"CATE variables prepared - T: {T.shape}, Y: {Y.shape}, W: {W.shape}, X: {X.shape}")
        
        return T, Y, W, X
    
    def get_feature_names_cate(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get feature names for CATE analysis (W and X variables)."""
        # W feature names
        W_numeric_names = ['External Network Size', 'Strong Ties', 'Diverse Ties',
                          'External Collaboration Hours', 'After-hours Meeting Hours',
                          'After-hours Email Hours', 'Available-to-focus Hours',
                          'Calendared Learning Time', 'Active Connected Hours',
                          'External 1:1 Meeting Hours', 'Weekend Collaboration Hours',
                          'Uninterrupted Hours']
        
        W_categorical_names = []
        cols_to_encode_W = ["Region", "Org"]
        if cols_to_encode_W:
            W_categorical_encoded = pd.get_dummies(data[cols_to_encode_W], drop_first=False)
            W_categorical_names = list(W_categorical_encoded.columns)
        
        W_feature_names = W_numeric_names + W_categorical_names
          # X feature names
        X_numeric_names = ["Internal Network Size"]
        X_categorical_names = []
        cols_to_encode_X = ["LevelDesignation", "Function"]
        if cols_to_encode_X:
            X_categorical_encoded = pd.get_dummies(data[cols_to_encode_X], drop_first=False)
            X_categorical_names = list(X_categorical_encoded.columns)
        
        X_feature_names = X_numeric_names + X_categorical_names
        
        return W_feature_names, X_feature_names
    
    def get_treatment_grid(self) -> np.ndarray:
        """Generate treatment values for effect estimation."""
        # If treatment range parameters are at default values, auto-detect from data
        if hasattr(self, '_data') and self._data is not None:
            treatment = self._data[self.config.treatment_var]
            actual_min = float(treatment.min())
            actual_max = float(treatment.max())
            
            # Use actual data range if defaults haven't been explicitly overridden
            if (self.config.treatment_min == 0 and self.config.treatment_max == 8 and 
                self.config.treatment_step == 1):
                # Auto-detect: use actual min/max with integer steps
                treatment_min = int(actual_min)
                treatment_max = int(actual_max)
                treatment_step = 1
                logger.info(f"Auto-detected treatment range: {treatment_min} to {treatment_max} "
                           f"(data range: {actual_min:.2f} to {actual_max:.2f})")
            else:
                # Use configured values (user has explicitly set them)
                treatment_min = self.config.treatment_min
                treatment_max = self.config.treatment_max
                treatment_step = self.config.treatment_step
        else:
            # Fallback to configured values if no data available
            treatment_min = self.config.treatment_min
            treatment_max = self.config.treatment_max
            treatment_step = self.config.treatment_step
        
        return np.arange(
            treatment_min, 
            treatment_max + treatment_step, 
            treatment_step
        ).reshape(-1, 1)
    
    def get_summary_stats(self, data: pd.DataFrame) -> Dict:
        """Get summary statistics for the treatment variable."""
        treatment = data[self.config.treatment_var]
        
        stats = {
            'min': float(treatment.min()),
            'max': float(treatment.max()),
            'mean': float(treatment.mean()),
            'std': float(treatment.std()),
            'percentiles': {}
        }
        
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:        stats['percentiles'][f'p{p}'] = float(np.percentile(treatment, p))
        
        return stats
    
    def get_unique_user_count(self, data: pd.DataFrame) -> int:
        """
        Get the number of unique PersonIDs in the dataset.
        
        This method handles both single-record-per-person datasets and 
        multi-week/longitudinal datasets where one person can have multiple rows.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing PersonID column
            
        Returns
        -------
        int
            Number of unique persons (not total rows)
        """
        if 'PersonID' not in data.columns:
            raise ValueError("Dataset must contain 'PersonID' column for user count calculation")
        
        unique_count = data['PersonID'].nunique()
        total_rows = len(data)
        
        # Log information about data structure
        logger.info(f"Dataset contains {total_rows} total records for {unique_count} unique users")
        if total_rows > unique_count:
            logger.info(f"Detected longitudinal data: {total_rows/unique_count:.1f} records per user on average")
        
        return unique_count
    
    def get_user_data_structure_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed information about the user data structure.
        
        This is useful for understanding whether we have longitudinal data
        (multiple weeks per user) or cross-sectional data (one record per user).
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing PersonID column
            
        Returns
        -------
        dict
            Dictionary containing structure information
        """
        if 'PersonID' not in data.columns:
            raise ValueError("Dataset must contain 'PersonID' column")
        
        unique_users = data['PersonID'].nunique()
        total_rows = len(data)
        records_per_user = data['PersonID'].value_counts()
        
        structure_info = {
            'total_records': total_rows,
            'unique_users': unique_users,
            'avg_records_per_user': total_rows / unique_users,
            'min_records_per_user': records_per_user.min(),
            'max_records_per_user': records_per_user.max(),
            'is_longitudinal': total_rows > unique_users,
            'data_type': 'longitudinal' if total_rows > unique_users else 'cross_sectional'
        }
        
        # Add distribution of records per user if longitudinal
        if structure_info['is_longitudinal']:
            structure_info['records_per_user_distribution'] = {
                'mean': float(records_per_user.mean()),
                'median': float(records_per_user.median()),
                'std': float(records_per_user.std())
            }
        
        return structure_info
