"""
Data processing utilities for treatment effect estimation.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class for treatment effect estimation."""
    
    def __init__(self, treatment_var: str, outcome_var: str):
        """
        Initialize DataProcessor.
        
        Parameters
        ----------
        treatment_var : str
            Name of the treatment variable column
        outcome_var : str
            Name of the outcome variable column
        """
        self.treatment_var = treatment_var
        self.outcome_var = outcome_var
        self._data = None  # Store loaded data for auto-detection
    
    def load_data(self, data_file: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        data_file : str
            Path to the CSV file to load
            
        Returns
        -------
        pd.DataFrame
            Loaded and filtered data
        """
        try:
            data_path = Path(data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            data = pd.read_csv(data_path, sep=None, engine='python')

            # Apply 95th percentile threshold filter for treatment variable
            treatment_95th = np.percentile(data[self.treatment_var], 95)
            original_count = len(data)
            data = data[data[self.treatment_var] <= treatment_95th]
            filtered_count = len(data)
            
            logger.info(f"Applied 95th percentile filter: treatment <= {treatment_95th:.2f}")
            logger.info(f"Data filtered from {original_count} to {filtered_count} rows ({filtered_count/original_count*100:.1f}% retained)")

            self._data = data  # Store for auto-detection
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate data contains required columns."""
        pass  # Temporarily disabled validation
    
    def prepare_variables_ate(self, data: pd.DataFrame, 
                             w_numeric_vars: Optional[List[str]] = None,
                             w_categorical_vars: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare variables for ATE analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        w_numeric_vars : list of str, optional
            List of numeric confounding variable names
        w_categorical_vars : list of str, optional
            List of categorical confounding variable names
        
        Returns
        -------
        T : np.ndarray
            Treatment variable (n_samples, 1)
        Y : np.ndarray
            Outcome variable (n_samples,)
        W : np.ndarray
            Confounding variables
        """
        # Extract treatment and outcome
        T = data[self.treatment_var].values.reshape(-1, 1)
        Y = data[self.outcome_var].values.ravel()  # Flatten Y for ATE analysis
        
        # For ATE: Use explicitly defined confounding variables
        # Prepare numeric variables
        W_numeric = data[w_numeric_vars] if w_numeric_vars else pd.DataFrame()
        
        # One-hot encode categorical variables
        W_categorical = pd.DataFrame()
        if w_categorical_vars:
            for var in w_categorical_vars:
                encoded = pd.get_dummies(data[var], prefix=var, drop_first=True)
                W_categorical = pd.concat([W_categorical, encoded], axis=1)
        
        # Combine all confounding variables
        if not W_categorical.empty and not W_numeric.empty:
            W = pd.concat([W_numeric, W_categorical], axis=1).values
        elif not W_numeric.empty:
            W = W_numeric.values
        elif not W_categorical.empty:
            W = W_categorical.values
        else:
            raise ValueError("No confounding variables (W) specified for ATE analysis")
        
        logger.info(f"ATE variables prepared - T: {T.shape}, Y: {Y.shape}, W: {W.shape}")
        
        # Log variable names for clarity
        W_names = (list(w_numeric_vars) if w_numeric_vars else []) + (list(W_categorical.columns) if not W_categorical.empty else [])
        logger.info(f"ATE W variables: {W_names}")
        
        return T, Y, W
    
    def prepare_variables_cate(self, data: pd.DataFrame,
                               w_numeric_vars: Optional[List[str]] = None,
                               w_categorical_vars: Optional[List[str]] = None,
                               x_numeric_vars: Optional[List[str]] = None,
                               x_categorical_vars: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare variables for CATE analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        w_numeric_vars : list of str, optional
            List of numeric confounding variable names
        w_categorical_vars : list of str, optional
            List of categorical confounding variable names
        x_numeric_vars : list of str, optional
            List of numeric heterogeneity variable names
        x_categorical_vars : list of str, optional
            List of categorical heterogeneity variable names
        
        Returns
        -------
        T : np.ndarray
            Treatment variable (n_samples, 1)
        Y : np.ndarray
            Outcome variable (n_samples,)
        W : np.ndarray
            Confounding variables
        X : np.ndarray
            Effect heterogeneity variables
        """
        # Extract treatment and outcome
        T = data[self.treatment_var].values.reshape(-1, 1)
        Y = data[self.outcome_var].values.ravel()  # Flatten Y for CATE analysis
        
        # Prepare confounding variables (W) for CATE
        W_numeric = data[w_numeric_vars] if w_numeric_vars else pd.DataFrame()
        
        # W categorical variables
        W_categorical = pd.DataFrame()
        if w_categorical_vars:
            for var in w_categorical_vars:
                unique_values = data[var].nunique()
                
                if unique_values == 2:
                    # Binary variable: encode as single column (0/1)
                    encoded_values = pd.Categorical(data[var]).codes
                    # Get the category that corresponds to code 1
                    categories = pd.Categorical(data[var]).categories
                    value_for_1 = categories[1] if len(categories) > 1 else categories[0]
                    column_name = f"{var}_{value_for_1}"
                    encoded = pd.DataFrame({column_name: encoded_values})
                else:
                    # Multi-category variable: use one-hot encoding with drop_first=False
                    encoded = pd.get_dummies(data[var], prefix=var, drop_first=False)
                
                W_categorical = pd.concat([W_categorical, encoded], axis=1)

        # Combine W variables
        if not W_categorical.empty and not W_numeric.empty:
            W = pd.concat([W_numeric, W_categorical], axis=1).values
        elif not W_numeric.empty:
            W = W_numeric.values
        elif not W_categorical.empty:
            W = W_categorical.values
        else:
            raise ValueError("No confounding variables (W) specified for CATE analysis")
            
        # Prepare effect heterogeneity variables (X)
        X_numeric = data[x_numeric_vars] if x_numeric_vars else pd.DataFrame()
        
        # X categorical variables  
        X_categorical = pd.DataFrame()
        if x_categorical_vars:
            X_categorical = pd.get_dummies(data[x_categorical_vars], drop_first=False)
        
        # Combine X variables
        if not X_categorical.empty and not X_numeric.empty:
            X = pd.concat([X_numeric, X_categorical], axis=1).values
        elif not X_numeric.empty:
            X = X_numeric.values
        elif not X_categorical.empty:
            X = X_categorical.values
        else:
            raise ValueError("No effect heterogeneity variables (X) specified for CATE analysis")
            
        logger.info(f"CATE variables prepared - T: {T.shape}, Y: {Y.shape}, W: {W.shape}, X: {X.shape}")
        
        # Log variable names for clarity
        W_names = (list(w_numeric_vars) if w_numeric_vars else []) + (list(W_categorical.columns) if not W_categorical.empty else [])
        X_names = (list(x_numeric_vars) if x_numeric_vars else []) + (list(X_categorical.columns) if not X_categorical.empty else [])
        logger.info(f"CATE W variables: {W_names}")
        logger.info(f"CATE X variables: {X_names}")
        
        return T, Y, W, X
    
    def get_feature_names_cate(self, data: pd.DataFrame,
                               w_numeric_vars: Optional[List[str]] = None,
                               w_categorical_vars: Optional[List[str]] = None,
                               x_numeric_vars: Optional[List[str]] = None,
                               x_categorical_vars: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Get feature names for CATE analysis (W and X variables).
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
        w_numeric_vars : list of str, optional
            List of numeric confounding variable names
        w_categorical_vars : list of str, optional
            List of categorical confounding variable names
        x_numeric_vars : list of str, optional
            List of numeric heterogeneity variable names
        x_categorical_vars : list of str, optional
            List of categorical heterogeneity variable names
            
        Returns
        -------
        tuple of (list, list)
            (W_feature_names, X_feature_names)
        """
        # W feature names
        W_numeric_names = list(w_numeric_vars) if w_numeric_vars else []
        W_categorical_names = []
        
        if w_categorical_vars:
            W_categorical_encoded = pd.get_dummies(data[w_categorical_vars], drop_first=False)
            W_categorical_names = list(W_categorical_encoded.columns)
        
        W_feature_names = W_numeric_names + W_categorical_names
        
        # X feature names
        X_numeric_names = list(x_numeric_vars) if x_numeric_vars else []
        X_categorical_names = []
        
        if x_categorical_vars:
            X_categorical_encoded = pd.get_dummies(data[x_categorical_vars], drop_first=False)
            X_categorical_names = list(X_categorical_encoded.columns)
        
        X_feature_names = X_numeric_names + X_categorical_names
        
        return W_feature_names, X_feature_names
    
    def get_treatment_grid(self, num_intervals: int = 10) -> np.ndarray:
        """
        Generate treatment values for effect estimation.
        
        Parameters
        ----------
        num_intervals : int, optional
            Number of intervals to split treatment range into (default: 10)
            
        Returns
        -------
        np.ndarray
            Array of treatment values
        """        
        # Calculate treatment range from actual data
        if hasattr(self, '_data') and self._data is not None:
            treatment = self._data[self.treatment_var]
            treatment_min = float(treatment.min())
            treatment_max = float(treatment.max())
            
            # Calculate step size based on number of intervals
            treatment_step = (treatment_max - treatment_min) / num_intervals
            
            logger.info(f"Auto-calculated treatment range: {treatment_min:.2f} to {treatment_max:.2f} "
                       f"with {num_intervals} intervals (step size: {treatment_step:.3f})")
        else:
            # Fallback: use sensible defaults if no data available
            treatment_min = 0
            treatment_max = 10
            treatment_step = (treatment_max - treatment_min) / num_intervals
            logger.warning(f"No data available for treatment range calculation. "
                          f"Using fallback range: {treatment_min} to {treatment_max} "
                          f"with {num_intervals} intervals")
        
        return np.arange(
            treatment_min, 
            treatment_max + treatment_step, 
            treatment_step
        ).reshape(-1, 1)
    
    def get_summary_stats(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the treatment variable.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input dataset
            
        Returns
        -------
        dict
            Dictionary with statistics (min, max, mean, std, percentiles)
        """
        treatment = data[self.treatment_var]
        
        stats = {
            'min': float(treatment.min()),
            'max': float(treatment.max()),
            'mean': float(treatment.mean()),
            'std': float(treatment.std()),
            'percentiles': {}
        }
        
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            stats['percentiles'][f'p{p}'] = float(np.percentile(treatment, p))
        
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
        if 'PersonId' not in data.columns:
            raise ValueError("Dataset must contain 'PersonID' column for user count calculation")
        
        unique_count = data['PersonId'].nunique()
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
        if 'PersonId' not in data.columns:
            raise ValueError("Dataset must contain 'PersonID' column")
        
        unique_users = data['PersonId'].nunique()
        total_rows = len(data)
        records_per_user = data['PersonId'].value_counts()
        
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
