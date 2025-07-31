"""
Treatment effect estimation models and utilities.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML, CausalForestDML
from custom_interpreter import CustomSingleTreeCateInterpreter
import logging
from config import ModelConfig

logger = logging.getLogger(__name__)

class TreatmentFeaturizer:
    """Factory class for treatment featurizers."""
    
    @staticmethod
    def create_featurizer(config: ModelConfig):
        """Create treatment featurizer based on configuration."""
        if config.featurizer_type == "spline":
            return SplineTransformer(
                degree=config.spline_degree,
                n_knots=config.spline_n_knots,
                include_bias=False,
                extrapolation='constant'
            )
        elif config.featurizer_type == "polynomial":
            return PolynomialFeatures(
                degree=config.poly_degree,
                interaction_only=config.poly_interaction_only,
                include_bias=False
            )
        else:
            raise ValueError(f"Unknown featurizer type: {config.featurizer_type}")

class TreatmentEffectEstimator:
    """Main class for treatment effect estimation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.featurizer = TreatmentFeaturizer.create_featurizer(config)
        self.estimator = None
        self.baseline_estimator = None
        self.is_fitted = False
        self.analysis_type = None
    
    def fit(self, Y: np.ndarray, T: np.ndarray, W: np.ndarray, X: Optional[np.ndarray] = None, analysis_type: str = "ate") -> None:
        """
        Fit the treatment effect estimator.
        
        Args:
            Y: Outcome variable
            T: Treatment variable
            W: Confounding variables
            X: Effect heterogeneity variables (optional)
            analysis_type: "ate" or "cate" to determine which estimator to use
        """
        try:
            if analysis_type == "ate":
                # For ATE: Use LinearDML
                self.estimator = LinearDML(
                    treatment_featurizer=self.featurizer,
                    cv=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                self.baseline_estimator = LinearDML(
                    cv=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                # Fit ATE estimators (no X variables)
                logger.info("Fitting ATE estimators using LinearDML")
                self.estimator.fit(Y=Y, T=T, W=W)
                self.baseline_estimator.fit(Y=Y, T=T, W=W)
                
            elif analysis_type == "cate":
                # For CATE: Use CausalForestDML (matching notebook after "Applied example")
                self.estimator = CausalForestDML(
                    model_y=RandomForestRegressor(
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    ),
                    model_t=RandomForestRegressor(
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    ),
                    treatment_featurizer=self.featurizer,
                    cv=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                
                # Create baseline without featurizer for comparison
                self.baseline_estimator = CausalForestDML(
                    model_y=RandomForestRegressor(
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    ),
                    model_t=RandomForestRegressor(
                        min_samples_leaf=self.config.min_samples_leaf,
                        random_state=self.config.random_state
                    ),
                    cv=self.config.cv_folds,
                    random_state=self.config.random_state
                )
                
                if X is None or X.shape[1] == 0:
                    raise ValueError("CATE analysis requires heterogeneity variables (X)")
                
                # Fit CATE estimators with X variables
                logger.info("Fitting CATE estimators using CausalForestDML")
                self.estimator.fit(Y=Y.flatten(), T=T, X=X, W=W)
                self.baseline_estimator.fit(Y=Y.flatten(), T=T, X=X, W=W)
            
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            self.is_fitted = True
            self.analysis_type = analysis_type
            logger.info(f"Estimators fitted successfully for {analysis_type.upper()} analysis")
            
        except Exception as e:
            logger.error(f"Error fitting estimators: {e}")
            raise
    
    def estimate_ate(self, T0: float = 0, T1_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Estimate Average Treatment Effect (ATE).
        
        Args:
            T0: Baseline treatment value
            T1_values: Treatment values to compare against baseline
            
        Returns:
            Dictionary containing ATE estimates and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before estimation")
        
        if T1_values is None:
            T1_values = np.arange(0, 8, 1).reshape(-1, 1)
        
        # Point estimates
        ate_featurized = self.estimator.effect(T0=T0, T1=T1_values)
        ate_baseline = self.baseline_estimator.effect(T0=T0, T1=T1_values)
        
        # Confidence intervals
        num_points = len(T1_values)
        lb_feat, ub_feat = self.estimator.effect_interval(
            T0=np.full((num_points, 1), T0), 
            T1=T1_values
        )
        lb_base, ub_base = self.baseline_estimator.effect_interval(
            T0=np.full((num_points, 1), T0), 
            T1=T1_values
        )
        
        return {
            'treatment_values': T1_values.flatten(),
            'ate_featurized': ate_featurized.flatten(),
            'ate_baseline': ate_baseline.flatten(),
            'ci_lower_featurized': lb_feat.flatten(),
            'ci_upper_featurized': ub_feat.flatten(),
            'ci_lower_baseline': lb_base.flatten(),
            'ci_upper_baseline': ub_base.flatten()
        }
    
    def estimate_cate(self, X: np.ndarray, T0: float = 0, T1_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Estimate Conditional Average Treatment Effect (CATE).
        
        Args:
            X: Covariates for heterogeneity
            T0: Baseline treatment value
            T1_values: Treatment values to compare against baseline
            
        Returns:
            Dictionary containing CATE estimates and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before estimation")
        
        if self.analysis_type != "cate":
            raise ValueError("Estimator must be fitted for CATE analysis")
        
        if T1_values is None:
            T1_values = np.arange(0, 8, 1).reshape(-1, 1)
        
        # Point estimates for each treatment value
        cate_results = {}
        
        for i, t1 in enumerate(T1_values.flatten()):
            # Estimate CATE for this treatment level
            cate_point = self.estimator.effect(X, T0=T0, T1=t1)
              # Get confidence intervals
            lb, ub = self.estimator.effect_interval(X, T0=T0, T1=t1)
            
            cate_results[f'T{t1}'] = {
                'treatment_value': t1,
                'cate_point': cate_point.flatten(),
                'ci_lower': lb.flatten(),            'ci_upper': ub.flatten()
            }
        
        return cate_results
    
    def interpret_cate(self, X: np.ndarray, feature_names: list, T: np.ndarray = None, person_ids: np.ndarray = None) -> CustomSingleTreeCateInterpreter:
        """
        Create CATE interpretation tree.
        
        Args:
            X: Covariates for heterogeneity
            feature_names: Names of features in X
            T: Treatment variable array (used to calculate percentiles for T0/T1)
            person_ids: PersonID array for unique person counting in tree nodes
            
        Returns:
            Fitted CATE interpreter
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before interpretation")
        
        try:
            # Calculate T0 (5th percentile) and T1 (95th percentile) from treatment data
            if T is not None:
                T0 = float(np.percentile(T.flatten(), 5))
                T1 = float(np.percentile(T.flatten(), 95))
                logger.info(f"Using T0={T0:.2f} (5th percentile) and T1={T1:.2f} (95th percentile)")
            else:
                # Fallback to default values if no treatment data provided
                T0, T1 = 1, 6
                logger.warning("No treatment data provided, using default T0=1, T1=6")
            
            interpreter = CustomSingleTreeCateInterpreter(
                include_model_uncertainty=True,
                max_depth=self.config.cate_max_depth,
                min_samples_leaf=self.config.cate_min_samples_leaf
            )
            interpreter.interpret(self.estimator, X, T0=T0, T1=T1, person_ids=person_ids)
            logger.info("CATE interpretation completed")
            return interpreter
            
        except Exception as e:
            logger.error(f"Error in CATE interpretation: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of fitted model."""
        if not self.is_fitted:
            return {"fitted": False}
        
        summary = {
            "fitted": True,
            "analysis_type": self.analysis_type,
            "estimator_type": "LinearDML" if self.analysis_type == "ate" else "CausalForestDML",
            "featurizer_type": self.config.featurizer_type,
            "cv_folds": self.config.cv_folds
        }
        
        # Add model-specific summary if available
        if hasattr(self.estimator, 'summary'):
            try:
                summary["econml_summary"] = str(self.estimator.summary())
            except Exception as e:
                logger.warning(f"Could not get model summary: {e}")
        
        return summary
