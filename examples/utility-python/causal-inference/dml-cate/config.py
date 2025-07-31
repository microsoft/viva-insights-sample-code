"""
Configuration module for treatment effect estimation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Treatment featurizer parameters
    featurizer_type: str = "spline"  # "spline" or "polynomial"
    spline_degree: int = 3
    spline_n_knots: int = 5
    poly_degree: int = 2
    poly_interaction_only: bool = False
    
    # Model parameters
    estimator_type: str = "linear_dml"  # "linear_dml" or "causal_forest_dml"
    cv_folds: int = 5
    random_state: int = 123
    min_samples_leaf: int = 20
    
    # CATE interpretation parameters
    cate_max_depth: int = 3
    cate_min_samples_leaf: int = 10

@dataclass
class DataConfig:
    """Configuration for data processing."""
    # File paths
    data_file: str = "data/synthetic_employees_data_v32.csv"
    output_dir: str = "outputs"
    
    # Treatment and outcome variables
    treatment_var: str = "Teams Copilot Usage"
    outcome_var: str = "Coding Productivity"
    
    # Feature configurations
    w_numeric_vars: List[str] = None  # Confounding variables (numeric)
    x_numeric_vars: List[str] = None  # Effect heterogeneity variables (numeric)
    w_categorical_vars: List[str] = None  # Confounding variables (categorical)
    x_categorical_vars: List[str] = None  # Effect heterogeneity variables (categorical)
    
    # Treatment effect evaluation range
    treatment_min: float = 0
    treatment_max: float = 8
    treatment_step: float = 1
    
    def __post_init__(self):
        """Set default values for feature lists."""
        if self.w_numeric_vars is None:
            self.w_numeric_vars = [
                'External Network Size', 'Strong Ties', 'Diverse Ties',
                'External Collaboration Hours', 'After-hours Meeting Hours',
                'After-hours Email Hours', 'Available-to-focus Hours',
                'Calendared Learning Time', 'Active Connected Hours',
                'External 1:1 Meeting Hours', 'Weekend Collaboration Hours',
                'Uninterrupted Hours'
            ]
        
        if self.x_numeric_vars is None:
            self.x_numeric_vars = ["Internal Network Size"]
        
        if self.w_categorical_vars is None:
            self.w_categorical_vars = ["Region", "Org"]
        
        if self.x_categorical_vars is None:
            self.x_categorical_vars = ["LevelDesignation", "Function"]

@dataclass
class OutputConfig:
    """Configuration for output generation."""
    save_csv: bool = True
    save_json: bool = True
    save_plots: bool = True
    plot_width: float = 10
    plot_height: float = 6
    confidence_alpha: float = 0.1
    group_user_count: int = 1000

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls(
            model=ModelConfig(),
            data=DataConfig(),
            output=OutputConfig()
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )
