"""
Output generation and visualization utilities.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class OutputManager:
    """Manages output generation for treatment effect analysis."""
    
    def __init__(self, output_dir: str, treatment_var: str, outcome_var: str,
                 save_csv: bool = True, save_json: bool = True, save_plots: bool = True,
                 plot_width: float = 10, plot_height: float = 6):
        """
        Initialize OutputManager.
        
        Parameters
        ----------
        output_dir : str
            Directory path for saving outputs
        treatment_var : str
            Name of treatment variable
        outcome_var : str
            Name of outcome variable
        save_csv : bool
            Whether to save results as CSV files (default: True)
        save_json : bool
            Whether to save results as JSON files (default: True)
        save_plots : bool
            Whether to save plots as PNG files (default: True)
        plot_width : float
            Width of plots in inches (default: 10)
        plot_height : float
            Height of plots in inches (default: 6)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.treatment_var = treatment_var
        self.outcome_var = outcome_var
        self.save_csv = save_csv
        self.save_json = save_json
        self.save_plots = save_plots
        self.plot_width = plot_width
        self.plot_height = plot_height
    
    def _get_timestamp(self) -> str:
        """Generate timestamp string for file names (YYYYMMDD_HHMM format)."""
        return datetime.now().strftime("%Y%m%d_%H%M")
    
    def save_ate_results(self, results: Dict[str, Any], treatment_name: str = None, user_count: int = None) -> None:
        """
        Save ATE estimation results.
        
        Parameters
        ----------
        results : dict
            Dictionary containing ATE results with keys: treatment_values, ate_featurized, etc.
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
        user_count : int, optional
            Number of unique users in analysis (required for JSON output)
        """
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        if user_count is None:
            raise ValueError("user_count parameter is required. Calculate using data_processor.get_unique_user_count(data)")
        
        # Create DataFrame for results - add p-values if available
        df_data = {
            'Treatment': results['treatment_values'],
            'ATE_Featurized': results['ate_featurized'],
            'ATE_Baseline': results['ate_baseline'],
            'CI_Lower_Featurized': results['ci_lower_featurized'],
            'CI_Upper_Featurized': results['ci_upper_featurized'],
            'CI_Lower_Baseline': results['ci_lower_baseline'],
            'CI_Upper_Baseline': results['ci_upper_baseline']
        }
        
        # Add p-values if they exist in results
        if 'p_values_featurized' in results:
            df_data['P_Value_Featurized'] = results['p_values_featurized']
        if 'p_values_baseline' in results:
            df_data['P_Value_Baseline'] = results['p_values_baseline']
        
        df_effects = pd.DataFrame(df_data)
        
        # Round numeric columns to 3 decimal places
        numeric_columns = ['ATE_Featurized', 'ATE_Baseline', 'CI_Lower_Featurized', 
                          'CI_Upper_Featurized', 'CI_Lower_Baseline', 'CI_Upper_Baseline']
        df_effects[numeric_columns] = df_effects[numeric_columns].round(3)
        
        # Round p-values to 4 decimal places if they exist
        p_value_columns = [col for col in df_effects.columns if 'P_Value' in col]
        if p_value_columns:
            df_effects[p_value_columns] = df_effects[p_value_columns].round(4)
        
        if self.save_csv:
            timestamp = self._get_timestamp()
            csv_path = self.output_dir / f'ate_results_{treatment_name.replace(" ", "_")}_{timestamp}.csv'
            df_effects.to_csv(csv_path, index=False)
            logger.info(f"ATE results saved to {csv_path}")
        
        if self.save_json:
            # Create JSON in the required format - exclude meaningless (0,0) comparison
            json_list = []
            for _, row in df_effects.iterrows():
                if int(row["Treatment"]) > 0:  # Exclude StartValue=0, EndValue=0 case
                    json_item = {
                        "TreatmentName": treatment_name,
                        "StartValue": 0,
                        "EndValue": int(row["Treatment"]),
                        "GroupId": "",
                        "GroupFilterCondition": "",
                        "GroupUserCount": user_count,
                        "TreatmentEffect": round(float(row["ATE_Featurized"]), 3),
                        "ConfidenceInterval": {
                            "Lower": round(float(row["CI_Lower_Featurized"]), 3),
                            "Upper": round(float(row["CI_Upper_Featurized"]), 3)
                        }
                    }
                    # Add p-value if available and not NaN
                    if 'P_Value_Featurized' in row and not pd.isna(row['P_Value_Featurized']):
                        json_item["P_Value"] = round(float(row["P_Value_Featurized"]), 4)
                    
                    json_list.append(json_item)
            
            timestamp = self._get_timestamp()
            json_path = self.output_dir / f'ate_results_{treatment_name.replace(" ", "_")}_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(json_list, f, indent=2)
            logger.info(f"ATE results saved to {json_path}")
    
    def save_cate_results_with_subgroups(self, interpreter, feature_names: List[str], treatment_name: str = None, user_count: int = None) -> None:
        """
        Save CATE estimation results with tree-based subgroups.
        
        Parameters
        ----------
        interpreter : object
            SingleTreePolicyInterpreter object with tree-based subgroups
        feature_names : list of str
            Names of features used in the treatment effect model
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
        user_count : int, optional
            Total number of unique users in dataset (required for JSON output)
        """
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        if user_count is None:
            raise ValueError("user_count parameter is required. Calculate using data_processor.get_unique_user_count(data)")
        
        try:
            # Extract subgroups from the decision tree
            subgroups = interpreter.get_tree_subgroups(feature_names)
            
            # Get T0 and T1 values from interpreter
            T0 = getattr(interpreter, 'T0_', 0)
            T1 = getattr(interpreter, 'T1_', 1)
            
            if self.save_csv:
                # Create DataFrame for CSV export with p-values if available
                df_subgroups = pd.DataFrame([
                    {
                        'GroupId': sg['group_id'],
                        'FilterCondition': sg['filter_condition'],
                        'UserCount': sg['user_count'],
                        'TreatmentEffect': sg['treatment_effect'],
                        'TreatmentEffectStd': sg['treatment_effect_std'],
                        'CI_Lower': sg.get('ci_lower', 0.0),
                        'CI_Upper': sg.get('ci_upper', 0.0),
                        'P_Value': sg.get('p_value', np.nan),
                        'T0': T0,
                        'T1': T1
                    }
                    for sg in subgroups
                ])
                
                # Round numeric columns to 3 decimal places
                numeric_columns = ['TreatmentEffect', 'TreatmentEffectStd', 'CI_Lower', 'CI_Upper']
                df_subgroups[numeric_columns] = df_subgroups[numeric_columns].round(3)
                
                # Round p-values to 4 decimal places if they exist
                if 'P_Value' in df_subgroups.columns:
                    df_subgroups['P_Value'] = df_subgroups['P_Value'].round(4)
                
                timestamp = self._get_timestamp()
                csv_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}_{timestamp}.csv'
                df_subgroups.to_csv(csv_path, index=False)
                logger.info(f"CATE subgroup results saved to {csv_path}")
            
            if self.save_json:
                # Create JSON in the required format
                json_list = []
                for sg in subgroups:
                    if sg['user_count'] > 0:  # Only include subgroups with users
                        json_item = {
                            "TreatmentName": treatment_name,
                            "StartValue": float(T0),
                            "EndValue": float(T1),
                            "GroupId": sg['group_id'],
                            "GroupFilterCondition": sg['filter_condition'],
                            "GroupUserCount": sg['user_count'],
                            "TreatmentEffect": round(sg['treatment_effect'], 3),
                            "TreatmentEffectStd": round(sg['treatment_effect_std'], 3),
                            "ConfidenceInterval": {
                                "Lower": round(sg.get('ci_lower', 0.0), 3),
                                "Upper": round(sg.get('ci_upper', 0.0), 3)
                            }
                        }
                        # Add p-value if available
                        if 'p_value' in sg and not np.isnan(sg['p_value']):
                            json_item["P_Value"] = round(sg['p_value'], 4)
                        
                        json_list.append(json_item)
                
                timestamp = self._get_timestamp()
                json_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}_{timestamp}.json'
                with open(json_path, 'w') as f:
                    json.dump(json_list, f, indent=2)
                logger.info(f"CATE subgroup results saved to {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving CATE subgroup results: {e}")
            # Fallback to original method
            logger.info("Falling back to aggregated CATE results")
            self.save_cate_results({}, treatment_name, user_count=user_count)

    def save_cate_results(self, results: Dict[str, Any], treatment_name: str = None, 
                         subgroup_info: Dict = None, user_count: int = None) -> None:
        """
        Save CATE estimation results (legacy method for backward compatibility).
        
        Parameters
        ----------
        results : dict
            Dictionary with treatment values and CATE predictions
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
        subgroup_info : dict, optional
            Information about subgroups (not currently used)
        user_count : int, optional
            Total number of unique users in dataset (required)
        """
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        if user_count is None:
            raise ValueError("user_count parameter is required. Calculate using data_processor.get_unique_user_count(data)")
        
        # Aggregate results across treatment values
        aggregated_results = []
        
        for t_key, t_results in results.items():
            treatment_value = t_results['treatment_value']
            mean_cate = np.mean(t_results['cate_point'])
            std_cate = np.std(t_results['cate_point'])
            mean_ci_lower = np.mean(t_results['ci_lower'])
            mean_ci_upper = np.mean(t_results['ci_upper'])
            
            aggregated_results.append({
                'Treatment': treatment_value,
                'Mean_CATE': mean_cate,
                'Std_CATE': std_cate,
                'Mean_CI_Lower': mean_ci_lower,
                'Mean_CI_Upper': mean_ci_upper,
                'N_Observations': len(t_results['cate_point'])
            })
        
        df_cate = pd.DataFrame(aggregated_results)
        
        if self.save_csv:
            timestamp = self._get_timestamp()
            csv_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}_{timestamp}.csv'
            df_cate.to_csv(csv_path, index=False)
            logger.info(f"CATE results saved to {csv_path}")
        
        if self.save_json:
            json_list = [
                {
                    "TreatmentName": treatment_name,
                    "StartValue": 0,
                    "EndValue": int(row["Treatment"]),
                    "GroupId": subgroup_info.get("group_id", "") if subgroup_info else "",
                    "GroupFilterCondition": subgroup_info.get("filter_condition", "") if subgroup_info else "",
                    "GroupUserCount": subgroup_info.get("user_count", user_count) if subgroup_info else user_count,
                    "TreatmentEffect": float(row["Mean_CATE"]),
                    "TreatmentEffectStd": float(row["Std_CATE"]),
                    "ConfidenceInterval": {
                        "Lower": float(row["Mean_CI_Lower"]),
                        "Upper": float(row["Mean_CI_Upper"])
                    }
                }
                for _, row in df_cate.iterrows()
            ]
            
            timestamp = self._get_timestamp()
            json_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}_{timestamp}.json'
            with open(json_path, 'w') as f:
                json.dump(json_list, f, indent=2)
            logger.info(f"CATE results saved to {json_path}")
    
    def create_ate_plot(self, results: Dict[str, Any], treatment_name: str = None) -> Optional[str]:
        """
        Create and save ATE visualization.
        
        Parameters
        ----------
        results : dict
            Dictionary containing ATE results with keys: treatment_values, ate_featurized, etc.
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
            
        Returns
        -------
        str or None
            Path to saved plot file if save_plots is True, None otherwise
        """
        if not self.save_plots:
            return None
        
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        plt.figure(figsize=(self.plot_width, self.plot_height))
        
        treatment_values = results['treatment_values']
        
        # Plot point estimates
        plt.plot(
            treatment_values,
            results['ate_featurized'],
            label='ATE with Featurized Treatment',
            linewidth=3,
            color='blue'
        )
        
        plt.plot(
            treatment_values,
            results['ate_baseline'],
            label='ATE without Featurized Treatment',
            linewidth=2,
            color='red',
            linestyle='--'
        )
        
        # Plot confidence intervals
        plt.fill_between(
            treatment_values,
            results['ci_lower_featurized'],
            results['ci_upper_featurized'],
            alpha=0.4,
            color='blue',
            label='95% CI (Featurized)'
        )
        
        plt.fill_between(
            treatment_values,
            results['ci_lower_baseline'],
            results['ci_upper_baseline'],
            alpha=0.4,
            color='red',
            label='95% CI (Baseline)'
        )
        
        plt.xlabel('Treatment Value')
        plt.ylabel('Treatment Effect (Y(treatment) - Y(0))')
        plt.title(f'Average Treatment Effect vs {treatment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = self._get_timestamp()
        plot_path = self.output_dir / f'ate_plot_{treatment_name.replace(" ", "_")}_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ATE plot saved to {plot_path}")
        return str(plot_path)
    
    def create_cate_interpretation_plot(self, interpreter, feature_names: List[str], 
                                      treatment_name: str = None) -> Optional[str]:
        """
        Create and save CATE interpretation tree plot.
        
        Parameters
        ----------
        interpreter : object
            SingleTreePolicyInterpreter object for visualization
        feature_names : list of str
            Names of features used in the treatment effect model
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
            
        Returns
        -------
        str or None
            Path to saved plot file if successful, None otherwise
        """
        if not self.save_plots:
            return None
        
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        try:
            plt.figure(figsize=(15, 10))
            interpreter.plot(feature_names=feature_names, fontsize=12)
            
            timestamp = self._get_timestamp()
            plot_path = self.output_dir / f'cate_interpretation_{treatment_name.replace(" ", "_")}_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"CATE interpretation plot saved to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.warning(f"Could not create CATE interpretation plot: {e}")
            return None
    
    def save_summary_report(self, summary_data: Dict[str, Any], treatment_name: str = None) -> None:
        """
        Save summary report with all analysis results.
        
        Parameters
        ----------
        summary_data : dict
            Dictionary containing summary statistics and analysis results
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
        """
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        timestamp = self._get_timestamp()
        report_path = self.output_dir / f'summary_report_{treatment_name.replace(" ", "_")}_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {report_path}")
    
    def print_summary_stats(self, stats: Dict[str, Any], treatment_name: str = None) -> None:
        """
        Print summary statistics to console.
        
        Parameters
        ----------
        stats : dict
            Dictionary containing min, max, mean, std, and percentile data
        treatment_name : str, optional
            Name of treatment variable (uses self.treatment_var if None)
        """
        if treatment_name is None:
            treatment_name = self.treatment_var
        
        print(f"\n{'='*50}")
        print(f"SUMMARY STATISTICS FOR {treatment_name.upper()}")
        print(f"{'='*50}")
        print(f"Min: {stats['min']:.2f}")
        print(f"Max: {stats['max']:.2f}")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Std Dev: {stats['std']:.2f}")
        print(f"\nPercentile Distribution:")
        for p, value in stats['percentiles'].items():
            print(f"  {p}: {value:.2f}")
        print(f"{'='*50}\n")
