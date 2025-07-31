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
from config import OutputConfig, DataConfig

logger = logging.getLogger(__name__)

class OutputManager:
    """Manages output generation for treatment effect analysis."""
    
    def __init__(self, output_config: OutputConfig, data_config: DataConfig):
        self.output_config = output_config
        self.data_config = data_config
        self.output_dir = Path(output_config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_ate_results(self, results: Dict[str, Any], treatment_name: str = None, user_count: int = None) -> None:
        """Save ATE estimation results."""
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        if user_count is None:
            user_count = self.output_config.group_user_count
        
        # Create DataFrame for results
        df_effects = pd.DataFrame({
            'Treatment': results['treatment_values'],
            'ATE_Featurized': results['ate_featurized'],
            'ATE_Baseline': results['ate_baseline'],
            'CI_Lower_Featurized': results['ci_lower_featurized'],
            'CI_Upper_Featurized': results['ci_upper_featurized'],
            'CI_Lower_Baseline': results['ci_lower_baseline'],
            'CI_Upper_Baseline': results['ci_upper_baseline']
        })
        
        # Round numeric columns to 3 decimal places
        numeric_columns = ['ATE_Featurized', 'ATE_Baseline', 'CI_Lower_Featurized', 
                          'CI_Upper_Featurized', 'CI_Lower_Baseline', 'CI_Upper_Baseline']
        df_effects[numeric_columns] = df_effects[numeric_columns].round(3)
        
        if self.output_config.save_csv:
            csv_path = self.output_dir / f'ate_results_{treatment_name.replace(" ", "_")}.csv'
            df_effects.to_csv(csv_path, index=False)
            logger.info(f"ATE results saved to {csv_path}")
        
        if self.output_config.save_json:
            # Create JSON in the required format - exclude meaningless (0,0) comparison
            json_list = [
                {
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
                for _, row in df_effects.iterrows()
                if int(row["Treatment"]) > 0  # Exclude StartValue=0, EndValue=0 case
            ]
            
            json_path = self.output_dir / f'ate_results_{treatment_name.replace(" ", "_")}.json'
            with open(json_path, 'w') as f:
                json.dump(json_list, f, indent=2)
            logger.info(f"ATE results saved to {json_path}")
    
    def save_cate_results_with_subgroups(self, interpreter, feature_names: List[str], treatment_name: str = None, user_count: int = None) -> None:
        """Save CATE estimation results with tree-based subgroups."""
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        if user_count is None:
            user_count = self.output_config.group_user_count
        
        try:
            # Extract subgroups from the decision tree
            subgroups = interpreter.get_tree_subgroups(feature_names)
            
            # Get T0 and T1 values from interpreter
            T0 = getattr(interpreter, 'T0_', 0)
            T1 = getattr(interpreter, 'T1_', 1)
            
            if self.output_config.save_csv:
                # Create DataFrame for CSV export
                df_subgroups = pd.DataFrame([
                    {
                        'GroupId': sg['group_id'],
                        'FilterCondition': sg['filter_condition'],
                        'UserCount': sg['user_count'],
                        'TreatmentEffect': sg['treatment_effect'],
                        'TreatmentEffectStd': sg['treatment_effect_std'],
                        'CI_Lower': sg.get('ci_lower', 0.0),
                        'CI_Upper': sg.get('ci_upper', 0.0),
                        'T0': T0,
                        'T1': T1
                    }
                    for sg in subgroups
                ])
                
                # Round numeric columns to 3 decimal places
                numeric_columns = ['TreatmentEffect', 'TreatmentEffectStd', 'CI_Lower', 'CI_Upper']
                df_subgroups[numeric_columns] = df_subgroups[numeric_columns].round(3)
                
                csv_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}.csv'
                df_subgroups.to_csv(csv_path, index=False)
                logger.info(f"CATE subgroup results saved to {csv_path}")
            
            if self.output_config.save_json:
                # Create JSON in the required format
                json_list = [
                    {
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
                    for sg in subgroups
                    if sg['user_count'] > 0  # Only include subgroups with users
                ]
                
                json_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}.json'
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
        """Save CATE estimation results (legacy method for backward compatibility)."""
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        if user_count is None:
            user_count = self.output_config.group_user_count
        
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
        
        if self.output_config.save_csv:
            csv_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}.csv'
            df_cate.to_csv(csv_path, index=False)
            logger.info(f"CATE results saved to {csv_path}")
        
        if self.output_config.save_json:
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
            
            json_path = self.output_dir / f'cate_results_{treatment_name.replace(" ", "_")}.json'
            with open(json_path, 'w') as f:
                json.dump(json_list, f, indent=2)
            logger.info(f"CATE results saved to {json_path}")
    
    def create_ate_plot(self, results: Dict[str, Any], treatment_name: str = None) -> Optional[str]:
        """Create and save ATE visualization."""
        if not self.output_config.save_plots:
            return None
        
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        plt.figure(figsize=(self.output_config.plot_width, self.output_config.plot_height))
        
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
        
        plot_path = self.output_dir / f'ate_plot_{treatment_name.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ATE plot saved to {plot_path}")
        return str(plot_path)
    
    def create_cate_interpretation_plot(self, interpreter, feature_names: List[str], 
                                      treatment_name: str = None) -> Optional[str]:
        """Create and save CATE interpretation tree plot."""
        if not self.output_config.save_plots:
            return None
        
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        try:
            plt.figure(figsize=(15, 10))
            interpreter.plot(feature_names=feature_names, fontsize=12)
            
            plot_path = self.output_dir / f'cate_interpretation_{treatment_name.replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"CATE interpretation plot saved to {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.warning(f"Could not create CATE interpretation plot: {e}")
            return None
    
    def save_summary_report(self, summary_data: Dict[str, Any], treatment_name: str = None) -> None:
        """Save summary report with all analysis results."""
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
        report_path = self.output_dir / f'summary_report_{treatment_name.replace(" ", "_")}.json'
        
        with open(report_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Summary report saved to {report_path}")
    
    def print_summary_stats(self, stats: Dict[str, Any], treatment_name: str = None) -> None:
        """Print summary statistics to console."""
        if treatment_name is None:
            treatment_name = self.data_config.treatment_var
        
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
