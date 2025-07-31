import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

class MinimalITSAAnalysis:
    """ITSA with proper control series and confounders for Copilot licensing."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._prepare_time_variables()
    
    def _prepare_time_variables(self):
        """Prepare time variables for ITSA."""
        self.data['TimeSinceLicensing'] = 0
        
        for person_id in self.data['PersonID'].unique():
            person_mask = self.data['PersonID'] == person_id
            person_data = self.data[person_mask]
            licensing_week = person_data['LicensingWeek'].iloc[0]
            
            if licensing_week != -1:  # Licensed individual
                post_mask = person_mask & (self.data['Week'] >= licensing_week)
                self.data.loc[post_mask, 'TimeSinceLicensing'] = (
                    self.data.loc[post_mask, 'Week'] - licensing_week
                )
    
    def run_controlled_itsa(self, matched_data: pd.DataFrame = None) -> dict:
        """ITSA with control series (never-licensed units or matched controls from DiD).
        
        Args:
            matched_data: Optional DataFrame with matched treated/control units from DiD analysis.
                         If provided, uses these matched controls instead of all never-licensed units.
        """
        # Use matched data if provided, otherwise use all data
        analysis_data = matched_data if matched_data is not None else self.data
        
        # Aggregate to weekly level by licensing group
        weekly_data = analysis_data.groupby(['Week', 'Licensed']).agg({
            'External Collaboration Hours': 'mean',
            'Post': 'max',  # Any licensed unit post-licensing
            'TimeSinceLicensing': 'max',
            'Internal Network Size': 'mean',
            'After-hours Meeting Hours': 'mean',
            'Available-to-focus Hours': 'mean'
        }).reset_index()
        
        # Create interaction terms
        weekly_data['LicensedXPost'] = weekly_data['Licensed'] * weekly_data['Post']
        weekly_data['LicensedXTimeSince'] = weekly_data['Licensed'] * weekly_data['TimeSinceLicensing']
        weekly_data['LicensedXWeek'] = weekly_data['Licensed'] * weekly_data['Week']
        
        # ITSA with control series and confounders
        formula = ("Q('External Collaboration Hours') ~ Week + Post + TimeSinceLicensing + "
                  "Licensed + LicensedXPost + LicensedXTimeSince + LicensedXWeek + "
                  "Q('Internal Network Size') + Q('After-hours Meeting Hours') + "
                  "Q('Available-to-focus Hours')")
        
        try:
            model = ols(formula, data=weekly_data).fit()
            
            return {
                'level_change': model.params.get('LicensedXPost', np.nan),
                'slope_change': model.params.get('LicensedXTimeSince', np.nan),
                'baseline_trend_diff': model.params.get('LicensedXWeek', np.nan),
                'level_p_value': model.pvalues.get('LicensedXPost', np.nan),
                'slope_p_value': model.pvalues.get('LicensedXTimeSince', np.nan),
                'baseline_trend_p_value': model.pvalues.get('LicensedXWeek', np.nan),
                'model': model,
                'r_squared': model.rsquared,
                'n_weeks': len(weekly_data),
                'control_type': 'matched_controls' if matched_data is not None else 'all_never_licensed',
                'n_treated_individuals': len(analysis_data[analysis_data['Licensed'] == 1]['PersonID'].unique()),
                'n_control_individuals': len(analysis_data[analysis_data['Licensed'] == 0]['PersonID'].unique())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_matched_controlled_itsa(self, did_analysis) -> dict:
        """ITSA using propensity score matched controls from DiD analysis.
        
        Args:
            did_analysis: MinimalDiDAnalysis object to obtain matched controls from
            
        Returns:
            ITSA results using matched control series
        """
        # Get matched data from DiD analysis
        matched_data = did_analysis.create_matched_control_group()
        
        # Create temporary ITSA analyzer with matched data to prepare time variables
        temp_itsa = MinimalITSAAnalysis(matched_data)
        
        # Run controlled ITSA with the prepared matched data
        result = temp_itsa.run_controlled_itsa()
        
        if 'error' not in result:
            result['method'] = 'ITSA with Propensity Score Matched Controls'
            result['control_type'] = 'matched_controls'
            
            # Add matching diagnostics
            original_n_licensed = len(self.data[self.data['Licensed'] == 1]['PersonID'].unique())
            original_n_unlicensed = len(self.data[self.data['Licensed'] == 0]['PersonID'].unique())
            matched_n_licensed = len(matched_data[matched_data['Licensed'] == 1]['PersonID'].unique())
            matched_n_unlicensed = len(matched_data[matched_data['Licensed'] == 0]['PersonID'].unique())
            
            result['matching_diagnostics'] = {
                'original_licensed': original_n_licensed,
                'original_unlicensed': original_n_unlicensed,
                'matched_licensed': matched_n_licensed,
                'matched_unlicensed': matched_n_unlicensed,
                'matching_ratio': matched_n_unlicensed / matched_n_licensed if matched_n_licensed > 0 else 0
            }
        
        return result
    
    def run_basic_itsa_licensed_only(self) -> dict:
        """Basic ITSA for licensed group only (no control series)."""
        licensed_data = self.data[self.data['Licensed'] == 1].groupby('Week').agg({
            'External Collaboration Hours': 'mean',
            'Post': 'max',
            'TimeSinceLicensing': 'max'
        }).reset_index()
        
        formula = "Q('External Collaboration Hours') ~ Week + Post + TimeSinceLicensing"
        
        try:
            model = ols(formula, data=licensed_data).fit()
            
            return {
                'level_change': model.params.get('Post', np.nan),
                'slope_change': model.params.get('TimeSinceLicensing', np.nan),
                'baseline_trend': model.params.get('Week', np.nan),
                'level_p_value': model.pvalues.get('Post', np.nan),
                'slope_p_value': model.pvalues.get('TimeSinceLicensing', np.nan),
                'baseline_trend_p_value': model.pvalues.get('Week', np.nan),
                'model': model,
                'r_squared': model.rsquared
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_autocorrelation(self) -> dict:
        """Check for autocorrelation in the time series."""
        # Get licensed group weekly means
        licensed_weekly = self.data[self.data['Licensed'] == 1].groupby('Week')['External Collaboration Hours'].mean()
        
        if len(licensed_weekly) < 4:
            return {'error': 'Insufficient data for autocorrelation testing'}
        
        # Calculate first-order autocorrelation
        autocorr = licensed_weekly.autocorr(lag=1)
        
        # Simple Durbin-Watson-like statistic
        residuals = licensed_weekly.diff().dropna()
        dw_stat = 2 * (1 - autocorr) if not np.isnan(autocorr) else np.nan
        
        return {
            'autocorrelation': autocorr,
            'durbin_watson_approx': dw_stat,
            'autocorrelation_present': abs(autocorr) > 0.3 if not np.isnan(autocorr) else None
        }
    
    def test_parallel_trends_itsa(self, matched_data: pd.DataFrame = None) -> dict:
        """Test parallel trends assumption for ITSA control and treatment series.
        
        Args:
            matched_data: Optional DataFrame with matched treated/control units from DiD analysis.
                         If provided, uses these matched controls instead of all never-licensed units.
        
        Returns:
            Dictionary with parallel trends test results
        """
        # Use matched data if provided, otherwise use all data
        analysis_data = matched_data if matched_data is not None else self.data
        
        # Get pre-intervention data for parallel trends testing
        pre_data = []
        
        for person_id in analysis_data['PersonID'].unique():
            person_data = analysis_data[analysis_data['PersonID'] == person_id]
            licensing_week = person_data['LicensingWeek'].iloc[0]
            
            if licensing_week != -1:  # Licensed
                person_pre = person_data[person_data['Week'] < licensing_week]
            else:  # Never licensed (control)
                person_pre = person_data[person_data['Week'] < 8]  # Use first 8 weeks as pre-period
            
            pre_data.append(person_pre)
        
        if not pre_data:
            return {'error': 'No pre-intervention data available for parallel trends testing'}
        
        pre_data = pd.concat(pre_data, ignore_index=True)
        
        if len(pre_data) == 0:
            return {'error': 'No pre-intervention observations found'}
        
        # Test for parallel trends: interaction between Licensed and Week should be non-significant
        formula = "Q('External Collaboration Hours') ~ Licensed + Week + Licensed*Week"
        
        try:
            model = ols(formula, data=pre_data).fit()
            interaction_pvalue = model.pvalues.get('Licensed:Week', np.nan)
            
            return {
                'pvalue': interaction_pvalue,
                'assumption_violated': interaction_pvalue < 0.05,
                'model': model,
                'n_pre_observations': len(pre_data),
                'formula': formula,
                'control_type': 'matched_controls' if matched_data is not None else 'all_never_licensed',
                'n_treated_pre': len(pre_data[pre_data['Licensed'] == 1]),
                'n_control_pre': len(pre_data[pre_data['Licensed'] == 0])
            }
        except Exception as e:
            return {'error': f'Parallel trends test failed: {str(e)}', 'formula': formula}

    def run_individual_itsa_summary(self) -> dict:
        """Summary of individual ITSA results."""
        individual_results = []
        
        licensed_individuals = self.data[
            (self.data['Licensed'] == 1) & (self.data['LicensingWeek'] != -1)
        ]['PersonID'].unique()
        
        for person_id in licensed_individuals:
            person_data = self.data[self.data['PersonID'] == person_id]
            intervention_week = person_data['InterventionWeek'].iloc[0]
            
            # Create person-specific indicators
            person_data = person_data.copy()
            person_data['PersonPost'] = (person_data['Week'] >= intervention_week).astype(int)
            person_data['PersonTimeSince'] = np.maximum(0, person_data['Week'] - intervention_week)
            
            try:
                formula = "`External Collaboration Hours` ~ Week + PersonPost + PersonTimeSince"
                model = ols(formula, data=person_data).fit()
                
                individual_results.append({
                    'person_id': person_id,
                    'level_change': model.params.get('PersonPost', np.nan),
                    'slope_change': model.params.get('PersonTimeSince', np.nan),
                    'intervention_week': intervention_week
                })
            except Exception:
                continue
        
        if individual_results:
            level_changes = [r['level_change'] for r in individual_results if not np.isnan(r['level_change'])]
            slope_changes = [r['slope_change'] for r in individual_results if not np.isnan(r['slope_change'])]
            
            return {
                'n_individuals': len(individual_results),
                'avg_level_change': np.mean(level_changes) if level_changes else np.nan,
                'avg_slope_change': np.mean(slope_changes) if slope_changes else np.nan,
                'individual_results': individual_results
            }
        else:
            return {'error': 'No valid individual ITSA results'}
    
    def run_enhanced_analysis(self) -> dict:
        """Run enhanced ITSA analysis with matched controls.
        
        This method attempts to run ITSA with matched controls, falling back to 
        basic ITSA if matching fails.
        
        Returns:
            Dictionary with ITSA results including level_change, slope_change, and p-values
        """
        try:
            # Try to create a minimal DiD analyzer for matching
            from did_analysis import MinimalDiDAnalysis
            did_analyzer = MinimalDiDAnalysis(self.data)
            
            # Run matched controlled ITSA
            result = self.run_matched_controlled_itsa(did_analyzer)
            
            # Check if we have valid results
            if 'error' not in result and 'level_change' in result:
                return result
            else:
                # Fall back to basic ITSA if enhanced fails
                return self.run_basic_itsa_licensed_only()
                
        except Exception as e:
            # Fall back to basic ITSA if enhanced analysis fails
            try:
                basic_result = self.run_basic_itsa_licensed_only()
                return basic_result
            except Exception as e2:
                return {
                    'error': f'Both enhanced and basic ITSA failed: Enhanced - {str(e)}, Basic - {str(e2)}',
                    'level_change': 0.0,
                    'slope_change': 0.0,
                    'level_p_value': 1.0,
                    'slope_p_value': 1.0
                }
