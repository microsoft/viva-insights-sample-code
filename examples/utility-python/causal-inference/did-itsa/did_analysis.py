import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class MinimalDiDAnalysis:
    """DiD analysis with Copilot licensing as treatment and training as confounder."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        # Update variable names for licensing
        self.data['LicensedXPost'] = self.data['Licensed'] * self.data['Post']
        
        # Define confounder groups from requirements document
        self.hr_confounders = [
            'C(LevelDesignation)', 'Tenure', 'C(Region)', 
            'C(Function)', 'C(Org)', 'C(SupervisorIndicator)'
        ]
        
        self.collaboration_confounders = [
            "Q('Internal Network Size')", "Q('External Network Size')",
            "Q('Strong Ties')", "Q('Diverse Ties')",
            "Q('After-hours Meeting Hours')", "Q('After-hours Email Hours')",
            "Q('Available-to-focus Hours')", "Q('Weekend Collaboration Hours')",
            "Q('Calendared Learning Time')", "Q('Active Connected Hours')",
            "Q('External 1:1 Meeting Hours')", "Q('Uninterrupted Hours')"
        ]
        
        # Updated training confounders (now time-varying)
        self.training_confounders = [
            "Training_Week",  # Which week of training (0 if not in training)
            "Q('Training_Duration_Total')"  # Total planned training duration
        ]
    
    def check_balance(self) -> pd.DataFrame:
        """Check covariate balance between licensed and unlicensed groups."""
        balance_results = []
        
        # Check balance for all confounders (excluding training variables for controls)
        numeric_vars = [
            'Tenure', 'Internal Network Size', 'External Network Size',
            'After-hours Meeting Hours', 'Calendared Learning Time',
            'Available-to-focus Hours', 'Active Connected Hours'
            # Note: Training variables excluded as controls never receive training
        ]
        
        categorical_vars = [
            'LevelDesignation', 'Region', 'Function', 'Org', 'SupervisorIndicator'
        ]
        
        # Pre-treatment balance check (use pre-licensing period)
        pre_data = self.data[self.data['Post'] == 0]
        
        for var in numeric_vars:
            if var in pre_data.columns:
                licensed_mean = pre_data[pre_data['Licensed'] == 1][var].mean()
                unlicensed_mean = pre_data[pre_data['Licensed'] == 0][var].mean()
                
                # Standardized difference
                licensed_std = pre_data[pre_data['Licensed'] == 1][var].std()
                unlicensed_std = pre_data[pre_data['Licensed'] == 0][var].std()
                pooled_std = np.sqrt((licensed_std**2 + unlicensed_std**2) / 2)
                
                std_diff = (licensed_mean - unlicensed_mean) / pooled_std if pooled_std > 0 else 0
                
                balance_results.append({
                    'Variable': var,
                    'Type': 'Numeric',
                    'Licensed_Mean': licensed_mean,
                    'Unlicensed_Mean': unlicensed_mean,
                    'Std_Difference': std_diff,
                    'Imbalanced': abs(std_diff) > 0.25
                })
        
        for var in categorical_vars:
            if var in pre_data.columns:
                licensed_dist = pre_data[pre_data['Licensed'] == 1][var].value_counts(normalize=True)
                unlicensed_dist = pre_data[pre_data['Licensed'] == 0][var].value_counts(normalize=True)
                
                all_categories = set(licensed_dist.index) | set(unlicensed_dist.index)
                tvd = sum(abs(licensed_dist.get(cat, 0) - unlicensed_dist.get(cat, 0)) 
                         for cat in all_categories) / 2
                
                balance_results.append({
                    'Variable': var,
                    'Type': 'Categorical',
                    'Licensed_Mean': 'N/A',
                    'Unlicensed_Mean': 'N/A',
                    'Std_Difference': tvd,
                    'Imbalanced': tvd > 0.15
                })
        
        return pd.DataFrame(balance_results)
    
    def calculate_balance(self, variable: str) -> float:
        """Calculate standardized mean difference for a single variable.
        
        Args:
            variable: Variable name to calculate balance for
            
        Returns:
            Standardized mean difference (SMD) or Total Variation Distance (TVD) for categorical
        """
        # Use pre-treatment data
        pre_data = self.data[self.data['Post'] == 0]
        
        if variable not in pre_data.columns:
            return 0.0
            
        # Check if variable is numeric or categorical
        if pd.api.types.is_numeric_dtype(pre_data[variable]):
            # Numeric variable - calculate SMD
            licensed_mean = pre_data[pre_data['Licensed'] == 1][variable].mean()
            unlicensed_mean = pre_data[pre_data['Licensed'] == 0][variable].mean()
            
            licensed_std = pre_data[pre_data['Licensed'] == 1][variable].std()
            unlicensed_std = pre_data[pre_data['Licensed'] == 0][variable].std()
            pooled_std = np.sqrt((licensed_std**2 + unlicensed_std**2) / 2)
            
            if pooled_std > 0:
                return (licensed_mean - unlicensed_mean) / pooled_std
            else:
                return 0.0
        else:
            # Categorical variable - calculate Total Variation Distance
            licensed_dist = pre_data[pre_data['Licensed'] == 1][variable].value_counts(normalize=True)
            unlicensed_dist = pre_data[pre_data['Licensed'] == 0][variable].value_counts(normalize=True)
            
            all_categories = set(licensed_dist.index) | set(unlicensed_dist.index)
            tvd = sum(abs(licensed_dist.get(cat, 0) - unlicensed_dist.get(cat, 0)) 
                     for cat in all_categories) / 2
            return tvd
    
    def create_matched_control_group(self, matching_vars: list = None) -> pd.DataFrame:
        """Create matched control group using propensity score matching.
        
        Args:
            matching_vars: Variables to match on. If None, uses HR + basic collaboration vars
        
        Returns:
            DataFrame with matched treated and control units (full longitudinal data)
        """
        if matching_vars is None:
            # Use key confounders for matching (NO training variables)
            matching_vars = [
                'Tenure', 'Internal Network Size', 'External Network Size',
                'After-hours Meeting Hours', 'Calendared Learning Time'
            ]
            # Add categorical variables
            categorical_matching = ['LevelDesignation', 'Function', 'Region']
        else:
            categorical_matching = []
        
        # Get pre-treatment data for matching (only use first observation per person)
        pre_data = self.data[self.data['Post'] == 0].copy()
        
        # Take only one observation per person (first week) for matching
        pre_data_first = pre_data.groupby('PersonID').first().reset_index()
        
        # Prepare data for matching
        treated_pre = pre_data_first[pre_data_first['Licensed'] == 1].copy()
        control_pre = pre_data_first[pre_data_first['Licensed'] == 0].copy()
        
        if len(treated_pre) == 0 or len(control_pre) == 0:
            return self.data  # Return original data if no matching possible
        
        # Create dummy variables for categorical variables
        for cat_var in categorical_matching:
            if cat_var in pre_data_first.columns:
                dummies = pd.get_dummies(pre_data_first[cat_var], prefix=cat_var)
                pre_data_first = pd.concat([pre_data_first, dummies], axis=1)
                matching_vars.extend(dummies.columns.tolist())
        
        # Ensure all matching variables exist and vary
        available_vars = []
        for var in matching_vars:
            if var in pre_data_first.columns:
                if pre_data_first[var].std() > 0:  # Only include variables with variation
                    available_vars.append(var)
        
        if len(available_vars) == 0:
            # Fall back to random sampling if no good matching variables
            control_sample = control_pre.sample(min(len(treated_pre), len(control_pre)), random_state=42)
            matched_control_ids = control_sample['PersonID'].tolist()
        else:
            # Propensity Score Matching
            matched_controls = self._propensity_score_matching(pre_data_first, available_vars)
            matched_control_ids = matched_controls['PersonID'].tolist()
        
        # Get ALL longitudinal observations for matched units
        treated_ids = treated_pre['PersonID'].tolist()
        all_matched_ids = treated_ids + matched_control_ids
        
        # Return full longitudinal data for matched individuals
        matched_data = self.data[
            self.data['PersonID'].isin(all_matched_ids)
        ].copy()
        
        return matched_data
    
    def _propensity_score_matching(self, pre_data: pd.DataFrame, matching_vars: list) -> pd.DataFrame:
        """Propensity score matching implementation with proper categorical encoding."""
        # Separate numeric and categorical variables from matching_vars
        numeric_vars = []
        categorical_vars = []
        
        for var in matching_vars:
            if var in pre_data.columns:
                if pre_data[var].dtype in ['object', 'category'] or var.startswith(('LevelDesignation_', 'Function_', 'Region_')):
                    # This is either a categorical variable or already a dummy
                    if '_' in var and var.split('_')[0] in ['LevelDesignation', 'Function', 'Region']:
                        numeric_vars.append(var)  # Already a dummy variable
                    else:
                        categorical_vars.append(var)  # Original categorical variable
                else:
                    numeric_vars.append(var)  # Numeric variable
        
        # Create feature matrix
        X_parts = []
        
        # Add numeric variables
        if numeric_vars:
            numeric_data = pre_data[numeric_vars].fillna(0)
            X_parts.append(numeric_data)
        
        # Add dummy variables for categorical variables
        for cat_var in categorical_vars:
            if cat_var in pre_data.columns:
                dummies = pd.get_dummies(pre_data[cat_var], prefix=cat_var, drop_first=True)
                X_parts.append(dummies)
        
        # Combine all features
        if len(X_parts) == 0:
            return pre_data[pre_data['Licensed'] == 0].sample(min(100, len(pre_data[pre_data['Licensed'] == 0])))
        
        X = pd.concat(X_parts, axis=1)
        y = pre_data['Licensed']
        
        # Check for sufficient variation
        if len(X.columns) == 0 or y.nunique() < 2:
            return pre_data[pre_data['Licensed'] == 0].sample(min(100, len(pre_data[pre_data['Licensed'] == 0])))
        
        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].std() == 0]
        if constant_cols:
            X = X.drop(columns=constant_cols)
        
        if len(X.columns) == 0:
            return pre_data[pre_data['Licensed'] == 0].sample(min(100, len(pre_data[pre_data['Licensed'] == 0])))
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit propensity score model with regularization
        ps_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        ps_model.fit(X_scaled, y)
        
        # Calculate propensity scores
        pre_data = pre_data.copy()
        pre_data['propensity_score'] = ps_model.predict_proba(X_scaled)[:, 1]
        
        # Impose common support (trim extreme propensity scores)
        ps_min = np.percentile(pre_data['propensity_score'], 5)
        ps_max = np.percentile(pre_data['propensity_score'], 95)
        pre_data = pre_data[(pre_data['propensity_score'] >= ps_min) & 
                           (pre_data['propensity_score'] <= ps_max)]
        
        # Match each treated unit to nearest control by propensity score
        treated_pre = pre_data[pre_data['Licensed'] == 1]
        control_pre = pre_data[pre_data['Licensed'] == 0]
        
        # Limit matching to reasonable sample size
        max_matches = min(len(treated_pre), len(control_pre), 200)
        
        matched_controls = []
        used_control_ids = set()
        
        for i, (_, treated_unit) in enumerate(treated_pre.iterrows()):
            if i >= max_matches:
                break
                
            treated_ps = treated_unit['propensity_score']
            
            # Find controls within caliper (0.1 standard deviation)
            ps_std = control_pre['propensity_score'].std()
            caliper = 0.1 * ps_std
            
            eligible_controls = control_pre[
                (np.abs(control_pre['propensity_score'] - treated_ps) <= caliper) &
                (~control_pre['PersonID'].isin(used_control_ids))
            ]
            
            if len(eligible_controls) > 0:
                # Find closest match within caliper
                control_distances = np.abs(eligible_controls['propensity_score'] - treated_ps)
                closest_control_idx = control_distances.idxmin()
                matched_control = eligible_controls.loc[closest_control_idx]
                matched_controls.append(matched_control)
                used_control_ids.add(matched_control['PersonID'])
        
        if len(matched_controls) == 0:
            # Fall back to simple random sample if no matches found
            return control_pre.sample(min(50, len(control_pre)))
        
        return pd.DataFrame(matched_controls)
    
    def run_basic_did(self) -> dict:
        """Basic DiD without confounders - Copilot licensing effect."""
        formula = "Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost"
        model = ols(formula, data=self.data).fit()
        
        return self._extract_results(model, 'Basic DiD (Licensing)')
    
    def run_hr_controlled_did(self) -> dict:
        """DiD controlling for HR attributes only."""
        confounders = ' + '.join(self.hr_confounders)
        formula = f"Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost + {confounders}"
        
        model = ols(formula, data=self.data).fit()
        return self._extract_results(model, 'HR-Controlled DiD (Licensing)')
    
    def run_training_controlled_did(self) -> dict:
        """DiD controlling for training confounders - isolates pure licensing effect."""
        confounders = ' + '.join(self.training_confounders)
        formula = f"Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost + {confounders}"
        
        try:
            model = ols(formula, data=self.data).fit()
            return self._extract_results(model, 'Training-Controlled DiD (Pure Licensing)')
        except Exception as e:
            return {'error': f'Model failed: {str(e)}'}
    
    def run_fully_controlled_did(self) -> dict:
        """DiD controlling for all confounders including training."""
        essential_confounders = (
            self.hr_confounders[:4] + 
            self.collaboration_confounders[:6] + 
            self.training_confounders
        )
        confounders = ' + '.join(essential_confounders)
        
        formula = f"Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost + {confounders}"
        
        try:
            model = ols(formula, data=self.data).fit()
            return self._extract_results(model, 'Fully-Controlled DiD (Licensing)')
        except Exception as e:
            return {'error': f'Model failed: {str(e)}'}
    
    def run_never_licensed_control_did(self) -> dict:
        """DiD using only never-licensed units as controls."""
        never_licensed_data = self.data[
            (self.data['Licensed'] == 0) |  # Never licensed
            (self.data['Licensed'] == 1)    # Licensed units
        ]
        
        formula = "Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost"
        model = ols(formula, data=never_licensed_data).fit()
        
        result = self._extract_results(model, 'Never-Licensed Control DiD')
        result['n_unlicensed'] = len(never_licensed_data[never_licensed_data['Licensed'] == 0]['PersonID'].unique())
        result['n_licensed'] = len(never_licensed_data[never_licensed_data['Licensed'] == 1]['PersonID'].unique())
        
        return result
    
    def run_training_effect_analysis(self) -> dict:
        """Separate analysis of training effects among licensed users only."""
        licensed_data = self.data[self.data['Licensed'] == 1].copy()
        
        if len(licensed_data) == 0:
            return {'error': 'No licensed users found'}
        
        # Create training treatment variables using Training_Week
        licensed_data['In_Training'] = (licensed_data['Training_Week'] > 0).astype(int)
        licensed_data['TrainingXPost'] = licensed_data['In_Training'] * licensed_data['Post']
        
        formula = "Q('External Collaboration Hours') ~ In_Training + Post + TrainingXPost"
        
        try:
            model = ols(formula, data=licensed_data).fit()
            return self._extract_results_training(model, 'Training Effect (Licensed Users Only)')
        except Exception as e:
            return {'error': f'Training analysis failed: {str(e)}'}
    
    def test_parallel_trends_stratified(self, strata_vars: list = None) -> dict:
        """Test parallel trends within balanced strata.
        
        Args:
            strata_vars: Variables to stratify on. If None, uses key categorical variables
            
        Returns:
            Dictionary with overall test and stratum-specific results
        """
        if strata_vars is None:
            strata_vars = ['LevelDesignation', 'Function']
        
        # Filter to available variables
        available_strata = [var for var in strata_vars if var in self.data.columns]
        
        if not available_strata:
            # Fall back to regular parallel trends test
            return self.test_parallel_trends()
        
        stratum_results = {}
        all_pvalues = []
        
        # Test parallel trends within each stratum
        pre_data = []
        for person_id in self.data['PersonID'].unique():
            person_data = self.data[self.data['PersonID'] == person_id]
            licensing_week = person_data['LicensingWeek'].iloc[0]
            
            if licensing_week != -1:  # Licensed
                person_pre = person_data[person_data['Week'] < licensing_week]
            else:  # Never licensed (control)
                person_pre = person_data[person_data['Week'] < 8]
            
            pre_data.append(person_pre)
        
        pre_data = pd.concat(pre_data, ignore_index=True)
        
        # Group by strata
        for stratum_values, stratum_data in pre_data.groupby(available_strata):
            if len(stratum_data) < 10:  # Skip small strata
                continue
                
            # Check if stratum has both treated and control units
            if stratum_data['Licensed'].nunique() < 2:
                continue
            
            stratum_name = '_'.join([f"{var}_{val}" for var, val in zip(available_strata, stratum_values)])
            
            try:
                formula = "Q('External Collaboration Hours') ~ Licensed + Week + Licensed*Week"
                model = ols(formula, data=stratum_data).fit()
                interaction_pvalue = model.pvalues.get('Licensed:Week', np.nan)
                
                stratum_results[stratum_name] = {
                    'pvalue': interaction_pvalue,
                    'assumption_violated': interaction_pvalue < 0.05,
                    'n_obs': len(stratum_data),
                    'n_licensed': len(stratum_data[stratum_data['Licensed'] == 1]),
                    'n_unlicensed': len(stratum_data[stratum_data['Licensed'] == 0])
                }
                
                if not np.isnan(interaction_pvalue):
                    all_pvalues.append(interaction_pvalue)
                    
            except Exception as e:
                stratum_results[stratum_name] = {'error': str(e)}
        
        # Overall assessment
        if all_pvalues:
            # Use minimum p-value (most conservative)
            overall_pvalue = min(all_pvalues)
            violations = sum(1 for p in all_pvalues if p < 0.05)
            
            return {
                'overall_pvalue': overall_pvalue,
                'overall_assumption_violated': overall_pvalue < 0.05,
                'strata_tested': len(all_pvalues),
                'strata_violations': violations,
                'violation_rate': violations / len(all_pvalues) if all_pvalues else 0,
                'stratum_results': stratum_results,
                'method': 'stratified_parallel_trends'
            }
        else:
            return {
                'error': 'No valid strata found for testing',
                'stratum_results': stratum_results
            }
    
    def test_parallel_trends(self) -> dict:
        """Test parallel trends for licensing with proper control group."""
        pre_data = []
        
        for person_id in self.data['PersonID'].unique():
            person_data = self.data[self.data['PersonID'] == person_id]
            licensing_week = person_data['LicensingWeek'].iloc[0]
            
            if licensing_week != -1:  # Licensed
                person_pre = person_data[person_data['Week'] < licensing_week]
            else:  # Never licensed (control)
                person_pre = person_data[person_data['Week'] < 8]
            
            pre_data.append(person_pre)
        
        pre_data = pd.concat(pre_data, ignore_index=True)
        
        formula = "Q('External Collaboration Hours') ~ Licensed + Week + Licensed*Week"
        
        try:
            model = ols(formula, data=pre_data).fit()
            interaction_pvalue = model.pvalues.get('Licensed:Week', np.nan)
            
            return {
                'pvalue': interaction_pvalue,
                'assumption_violated': interaction_pvalue < 0.05,
                'model': model,
                'n_pre_observations': len(pre_data),
                'formula': formula
            }
        except Exception as e:
            return {'error': str(e), 'formula': formula}
        """Test parallel trends for licensing with proper control group."""
        pre_data = []
        
        for person_id in self.data['PersonID'].unique():
            person_data = self.data[self.data['PersonID'] == person_id]
            licensing_week = person_data['LicensingWeek'].iloc[0]
            
            if licensing_week != -1:  # Licensed
                person_pre = person_data[person_data['Week'] < licensing_week]
            else:  # Never licensed (control)
                person_pre = person_data[person_data['Week'] < 8]
            
            pre_data.append(person_pre)
        
        pre_data = pd.concat(pre_data, ignore_index=True)
        
        formula = "Q('External Collaboration Hours') ~ Licensed + Week + Licensed*Week"
        
        try:
            model = ols(formula, data=pre_data).fit()
            interaction_pvalue = model.pvalues.get('Licensed:Week', np.nan)
            
            return {
                'pvalue': interaction_pvalue,
                'assumption_violated': interaction_pvalue < 0.05,
                'model': model,
                'n_pre_observations': len(pre_data),
                'formula': formula
            }
        except Exception as e:
            return {'error': str(e), 'formula': formula}
    
    def run_cohort_specific_did(self) -> dict:
        """DiD analysis by licensing week cohorts."""
        cohort_results = {}
        
        licensing_weeks = self.data[self.data['Licensed'] == 1]['LicensingWeek'].unique()
        licensing_weeks = licensing_weeks[licensing_weeks != -1]
        
        for licensing_week in licensing_weeks:
            cohort_data = self.data.copy()
            
            cohort_data['CohortLicensed'] = (
                (cohort_data['Licensed'] == 1) & 
                (cohort_data['LicensingWeek'] == licensing_week)
            ).astype(int)
            
            cohort_data['CohortPost'] = (cohort_data['Week'] >= licensing_week).astype(int)
            cohort_data['CohortLicensedXPost'] = cohort_data['CohortLicensed'] * cohort_data['CohortPost']
            
            cohort_subset = cohort_data[
                (cohort_data['Licensed'] == 0) |
                (cohort_data['LicensingWeek'] == licensing_week)
            ]
            
            if len(cohort_subset) > 0:
                formula = "Q('External Collaboration Hours') ~ CohortLicensed + CohortPost + CohortLicensedXPost"
                try:
                    model = ols(formula, data=cohort_subset).fit()
                    cohort_results[f'week_{licensing_week}'] = {
                        'ate': model.params['CohortLicensedXPost'],
                        'se': model.bse['CohortLicensedXPost'],
                        'pvalue': model.pvalues['CohortLicensedXPost'],
                        'n_licensed': len(cohort_subset[cohort_subset['CohortLicensed'] == 1]),
                        'n_unlicensed': len(cohort_subset[cohort_subset['CohortLicensed'] == 0])
                    }
                except Exception as e:
                    cohort_results[f'week_{licensing_week}'] = {'error': str(e)}
        
        return cohort_results
    
    def _extract_results(self, model, model_name: str) -> dict:
        """Extract standardized results from licensing model."""
        return {
            'model_name': model_name,
            'ate': model.params['LicensedXPost'],
            'se': model.bse['LicensedXPost'],
            'pvalue': model.pvalues['LicensedXPost'],
            'ci_lower': model.conf_int().loc['LicensedXPost', 0],
            'ci_upper': model.conf_int().loc['LicensedXPost', 1],
            'r_squared': model.rsquared,
            'n_obs': len(model.fittedvalues)
        }
    
    def _extract_results_training(self, model, model_name: str) -> dict:
        """Extract standardized results from training model."""
        return {
            'model_name': model_name,
            'ate': model.params['TrainingXPost'],
            'se': model.bse['TrainingXPost'],
            'pvalue': model.pvalues['TrainingXPost'],
            'ci_lower': model.conf_int().loc['TrainingXPost', 0],
            'ci_upper': model.conf_int().loc['TrainingXPost', 1],
            'r_squared': model.rsquared,
            'n_obs': len(model.fittedvalues)
        }
    
    def run_matched_did(self) -> dict:
        """DiD analysis using propensity score matched control group.
            
        Returns:
            DiD results using matched sample
        """
        # Create matched dataset using propensity score matching
        matched_data = self.create_matched_control_group()
        
        # Create temporary analysis object with matched data
        matched_analysis = MinimalDiDAnalysis(matched_data)
        
        # Run basic DiD on matched sample
        result = matched_analysis.run_basic_did()
        result['model_name'] = 'Propensity Score Matched DiD'
        
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
    
    def run_weighted_did(self, weight_vars: list = None) -> dict:
        """DiD analysis using inverse propensity weighting.
        
        Args:
            weight_vars: Variables to use for propensity score estimation
            
        Returns:
            Weighted DiD results
        """
        if weight_vars is None:
            # Separate numeric and categorical variables
            numeric_vars = [
                'Tenure', 'Internal Network Size', 'External Network Size',
                'After-hours Meeting Hours', 'Calendared Learning Time'
            ]
            categorical_vars = ['LevelDesignation', 'Function', 'Region']
        else:
            # Assume user provided variables are all numeric unless specified
            numeric_vars = weight_vars
            categorical_vars = []
        
        # Get pre-treatment data for propensity score estimation
        pre_data = self.data[self.data['Post'] == 0].copy()
        
        # Start with numeric variables
        available_numeric_vars = [var for var in numeric_vars if var in pre_data.columns]
        
        if len(available_numeric_vars) == 0 and len(categorical_vars) == 0:
            return {'error': 'No valid variables for propensity score estimation'}
        
        # Create feature matrix starting with numeric variables
        X_numeric = pre_data[available_numeric_vars].fillna(0) if available_numeric_vars else pd.DataFrame()
        
        # Add dummy variables for categorical variables
        X_categorical_list = []
        for cat_var in categorical_vars:
            if cat_var in pre_data.columns:
                # Create dummy variables (drop first to avoid multicollinearity)
                dummies = pd.get_dummies(pre_data[cat_var], prefix=cat_var, drop_first=True)
                X_categorical_list.append(dummies)
        
        # Combine numeric and categorical features
        if len(X_categorical_list) > 0:
            X_categorical = pd.concat(X_categorical_list, axis=1)
            if len(X_numeric.columns) > 0:
                X = pd.concat([X_numeric, X_categorical], axis=1)
            else:
                X = X_categorical
        else:
            X = X_numeric
        
        if len(X.columns) == 0:
            return {'error': 'No valid features after preprocessing'}
        
        y = pre_data['Licensed']
        
        # Check for sufficient variation
        if y.nunique() < 2:
            return {'error': 'Insufficient treatment variation for weighting'}
        
        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].std() == 0]
        if constant_cols:
            X = X.drop(columns=constant_cols)
        
        if len(X.columns) == 0:
            return {'error': 'No varying covariates for propensity estimation'}
        
        # Estimate propensity scores with regularization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ps_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        ps_model.fit(X_scaled, y)
        
        # Calculate propensity scores for full dataset
        # Need to create the same feature matrix for full dataset
        full_X_numeric = self.data[available_numeric_vars].fillna(0) if available_numeric_vars else pd.DataFrame()
        
        # Create dummy variables for full dataset
        full_X_categorical_list = []
        for cat_var in categorical_vars:
            if cat_var in self.data.columns:
                # Use same categories as in training data to avoid dimension mismatch
                train_categories = pre_data[cat_var].unique()
                dummies = pd.get_dummies(self.data[cat_var], prefix=cat_var, drop_first=True)
                # Ensure same columns as training data
                for col in X.columns:
                    if col.startswith(cat_var + '_') and col not in dummies.columns:
                        dummies[col] = 0
                full_X_categorical_list.append(dummies[dummies.columns.intersection(X.columns)])
        
        # Combine for full dataset
        if len(full_X_categorical_list) > 0:
            full_X_categorical = pd.concat(full_X_categorical_list, axis=1)
            if len(full_X_numeric.columns) > 0:
                full_X = pd.concat([full_X_numeric, full_X_categorical], axis=1)
            else:
                full_X = full_X_categorical
        else:
            full_X = full_X_numeric
        
        # Ensure same column order as training data
        full_X = full_X[X.columns]
        full_X_scaled = scaler.transform(full_X)
        propensity_scores = ps_model.predict_proba(full_X_scaled)[:, 1]
        
        # Trim extreme propensity scores for stability
        ps_min = np.percentile(propensity_scores, 5)
        ps_max = np.percentile(propensity_scores, 95)
        propensity_scores = np.clip(propensity_scores, ps_min, ps_max)
        
        # Calculate inverse propensity weights
        weights = np.where(
            self.data['Licensed'] == 1,
            1 / propensity_scores,  # For treated units
            1 / (1 - propensity_scores)  # For control units
        )
        
        # Aggressive weight trimming to prevent extreme values
        weights = np.clip(weights, 0.2, 5.0)
        
        # Normalize weights to sum to sample size
        treated_mask = self.data['Licensed'] == 1
        control_mask = self.data['Licensed'] == 0
        
        treated_weights = weights[treated_mask]
        control_weights = weights[control_mask]
        
        n_treated = len(treated_weights)
        n_control = len(control_weights)
        
        # Normalize within groups
        if treated_weights.sum() > 0:
            treated_weights = treated_weights * (n_treated / treated_weights.sum())
        if control_weights.sum() > 0:
            control_weights = control_weights * (n_control / control_weights.sum())
        
        # Reconstruct full weight vector
        weights_normalized = np.zeros(len(self.data))
        weights_normalized[treated_mask] = treated_weights
        weights_normalized[control_mask] = control_weights
        
        # Final check for problematic weights
        if np.any(np.isnan(weights_normalized)) or np.any(np.isinf(weights_normalized)):
            return {'error': 'Invalid weights generated'}
        
        # Run weighted regression
        weighted_data = self.data.copy()
        weighted_data['weights'] = weights_normalized
        
        formula = "Q('External Collaboration Hours') ~ Licensed + Post + LicensedXPost"
        
        try:
            model = ols(formula, data=weighted_data, weights=weighted_data['weights']).fit()
            result = self._extract_results(model, 'Weighted DiD (IPW)')
            
            # Add weighting diagnostics
            result['weight_diagnostics'] = {
                'mean_weight_treated': treated_weights.mean(),
                'mean_weight_control': control_weights.mean(),
                'min_weight': weights_normalized.min(),
                'max_weight': weights_normalized.max(),
                'effective_sample_size': len(weights_normalized) ** 2 / np.sum(weights_normalized ** 2),
                'features_used': list(X.columns)
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Weighted DiD failed: {str(e)}'}
    
    def run_robust_did_suite(self) -> dict:
        """Run comprehensive suite of robust DiD methods.
        
        Returns:
            Dictionary with results from all robust methods
        """
        results = {}
        
        # Original methods
        results['basic_did'] = self.run_basic_did()
        results['fully_controlled_did'] = self.run_fully_controlled_did()
        
        # Robust methods
        try:
            results['matched_did'] = self.run_matched_did()
        except Exception as e:
            results['matched_did'] = {'error': str(e)}
        
        try:
            results['weighted_did'] = self.run_weighted_did()
        except Exception as e:
            results['weighted_did'] = {'error': str(e)}
        
        # Parallel trends tests
        results['parallel_trends_basic'] = self.test_parallel_trends()
        
        try:
            results['parallel_trends_stratified'] = self.test_parallel_trends_stratified()
        except Exception as e:
            results['parallel_trends_stratified'] = {'error': str(e)}
        
        return results
    
    def run_progressive_analysis(self) -> dict:
        """Run progressive DiD analysis with increasing levels of control.
        
        Returns:
            Dictionary with results from progressive specifications
        """
        results = {}
        
        # Progressive control strategy
        try:
            results['Simple DiD'] = self.run_basic_did()
        except Exception as e:
            results['Simple DiD'] = None
            
        try:
            results['Enhanced DiD'] = self.run_hr_controlled_did()
        except Exception as e:
            results['Enhanced DiD'] = None
            
        try:
            results['Robust DiD'] = self.run_fully_controlled_did()
        except Exception as e:
            results['Robust DiD'] = None
            
        return results
