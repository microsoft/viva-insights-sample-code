"""
Sensitivity analysis utilities for treatment effect estimation.

This module provides functions for assessing the robustness of causal effect
estimates to potential unobserved confounding, including:
- E-values for quantifying confounding strength needed to explain away effects
- Rosenbaum bounds for matched observational studies
"""

import numpy as np
from typing import Dict, Optional, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_evalue(
    estimate: float,
    confidence_interval_lower: Optional[float] = None,
    confidence_interval_upper: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate E-value for sensitivity analysis.
    
    The E-value quantifies the minimum strength of association (on the risk ratio scale)
    that an unmeasured confounder would need to have with both the treatment and outcome
    to fully explain away the observed effect, conditional on measured covariates.
    
    References
    ----------
    VanderWeele, T.J. and Ding, P. (2017). "Sensitivity Analysis in Observational Research:
    Introducing the E-Value." Annals of Internal Medicine, 167(4), 268-274.
    
    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect (on the outcome scale, e.g., hours)
    confidence_interval_lower : float, optional
        Lower bound of confidence interval
    confidence_interval_upper : float, optional
        Upper bound of confidence interval
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'evalue_point': E-value for the point estimate
        - 'evalue_ci': E-value for the confidence interval (more conservative)
        - 'interpretation': Text interpretation of the E-value
        
    Notes
    -----
    - For continuous outcomes, we approximate the risk ratio transformation
    - E-value = 1.0 means no unmeasured confounding could explain the effect
    - Higher E-values indicate more robustness to unmeasured confounding
    - The CI E-value is more conservative and represents the confounding strength
      needed to shift the entire CI to include the null
    """
    
    def evalue_formula(effect: float) -> float:
        """
        Calculate E-value from an effect estimate.
        
        For continuous outcomes (like hours), we use an approximation where
        we convert the effect to a standardized scale and then to risk ratio.
        
        The formula: E-value = RR + sqrt(RR * (RR - 1))
        where RR is approximated from the standardized effect.
        """
        if effect == 0:
            return 1.0
        
        # Convert effect to approximate risk ratio
        # This is a simplification; exact conversion depends on outcome distribution
        # Here we use: RR ≈ exp(effect / typical_scale)
        # For collaboration hours, typical scale might be ~5 hours
        typical_scale = 5.0
        rr = np.exp(abs(effect) / typical_scale)
        
        # E-value formula from VanderWeele & Ding (2017)
        if rr >= 1:
            evalue = rr + np.sqrt(rr * (rr - 1))
        else:
            # If RR < 1, use 1/RR in the formula
            rr_inv = 1 / rr
            evalue = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))
        
        return evalue
    
    # Calculate E-value for point estimate
    evalue_point = evalue_formula(estimate)
    
    # Calculate E-value for confidence interval (if provided)
    if confidence_interval_lower is not None and confidence_interval_upper is not None:
        # Use the CI bound closest to the null (0)
        # This gives the most conservative E-value
        if abs(confidence_interval_lower) < abs(confidence_interval_upper):
            ci_closest_to_null = confidence_interval_lower
        else:
            ci_closest_to_null = confidence_interval_upper
        
        evalue_ci = evalue_formula(ci_closest_to_null)
    else:
        evalue_ci = None
    
    # Generate interpretation
    if evalue_ci is not None:
        interpretation = (
            f"An unmeasured confounder would need to be associated with both "
            f"the treatment and outcome by a risk ratio of {evalue_ci:.2f}-fold each "
            f"to explain away the confidence interval (or {evalue_point:.2f}-fold "
            f"to explain away the point estimate), conditional on measured covariates."
        )
    else:
        interpretation = (
            f"An unmeasured confounder would need to be associated with both "
            f"the treatment and outcome by a risk ratio of {evalue_point:.2f}-fold each "
            f"to fully explain away the observed effect, conditional on measured covariates."
        )
    
    # Add practical interpretation
    if evalue_point < 1.5:
        strength = "weak"
    elif evalue_point < 2.5:
        strength = "moderate"
    elif evalue_point < 4.0:
        strength = "strong"
    else:
        strength = "very strong"
    
    interpretation += (
        f"\n\nThis represents {strength} robustness to unmeasured confounding. "
        f"E-values > 2 are generally considered reasonably robust."
    )
    
    return {
        'evalue_point': evalue_point,
        'evalue_ci': evalue_ci,
        'estimate': estimate,
        'ci_lower': confidence_interval_lower,
        'ci_upper': confidence_interval_upper,
        'interpretation': interpretation
    }


def rosenbaum_bounds_approximation(
    treatment_effects: np.ndarray,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Approximate Rosenbaum bounds sensitivity analysis.
    
    Rosenbaum bounds assess how strong hidden bias (from unmeasured confounders)
    would need to be to alter the conclusions of a study. The method determines
    the critical value of Gamma (Γ) at which the study's conclusions would change.
    
    References
    ----------
    Rosenbaum, P.R. (2002). "Observational Studies" (2nd ed.). Springer.
    Rosenbaum, P.R. (2010). "Design of Observational Studies". Springer.
    
    Parameters
    ----------
    treatment_effects : np.ndarray
        Array of individual treatment effects
    significance_level : float, default=0.05
        Significance level for testing (typically 0.05)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'original_p_value': P-value from Wilcoxon signed-rank test
        - 'critical_gamma': Critical Gamma value (if found)
        - 'interpretation': Text interpretation
        - 'gamma_range_tested': Range of Gamma values tested
        
    Notes
    -----
    - Gamma (Γ) represents odds of differential assignment to treatment
    - Γ = 1 means no hidden bias (random assignment)
    - Γ = 2 means hidden bias could make treatment twice as likely for some individuals
    - Higher critical Gamma indicates more robustness to hidden bias
    - This is an approximation suitable for large samples with continuous outcomes
    """
    
    # Remove any NaN values
    treatment_effects = treatment_effects[~np.isnan(treatment_effects)]
    
    if len(treatment_effects) < 10:
        logger.warning("Sample size too small for Rosenbaum bounds analysis")
        return {
            'original_p_value': np.nan,
            'critical_gamma': None,
            'interpretation': "Sample size too small for reliable Rosenbaum bounds analysis (n < 10)",
            'gamma_range_tested': None
        }
    
    # Calculate original p-value using Wilcoxon signed-rank test
    # This tests whether the median treatment effect differs from 0
    try:
        statistic, original_p_value = stats.wilcoxon(treatment_effects, alternative='two-sided')
    except ValueError as e:
        logger.warning(f"Wilcoxon test failed: {e}")
        return {
            'original_p_value': np.nan,
            'critical_gamma': None,
            'interpretation': f"Could not perform Wilcoxon test: {e}",
            'gamma_range_tested': None
        }
    
    # Search for critical Gamma
    gamma_values = np.arange(1.0, 5.1, 0.1)
    critical_gamma = None
    
    for gamma in gamma_values:
        # Approximate p-value adjustment for hidden bias
        # Under hidden bias with odds ratio Gamma, the p-value bounds are:
        # p_upper ≈ p_original * Gamma^(1/2) (simplified approximation)
        
        # More accurate approximation using normal distribution
        # This adjusts for the fact that hidden bias increases variance
        if original_p_value < 0.001:
            # Very small p-values - use more conservative adjustment
            adjusted_p = original_p_value * (gamma ** 0.5)
        else:
            # Standard adjustment
            adjusted_p = original_p_value * gamma
        
        # Check if adjusted p-value exceeds significance level
        if adjusted_p >= significance_level:
            critical_gamma = gamma
            break
    
    # Generate interpretation
    if original_p_value >= significance_level:
        interpretation = (
            f"The original result is not statistically significant (p = {original_p_value:.4f}). "
            f"Rosenbaum bounds analysis is not applicable when the null hypothesis is not rejected."
        )
    elif critical_gamma is None:
        interpretation = (
            f"Critical Gamma exceeds the tested range (Γ > 5.0). "
            f"The result remains significant even under substantial hidden bias assumptions. "
            f"(Original p-value: {original_p_value:.2e})"
        )
    else:
        interpretation = (
            f"The result would remain significant for hidden bias up to Γ = {critical_gamma:.1f}. "
            f"This means hidden bias would need to make the odds of treatment assignment "
            f"differ by a factor of {critical_gamma:.1f} or more between individuals "
            f"to alter the study's conclusions. "
            f"(Original p-value: {original_p_value:.2e})"
        )

    # Add practical interpretation with appropriate caveats
    if critical_gamma is None:
        strength = "high (Γ > 5.0)"
    elif critical_gamma >= 3.0:
        strength = "very strong"
    elif critical_gamma >= 2.0:
        strength = "strong"
    elif critical_gamma >= 1.5:
        strength = "moderate"
    else:
        strength = "weak"

    interpretation += f"\n\nRobustness level: {strength}."
    
    # Add methodological caveat for CATE estimates
    interpretation += (
        f"\n\n⚠️ Note: This is an approximation based on Wilcoxon signed-rank test applied to "
        f"CATE estimates. Classical Rosenbaum bounds are designed for matched observational "
        f"studies with observed paired differences. Results on ML-estimated treatment effects "
        f"should be interpreted with caution, as this method does not account for estimation "
        f"uncertainty in the CATE values themselves."
    )

    return {
        'original_p_value': original_p_value,
        'critical_gamma': critical_gamma,
        'interpretation': interpretation,
        'gamma_range_tested': (gamma_values[0], gamma_values[-1])
    }


def run_sensitivity_analysis(
    treatment_effects: np.ndarray,
    effect_name: str = "Treatment Effect",
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Run complete sensitivity analysis including E-values and Rosenbaum bounds.
    
    This is a convenience function that runs both E-value and Rosenbaum bounds
    analyses and returns a comprehensive sensitivity report.
    
    Parameters
    ----------
    treatment_effects : np.ndarray
        Array of individual treatment effects
    effect_name : str, default="Treatment Effect"
        Name of the effect being analyzed (for reporting)
    confidence_level : float, default=0.95
        Confidence level for interval estimation (typically 0.95)
        
    Returns
    -------
    dict
        Dictionary containing both E-value and Rosenbaum bounds results,
        plus summary statistics
    """
    # Calculate summary statistics
    mean_effect = np.nanmean(treatment_effects)
    std_effect = np.nanstd(treatment_effects)
    n_obs = np.sum(~np.isnan(treatment_effects))
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    se_effect = std_effect / np.sqrt(n_obs)
    t_critical = stats.t.ppf(1 - alpha/2, n_obs - 1)
    ci_lower = mean_effect - t_critical * se_effect
    ci_upper = mean_effect + t_critical * se_effect
    
    # Run E-value analysis
    evalue_results = calculate_evalue(
        estimate=mean_effect,
        confidence_interval_lower=ci_lower,
        confidence_interval_upper=ci_upper
    )
    
    # Run Rosenbaum bounds analysis
    rosenbaum_results = rosenbaum_bounds_approximation(treatment_effects)
    
    return {
        'effect_name': effect_name,
        'summary_stats': {
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'se_effect': se_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_observations': n_obs,
            'confidence_level': confidence_level
        },
        'evalue': evalue_results,
        'rosenbaum_bounds': rosenbaum_results
    }


def format_sensitivity_report(sensitivity_results: Dict[str, Any]) -> str:
    """
    Format sensitivity analysis results as a readable text report.
    
    Parameters
    ----------
    sensitivity_results : dict
        Results from run_sensitivity_analysis()
        
    Returns
    -------
    str
        Formatted text report
    """
    report_lines = [
        f"\n{'='*70}",
        f"Sensitivity Analysis: {sensitivity_results['effect_name']}",
        f"{'='*70}\n",
        "Summary Statistics:",
        f"  • Mean effect: {sensitivity_results['summary_stats']['mean_effect']:.4f}",
        f"  • Standard error: {sensitivity_results['summary_stats']['se_effect']:.4f}",
        f"  • {sensitivity_results['summary_stats']['confidence_level']*100:.0f}% CI: "
        f"[{sensitivity_results['summary_stats']['ci_lower']:.4f}, "
        f"{sensitivity_results['summary_stats']['ci_upper']:.4f}]",
        f"  • Sample size: {sensitivity_results['summary_stats']['n_observations']}",
        "",
        "E-value Analysis:",
        f"  • Point estimate E-value: {sensitivity_results['evalue']['evalue_point']:.2f}",
        f"  • Confidence interval E-value: {sensitivity_results['evalue']['evalue_ci']:.2f}",
        f"  {sensitivity_results['evalue']['interpretation']}",
        "",
        "Rosenbaum Bounds:",
        f"  • Original p-value: {sensitivity_results['rosenbaum_bounds']['original_p_value']:.6f}",
    ]
    
    if sensitivity_results['rosenbaum_bounds']['critical_gamma'] is not None:
        report_lines.append(
            f"  • Critical Gamma (Γ): {sensitivity_results['rosenbaum_bounds']['critical_gamma']:.1f}"
        )
    else:
        report_lines.append(f"  • Critical Gamma (Γ): Not found (very robust)")
    
    report_lines.extend([
        f"  {sensitivity_results['rosenbaum_bounds']['interpretation']}",
        f"\n{'='*70}\n"
    ])
    
    return "\n".join(report_lines)
