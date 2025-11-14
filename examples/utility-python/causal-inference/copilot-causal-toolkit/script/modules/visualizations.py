"""
Visualization utilities for causal inference analysis.

This module provides functions for:
- Dose-response curve plotting with confidence intervals
- Marginal effects visualization
- Subgroup comparison plots
- CATE tree visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_dose_response_curve(
    treatment_values: np.ndarray,
    ate_values: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    title: str = "Dose-Response Curve",
    xlabel: str = "Copilot Actions per Week",
    ylabel: str = "Change in Outcome (hours)",
    output_path: Optional[str] = None,
    baseline_values: Optional[np.ndarray] = None,
    baseline_ci_lower: Optional[np.ndarray] = None,
    baseline_ci_upper: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot dose-response curve with confidence intervals.
    
    Parameters
    ----------
    treatment_values : np.ndarray
        Treatment levels (x-axis)
    ate_values : np.ndarray
        ATE estimates at each treatment level (y-axis)
    ci_lower : np.ndarray
        Lower confidence interval bounds
    ci_upper : np.ndarray
        Upper confidence interval bounds
    title : str, default="Dose-Response Curve"
        Plot title
    xlabel : str, default="Copilot Actions per Week"
        X-axis label
    ylabel : str, default="Change in Outcome (hours)"
        Y-axis label
    output_path : str, optional
        Path to save the plot (if None, won't save)
    baseline_values : np.ndarray, optional
        Baseline model ATE values for comparison
    baseline_ci_lower : np.ndarray, optional
        Baseline model lower CI
    baseline_ci_upper : np.ndarray, optional
        Baseline model upper CI
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main ATE curve with confidence band
    ax.plot(treatment_values, ate_values, 
            linewidth=2.5, color='#0078D4', label='Featurized Model', marker='o')
    ax.fill_between(treatment_values, ci_lower, ci_upper, 
                     alpha=0.2, color='#0078D4', label='95% CI')
    
    # Plot baseline if provided
    if baseline_values is not None:
        ax.plot(treatment_values, baseline_values,
                linewidth=2, color='#D83B01', linestyle='--', 
                label='Baseline Model', marker='s', alpha=0.7)
        if baseline_ci_lower is not None and baseline_ci_upper is not None:
            ax.fill_between(treatment_values, baseline_ci_lower, baseline_ci_upper,
                           alpha=0.15, color='#D83B01')
    
    # Add zero reference line
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for interpretation
    if ate_values[-1] < 0:
        effect_direction = "Negative effect (reduction)"
    elif ate_values[-1] > 0:
        effect_direction = "Positive effect (increase)"
    else:
        effect_direction = "No effect"
    
    ax.text(0.02, 0.98, effect_direction, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dose-response plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_marginal_effects(
    treatment_values: np.ndarray,
    marginal_effects: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    title: str = "Marginal Effects",
    xlabel: str = "Copilot Actions per Week",
    ylabel: str = "Marginal Effect (hours per additional action)",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot marginal effects (derivatives of dose-response curve).
    
    Parameters
    ----------
    treatment_values : np.ndarray
        Treatment levels
    marginal_effects : np.ndarray
        Marginal effect estimates
    ci_lower : np.ndarray
        Lower confidence bounds
    ci_upper : np.ndarray
        Upper confidence bounds
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_path : str, optional
        Path to save plot
    figsize : tuple
        Figure size
    show_plot : bool
        Whether to display plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot marginal effects
    ax.plot(treatment_values, marginal_effects,
            linewidth=2.5, color='#107C10', marker='o', label='Marginal Effect')
    ax.fill_between(treatment_values, ci_lower, ci_upper,
                     alpha=0.2, color='#107C10', label='95% CI')
    
    # Zero reference line
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved marginal effects plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_subgroup_comparison(
    subgroup_df: pd.DataFrame,
    effect_col: str = 'mean_effect',
    subgroup_col: str = 'name',
    ci_lower_col: str = 'ci_lower',
    ci_upper_col: str = 'ci_upper',
    title: str = "Treatment Effects by Subgroup",
    xlabel: str = "Treatment Effect (hours)",
    output_path: Optional[str] = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 8),
    show_plot: bool = True
) -> plt.Figure:
    """
    Create horizontal bar plot comparing treatment effects across subgroups.
    
    Parameters
    ----------
    subgroup_df : pd.DataFrame
        Dataframe with subgroup analysis results
    effect_col : str, default='mean_effect'
        Column containing effect estimates
    subgroup_col : str, default='name'
        Column containing subgroup names
    ci_lower_col : str, default='ci_lower'
        Column with lower CI bounds
    ci_upper_col : str, default='ci_upper'
        Column with upper CI bounds
    title : str
        Plot title
    xlabel : str
        X-axis label
    output_path : str, optional
        Path to save plot
    top_n : int, default=10
        Number of top subgroups to show
    figsize : tuple
        Figure size
    show_plot : bool
        Whether to display plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Take top N subgroups
    plot_data = subgroup_df.head(top_n).copy()
    
    # Calculate error bars
    plot_data['error_lower'] = plot_data[effect_col] - plot_data[ci_lower_col]
    plot_data['error_upper'] = plot_data[ci_upper_col] - plot_data[effect_col]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(plot_data))
    colors = ['#D13438' if x < 0 else '#107C10' for x in plot_data[effect_col]]
    
    ax.barh(y_pos, plot_data[effect_col], color=colors, alpha=0.7)
    
    # Add error bars
    ax.errorbar(
        plot_data[effect_col], y_pos,
        xerr=[plot_data['error_lower'], plot_data['error_upper']],
        fmt='none', ecolor='black', capsize=5, linewidth=2
    )
    
    # Add zero reference line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data[subgroup_col], fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        value = row[effect_col]
        ax.text(value, i, f' {value:.3f}', 
                va='center', ha='left' if value > 0 else 'right',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subgroup comparison plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_treatment_distribution(
    data: pd.DataFrame,
    treatment_var: str,
    title: str = "Treatment Distribution",
    xlabel: str = "Copilot Actions per Week",
    output_path: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot histogram of treatment variable distribution.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with treatment variable
    treatment_var : str
        Name of treatment column
    title : str
        Plot title
    xlabel : str
        X-axis label
    output_path : str, optional
        Path to save plot
    bins : int, default=30
        Number of histogram bins
    figsize : tuple
        Figure size
    show_plot : bool
        Whether to display plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    treatment = data[treatment_var].dropna()
    
    # Create histogram
    ax.hist(treatment, bins=bins, color='#0078D4', alpha=0.7, edgecolor='black')
    
    # Add summary statistics as vertical lines
    mean_val = treatment.mean()
    median_val = treatment.median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add text box with summary stats
    stats_text = f'n = {len(treatment):,}\nStd = {treatment.std():.2f}\nMin = {treatment.min():.1f}\nMax = {treatment.max():.1f}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved treatment distribution plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_outcome_vs_treatment_scatter(
    data: pd.DataFrame,
    treatment_var: str,
    outcome_var: str,
    title: str = "Outcome vs Treatment",
    xlabel: str = "Copilot Actions per Week",
    ylabel: str = "Outcome (hours)",
    output_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True
) -> plt.Figure:
    """
    Create scatter plot of outcome vs treatment with trend line.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with treatment and outcome
    treatment_var : str
        Treatment variable column
    outcome_var : str
        Outcome variable column
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    output_path : str, optional
        Path to save plot
    sample_size : int, optional
        Max number of points to plot (for large datasets)
    figsize : tuple
        Figure size
    show_plot : bool
        Whether to display plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Sample data if needed
    plot_data = data[[treatment_var, outcome_var]].dropna()
    if sample_size and len(plot_data) > sample_size:
        plot_data = plot_data.sample(n=sample_size, random_state=42)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(plot_data[treatment_var], plot_data[outcome_var],
               alpha=0.3, s=20, color='#0078D4')
    
    # Add trend line
    z = np.polyfit(plot_data[treatment_var], plot_data[outcome_var], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_data[treatment_var].min(), 
                         plot_data[treatment_var].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, 
            label=f'Trend: y = {z[0]:.3f}x + {z[1]:.2f}')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add correlation coefficient
    corr = plot_data[treatment_var].corr(plot_data[outcome_var])
    ax.text(0.02, 0.98, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def create_analysis_summary_plot(
    ate_results: Dict[str, Any],
    subgroup_results: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    show_plot: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple subplots.
    
    Parameters
    ----------
    ate_results : dict
        Dictionary with ATE analysis results
    subgroup_results : pd.DataFrame
        Subgroup analysis results
    output_path : str, optional
        Path to save plot
    figsize : tuple
        Figure size
    show_plot : bool
        Whether to display plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Dose-response curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ate_results['treatment_values'], ate_results['ate_featurized'],
             linewidth=2.5, color='#0078D4', marker='o', label='ATE')
    ax1.fill_between(ate_results['treatment_values'],
                      ate_results['ci_lower_featurized'],
                      ate_results['ci_upper_featurized'],
                      alpha=0.2, color='#0078D4', label='95% CI')
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('Treatment Level', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Effect Size', fontsize=11, fontweight='bold')
    ax1.set_title('Overall Dose-Response Curve', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Top subgroups
    ax2 = fig.add_subplot(gs[1, 0])
    top_5 = subgroup_results.head(5)
    colors = ['#D13438' if x < 0 else '#107C10' for x in top_5['mean_effect']]
    ax2.barh(range(len(top_5)), top_5['mean_effect'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_5)))
    ax2.set_yticklabels(top_5['name'], fontsize=8)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Effect Size', fontsize=11, fontweight='bold')
    ax2.set_title('Top 5 Subgroups', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Subplot 3: Effect distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(subgroup_results['mean_effect'], bins=20, 
             color='#0078D4', alpha=0.7, edgecolor='black')
    ax3.axvline(subgroup_results['mean_effect'].mean(), 
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Effect Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Subgroup Effects', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Causal Inference Analysis Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig
