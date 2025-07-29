---
layout: default
title: "Causal Inference: Experimentation Guide"
permalink: /causal-inference-experimentation-guide/
---

# Guidance on Causal Inference with Copilot Analytics

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide default Minima navigation to prevent duplicates */
.site-header .site-nav,
.site-header .trigger,
.site-header .page-link {
  display: none !important;
}
</style>

This is a comprehensive guide on how to run a causal inference (or treatment effect estimation) analysis with Copilot Analytics. 

## What is causal inference?

Causal inference is a statistical methodology that helps us determine whether one event actually causes another, rather than just observing that they happen together. In the context of organizational data, it allows us to distinguish between correlation and causation by controlling for confounding variables and establishing the direction of causality.

Unlike simple comparisons that might show "users with Copilot are more productive," causal inference answers the more precise question: "how much more productive would users become *if* they were given Copilot access?" This distinction is crucial for making informed business decisions about technology investments.

## Why run causal inference for Copilot Analytics?

Causal inference is essential for Copilot Analytics because:

1. Investment Justification: Provide robust evidence of Copilot's ROI by isolating its true impact from other factors like training, team composition, or seasonal trends.

2. Targeted Deployment: Identify which employee segments (e.g., senior developers, specific functions, regions) benefit most from Copilot, enabling strategic rollout decisions.

3. Policy Optimization: Understand whether observed productivity gains come from Copilot licensing itself, training programs, or their interaction, informing enablement strategies.

4. Confounding Control: Account for selection bias where high-performing teams might be more likely to adopt new tools, ensuring accurate impact measurement.

5. Temporal Dynamics: Distinguish between immediate adoption effects and sustained productivity improvements over time.

## Outcomes of the causal inference analysis

At the end of the causal inference analysis, we will be able to understand:

* Average Treatment Effects (ATE): The overall impact of Copilot licensing on external collaboration hours across the entire organization
* Conditional Average Treatment Effects (CATE): How different groups in an organization benefit from using Copilot, enabling targeted enablement strategies  
* Heterogeneous Impact Detection: Which employee characteristics (level, function, region, network size) predict higher or lower treatment effects
* Confidence Intervals: Statistical uncertainty bounds around our estimates to support decision-making
* Subgroup Discovery: Automatically identified cohorts where Copilot has the most meaningful impact

## Experiment design

In causal inference, it is important to identify three sets of variables as part of the design:

* outcome (Y): The dependent variable we want to measure the effect on
* treatment (X): The intervention or policy change we want to evaluate  
* confounders (W): Variables that affect both treatment assignment and outcomes, which must be controlled for

## The outcome variable

In our analysis, we will be using external collaboration hours as the outcome variable. This variable is selected as it is a good success measure proxy for external-facing or sales-focused employees, and it is also available as a native Viva Insights metric that doesn't require further importing. 

External collaboration hours measures the time employees spend in meetings, emails, and other collaborative activities with people outside their immediate organization. This metric is particularly valuable because:

- Business Impact: It directly relates to client engagement, partnership building, and revenue-generating activities
- Measurable: Automatically tracked through Microsoft 365 without requiring manual data collection
- Sensitive to Change: Likely to show measurable differences as employees become more efficient with Copilot assistance

For non-sales focused scenarios, other outcome variables such as number of tickets closed, total hours, projects closed, coding productivity, or meeting efficiency may be used instead.

## The treatment variable  

The treatment variable represents Copilot usage intensity, measured as continuous variables such as:

- "Teams Copilot Usage" (hours per week)
- "GitHub Copilot Usage" (hours per week)  
- "M365 Copilot Usage" (hours per week)

We model treatment as continuous rather than binary (user vs. non-user) because:

- Dosage Effects: Higher usage levels may yield different benefits than minimal usage
- Realistic Adoption: Reflects actual usage patterns where employees adopt tools gradually
- Non-linear Relationships: Allows detection of diminishing returns or threshold effects

## Confounder variables and their importance

Confounders are variables that influence both who gets treated (Copilot access/usage) and the outcome (external collaboration hours). Controlling for confounders is critical because without them, we might incorrectly attribute changes to Copilot when they're actually due to other factors.

Key confounding categories include:

- HR Attributes: Level designation, tenure, region, function, organization, and manager status affect both likelihood of Copilot access and baseline collaboration patterns.

- Collaboration Metrics: Network size, strong ties, meeting hours, and focus time capture individual work styles that predict both Copilot adoption and collaboration outcomes.

- Temporal Factors: Seasonal trends, organizational changes, and training programs that coincide with Copilot rollout.

- Selection Bias: High-performing teams or individuals may be more likely to adopt new tools, creating spurious correlations.

Without proper confounder control, we risk measuring "who gets Copilot" effects rather than "what Copilot does" effects.

## Choosing the right organizational attributes

Selecting appropriate organizational attributes for the analysis is crucial for obtaining valid causal estimates. The choice of variables should be guided by:

- Domain Knowledge: Include variables that organizational leaders and HR experts identify as important predictors of both Copilot adoption and collaboration patterns.

- Data Availability: Ensure all selected attributes are consistently available across the study period and population.

- Balance vs. Relevance: Include enough variables to control confounding, but avoid including variables that are outcomes themselves or perfect predictors of treatment.

### Importance of validating the data

Before running any causal analysis, it's essential to validate that your data meets the necessary requirements:

- Completeness: Check for missing values in key variables, especially treatment, outcome, and critical confounders. High missingness can bias results.

- Temporal Alignment: Ensure measurement timing is correct - confounders should be measured before or at treatment initiation, not after.

- Treatment Variation: Verify sufficient variation in treatment levels. If everyone has similar Copilot usage, detecting effects becomes difficult.

- Control Group: Confirm that you have adequate control observations (non-users or low-usage users) for comparison.

- Outliers: Identify and handle extreme values that might disproportionately influence results.

- Covariate Balance: Check whether treatment and control groups are similar on observable characteristics before intervention.

## Techniques 

There are many different types of causal inference techniques. In this analysis, we will be focusing on: 

1. **Difference in differences (DiD)**
2. **Interrupted time-series analysis (ITSA)**
3. **Double machine learning causal forest (CausalForestDML)**

### Why these techniques?

These three methods complement each other and provide increasing levels of sophistication:

* Progressive Validation: Starting with DiD establishes a baseline estimate, ITSA adds temporal nuance, and CausalForestDML enables subgroup discovery.

* Assumption Robustness: Each method makes different assumptions, so agreement across methods strengthens confidence in results.

* Stakeholder Communication: DiD provides intuitive explanations, while CausalForestDML offers actionable subgroup insights.

* Methodological Rigor: This multi-method approach follows best practices in causal inference research.

### DiD (Difference-in-Differences)

#### What it is

DiD compares changes in outcomes between treatment and control groups over time, controlling for both group-specific and time-specific factors.

#### How it works

The method estimates the treatment effect as: (Treatment Group Post - Treatment Group Pre) - (Control Group Post - Control Group Pre).

#### Key assumptions

- Parallel trends: Treatment and control groups would have followed similar trends in the absence of treatment
- No spillover effects: Treatment of one group doesn't affect control group outcomes  
- Treatment timing: The intervention timing is clearly defined and exogenous

#### Why use DiD for Copilot

- Handles selection bias where certain teams are more likely to get Copilot access
- Controls for time trends that affect all employees (e.g., seasonal patterns, organizational changes)
- Provides interpretable estimates that stakeholders can easily understand
- Works well with staggered treatment rollouts common in enterprise software deployments

### ITSA (Interrupted Time Series Analysis)

#### What it is
ITSA models the outcome trend before intervention and detects changes in level and slope after intervention, using control series for comparison.

#### How it works
Fits separate regression lines to pre- and post-intervention periods, testing for:

- Level change: Immediate jump in outcome at intervention time
- Slope change: Change in the trend (rate of improvement) post-intervention

#### Key assumptions

- Stable trend: The pre-intervention trend would have continued without treatment
- No other interventions: No major confounding events occur at the same time
- Sufficient data points: Adequate observations before and after intervention

#### Why use ITSA for Copilot

- Captures both immediate and gradual effects of Copilot adoption
- Reveals whether benefits are sustained or fade over time
- Handles autocorrelation in weekly/monthly measurements common in collaboration data
- Identifies the time course of impact, informing training and support strategies

### CausalForestDML (Double Machine Learning + Causal Forest)

#### What it is
A machine learning approach that estimates heterogeneous treatment effects using random forests while maintaining statistical rigor through double machine learning.

#### How it works

1. Double ML: Uses machine learning to model both outcome and treatment, removing bias from model misspecification
2. Causal Forest: Builds an ensemble of trees to estimate treatment effects that vary by individual characteristics  
3. Tree Interpretation: Applies decision tree algorithms to discover which subgroups have high vs. low treatment effects

#### Key assumptions

- Unconfoundedness: All relevant confounders are observed and included
- Overlap: Every individual has positive probability of receiving any treatment level
- Consistency: Treatment effects are well-defined and stable

#### Why use CausalForestDML for Copilot

- Personalization: Identifies which employee characteristics predict higher Copilot benefits
- Targeting: Enables strategic rollout to segments most likely to benefit
- Non-linear effects: Captures complex interactions between employee attributes and treatment response
- Individual-level estimates: Provides treatment effect estimates for each employee, supporting personalized recommendations
- Robustness: Double ML framework reduces bias from flexible machine learning models

We will use CausalForestDML for estimating Conditional Average Treatment Effects (CATEs), enabling us to detect heterogeneous impacts of Copilot usage at personal level. To support interpretability and subgroup discovery, we will also apply SingleTreeCateInterpreter algorithm, which extracts representative cohorts based on HR attributes and/or collaboration metrics. This combined approach allows us to identify where Copilot usage has the most meaningful effect and supports targeted enablement strategies.

## Running the analysis 

### Pre-requisites

#### Required Software

- Python 3.8+: Download from [python.org](https://python.org) or install via your preferred package manager
- VS Code (Recommended): Provides excellent support for Jupyter notebooks and Python development
- Git: For cloning the repository and version control

#### Python Packages

The analysis requires several specialized packages. Install them using:
```bash
pip install -r requirements.txt
```

Key dependencies include:

- `econml`: Microsoft's causal inference library
- `pandas` & `numpy`: Data manipulation and numerical computing
- `scikit-learn`: Machine learning utilities  
- `matplotlib` & `seaborn`: Visualization
- `scipy` & `statsmodels`: Statistical analysis

#### What is a Jupyter Notebook?

Jupyter notebooks are interactive documents that combine code, visualizations, and explanatory text. They're ideal for data analysis because you can:

- Run code step-by-step and see immediate results
- Document your analysis process with markdown text
- Create inline plots and tables
- Share reproducible analysis workflows

Jupyter notebooks have the extension `.ipynb`.

### Data requirements

This section outlines the data requirements for a quasi-experimental analysis using the CausalForestDML methodology to estimate heterogeneous effects of Copilot usage on External Collaboration Hours.

All relevant confounding variables (covariates) must be collected to control for other factors that could influence the outcome. This data will be measured on a weekly basis where applicable. The required data fall into three categories:

1. HR attributes 
2. Collaboration metrics (Viva Insights)
3. Copilot training information

💡 **Note**: Each HR attribute is a potential confounder to account for differences in workforce composition (e.g. seniority, region) between groups. The Viva Insights (VI) metrics are weekly collaboration measures that capture work patterns (network sizes, after-hours work, focus time, etc.) including Copilot usage activity. The Copilot training data indicates which users received Copilot training and how many hours, providing context on enablement efforts.

### How to run

#### Step 1: Download the Person Query from Viva Insights

1. Navigate to Viva Insights Person Query 
   - Access your Viva Insights workspace
   - Go to `Analyze` > `Query designer` > `Person query`

2. Set the time range  
   - Select 1 month before the intervention (pre-intervention baseline) and 3 months after the intervention
   - The intervention date should be when Copilot licenses were first assigned to users
   - This 4-month window provides sufficient data for before/after comparison

3. Select required metrics by category  
   Click "Add metrics" and choose from these specific groupings:

   | **Metric Category** | **Required Fields** |
   |---------------------|---------------------|
   | **Collaboration network** | Internal Network Size; External Network Size; Strong Ties; Diverse Ties |
   | **After hours collaboration** | After-hours Meeting Hours; After-hours Email Hours; Available-to-focus Hours |
   | **Collaboration by day of week** | Weekend Collaboration Hours |
   | **Learning time** | Calendared Learning Time |
   | **Collaboration activity** | Active Connected Hours |
   | **External collaboration** | External 1:1 Meeting Hours |
   | **Focus metrics** | Uninterrupted Hours |
   | **Microsoft 365 Copilot** | **Select all metrics** (this captures all Copilot usage data) |

4. Configure analysis attributes
   - Set `IsActive = True` to include only active employees
   - Select the following employee attributes:

   | Attribute Category | Required Fields |
   |------------------------|---------------------|
   | HR Attributes | Level Designation; Tenure; Region; Function; Org; SupervisorIndicator (Manager vs. Individual Contributor) |
   | Additional Data (Copilot training) | Copilot Training Participation (Yes/No); Copilot Training Duration (hours per week, if available) |

5. Export and save the data as a CSV file

#### Step 2: Update the Jupyter notebook with new file paths

1. Open `Treatment_Featurization_VI_v42_spline_simdata.ipynb` in VS Code or Jupyter
2. Locate the data loading cell (typically near the top):
   ```python
   data = pd.read_csv("data/synthetic_employees_data_v32.csv")
   ```
3. Replace the file path with your exported Viva Insights data:
   ```python
   data = pd.read_csv("path/to/your/viva_insights_export.csv")
   ```
4. Update any outcome or treatment variable names to match your data column names
5. Run all cells in sequence to perform the analysis

#### Alternative: Using Command-Line Tools

For more advanced users, you can use the specialized command-line tools in the `vi_ate_cate` folder:

```bash
# For Average Treatment Effect analysis
python main.py ate --data-file "your_data.csv" --treatment-var "Teams Copilot Usage"

# For subgroup analysis (CATE)  
python main.py cate --data-file "your_data.csv" --treatment-var "Teams Copilot Usage"
```

## Evaluating the outputs

The analysis generates several types of outputs across the different methodologies. Understanding these outputs is crucial for making informed decisions about Copilot deployment and optimization.

#### Average Treatment Effect (ATE)

#### What is it?

The Average Treatment Effect represents the expected change in outcome (external collaboration hours) for a randomly selected individual if they were to increase their Copilot usage from a baseline level (typically 0) to a specific treatment level.

#### Key output files:

- `ate_results_{treatment}.csv`: Detailed results for each treatment level tested
- `ate_results_{treatment}.json`: Summary statistics and key insights
- `ate_plot_{treatment}.png`: Visual treatment effect curve showing how effects vary by usage intensity
- `summary_report_{treatment}.json`: Complete analysis metadata including model performance

#### How to interpret:

- Effect Size: The numerical impact (e.g., "+0.5 hours per week")
- Confidence Intervals: Range of plausible effect sizes (typically 95% CI)  
- P-values: Statistical significance of the effect
- Treatment Curve: How benefits change with different usage levels (linear, diminishing returns, threshold effects)

#### Individual vs. Average vs. Conditional Effects:

- Individual Treatment Effect (ITE): The effect for a specific person (usually unobservable)
- Average Treatment Effect (ATE): The average effect across the entire population
- Conditional Average Treatment Effect (CATE): The average effect within specific subgroups (e.g., senior vs. junior employees)

#### DiD Analysis Outputs

The DiD analysis provides several specifications with increasing levels of control:

- Basic DiD: Raw before/after comparison between treatment and control groups
- HR-Controlled DiD: Controls for organizational factors (level, region, function)
- Training-Controlled DiD: Isolates licensing effects by controlling for training participation
- Fully-Controlled DiD: Most comprehensive specification including all confounders

#### Key metrics to examine:

- Treatment coefficient: The estimated DiD effect
- P-value: Statistical significance (typically want p < 0.05)
- Confidence intervals: Uncertainty bounds around the estimate
- R-squared: How well the model explains outcome variation
- Parallel trends test: Whether the assumption is satisfied (p > 0.05 preferred)

#### ITSA Analysis Outputs

ITSA reveals the temporal dynamics of treatment effects:

- Level change: Immediate jump in outcome at intervention time
- Slope change: Change in trend (acceleration/deceleration) post-intervention
- Sustained effects: Whether benefits persist or fade over time

#### Interpretation guidelines:

- Immediate effects: Look for significant level changes
- Growing benefits: Positive slope changes indicate accelerating gains
- Diminishing returns: Negative slope changes suggest benefits plateau
- Persistence: Compare early vs. late post-intervention periods

#### CATE Analysis Outputs

CATE analysis identifies heterogeneous treatment effects across subgroups:

#### Key output files:

- `cate_results_{treatment}.csv`: Subgroup-specific treatment effects
- `cate_results_{treatment}.json`: Decision tree subgroups with effect sizes
- `cate_interpretation_{treatment}.png`: Decision tree visualization showing subgroup rules
- Individual-level effect estimates for personalized recommendations

#### Subgroup interpretation:

- High-impact subgroups: Groups with large positive effects (priority for rollout)
- Low/negative-impact subgroups: Groups where Copilot may not be beneficial
- Subgroup conditions: The specific characteristics defining each group
- Sample sizes: Ensure subgroups are large enough for reliable inference

## Best practices

#### Statistical Rigor

- Review confidence intervals: Don't rely solely on point estimates. Wide confidence intervals suggest high uncertainty.
- Multiple testing correction: When examining many subgroups, adjust p-values to avoid false discoveries.
- Effect size vs. significance: A statistically significant but tiny effect may not be practically meaningful.
- Robustness checks: Compare results across different specifications (basic vs. controlled models).

#### Domain Validation  

- Validate findings with domain knowledge: Do the results align with organizational intuition about who benefits from Copilot?
- Triangulate with qualitative evidence: Supplement quantitative results with user interviews and case studies.
- Consider external validity: Will these results generalize to other teams, time periods, or organizations?

#### Implementation Considerations

- Prioritize high-confidence subgroups: Focus rollout on groups with large, statistically significant effects.
- Monitor temporal patterns: Use ITSA results to understand optimal training timing and support needs.
- Iterative refinement: Use initial results to inform subsequent analysis waves with better data or refined questions.
- Stakeholder communication: Present results in business terms (e.g., "productivity gains") rather than statistical jargon.
