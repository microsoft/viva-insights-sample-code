---
layout: default
title: "Copilot Causal Toolkit"
permalink: /copilot-causal-toolkit/
---

# Copilot Causal Toolkit

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

The [Copilot Causal Toolkit](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/copilot-causal-toolkit) enables you to run causal inference analysis with Viva Insights data. For example, this can be used to address: 

* Does Copilot usage increase the time that our sellers spend with our customers?
* Does Copilot usage impact our team's wellbeing positively? 

Two key scenarios - Seller Productivity and Burnout Prevention - are covered in this toolkit. Both involve using Copilot usage (`Total_Copilot_actions_taken`) as a treatment variable, and evaluating the treatment effect of using Copilot on an outcome variable (e.g. external or after-hours collaboration hours). 

You can access the toolkit [here](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/copilot-causal-toolkit) on GitHub.

**Python** and **Jupyter notebooks** (.ipynb) are used for running the analysis. It is therefore recommended that you read the pre-requisites section and ensure that all of these are satisfied. 

## Directory

The `copilot-causal-toolkit` directory is organized as follows:

```
copilot-causal-toolkit/
├── data/                          # Directory for input data files
│   └── (Place your CSV files here - Person Query or Super Users Report exports)
│
├── script/                        # Main analysis scripts and modules
│   ├── CI-DML_AftCollabHours_PQ.ipynb     # After-hours collab analysis (Person Query)
│   ├── CI-DML_AftCollabHours_SUR.ipynb    # After-hours collab analysis (Super Users Report)
│   ├── CI-DML_ExtCollabHours_PQ.ipynb     # External collab analysis (Person Query)
│   ├── CI-DML_ExtCollabHours_SUR.ipynb    # External collab analysis (Super Users Report)
│   └── modules/                   # Helper modules for DML analysis
│       ├── data_processor.py      # Data cleaning and preprocessing functions
│       ├── data_filter.py         # Data filtering and aggregation utilities
│       ├── estimator.py           # DML estimation and causal inference functions
│       ├── output_manager.py      # Output generation functions
│       ├── subgroup_analysis.py   # Subgroup identification and comparison
│       ├── sensitivity_analysis.py # E-values and Rosenbaum bounds
│       └── visualizations.py      # Plotting and visualization functions
│
├── output/                        # Directory for analysis outputs
│   └── (Analysis results, plots, and reports will be saved here)
│
└── README.md                      # Documentation file
```

**Key directories:**

- `data/`: Place your Viva Insights data exports here (CSV format)
- `script/`: Contains the Jupyter notebooks for running causal inference analysis
- `script/modules/`: Python modules with reusable functions for the analysis pipeline
- `output/`: All analysis outputs (results tables, plots, HTML reports) are saved here

This directory contains four main Jupyter notebooks (.ipynb) for running causal inference analysis on top of Viva Insights data. Two work directly with a Person Query (PQ) schema: 

* `CI-DML_AftCollabHours_PQ.ipynb` (Uses After-hours collaboration hours as outcome)
* `CI-DML_ExtCollabHours_PQ.ipynb` (Uses External collaboration hours as outcome)

And two of these work from an output from the Super Users Report (SUR):

* `CI-DML_AftCollabHours_SUR.ipynb` (Uses After-hours collaboration hours as outcome)
* `CI-DML_ExtCollabHours_SUR.ipynb` (Uses External collaboration hours as outcome)

We recommend creating a copy of the template that you wish to use and renaming them to match the scenario of your analysis. 

## Getting Started

### Step 1: Download or Clone the Repository

First, you'll need to get a copy of the causal inference package on your local computer:

**Option A: Download as ZIP** (Recommended for beginners)
1. Click the green "Code" button at the top of the repository
2. Select "Download ZIP"
3. Extract the ZIP file to a location on your computer (e.g., `C:\Users\YourName\Documents\repository-name`)
4. Navigate to the `copilot-causal-toolkit` subdirectory within the extracted folder
5. Remember this `copilot-causal-toolkit` location—it will be your **working directory**

**Option B: Clone with Git** (If you have Git installed)
```bash
git clone [repository-url]
cd [repository-name]/copilot-causal-toolkit
```

**Note:** The `copilot-causal-toolkit` folder may be located within a larger repository. Make sure you're working from the `copilot-causal-toolkit` subdirectory specifically, as this is where the analysis notebooks and modules are located.

### Step 2: Open the Project in Your Editor

**Using Visual Studio Code (Recommended):**
1. Open VS Code
2. Go to `File` → `Open Folder...`
3. Navigate to and select the `copilot-causal-toolkit` folder specifically (not the parent repository)
   - If you cloned the full repository, make sure to navigate into the `copilot-causal-toolkit` subdirectory
   - Example path: `C:\Users\YourName\Documents\repository-name\copilot-causal-toolkit`
4. VS Code will now show the copilot-causal-toolkit structure in the sidebar (with `data/`, `script/`, `output/` folders)

**Important:** Make sure you open the `copilot-causal-toolkit` folder itself, not the parent repository folder. This ensures all relative file paths in the notebooks work correctly.

Once your project is open, you can proceed to set up the prerequisites and data.

## Setting up

### Pre-requisites

Before running the causal inference analysis, ensure you have the following installed on your computer:

#### 1. Python Installation

You'll need **Python 3.8 or higher** (Python 3.9 or 3.10 recommended). 

To check if Python is already installed:

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac) and run:

```bash
python --version
```

If you don't have Python or need to upgrade, download it from [python.org](https://www.python.org/downloads/). During installation, make sure to check the box that says "Add Python to PATH".

#### 2. Jupyter Notebook

Jupyter allows you to run `.ipynb` notebook files interactively. Install it using:

```bash
pip install jupyter
```

Alternatively, you can use **Visual Studio Code** with the Python and Jupyter extensions, which provides a more integrated experience.

#### 3. Required Python Packages

The analysis requires several Python packages. Copy and paste the following command to install all required packages at once:

```bash
pip install numpy pandas matplotlib scipy scikit-learn econml vivainsights
```

**Package purposes:**

- `numpy` & `pandas` - Data manipulation and numerical computing
- `matplotlib` - Creating plots and visualizations
- `scipy` - Statistical functions and tests
- `scikit-learn` - Machine learning models and preprocessing
- `econml` - Causal inference methods (Double Machine Learning)
- `vivainsights` - Microsoft Viva Insights data loading and analysis

**Installation time:** This may take 5-10 minutes depending on your internet connection.

#### 4. Verify Installation

To verify all packages are installed correctly, run:

```python
import numpy
import pandas
import matplotlib
import scipy
import sklearn
import econml
import vivainsights
print("✓ All packages installed successfully!")
```

You can run this in a Python terminal or in a Jupyter notebook cell.

#### 5. Recommended: Virtual Environment (Optional but Best Practice)

If you're familiar with Python environments, we recommend creating a virtual environment to avoid package conflicts:

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Then install packages
pip install numpy pandas matplotlib scipy scikit-learn econml vivainsights jupyter
```

### Downloading the data

<!-- Where to get the data, and where to save it -->

There are two means of obtaining data for this causal inference analysis: 

1. export a Person Query as a csv file from Viva Insights
2. export a csv file from an existing Super Users Report (if available)

For best results, we generally recommend the first method as this ensures that we have comprehensive coverage of all the covariates required in order to produce a causal inference result that we can be confident in. The second method can be quicker and immediately extracted from a Super Users report as it does not require the running and setting up of a fresh query. 

#### Method 1: Export query from Viva Insights

1. Open: <https://analysis.insights.viva.office.com/Analysis/CreateAnalysis>
2. Select Person Query → 'Set up analysis'.
3. Configure:
   * Time period: Last 6 months (rolling)
   * Group by: Week
   * Metrics: See sub-step 4 for required attribute selection.
   * Filter: Is Active = True (if available) - You can validate the number of employees here.
   * Attributes: Include Organization and Function Type (others optional) - this is the last box on this page.
4. Select required metrics (see below section on what columns to include)
5. Save and Run query. Wait until **Status = Completed**.
  
#### Method 2: Export query from Super Users Report

This second method assumes that you already have a Super Users Report (.pbit/.pbix) populated with data from Viva Insights. The Super Users Report contains pre-aggregated data that can be quickly exported without setting up a new Person Query.

**Steps to export data from Power BI:**

1. **Open your Super Users Report** in Power BI Desktop
   - If you don't have the report file, ask your Viva Insights admin or check your organization's shared workspace

2. **Navigate to the data view**
   - In Power BI Desktop, click the **Data** icon (table icon) on the left sidebar
   - This shows all tables in your report

3. **Find the appropriate data table**
   - Look for a table containing person-level data with columns like:
     - `PersonId`
     - `Date` (the metric date/week)
     - `Total_Copilot_actions_taken`
     - Collaboration metrics (hours, meeting hours, etc.)
     - Organizational attributes
   - Common table names: `PersonMetrics`, `UserData`, or similar

4. **Export the table to CSV**
   - **Option A:** Right-click on the table name → Select **"Copy table"** → Paste into Excel → Save as CSV
   - **Option B:** Click on the table → Go to **Home** tab → Click **"Transform data"** to open Power Query Editor → In Power Query: Right-click the table → **"Export"** → Choose CSV format

5. **Verify the export**
   - Open the CSV file to ensure it contains:
     - Multiple weeks of data (ideally 12-26 weeks)
     - All required columns listed in the scenarios below
     - No excessive blank rows or formatting issues

6. **Save to the data folder**
   - Move the exported CSV file to `copilot-causal-toolkit/data/`
   - Rename it something descriptive (e.g., `SuperUsersReport_Export_2025.csv`)

**Important notes for Super Users Report data:**

- The Super Users Report uses `Date` instead of `MetricDate` as the date column
- The SUR notebooks (ending in `_SUR.ipynb`) are specifically designed to handle this schema difference
- Some metrics available in Person Query may not be available in the Super Users Report (e.g., `Available_to_focus_hours`, `Weekend_collaboration_hours`)
- If critical metrics are missing, consider using Method 1 (Person Query) instead

**Alternative: Export from Power BI Service (Online)**

If your report is published to Power BI Service (online):

1. Open the report at powerbi.com
2. Navigate to the relevant page showing the data table
3. Click the **"..."** (More options) on a visual showing the underlying data
4. Select **"Export data"** → Choose **"Underlying data"** → **".csv"** format
5. Download and save to `copilot-causal-toolkit/data/`

**Note:** Some Power BI reports may have export restrictions depending on your organization's policies. Contact your Power BI admin if you cannot export data.

#### Columns to include: Seller Productivity Scenario

In this scenario, the goal is to understand how increasing Copilot usage increases or decreases time spent collaborating with external stakeholders and customers.

* **Outcome variable:** `External_collaboration_hours` - Total hours spent in meetings, emails, chats, and calls with people outside the organization
* **Treatment variable:** `Total_Copilot_actions_taken` - Total number of Copilot interactions per person per week
* **Confounder variables (time-varying controls):**
  - `Meeting_hours` - Time spent in meetings
  - `Email_hours` - Time spent sending and reading emails  
  - `Chat_hours` - Time spent in instant messages/chats
  - `Collaboration_hours` - Total collaboration time (meetings + emails + chats)
  - `Internal_network_size` - Number of distinct internal collaborators
  - `Networking_outside_organization` - Number of external contacts interacted with
  - `Total_focus_hours` - Time available for deep work without interruptions
  - Other relevant behavioral and network metrics from your Person Query
* **Organizational attributes:** These are used for heterogeneity analysis (identifying which subgroups benefit most/least from Copilot). Include as many as possible, typically covering dimensions like `Organization`, `Function`, `Level`, `IsManager`, and `Area`. The exact names can differ based on your organization's HR data structure—just ensure the variable names in the notebook are updated accordingly in the configuration section.

#### Columns to include: Burnout Prevention Scenario

In this scenario, the goal is to understand how increasing Copilot usage increases or decreases after-hours work patterns, which can impact employee wellbeing and burnout risk.

* **Outcome variable:** `After_hours_collaboration_hours` - Time spent in work-related activities outside standard business hours
* **Treatment variable:** `Total_Copilot_actions_taken` - Total number of Copilot interactions per person per week
* **Confounder variables (time-varying controls):**
  - `Meeting_hours` - Time spent in meetings
  - `Email_hours` - Time spent sending and reading emails
  - `Chat_hours` - Time spent in instant messages/chats
  - `Collaboration_hours` - Total collaboration time (meetings + emails + chats)
  - `Internal_network_size` - Number of distinct internal collaborators
  - `Networking_outside_organization` - Number of external contacts interacted with
  - `Total_focus_hours` - Time available for deep work without interruptions
  - `Workweek_span` - Total hours between first and last work activity
  - Other relevant behavioral and network metrics from your Person Query
* **Organizational attributes:** These are used for heterogeneity analysis (identifying which subgroups experience the largest changes in after-hours work). Include as many as possible, typically covering dimensions like `Organization`, `Function`, `Level`, `IsManager`, and `Area`. The exact names can differ based on your organization's HR data structure—just ensure the variable names in the notebook are updated accordingly in the configuration section.

To examine the organizational or HR attributes you have in your dataset, you can run the following scripts to explore them: 
```python
hrvar_str = vi.extract_hr(data, return_type = 'vars').columns

for hr_var in hrvar_str:
    hrvar_table = vi.hrvar_count(data, hrvar = hr_var, return_type = 'table')
    print(f"\nValue counts for {hr_var}:")
    print(hrvar_table)

for hr_var in hrvar_str:
    vi.hrvar_count(data = data, hrvar = hr_var, return_type = 'plot')
```

### Changing parameters in the scripts

Before running the analysis, you'll need to customize a few parameters in the notebook to match your data and organizational structure. Open your chosen notebook and look for these configuration sections:

#### 1. File Paths (Setup and Imports)

Update the data file path to match your actual CSV filename:

```python
# Update this line to match your data file name
data_file_path = os.path.join(script_dir, '..', 'data', 'PersonQuery.Csv')
# For example: 'MyCompany_PersonQuery_2025.csv' or 'SuperUsersReport_Export.csv'

# Update the output directory name
output_base_dir = os.path.join(script_dir, '..', 'output', 'Subgroup Analysis - [YOUR COMPANY]')
# Replace [YOUR COMPANY] with your organization name, e.g., 'Contoso' or 'Fabrikam'
```

#### 2. Analysis Configuration (For After-Hours notebooks only)

For after-hours collaboration analysis, decide whether you want to find subgroups with negative effects (reductions) or positive effects (increases):

```python
# Toggle to control whether to find subgroups with NEGATIVE or POSITIVE effects
FIND_NEGATIVE_EFFECTS = True  # Set to True for reductions, False for increases
```

#### 3. Organizational Attributes (Variable Definitions)

Update these variable lists to match the column names in your data. Look for sections defining:

**SUBGROUP_VARS** - Organizational attributes for subgroup analysis and heterogeneity analysis:
```python
SUBGROUP_VARS = [
    'FunctionType',      # e.g., 'Function', 'Department', 'Division'
    'IsManager',         # Typically consistent across organizations
    'LevelDesignation',  # e.g., 'Level', 'Grade', 'Band'
    'Organization'       # Update these to match your data
]
```

*Note: `SUBGROUP_VARS` serves a dual purpose - these variables are used both for creating person-level aggregations and for identifying subgroups with heterogeneous treatment effects. Include 2-4 key demographic/organizational attributes that you want to analyze.*

**NETWORK_VARS** - Network metrics (usually consistent, but verify):
```python
NETWORK_VARS = [
    'Internal_network_size',
    'External_network_size',
    'Strong_ties',
    'Diverse_ties'
]
```

**COLLABORATION_VARS** - Behavioral metrics (usually consistent):
```python
COLLABORATION_VARS = [
    'Collaboration_hours',
    'Available_to_focus_hours',
    'Active_connected_hours',
    'Uninterrupted_hours'
]
```

#### 4. Treatment and Outcome Variables (Usually no changes needed)

These are typically standard, but verify they exist in your data:
- Treatment: `Total_Copilot_actions_taken`
- Outcome: `External_collaboration_hours` or `After_hours_collaboration_hours`

#### 5. Quick Checklist Before Running

- [ ] Data file is in the `data/` folder
- [ ] `data_file_path` matches your CSV filename exactly (case-sensitive)
- [ ] `output_base_dir` has your organization name (optional but recommended)
- [ ] All variables in `SUBGROUP_VARS` exist in your dataset
- [ ] `FIND_NEGATIVE_EFFECTS` is set appropriately (after-hours notebooks only)

## Running the analysis

Once everything is in place, you can choose to either run the notebook in its entirety or cell-by-cell.

### Recommended: Run Cell-by-Cell

If you're running this for the first time, we **strongly recommend running cell-by-cell**. This allows you to:

- Catch errors early (e.g., missing columns, incorrect file paths)
- Review intermediate outputs to ensure data loaded correctly
- Understand what each step is doing
- Stop and adjust parameters if needed

**How to run cell-by-cell:**

- **In Jupyter Notebook:** Click inside a cell and press `Shift + Enter` to run it and move to the next cell
- **In VS Code:** Click the ▶️ play button on the left side of each cell, or use `Shift + Enter`

### Running All Cells at Once

Once you're confident the notebook is configured correctly, you can run all cells:

- **In Jupyter Notebook:** Go to `Cell` → `Run All`
- **In VS Code:** Click `Run All` at the top of the notebook

**Expected runtime:** 10-30 minutes depending on data size (larger datasets take longer).

### Troubleshooting Common Errors

If you encounter errors while running the notebook, here's how to resolve them:

#### Error: "FileNotFoundError: [Errno 2] No such file or directory"

- **Cause:** The data file path is incorrect or the file doesn't exist
- **Solution:** 
  1. Check that your CSV file is in the `copilot-causal-toolkit/data/` folder
  2. Verify the filename in `data_file_path` matches exactly (including capitalization)
  3. Try using the full absolute path: `data_file_path = r"C:\Users\YourName\...\data\file.csv"`

#### Error: "KeyError: 'ColumnName'" or "Column not found"

- **Cause:** A variable specified in the configuration doesn't exist in your data
- **Solution:**
  1. Check the list of columns in your data (the notebook prints this early on)
  2. Update `SUBGROUP_VARS`, `NETWORK_VARS`, `COLLABORATION_VARS` to match your actual column names
  3. Remove any variables from the lists that don't exist in your dataset

#### Error: "ValueError: could not convert string to float"

- **Cause:** Data type mismatch or missing values in numeric columns
- **Solution:** Check for non-numeric values in metrics like `Total_Copilot_actions_taken` or the outcome variable

#### Error: "MemoryError" or notebook becomes unresponsive

- **Cause:** Dataset is too large for available memory
- **Solution:** 
  1. Filter your data to a smaller time period
  2. Reduce the number of subgroups analyzed
  3. Close other applications to free up memory

#### Using GitHub Copilot for Help

If you have **GitHub Copilot** or **Copilot Chat** installed in VS Code:

1. **Select the error message** or problematic code
2. **Open Copilot Chat** (Ctrl+Shift+I or Cmd+Shift+I)
3. **Ask specific questions** like:
   - "Why am I getting this error?"
   - "How do I fix this KeyError for the column 'Organization'?"
   - "Explain what this cell is doing"
   - "How do I change this variable list to use different column names?"

Copilot can help explain error messages, suggest fixes, and guide you through modifying the code to match your data structure.

#### Still Stuck?

- Review the cell outputs carefully—error messages often indicate exactly what's wrong
- Check that all prerequisites are installed correctly (`pip list` to verify)
- Ensure your data follows the expected format (Person Query or Super Users Report schema)
- Try running the notebook on a small sample of data first to isolate issues

## Interpreting the outputs

After running the analysis successfully, you'll find results in the `output/` directory. The structure looks like this:

```
output/
└── Subgroup Analysis - [YOUR COMPANY]/
    ├── significant_subgroups_[timestamp].csv
    ├── sensitivity_analysis_results_[timestamp].json
    └── [SubgroupName]/
        ├── ate_plot_[timestamp].png
        ├── ate_results_[treatment]_[timestamp].csv
        ├── definition.txt
        └── transition_matrix_[treatment]_[timestamp].csv
```

### Main Output Files

#### 1. Significant Subgroups CSV (`significant_subgroups_[timestamp].csv`)

This file summarizes which subgroups show the strongest treatment effects (positive or negative).

**Key columns:**

- `subgroup_name` - The demographic group analyzed (e.g., "Organization_Sales__and__IsManager_True")
- `sample_size` - Number of person-weeks in this subgroup
- `ate_mean` - Average Treatment Effect: the average change in outcome per additional Copilot action
- `ate_lb` / `ate_ub` - 95% confidence interval bounds
- `statistically_significant` - Whether the effect is statistically significant (True/False)

**How to interpret:**

- **Positive ATE:** Increasing Copilot usage increases the outcome (more external/after-hours collaboration)
- **Negative ATE:** Increasing Copilot usage decreases the outcome (less external/after-hours collaboration)
- **Magnitude:** A larger absolute value means a stronger effect
- **Example:** `ate_mean = -0.15` means each additional Copilot action reduces after-hours collaboration by 0.15 hours per week

**Business interpretation:**

- For **External Collaboration (seller productivity):** Positive effects suggest Copilot helps people spend more time with customers/partners
- For **After-Hours Collaboration (wellbeing):** Negative effects suggest Copilot helps reduce evening/weekend work

#### 2. Subgroup-Specific Directories

For each significant subgroup, you'll find a dedicated folder with detailed results:

##### a. ATE Plot (`ate_plot_[timestamp].png`)

A visualization showing the **dose-response curve**: how the outcome changes at different levels of Copilot usage.

**What to look for:**

- **Slope:** Indicates whether effects are positive (upward) or negative (downward)
- **Linearity:** Is the effect consistent across all usage levels, or does it plateau/accelerate?
- **Confidence bands:** Shaded areas show statistical uncertainty (narrower = more confident)
- **Treatment grid points:** The x-axis shows specific Copilot usage levels (e.g., 5, 10, 15, 20 actions/week)

**Example interpretation:**

- A steadily declining curve with narrow confidence bands suggests Copilot reliably reduces the outcome
- A flat curve suggests no effect at any usage level
- An upward curve that plateaus suggests diminishing returns at high usage levels

##### b. ATE Results Table (`ate_results_[treatment]_[timestamp].csv`)

Detailed numeric results for each treatment level analyzed.

**Key columns:**

- `treatment_level` - Copilot usage level (e.g., 5, 10, 15 actions per week)
- `ate` - Estimated effect at this usage level
- `ate_lb` / `ate_ub` - Confidence interval bounds
- `comparison` - Which usage levels are being compared (e.g., "5 → 10")
- `effect_size` - Magnitude of change between usage levels

**How to use:**

- Compare effects at different usage levels to find optimal Copilot adoption targets
- Check confidence intervals to assess reliability of estimates
- Identify the "sweet spot" where effects are strongest

##### c. Definition File (`definition.txt`)

A plain-text description of:
- The subgroup definition (which demographic attributes and values)
- Sample size and time period
- Variables included in the analysis

**Purpose:** Makes it easy to communicate findings to stakeholders ("This analysis looked at Sales managers with 500+ employees...")

##### d. Transition Matrix (`transition_matrix_[treatment]_[timestamp].csv`)

Shows how people in this subgroup move between different Copilot usage levels over time.

**What it reveals:**
- **Adoption patterns:** Are people increasing usage over time?
- **Stickiness:** Do people who start using Copilot continue using it?
- **Volatility:** How much do usage levels fluctuate week-to-week?

**Example:** If the matrix shows most people moving from "Low" → "Medium" usage, this suggests successful adoption.

#### 3. Sensitivity Analysis Results (`sensitivity_analysis_results_[timestamp].json`)

A JSON file containing robustness checks that test whether results hold under different assumptions.

**What's tested:**
- Different model specifications (e.g., varying the number of controls)
- Alternative treatment definitions (e.g., different usage thresholds)
- Subsampling (checking if results are driven by outliers)

**How to interpret:**
- If results are similar across sensitivity tests, you can be more confident in the findings
- If results vary dramatically, the effect may be sensitive to modeling choices (interpret with caution)

**Note:** This is more technical—focus on the main ATE results unless you need to defend methodological choices.

### Summary: What to Focus On

For a quick analysis:
1. **Start with** `significant_subgroups_[timestamp].csv` to identify which groups show effects
2. **Dive into** each subgroup's `ate_plot_[timestamp].png` to visualize the dose-response
3. **Use** `ate_results_[timestamp].csv` for precise numbers to share with stakeholders
4. **Reference** `definition.txt` when explaining which populations were analyzed

### Making Sense of the Results

**Statistical Significance:**

- Effects with confidence intervals that **don't cross zero** are statistically significant
- Larger sample sizes generally produce narrower confidence intervals (more precise estimates)
- Non-significant results could mean: (a) no real effect, or (b) insufficient data to detect an effect

**Practical Significance:**

- Even if an effect is statistically significant, consider whether the magnitude matters in practice
- Example: A reduction of 0.05 hours/week (3 minutes) may be statistically significant but not practically meaningful
- Focus on effects where the magnitude aligns with business goals

**Next Steps After Reviewing Results:**

- Identify which subgroups show the most promising effects (positive or negative)
- Consider targeted interventions or training for subgroups with weak effects
- Monitor whether effects persist over time by re-running the analysis periodically
- Share visualizations and key findings with leadership to inform Copilot adoption strategy

## Methodology

This analysis uses **Double Machine Learning (DML)** with **two-way fixed effects residualization** to estimate the causal effect of Copilot usage on workplace outcomes. This section provides an overview of the methodology for those interested in understanding the technical approach.

### Overview: What is Causal Inference?

Traditional correlation analysis can tell us that Copilot users have different collaboration patterns, but it cannot tell us whether Copilot *causes* those differences. People who adopt Copilot early might already be different from non-adopters in ways that affect their work patterns.

**Causal inference** methods allow us to estimate what would happen if we *increased* someone's Copilot usage, accounting for:
- Pre-existing differences between people (some are naturally more productive)
- Time trends (work patterns change over time for everyone)
- Confounding factors (people with more meetings might also use Copilot more)

### The Double Machine Learning (DML) Framework

We use two primary DML estimators in this analysis:

#### 1. LinearDML - Average Treatment Effects (ATE)

**Purpose:** Estimate the average dose-response relationship across all individuals.

**How it works:**
1. Uses machine learning to model both:
   - How Copilot usage depends on covariates (propensity modeling)
   - How outcomes depend on covariates (outcome modeling)
2. Removes the predictable parts using these models (residualization)
3. Estimates the causal effect from the remaining "unexplained" variation

**Key features:**
- **Spline featurization:** Models non-linear dose-response curves (e.g., diminishing returns at high usage)
- **Cross-fitting:** Uses sample splitting to avoid overfitting and ensure valid confidence intervals
- **Doubly robust:** Estimates remain valid even if one of the two models (propensity or outcome) is misspecified

**Output:** A dose-response curve showing how outcomes change at different Copilot usage levels (e.g., 5, 10, 15, 20 actions/week)

#### 2. CausalForestDML - Heterogeneous Treatment Effects (CATE)

**Purpose:** Identify which subgroups experience different treatment effects (heterogeneity analysis).

**How it works:**
1. First applies the same residualization as LinearDML
2. Uses a random forest to discover which combinations of demographic attributes predict larger/smaller effects
3. Identifies subgroups automatically without pre-specifying which groups to analyze

**Key features:**
- **Adaptive subgroup discovery:** Finds meaningful subgroups based on the data
- **Individual Treatment Effect (ITE) estimation:** Provides person-specific effect estimates
- **Tree-based interpretation:** Results can be visualized as decision trees showing how effects vary

**Output:** Rankings of subgroups by treatment effect magnitude, with confidence intervals for each group

### Data Aggregation Approach

Before running DML, the notebooks **aggregate the longitudinal data by person**:

**Why this matters:**
- **Simplifies analysis:** Converts panel data (person-weeks) to cross-sectional data (person-level averages)
- **Reduces noise:** Smooths out week-to-week fluctuations by averaging over time
- **Person-level interpretation:** Effects represent differences between individuals rather than within-person changes

**Aggregation approach:**
For each person, we:
1. Calculate the mean of all numeric variables (treatment, outcome, controls) across all observed weeks
2. Keep demographic attributes (which are time-invariant by nature)
3. Result is one row per person with their average metrics

**What this means for interpretation:**
- Estimates compare people with different **average** Copilot usage levels
- Cannot distinguish whether effects come from *adopting* Copilot vs. pre-existing differences
- Results show **cross-sectional associations** adjusted for observed covariates
- This is more descriptive than causal, but DML's double-robustness provides some protection against confounding

### Key assumptions:

1. **Unconfoundedness:** We observe all important factors that affect both Copilot usage and outcomes (addressed by including comprehensive behavioral and demographic covariates)
2. **Positivity:** Every individual has some probability of any usage level (checked via treatment distribution plots)
3. **Stable Unit Treatment Value (SUTVA):** One person's Copilot usage doesn't affect another's outcomes (potential concern with team-level spillovers)

### Model Specifications Used

**Machine learning models for nuisance functions:**
- **Model for treatment (T):** Random Forest Regressor with 100 trees
- **Model for outcome (Y):** Random Forest Regressor with 100 trees
- **Final stage:** Linear regression (LinearDML) or Random Forest (CausalForestDML)

**Treatment featurization:**
- **Spline transformation:** 4 knots, degree 3 polynomial (captures smooth non-linear curves)
- **Alternative:** Polynomial features up to degree 2 (for comparison)

**Control variables included:**
- Time-varying: Collaboration hours, meeting hours, email hours, network size, focus time
- Time-invariant (via fixed effects): Person-specific baselines, week-specific trends
- Demographic: Organization, function, level, manager status (for heterogeneity analysis)

### Interpreting the Results

**What the ATE tells you:**
- The average effect of increasing Copilot usage by 1 action per week
- How effects vary across the usage spectrum (5 actions vs. 20 actions)
- Whether there are diminishing returns at high usage levels

**What the CATE tells you:**
- Which subgroups benefit most (or least) from Copilot
- Whether effects are positive for some groups and negative for others
- Where to target adoption efforts for maximum impact

**Confidence intervals:**
- 95% confidence intervals are provided for all estimates
- Width reflects statistical uncertainty (wider = less certain)
- Non-overlap with zero indicates statistical significance at the 5% level

### Robustness and Sensitivity

The analysis includes several robustness checks:

1. **Alternative model specifications:** Re-run with different ML algorithms
2. **Different control variables:** Test sensitivity to covariate selection
3. **Varying featurization:** Compare spline vs. polynomial treatment modeling
4. **Subsampling:** Check if results are driven by outliers or specific time periods

Results stored in `sensitivity_analysis_results_[timestamp].json` document how stable the findings are across these variations.

### Limitations and Caveats

**What this analysis CAN tell you:**
- Whether Copilot usage increases or decreases outcomes on average
- Which usage levels show the strongest effects
- Which subgroups experience different effects

**What this analysis CANNOT tell you:**
- Whether observed effects will persist beyond the study period (external validity)
- Mechanisms explaining *why* Copilot has these effects (mediation analysis needed)
- Long-term effects beyond 6 months (limited by data timeframe)
- Team-level or organizational spillover effects (individual-level analysis)

**Recommended interpretation:**
- Treat results as **descriptive of the observed period** rather than universal laws
- Consider whether identified subgroups reflect genuine heterogeneity or chance variation
- Validate findings by re-running analysis on new data as it becomes available
- Use results to inform hypotheses for further investigation, not as definitive proof