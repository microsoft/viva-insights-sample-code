# Output Spec: Exploratory Analysis Notebook

This specification defines the structure, code style, and content expectations for an **exploratory analysis notebook** generated from Viva Insights data. Use this spec as context for coding agents when you want them to produce a well-structured, reproducible notebook suitable for peer review and iterative analysis.

Apply the [shared analysis policy](../../skills/viva-insights-analysis/reference/analysis-policy.md).
Keep row-level working data local; public reports contain only protected aggregates.

---

## Target format

| Property | Requirement |
|----------|-------------|
| **File type** | Jupyter Notebook (`.ipynb`) for Python, or R Markdown (`.Rmd`) for R |
| **Language** | Python (pandas, matplotlib/seaborn) or R (tidyverse, ggplot2) |
| **Audience** | Data analysts, data scientists, peer reviewers |
| **Purpose** | Exploratory data analysis (EDA) — understand the data, surface patterns, document findings |

> **Note:** If using Python, the notebook should be runnable in JupyterLab, VS Code, or Google Colab. If using R, the `.Rmd` file should knit to HTML without errors.

---

## Recommended structure

### Cell 1: Title and description (Markdown)

```markdown
# Copilot Adoption: Exploratory Analysis

**Author:** [Name/Team]
**Date:** [Date]
**Data source:** Viva Insights person query export
**Period:** [Start date] – [End date]

## Objective

[1-2 sentences describing the goal of this analysis. E.g., "Explore Copilot adoption patterns
across the organization, identify high- and low-adoption segments, and surface hypotheses for
further investigation."]
```

### Cell 2: Setup and imports (Code)

Load all required packages upfront. Pin versions if reproducibility is critical.

**Python:**
```python
import vivainsights as vi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('colorblind')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

ORG_MINIMUM = 5  # Replace with the actual organisational requirement.
WORKFLOW_FLOOR = 5  # Use 10 for joined-query, dashboard or causal publication.
PRIVACY_THRESHOLD = max(ORG_MINIMUM, WORKFLOW_FLOOR)
DATA_PATH = './data/person_query.csv'
```

**R:**
```r
library(tidyverse)
library(lubridate)
library(vivainsights)
library(scales)

theme_set(theme_minimal(base_size = 12))

ORG_MINIMUM <- 5  # Replace with the actual organisational requirement.
WORKFLOW_FLOOR <- 5  # Use 10 for joined-query, dashboard or causal publication.
PRIVACY_THRESHOLD <- max(ORG_MINIMUM, WORKFLOW_FLOOR)
DATA_PATH <- "./data/person_query.csv"
```

### Cell 3: Data loading (Code)

Load the data using `import_query()` from the **vivainsights** package. This function reads in a Person Query CSV and cleans column names (replaces spaces and special characters with underscores), which helps handle localization differences.

**Python:**
```python
df = vi.import_query(DATA_PATH)

print(f"Shape: {df.shape}")
print(f"Date range: {df['MetricDate'].min()} to {df['MetricDate'].max()}")
print(f"Unique persons: {df['PersonId'].nunique()}")
print(f"Unique periods: {df['MetricDate'].nunique()}")
print(f"\nColumn names:\n{list(df.columns)}")
```

**R:**
```r
df <- import_query(DATA_PATH)

cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("Date range:", as.character(min(df$MetricDate)), "to", as.character(max(df$MetricDate)), "\n")
cat("Unique persons:", n_distinct(df$PersonId), "\n")
cat("Unique periods:", n_distinct(df$MetricDate), "\n")
cat("\nColumn names:\n")
print(names(df))
```

### Cell 4: Data validation (Code + Markdown)

Verify the panel structure and data quality before any analysis. This cell should:

1. Check for duplicate `(PersonId, MetricDate)` pairs
2. Read configured query grouping, week start and measurement windows; gaps are diagnostics only
3. Check panel completeness
4. Summarize missing values in key columns
5. Identify licensed vs. unlicensed users
6. Use `extract_hr()` to identify organizational attribute columns

```python
# Panel structure check
dupes = df.duplicated(subset=['PersonId', 'MetricDate'], keep=False).sum()
print(f"Duplicate person-periods: {dupes}")

# Query configuration is required; sparse daily activity can look weekly.
CONFIGURED_GROUPING = None  # Set from query configuration, not observed dates.
if CONFIGURED_GROUPING not in {'day', 'week', 'month'}:
    raise ValueError("Confirm configured query grouping before aggregation")
print(f"Configured grouping: {CONFIGURED_GROUPING}")

# Panel completeness
expected = df['PersonId'].nunique() * df['MetricDate'].nunique()
actual = len(df)
print(f"Panel completeness: {actual/expected:.1%} ({actual}/{expected})")

# Identify organizational attribute columns using extract_hr()
hr_cols = vi.extract_hr(df)
print(f"\nHR/organizational attributes: {hr_cols}")

# Identify Copilot metric columns (columns containing 'Copilot')
copilot_cols = [c for c in df.columns if 'Copilot' in c]
print(f"Copilot columns found: {copilot_cols}")
for col in copilot_cols:
    null_pct = df[col].isna().mean()
    print(f"  {col}: {null_pct:.1%} null")

# Explicit eligibility, independent of activity. Preserve unknown evidence.
df['is_licensed'] = pd.Series(pd.NA, index=df.index, dtype='boolean')
if 'Total_Copilot_enabled_days' in df:
    enabled = pd.to_numeric(df['Total_Copilot_enabled_days'], errors='raise').astype('Float64')
    df['is_licensed'] = enabled.gt(0)
print(f"\nKnown eligible person-periods: {df['is_licensed'].sum()}")
print(f"Unknown eligibility: {df['is_licensed'].isna().sum()}")
```

This example assumes the enabled-days definition establishes period eligibility.
Otherwise map explicit licence evidence or omit adoption rates. An observed
null cell remains unknown. Only absent activity rows inside a supplied complete
query's configured scope contribute zero; retain unavailable/out-of-scope reasons.

**Follow with a Markdown cell** summarizing the validation findings:

```markdown
### Validation Summary

- Panel structure: [confirmed/issues found]
- Granularity: [weekly/daily]
- Completeness: [X%]
- Licensed users: [X% of person-weeks]
- Data quality issues: [none/describe]
```

### Cells 5–N: Exploratory analysis sections (Code + Markdown, alternating)

Each analysis section should follow this pattern:
1. **Markdown cell:** State the question being explored
2. **Code cell:** Compute and visualize
3. **Markdown cell:** Interpret the results

**Example sections to include:**

#### Section: Overall adoption trends

```markdown
## Adoption Trends Over Time

How has Copilot adoption evolved across the reporting period?
```

```python
actions = pd.to_numeric(df['Copilot_Actions'], errors='raise').astype('Float64')
df['is_active'] = df['is_licensed'] & actions.gt(0)
eligible = df[df['is_licensed'].fillna(False)]
weekly = eligible.groupby('MetricDate').agg(
    licensed=('PersonId', 'nunique'),
    active=('is_active', 'sum'),
    activity_known=('is_active', lambda s: s.notna().all())
).reset_index()
weekly['adoption_rate'] = weekly['active'] / weekly['licensed']
weekly.loc[~weekly['activity_known'], 'adoption_rate'] = float('nan')
weekly.loc[weekly['licensed'] < PRIVACY_THRESHOLD, ['licensed', 'active', 'adoption_rate']] = float('nan')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weekly['MetricDate'], weekly['adoption_rate'] * 100, marker='o')
ax.set_xlabel('Week')
ax.set_ylabel('Adoption Rate (%)')
ax.set_title('Weekly Copilot Adoption Rate')
plt.tight_layout()
plt.show()
```

#### Section: Segmentation analysis

```markdown
## Adoption by Organization

Which organizations have the highest and lowest Copilot adoption?
```

#### Section: Usage depth analysis

```markdown
## Usage Depth: Actions Per Active User

Among active users, how intensively are they using Copilot?
```

#### Section: Correlation exploration

```markdown
## Relationship Between Collaboration and Copilot Usage

Do employees with higher collaboration hours tend to use Copilot more or less?
```

### Final cell: Key findings and next steps (Markdown)

```markdown
## Key Findings

1. **[Finding 1]:** [Description with supporting numbers]
2. **[Finding 2]:** [Description with supporting numbers]
3. **[Finding 3]:** [Description with supporting numbers]

## Next Steps

- [ ] [Follow-up analysis or investigation]
- [ ] [Data to collect or request]
- [ ] [Stakeholders to share findings with]

## Caveats

- [Note any data quality issues discovered]
- [Note any assumptions made]
- [Note any limitations of the analysis]
```

---

## Code style guidelines

### General principles
- **One idea per cell.** Each code cell should do one thing (load data, compute a metric, create a chart). Avoid cells that do multiple unrelated things.
- **Clear variable names.** Use descriptive names (`weekly_adoption` not `wa`, `licensed_users` not `lu`).
- **Comments for "why," not "what."** Comment non-obvious logic, not every line:

```python
# Good: explains the business logic
# This field's documented definition establishes eligibility, not activity.
df['is_licensed'] = df['Total_Copilot_enabled_days'].astype('Float64').gt(0)

# Bad: restates the code
# Set is_licensed
df['is_licensed'] = df['Total_Copilot_enabled_days'].astype('Float64').gt(0)
```

### Visualizations
- Every analysis step that produces a quantitative result should be accompanied by a visualization
- Use consistent chart styling across the notebook
- Include chart titles, axis labels, and legends
- Use colorblind-friendly palettes (`sns.set_palette('colorblind')` or `scale_color_brewer()`)
- Suppress segments below the privacy threshold in all charts and tables

### Print statements
- Print key numbers and shapes after each computation (helps with debugging and auditing)
- Use f-strings with formatting for readability:

```python
print(f"Adoption rate: {adoption:.1%}")
print(f"Active users: {active:,}")
```

---

## Notebook-specific considerations

### Cell ordering
- Cells must be executable top-to-bottom without errors (restart kernel and run all as a final check)
- Do not depend on cells being run out of order or multiple times
- Define all variables before using them

### Markdown cells for narration
- Use Markdown cells between code cells to create a narrative flow
- Each Markdown cell should either pose a question (before a code cell) or interpret results (after a code cell)
- Use headers (`##`, `###`) to create a navigable structure in the notebook sidebar

### Output control
- Suppress verbose library output (e.g., `warnings.filterwarnings('ignore')` for known warnings)
- Limit DataFrame displays to reasonable sizes (e.g., `.head(20)`)
- Clear large intermediate variables if memory is a concern

### Reproducibility
- Set random seeds if any stochastic methods are used
- Record package versions in the setup cell (or include a `requirements.txt`)
- Note the Python/R version used

---

## How to instruct a coding agent

When asking a coding agent to produce this output, include context from this spec. Here is a sample instruction snippet:

```
OUTPUT FORMAT INSTRUCTIONS:
Produce a Jupyter notebook (.ipynb) [or R Markdown (.Rmd)] with the following structure:

1. Title cell (Markdown): title, author, date, data source, objective
2. Setup cell (Code): imports (including vivainsights), configuration, constants
3. Data loading cell (Code): load CSV with `import_query()` from vivainsights, print shape and column names
4. Data validation section (Code + Markdown):
   - Check panel structure (unique PersonId × MetricDate)
   - Detect granularity (weekly vs daily)
   - Check panel completeness
   - Summarize missing values
   - Create is_licensed flag
   - Use extract_hr() to identify organizational attribute columns
   - Markdown summary of validation findings
5. Analysis sections (each = Markdown question → Code → Markdown interpretation):
   - Overall adoption trends (line chart)
   - Segmentation by Organization, FunctionType, LevelDesignation (bar charts)
   - Usage depth (distribution of Copilot actions among active users)
   - Correlation with collaboration metrics (scatter plot)
6. Key findings and next steps (Markdown): numbered findings, next steps checklist, caveats

Style: one idea per cell, clear variable names, visualization after each analysis step,
colorblind-friendly palette, suppress segments with < 10 users.
Cells must be executable top-to-bottom without errors.
Save as "copilot_adoption_eda_YYYYMMDD.ipynb".
```
