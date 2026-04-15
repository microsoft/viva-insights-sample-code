# Output Spec: Exploratory Analysis Notebook

This specification defines the structure, code style, and content expectations for an **exploratory analysis notebook** generated from Viva Insights data. Use this spec as context for coding agents when you want them to produce a well-structured, reproducible notebook suitable for peer review and iterative analysis.

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

PRIVACY_THRESHOLD = 10  # Minimum group size for reporting
DATA_PATH = './data/person_query.csv'
```

**R:**
```r
library(tidyverse)
library(lubridate)
library(vivainsights)
library(scales)

theme_set(theme_minimal(base_size = 12))

PRIVACY_THRESHOLD <- 10
DATA_PATH <- "./data/person_query.csv"
```

### Cell 3: Data loading (Code)

Load the data and print basic information.

```python
df = pd.read_csv(DATA_PATH)
df['MetricDate'] = pd.to_datetime(df['MetricDate'])
df['PersonId'] = df['PersonId'].astype(str)

print(f"Shape: {df.shape}")
print(f"Date range: {df['MetricDate'].min()} to {df['MetricDate'].max()}")
print(f"Unique persons: {df['PersonId'].nunique()}")
print(f"Unique periods: {df['MetricDate'].nunique()}")
print(f"\nColumn names:\n{list(df.columns)}")
```

### Cell 4: Data validation (Code + Markdown)

Verify the panel structure and data quality before any analysis. This cell should:

1. Check for duplicate `(PersonId, MetricDate)` pairs
2. Identify the time granularity (weekly vs. daily)
3. Check panel completeness
4. Summarize missing values in key columns
5. Identify licensed vs. unlicensed users

```python
# Panel structure check
dupes = df.duplicated(subset=['PersonId', 'MetricDate'], keep=False).sum()
print(f"Duplicate person-periods: {dupes}")

# Granularity detection
dates = sorted(df['MetricDate'].unique())
gap = (pd.to_datetime(dates[1]) - pd.to_datetime(dates[0])).days
print(f"Time granularity: {'weekly' if gap >= 7 else 'daily'} ({gap}-day gaps)")

# Panel completeness
expected = df['PersonId'].nunique() * df['MetricDate'].nunique()
actual = len(df)
print(f"Panel completeness: {actual/expected:.1%} ({actual}/{expected})")

# Missing values in Copilot columns
copilot_cols = [c for c in df.columns if 'Copilot' in c]
print(f"\nCopilot columns found: {copilot_cols}")
for col in copilot_cols:
    null_pct = df[col].isna().mean()
    print(f"  {col}: {null_pct:.1%} null")

# Licensed vs unlicensed
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)
df['is_active'] = df['is_licensed'] & (df.get('Copilot_Actions', pd.Series(dtype=float)) > 0)
print(f"\nLicensed person-weeks: {df['is_licensed'].sum()} ({df['is_licensed'].mean():.1%})")
print(f"Active person-weeks: {df['is_active'].sum()} ({df['is_active'].mean():.1%})")
```

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
weekly = df[df['is_licensed']].groupby('MetricDate').agg(
    licensed=('PersonId', 'nunique'),
    active=('is_active', 'sum')
).reset_index()
weekly['adoption_rate'] = weekly['active'] / weekly['licensed']

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
# Licensed = has at least one non-null Copilot metric (null means unlicensed, not inactive)
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)

# Bad: restates the code
# Set is_licensed to True if any copilot column is not null
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)
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
2. Setup cell (Code): imports, configuration, constants
3. Data loading cell (Code): load CSV, parse dates, print shape and column names
4. Data validation section (Code + Markdown):
   - Check panel structure (unique PersonId × MetricDate)
   - Detect granularity (weekly vs daily)
   - Check panel completeness
   - Summarize missing values
   - Create is_licensed and is_active flags
   - Markdown summary of validation findings
5. Analysis sections (each = Markdown question → Code → Markdown interpretation):
   - Overall adoption trends (line chart)
   - Segmentation by Organization, FunctionType, LevelDesignation (bar charts)
   - Usage depth (distribution of Copilot_Actions among active users)
   - Correlation with collaboration metrics (scatter plot)
6. Key findings and next steps (Markdown): numbered findings, next steps checklist, caveats

Style: one idea per cell, clear variable names, visualization after each analysis step,
colorblind-friendly palette, suppress segments with < 10 users.
Cells must be executable top-to-bottom without errors.
Save as "copilot_adoption_eda_YYYYMMDD.ipynb".
```
