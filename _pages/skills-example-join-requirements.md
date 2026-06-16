---
layout: page
title: "Skills data join requirements"
permalink: /skills-data-join-requirements/
---
### Development Environment Setup

For performing data joins and analyses, this tutorial covers example scripts in both the R and Python analysis environments. Here are the pre-requisites for them respectively: 

#### Option 1: R Environment

To run the R code in this tutorial, you must have R and ideally a code editor installed: 

- Install R: Download from [https://cran.r-project.org/](https://cran.r-project.org/)
- Install RStudio (recommended): Download from [https://www.rstudio.com/](https://www.rstudio.com/). Alternatively, you may choose to use [Visual Studio Code](https://code.visualstudio.com/) as your IDE. 

**Required R Packages:**

The next step is to install the required R packages: 

- **[vivainsights](https://microsoft.github.io/vivainsights/)**: Microsoft's R package for Viva Insights analysis
- **[tidyverse](https://www.tidyverse.org/)**: Collection of R packages for data science
- **[here](https://here.r-lib.org/)**: Easy file path management

If these are not already installed, you can install them in R with: 
```r
install.packages(c("vivainsights", "tidyverse", "here"))
```

#### Option 2: Python Environment  

To run the Python code in this tutorial, you must have Python and ideally a code editor installed: 

- Install Python: Download from [https://www.python.org/downloads/](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/products/distribution)
- Install Jupyter (recommended): Comes with Anaconda or install via `pip install jupyter`. Alternatively, you may choose to use [Visual Studio Code](https://code.visualstudio.com/) as your IDE. 

**Required Python Packages:**

Ensure that you have the following python packages installed: 

- **[vivainsights](https://microsoft.github.io/vivainsights-py/)**: Microsoft's Python package for Viva Insights analysis
- **[pandas](https://pandas.pydata.org/)**: Data manipulation and analysis library
- **[numpy](https://numpy.org/)**: Fundamental package for scientific computing
- **[plotly](https://plotly.com/)**: Interactive visualization library
- **[scipy](https://scipy.org/)**: Scientific computing tools

If not, you can install them in the Command Prompt with: 

```bash
pip install vivainsights pandas numpy plotly scipy
```

Once the developer pre-requisites are satisfied, see [how to load and join data]({{ site.baseurl }}/skills-data-join/#data-loading-and-joining).

### Required Viva Insights schema and join keys

Before running the joins, confirm each input table is present and contains its join key(s). Missing or non-unique keys are the most common cause of failed or exploded joins:

| Table | Role | Required key column(s) | Notes |
|-------|------|------------------------|-------|
| `MetricOutput` (Person Query) | Main / left table | `PersonId`, `MetricDate`, `MetricPrimaryKey`, `PeopleHistoricalId` | Add any outcome metrics (e.g. `Collaboration_hours`, `Total_Copilot_actions_taken`) here. One row per person-week. |
| `HR` | Org attributes | `PeopleHistoricalId` | One row per person-history; should be unique on the key. |
| `PersonSkillsMappingMetadata` | Bridge | `MetricPrimaryKey`, `SkillHistoricalId` | One-to-many: expands rows to person-skill grain. |
| `PersonSkills` | Skill instances | `SkillHistoricalId`, `SkillId` | Links a person's skill record to the skills catalog. |
| `SkillsLibrary` | Skills catalog | `SkillId` | One row per skill; should be unique on `SkillId`. |

Minimal pre-join checklist:

- ✅ Every table above loaded without error and is non-empty.
- ✅ Each join key column exists and has no unexpected `NA`s.
- ✅ The "one" side of each join (`HR`, `SkillsLibrary`) is **unique** on its key (`df %>% count(key) %>% filter(n > 1)` returns no rows).
- ✅ You have recorded the row count of `MetricOutput` so you can verify expected row growth after each join. 