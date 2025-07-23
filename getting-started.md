---
layout: default
title: "Getting Started"
permalink: /getting-started/
---

# Getting Started with Viva Insights Analytics

{% include custom-navigation.html %}

<style>
/* Hide default Minima navigation to prevent duplicates */
.site-header .site-nav,
.site-header .trigger,
.site-header .page-link {
  display: none !important;
}
</style>

Welcome to Viva Insights sample code! This guide will help you set up your development environment and get started with analyzing Viva Insights data using R or Python.

---

## 🔧 Setting Up R

### Prerequisites

Before installing R packages, ensure you have:

- **R version 4.0.0 or higher** - [Download R](https://cran.r-project.org/)
- **RStudio** (recommended) - [Download RStudio](https://www.rstudio.com/products/rstudio/download/)

### Install Required R Packages

```r
# Install core packages from CRAN
install.packages(c(
  "tidyverse",    # Complete data science toolkit 
  "vivainsights", # Microsoft Viva Insights R package
  "igraph",       # Network analysis
  "visNetwork"   # Interactive networks
))
```

### Verify R Installation

```r
# Load the vivainsights package
library(vivainsights)

# Check package version
packageVersion("vivainsights")

# View available functions
help(package = "vivainsights")
```

### R Environment Setup Tips

1. **Set up a project**: Create an RStudio project for your Viva Insights analysis - [Learn how to use RStudio projects](https://martinctc.github.io/blog/rstudio-projects-and-working-directories-a-beginner's-guide/)
2. **Configure data paths**: Set up consistent paths for your data files
3. **Enable auto-completion**: RStudio provides excellent auto-completion for vivainsights functions

---

## 🐍 Setting Up Python

### Prerequisites
- **Python 3.7 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (usually included with Python)
- **Virtual environment** (recommended)

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv viva-insights-env

# Activate virtual environment
# On Windows:
viva-insights-env\Scripts\activate
# On macOS/Linux:
source viva-insights-env/bin/activate
```

### Install Required Python Packages

```bash
# Core data science packages
pip install pandas numpy matplotlib seaborn plotly

# Network analysis
pip install networkx

# Jupyter notebooks
pip install jupyter

# Statistical analysis
pip install scipy scikit-learn

# Install vivainsights Python package
pip install vivainsights
```

### Verify Python Installation

```python
# Import vivainsights
import vivainsights as vi

# Check version
print(vi.__version__)

# Import other key packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

print("All packages imported successfully!")
```

### Python Environment Setup Tips

1. **Use Jupyter notebooks**: Great for exploratory analysis
   ```bash
   jupyter notebook
   ```
2. **Configure matplotlib**: Set up plotting preferences
   ```python
   import matplotlib.pyplot as plt
   plt.style.use('seaborn-v0_8')
   ```
3. **Set pandas options**: Configure display options
   ```python
   pd.set_option('display.max_columns', None)
   pd.set_option('display.width', None)
   ```

---

## 📊 Using Viva Insights Queries with R/Python

### Understanding Viva Insights Query Types

Viva Insights provides several types of queries that you can export and analyze:

1. **Person Query**: Individual-level metrics aggregated by person and date
2. **Meeting Query**: Meeting-level data including attendees, duration, and patterns
3. **Group-to-Group Query**: Collaboration between organizational groups
4. **Person-to-Person Query**: Direct collaboration between individuals
5. **Person-to-Group Query**: Individual collaboration with organizational groups

### Basic Workflow

#### 1. Export Data from Viva Insights
- Log into your Viva Insights tenant
- Navigate to **Analyze** > **Query designer**
- Select your query type and configure parameters
- Export results as CSV files

#### 2. Load and Explore Data

The following code snippet demonstrates how to load the required packages, load the .csv query into your environment, run some quick checks on the structure of the data, and create a simple visualization: 

**R Example:**
```r
library(vivainsights)
library(dplyr)

# Load person query data and standardizes variable names
person_data <- import_query("path/to/your/person_query.csv")

# Quick exploration
glimpse(person_data)
summary(person_data)

# Use vivainsights functions
person_data %>% create_bar(metric = "Email_hours")
```

**Python Example:**
```python
import vivainsights as vi
import pandas as pd

# Load person query data and standardizes variable names
person_data = vi.import_query("path/to/your/person_query.csv")

# Quick exploration
print(person_data.info())
print(person_data.describe())

# Use vivainsights functions
vi.create_bar(person_data, metric="Email_hours")
```

Whilst you can use `read.csv()` (R) or `pd.read_csv()` (Python) for reading in the .csv query into your R or Python environment, we recommend using the `import_query()` function instead from the **vivainsights** package. `import_query()` standardizes variable names and 'cleans' special characters, ensuring that you minimize the number of errors arising from variable name mismatches. 

If you do not have a person query handy, you can also try out the code using our inbuilt sample datasets from the two packages: 
```r
person_data = pq_data # Use in place of csv load
```

```python
person_data = vi.load_pq_data() # Use in place of csv load
```

#### 3. Common Analysis Patterns

**Time Trend Analysis:**
```r
# R
person_data %>% create_trend(metric = "Collaboration_hours")
```
```python
# Python
vi.create_trend(person_data, metric="Collaboration_hours")
```

**Distribution Analysis:**
```r
# R
person_data %>% create_boxplot(metric = "Meeting_hours", hrvar = "Organization")
```
```python
# Python
vi.create_boxplot(person_data, metric="Meeting_hours", hrvar="Organization")
```

**Network Analysis:**
```r
# R - Group-to-Group networks
g2g_data <- import_query("path/to/g2g_query.csv")
g2g_data %>% network_g2g(primary = "Organization", secondary = "LevelDesignation")
```
```python
# Python - Group-to-Group networks
g2g_data = vi.import_query("path/to/g2g_query.csv")
vi.network_g2g(g2g_data, primary="Organization", secondary="LevelDesignation")
```

### Data Preparation Best Practices

1. **Date Formatting**: Ensure date columns are properly formatted
   ```r
   # R
   person_data$Date <- as.Date(person_data$Date)
   ```
   ```python
   # Python
   person_data['Date'] = pd.to_datetime(person_data['Date'])
   ```

2. **Handle Missing Values**: Clean your data appropriately
   ```r
   # R
   person_data <- person_data %>% filter(!is.na(Email_hours))
   ```
   ```python
   # Python
   person_data = person_data.dropna(subset=['Email_hours'])
   ```

3. **Standardize Organizational Attributes**: Ensure consistent naming
   ```r
   # R
   person_data <- person_data %>% 
     mutate(Organization = str_trim(Organization))
   ```
   ```python
   # Python
   person_data['Organization'] = person_data['Organization'].str.strip()
   ```

### Advanced Integration Techniques

#### Custom Metrics and KPIs
```r
# R - Create custom collaboration intensity metric
person_data <- person_data %>%
  mutate(
    Collaboration_intensity = 
      (Email_hours + Meeting_hours + Instant_messages_hours) / 
      Total_focus_hours
  )
```

#### Multi-Query Analysis
```r
# R - Combine person and meeting data
person_summary <- person_data %>%
  group_by(PersonId, Date) %>%
  summarise(avg_collab_hours = mean(Collaboration_hours))

meeting_summary <- meeting_data %>%
  group_by(PersonId, Date) %>%
  summarise(total_meetings = n())

combined_data <- left_join(person_summary, meeting_summary, 
                          by = c("PersonId", "Date"))
```

#### Automated Reporting
```r
# R - Create automated reports
create_report <- function(data_path, output_path) {
  data <- import_query(data_path)
  
  # Generate multiple visualizations
  bar_plot <- data %>% create_bar(metric = "Email_hours")
  trend_plot <- data %>% create_trend(metric = "Collaboration_hours")
  
  # Save plots
  ggsave(paste0(output_path, "/collaboration_bar.png"), bar_plot)
  ggsave(paste0(output_path, "/trend_analysis.png"), trend_plot)
}
```

---

## 🚀 Next Steps

Once you have your environment set up:

1. **Explore the Examples**: Browse through our [utility scripts](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples) for inspiration
2. **Start with Person Queries**: These provide the most comprehensive individual-level insights
3. **Try Network Analysis**: Use group-to-group queries to understand organizational collaboration patterns
4. **Build Custom Dashboards**: Combine multiple visualizations into executive-ready reports
5. **Join the Community**: Share your insights and get help from other practitioners

### Helpful Resources

- **[vivainsights R Documentation](https://microsoft.github.io/vivainsights/)**
- **[vivainsights Python Documentation](https://github.com/microsoft/vivainsights-py)**
- **[Viva Insights Documentation](https://docs.microsoft.com/en-us/viva/insights/)**
- **[Sample Code Repository](https://github.com/microsoft/viva-insights-sample-code)**

### Need Help?

- **Issues or Bugs**: [Open an issue](https://github.com/microsoft/viva-insights-sample-code/issues) on GitHub
- **Feature Requests**: Share your ideas in our [discussions](https://github.com/microsoft/viva-insights-sample-code/discussions)
- **Community**: Connect with other users and contributors

---

*Ready to dive deeper? Check out our [Essentials]({{ site.baseurl }}/essentials/) for basic analysis scripts, or jump into [Advanced Analytics]({{ site.baseurl }}/advanced/) for sophisticated modeling techniques.*
