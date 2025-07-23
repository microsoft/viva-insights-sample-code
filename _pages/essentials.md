---
layout: page
title: "Essentials"
permalink: /essentials/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide any default Minima navigation that might appear */
.site-header .site-nav,
.trigger,
.page-link:not(.dropdown-toggle):not(.btn) {
  display: none !important;
}

/* Ensure our custom navigation is visible */
.custom-nav {
  display: block !important;
}
</style>

# Essential Viva Insights Scripts

This page provides some essential scripts to let you get started with analysis in Viva Insights. Using the R and Python scripts below, you can: 

* perform exploratory data analysis and identify key interesting hypotheses for your organization
* run a range of custom visualizations on your Viva Insights data
* create custom KPIs or segments using a combination of Viva Insights metrics and organizational / survey data

## Utility Scripts

### R Utilities
Essential functions and utilities for R-based analysis.

**📁 [Utility code for R](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-r)**
- **Purpose**: Collection of utility functions for common Viva Insights analysis tasks
- **Language**: R
- **Format**: Multiple R files
- **Prerequisites**: vivainsights R package

---

### Python Utilities
Essential functions and utilities for Python-based analysis.

**📁 [Utility code for Python](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python)**
- **Purpose**: Collection of utility functions for common Viva Insights analysis tasks
- **Language**: Python
- **Format**: Multiple Python files
- **Prerequisites**: vivainsights Python package

---

## Visualization Scripts

### Creating Essential Visualizations (R)
**📄 [create-example-visuals.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/create-example-visuals.R)**
- **Purpose**: Generate standard Viva Insights visualizations
- **Language**: R
- **Prerequisites**: vivainsights R package, ggplot2
- **Key Functions**: Bar charts, line plots, network diagrams
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/create-example-visuals.R)**

### Creating Essential Visualizations (Python)
**📄 [create-example-visuals.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/create-example-visuals.py)**
- **Purpose**: Generate standard Viva Insights visualizations
- **Language**: Python
- **Prerequisites**: vivainsights Python package, matplotlib, seaborn
- **Key Functions**: Bar charts, line plots, network diagrams
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/create-example-visuals.py)**

---

## Custom KPI Generation

### Generate Custom KPIs from Viva Insights (R)
**📄 [generate-custom-kpi.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/generate-custom-kpi/generate-custom-kpi.md)**
- **Purpose**: Tutorial for creating custom key performance indicators
- **Language**: R
- **Format**: Markdown tutorial with code examples
- **Prerequisites**: vivainsights R package
- **[📥 Download Script](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/generate-custom-kpi/generate_kpis.R)**

---

## Getting Started Tutorials

### Introduction to Viva Insights with Python
**📁 [Introduction to Viva Insights with Python](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/intro-to-vivainsights-py)**
- **Purpose**: Comprehensive introduction to Python-based Viva Insights analysis
- **Language**: Python
- **Format**: Jupyter Notebooks
- **Prerequisites**: vivainsights Python package
- **Key Topics**: Data loading, basic analysis, visualization

**Included Notebooks:**
- **[📓 demo-vivainsights-py.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/intro-to-vivainsights-py/demo-vivainsights-py.ipynb)**: General introduction
- **[📓 demo-ona-vivainsights-py.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/intro-to-vivainsights-py/demo-ona-vivainsights-py.ipynb)**: Organizational Network Analysis

---

## Need Help?

- **R Package Documentation**: [vivainsights R](https://microsoft.github.io/vivainsights/)
- **Python Package Documentation**: [vivainsights Python](https://microsoft.github.io/vivainsights-py/)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)
