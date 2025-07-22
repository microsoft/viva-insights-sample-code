---
layout: page
title: "Essentials"
permalink: /essentials/
---

<link rel="stylesheet" href="{{ "/assets/css/custom-nav.css" | relative_url }}">

<script>
document.addEventListener('DOMContentLoaded', function() {
  const nav = document.querySelector('.site-nav .trigger');
  if (nav) {
    const baseUrl = '/viva-insights-sample-code';
    nav.innerHTML = `
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/essentials/">
          Essentials <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-r">R Utilities</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python">Python Utilities</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/create-example-visuals.R">Create Visuals (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/create-example-visuals.py">Create Visuals (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/generate-custom-kpi/generate-custom-kpi.md">Custom KPIs (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/intro-to-vivainsights-py">Intro to Python</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/advanced/">
          Advanced <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/top-performers-rf.ipynb">Top Performers (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/top-performers-rf.Rmd">Top Performers (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/information-value.ipynb">Information Value (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/information-value.Rmd">Information Value (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/pairwise-chisq.py">Chi-Square Tests (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/pairwise_chisq.Rmd">Chi-Square Tests (R)</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/network/">
          Network <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-g2g.py">Group-to-Group (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-g2g.Rmd">Group-to-Group (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-p2p.py">Person-to-Person (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-p2p.Rmd">Person-to-Person (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/extending-vivainsights-with-R/example_ONA.R">ONA Examples (R)</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/copilot/">
          Copilot <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/copilot-analytics-examples.R">Analysis Scripts (R)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.py">Analysis Scripts (Python)</a>
          <a href="https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.ipynb">Jupyter Notebook</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/dax/calculated-columns">DAX Scripts</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md">Usage Segmentation Guide</a>
        </div>
      </div>
      <a class="page-link" href="https://github.com/microsoft/viva-insights-sample-code" target="_blank">
        GitHub
      </a>
    `;
  }
});
</script>

# Essential Viva Insights Scripts

Get started with these fundamental scripts for Viva Insights analysis.

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
