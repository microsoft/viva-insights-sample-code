---
layout: page
title: "Essentials"
eyebrow: "Essentials"
description: "Essential R and Python scripts for getting started with Viva Insights: utilities, custom visualizations, and custom KPI generation."
permalink: /essentials/
css: "/assets/css/lang-switch.css"
---
# Essential Viva Insights Scripts

This page provides some essential scripts to let you get started with analysis in Viva Insights. Using the R and Python scripts below, you can: 

* perform exploratory data analysis and identify key interesting hypotheses for your organization
* run a range of custom visualizations on your Viva Insights data
* create custom KPIs or segments using a combination of Viva Insights metrics and organizational / survey data

<div class="lang-switch" role="group" aria-label="Choose code language for this page">
  <span class="lang-switch-label">Show code in:</span>
  <div class="lang-switch-group">
    <button type="button" class="lang-switch-btn" data-lang-btn="r">R</button>
    <button type="button" class="lang-switch-btn" data-lang-btn="python">Python</button>
  </div>
  <span class="lang-switch-note">Remembers your choice on this device.</span>
</div>
<script>
(function () {
  var stored = null;
  try { stored = window.localStorage.getItem('vi-lang-pref'); } catch (e) {}
  document.documentElement.setAttribute('data-lang', stored === 'python' ? 'python' : 'r');
})();
</script>

## Utility Scripts

Most of the utility scripts formerly listed here now have their own catalogued entry, with a fuller description, prerequisites, and a direct download link, on the topic page where they're used:

- **[Advanced Analytics]({{ site.baseurl }}/advanced/)**: top-performer modeling, information value, pairwise chi-square tests, collaboration by time of day, workplace intervention evaluation, meeting engagement drivers
- **[Network Analysis]({{ site.baseurl }}/network/)**: group-to-group and person-to-person network customization
- **[Copilot Analytics]({{ site.baseurl }}/copilot/)**: Copilot usage analysis, usage segment trends, difference-in-differences and event-study impact analysis

If you'd rather browse everything in one place, the raw folders are on GitHub: [utility-r](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-r) and [utility-python](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python).

---

## Visualization Scripts

### Creating Essential Visualizations

<div data-lang-block="r" markdown="1">
**📄 [create-example-visuals.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/create-example-visuals.R)**
- **Purpose**: Generate standard Viva Insights visualizations
- **Prerequisites**: vivainsights R package, ggplot2
- **Key Functions**: Bar charts, line plots, network diagrams
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/create-example-visuals.R)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [create-example-visuals.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/create-example-visuals.py)**
- **Purpose**: Generate standard Viva Insights visualizations
- **Prerequisites**: vivainsights Python package, matplotlib, seaborn
- **Key Functions**: Bar charts, line plots, network diagrams
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/create-example-visuals.py)**
</div>

---

## Custom KPI Generation

### Generate Custom KPIs from Viva Insights (R)
**📄 [generate-custom-kpi.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/generate-custom-kpi/generate-custom-kpi.md)**
- **Purpose**: Tutorial for creating custom key performance indicators
- **Language**: R
- **Format**: Markdown tutorial with code examples
- **Prerequisites**: vivainsights R package
- **📖 Full tutorial**: [Generate Custom KPIs in R]({{ site.baseurl }}/generate-custom-kpi/), a step-by-step walkthrough on this site
- **[📥 Download Script](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/generate-custom-kpi/generate_kpis.R)**

---

## Introductory Notebooks

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

There is no R equivalent notebook yet. R users can start instead from the [Getting Started]({{ site.baseurl }}/getting-started/) page, which covers the same import and first-analysis steps.

---

## Related pages

- [Getting Started]({{ site.baseurl }}/getting-started/): set up your R or Python environment and run your first analysis
- [Generate Custom KPIs in R]({{ site.baseurl }}/generate-custom-kpi/): full walkthrough of the custom-KPI workflow
- [Joining People Skills Data]({{ site.baseurl }}/skills-data-join/): combine People Skills data with Viva Insights metrics
- [Advanced Analytics]({{ site.baseurl }}/advanced/): machine learning, regression, and statistical testing
- [Network Analysis]({{ site.baseurl }}/network/): organizational network analysis (ONA)
- [Frontier]({{ site.baseurl }}/frontier-analytics/): turn an export into a finished dashboard or report with a coding agent

---

## Need Help?

- **R Package Documentation**: [vivainsights R](https://microsoft.github.io/vivainsights/)
- **Python Package Documentation**: [vivainsights Python](https://microsoft.github.io/vivainsights-py/)
- **Common data pitfalls**: [Data pitfalls reference](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/viva-insights-analysis/reference/data-pitfalls.md), for issues like `IsManager` arriving as text or the privacy suppression threshold
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)

<script src="{{ '/assets/js/lang-switch.js' | relative_url }}"></script>
