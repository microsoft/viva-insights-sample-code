---
layout: home
title: "Viva Insights Sample Code Library"
---

<link rel="stylesheet" href="{{ "/assets/css/custom-nav.css" | relative_url }}">

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Add dropdown functionality after page loads
  const nav = document.querySelector('.site-nav .trigger');
  if (nav) {
    nav.innerHTML = `
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="/essentials/">
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
        <a class="page-link dropdown-toggle" href="/analytics/">
          Analytics <span class="dropdown-arrow">▼</span>
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
        <a class="page-link dropdown-toggle" href="/network/">
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
        <a class="page-link dropdown-toggle" href="/copilot/">
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
    `;
  }
});
</script>

# Welcome to the Viva Insights Sample Code Library

This repository contains sample code for the Viva Insights open-source packages, including code that is used in tutorials or demos. The sample code shown here covers more elaborate scenarios and examples that leverage, but are not included in the R and Python packages themselves.

## Quick Links

- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [wpa R package (legacy)](https://microsoft.github.io/wpa/)

## How to Use This Library

Browse the categories below to find sample code that matches your needs. Each script includes:
- **Purpose**: What the script accomplishes
- **Prerequisites**: Required packages and data formats
- **Usage**: How to run and customize the code
- **Download**: Direct link to the raw script file

---

## Categories

### 🔧 [Essentials](essentials.html)
Essential utility functions and visualizations for getting started with Viva Insights analysis.

### 📊 [Advanced Analytics](analytics.html)
Regression models, machine learning, and statistical analysis techniques.

### 🔗 [Network Analysis](network.html)
Collaboration network visualization and analysis scripts.

### 🤖 [Copilot Analytics](copilot.html)
Analysis and visualization scripts specifically for Microsoft Copilot usage data.

---

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA). For details, visit [Microsoft CLA](https://cla.opensource.microsoft.com).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/microsoft/viva-insights-sample-code/blob/main/LICENSE) file for details.
