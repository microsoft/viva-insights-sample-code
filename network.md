---
layout: page
title: "Network Analysis"
permalink: /network/
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
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/create-example-visuals.R">Create Visuals (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/create-example-visuals.py">Create Visuals (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/generate-custom-kpi/generate-custom-kpi.md">Custom KPIs (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/intro-to-vivainsights-py">Intro to Python</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/advanced/">
          Advanced <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/top-performers-rf.ipynb">Top Performers (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/top-performers-rf.Rmd">Top Performers (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/information-value.ipynb">Information Value (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/information-value.Rmd">Information Value (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/pairwise-chisq.py">Chi-Square Tests (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/pairwise_chisq.Rmd">Chi-Square Tests (R)</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/network/">
          Network <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-g2g.py">Group-to-Group (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-g2g.Rmd">Group-to-Group (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-p2p.py">Person-to-Person (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-p2p.Rmd">Person-to-Person (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/extending-vivainsights-with-R/example_ONA.R">ONA Examples (R)</a>
        </div>
      </div>
      <div class="dropdown">
        <a class="page-link dropdown-toggle" href="${baseUrl}/copilot/">
          Copilot <span class="dropdown-arrow">▼</span>
        </a>
        <div class="dropdown-content">
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-analytics-examples.R">Analysis Scripts (R)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.py">Analysis Scripts (Python)</a>
          <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.ipynb">Jupyter Notebook</a>
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

# Network Analysis Scripts

Collaboration network visualization and analysis scripts for understanding organizational connectivity.

## Group-to-Group Network Analysis

### Customizing Group-to-Group Networks (Python)
**📄 [custom-network-g2g.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-g2g.py)**
- **Purpose**: Create customized group-to-group collaboration network visualizations
- **Language**: Python
- **Prerequisites**: vivainsights Python package, networkx, matplotlib
- **Key Features**: Custom styling, filtering, layout algorithms, export options
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-g2g.py)**

### Customizing Group-to-Group Networks (R)
**📄 [custom-network-g2g.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-g2g.Rmd)**
- **Purpose**: Create customized group-to-group collaboration network visualizations
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, igraph, ggplot2
- **Key Features**: Custom styling, filtering, layout algorithms, export options
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-g2g.Rmd)**

---

## Person-to-Person Network Analysis

### Customizing Person-to-Person Networks (Python)
**📄 [custom-network-p2p.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-p2p.py)**
- **Purpose**: Create customized person-to-person collaboration network visualizations
- **Language**: Python
- **Prerequisites**: vivainsights Python package, networkx, matplotlib
- **Key Features**: Individual-level analysis, community detection, centrality measures
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-p2p.py)**

### Customizing Person-to-Person Networks (R)
**📄 [custom-network-p2p.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-p2p.Rmd)**
- **Purpose**: Create customized person-to-person collaboration network visualizations
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, igraph, ggplot2
- **Key Features**: Individual-level analysis, community detection, centrality measures
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-p2p.Rmd)**

---

## Organizational Network Analysis Examples

### Extended ONA Analysis (R)
**📄 [example_ONA.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/extending-vivainsights-with-R/example_ONA.R)**
- **Purpose**: Comprehensive organizational network analysis workflows
- **Language**: R
- **Prerequisites**: vivainsights R package, igraph, dplyr
- **Key Features**: Network metrics, clustering, centrality analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/extending-vivainsights-with-R/example_ONA.R)**

### ONA Group Analysis (R)
**📄 [example_ONA_groups.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/extending-vivainsights-with-R/example_ONA_groups.R)**
- **Purpose**: Group-based organizational network analysis
- **Language**: R
- **Prerequisites**: vivainsights R package, igraph, dplyr
- **Key Features**: Inter-group dynamics, group-level metrics, comparative analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/extending-vivainsights-with-R/example_ONA_groups.R)**

---

## Example Visualizations

### Generated Network Visualizations
**📁 [Network Visualization Examples](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-r/example-visuals)**

**Group-to-Group Networks**:
- **[🖼️ network_g2g.svg](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/example-visuals/network_g2g.svg)**: Sample group-to-group network
- **[🖼️ network_g2g.svg (Python)](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/example-visuals/network_g2g.svg)**: Python-generated network

**Person-to-Person Networks**:
- **[🖼️ network_p2p.svg](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/example-visuals/network_p2p.svg)**: Sample person-to-person network
- **[🖼️ network_p2p.svg (Python)](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/example-visuals/network_p2p.svg)**: Python-generated network

---

## Network Analysis Workflows

### 1. Group-to-Group Analysis Workflow
1. **Data Preparation**: Load group-based collaboration data
2. **Network Construction**: Build group interaction networks
3. **Visualization**: Create network diagrams with custom styling
4. **Analysis**: Calculate group-level network metrics
5. **Interpretation**: Identify collaboration patterns and bottlenecks

### 2. Person-to-Person Analysis Workflow
1. **Data Preparation**: Load person-level collaboration data
2. **Network Construction**: Build individual interaction networks
3. **Community Detection**: Identify informal organizational clusters
4. **Centrality Analysis**: Find key connectors and influencers
5. **Visualization**: Create person-level network maps

### 3. Organizational Network Analysis Workflow
1. **Multi-Level Analysis**: Combine group and person-level insights
2. **Temporal Analysis**: Track network changes over time
3. **Comparative Analysis**: Compare networks across departments/teams
4. **Recommendations**: Provide actionable insights for collaboration improvement

---

## Key Network Metrics

### Group-Level Metrics
- **Density**: How interconnected groups are
- **Centrality**: Which groups are most central to collaboration
- **Clustering**: How groups form collaborative clusters
- **Modularity**: Strength of group boundaries

### Person-Level Metrics
- **Betweenness Centrality**: Key bridge-builders
- **Closeness Centrality**: Well-connected individuals
- **Degree Centrality**: Number of direct connections
- **Eigenvector Centrality**: Influence through connections

---

## Customization Options

### Visual Customization
- **Node Styling**: Size, color, shape based on attributes
- **Edge Styling**: Width, color, style based on interaction strength
- **Layout Algorithms**: Force-directed, hierarchical, circular
- **Labeling**: Custom node and edge labels
- **Export Formats**: SVG, PNG, PDF for presentations

### Analysis Customization
- **Filtering**: Focus on specific groups, time periods, or interaction types
- **Thresholding**: Filter weak connections for clarity
- **Aggregation**: Roll up data to different organizational levels
- **Comparison**: Side-by-side network comparisons

---

## Prerequisites

### Python Environment
```bash
pip install vivainsights networkx matplotlib seaborn plotly pandas numpy
```

### R Environment
```r
install.packages(c("vivainsights", "igraph", "ggplot2", "dplyr", "visNetwork"))
```

---

## Best Practices

1. **Data Quality**: Ensure clean, complete collaboration data
2. **Privacy**: Anonymize person-level data when appropriate
3. **Interpretation**: Focus on actionable insights, not just metrics
4. **Validation**: Cross-check network insights with qualitative feedback
5. **Temporal Analysis**: Track network changes over time for trends

---

## Need Help?

- **Network Analysis**: [NetworkX Documentation](https://networkx.org/documentation/stable/) | [igraph R Documentation](https://igraph.org/r/)
- **Visualization**: [Matplotlib](https://matplotlib.org/) | [ggplot2](https://ggplot2.tidyverse.org/)
- **Viva Insights**: [Package Documentation](https://microsoft.github.io/vivainsights/)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)
