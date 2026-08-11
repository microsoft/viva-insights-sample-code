---
layout: page
title: "Network Analysis"
eyebrow: "Advanced analytics · Network"
description: "Organizational network analysis (ONA) with Viva Insights. Visualize and analyze collaboration networks at the group-to-group and person-to-person level in R and Python."
permalink: /network/
css_files:
  - "/assets/css/lang-switch.css"
  - "/assets/css/topic-switch.css"
---
# Network Analysis Scripts

Organizational Network Analysis (ONA) maps the relationships and interactions between people, teams, and departments based on actual collaboration patterns, rather than formal reporting structures — revealing the informal networks that drive real work and innovation, in ways a traditional org chart cannot. See [Key Use Cases for Network Analysis](#key-use-cases-for-network-analysis) further down the page for the full range of ways this is applied.

Viva Insights makes network metrics available through four query types. Sample scripts on this page currently cover two of them — pick a card to jump straight to one, or read on for the full picture.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="#group-to-group-network-analysis">
    <span class="vi-card-icon">🏢</span>
    <span class="vi-card-title">Group-to-Group Query</span>
    <span class="vi-card-desc">An edgelist with each row representing the collaboration of one grouping (organizational) attribute with another.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#person-to-person-network-analysis">
    <span class="vi-card-icon">🧑‍🤝‍🧑</span>
    <span class="vi-card-title">Person-to-Person Query</span>
    <span class="vi-card-desc">An edgelist with each row representing the collaboration of one person with another.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics">
    <span class="vi-card-icon">👤</span>
    <span class="vi-card-title">Person Query (Network Metrics)</span>
    <span class="vi-card-desc">Person-date level metrics such as Strong ties, Diverse ties, Influencer score, and Internal network size. No sample script in this repo yet.</span>
    <span class="vi-card-more">Metrics reference →</span>
  </a>
  <a class="vi-card" href="https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics">
    <span class="vi-card-icon">🕸️</span>
    <span class="vi-card-title">Person-to-Group Query</span>
    <span class="vi-card-desc">An edgelist with each row representing one person's collaboration with respect to a grouping (organizational) attribute. No sample script in this repo yet.</span>
    <span class="vi-card-more">Metrics reference →</span>
  </a>
</div>

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

<div class="topic-switch" role="group" aria-label="Focus on a network query type">
  <span class="topic-switch-label">Focus on:</span>
  <div class="topic-switch-group">
    <button type="button" class="topic-switch-btn" data-topic-btn="all">Both</button>
    <button type="button" class="topic-switch-btn" data-topic-btn="g2g">Group-to-Group</button>
    <button type="button" class="topic-switch-btn" data-topic-btn="p2p">Person-to-Person</button>
  </div>
  <span class="topic-switch-note">Remembers your choice on this device.</span>
</div>
<script>
(function () {
  var stored = null;
  try { stored = window.localStorage.getItem('vi-network-topic-pref'); } catch (e) {}
  document.documentElement.setAttribute('data-topic', (stored === 'g2g' || stored === 'p2p') ? stored : 'all');
})();
</script>

---

<div data-topic-block="g2g" markdown="1">

## Group-to-Group Network Analysis

### Customizing Group-to-Group Networks

<div data-lang-block="r" markdown="1">
**📄 [custom-network-g2g.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-g2g.Rmd)**
- **Purpose**: Create customized group-to-group collaboration network visualizations
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, igraph, ggplot2
- **Key Features**: Custom styling, filtering, layout algorithms, export options
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-g2g.Rmd)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [custom-network-g2g.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-g2g.py)**
- **Purpose**: Create customized group-to-group collaboration network visualizations
- **Prerequisites**: vivainsights Python package, networkx, matplotlib
- **Key Features**: Custom styling, filtering, layout algorithms, export options
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-g2g.py)**
</div>

### Extended Group-to-Group Analysis (R)

A deeper walkthrough of `network_g2g()`: interaction matrices, igraph objects, sankey visualizations, and overlaying organizational colours and sizes onto the network plot.

**📄 [example_ONA_groups.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/extending-vivainsights-with-R/example_ONA_groups.R)**
- **Purpose**: Group-based organizational network analysis
- **Language**: R
- **Prerequisites**: vivainsights R package, igraph, dplyr
- **Key Features**: Inter-group dynamics, group-level metrics, comparative analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/extending-vivainsights-with-R/example_ONA_groups.R)**

### Group-to-Group Example Visualizations

- **[🖼️ network_g2g.svg](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/example-visuals/network_g2g.svg)**: Sample group-to-group network (R)
- **[🖼️ network_g2g.svg (Python)](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/example-visuals/network_g2g.svg)**: Python-generated network

</div>

---

<div data-topic-block="p2p" markdown="1">

## Person-to-Person Network Analysis

### Customizing Person-to-Person Networks

<div data-lang-block="r" markdown="1">
**📄 [custom-network-p2p.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/custom-network-p2p.Rmd)**
- **Purpose**: Create customized person-to-person collaboration network visualizations
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, igraph, ggplot2
- **Key Features**: Individual-level analysis, community detection, centrality measures
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/custom-network-p2p.Rmd)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [custom-network-p2p.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/custom-network-p2p.py)**
- **Purpose**: Create customized person-to-person collaboration network visualizations
- **Prerequisites**: vivainsights Python package, networkx, matplotlib
- **Key Features**: Individual-level analysis, community detection, centrality measures
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/custom-network-p2p.py)**
</div>

### Extended Person-to-Person Analysis (R)

A deeper walkthrough of `network_p2p()`: Louvain and Leiden community detection, closeness/degree/betweenness centrality, sankey visualizations, and a fast plotting method for large graphs.

**📄 [example_ONA.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/extending-vivainsights-with-R/example_ONA.R)**
- **Purpose**: Comprehensive organizational network analysis workflows
- **Language**: R
- **Prerequisites**: vivainsights R package, igraph, dplyr
- **Key Features**: Network metrics, clustering, centrality analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/extending-vivainsights-with-R/example_ONA.R)**

### Person-to-Person Example Visualizations

- **[🖼️ network_p2p.svg](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/example-visuals/network_p2p.svg)**: Sample person-to-person network (R)
- **[🖼️ network_p2p.svg (Python)](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/example-visuals/network_p2p.svg)**: Python-generated network

</div>

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

## Key Use Cases for Network Analysis

Network analysis with Viva Insights data is particularly valuable for:

- **Change Management**: Identify key influencers and communication pathways to ensure successful organizational transformations
- **Organizational Design**: Understand how work actually flows across teams and departments to optimize organizational structure
- **Talent Development**: Discover high-potential employees who serve as connectors and bridge-builders across the organization
- **Innovation & Knowledge Sharing**: Map how expertise and information flow to identify bottlenecks and opportunities for better collaboration
- **Merger & Acquisition Integration**: Visualize collaboration patterns between merged entities and track integration progress
- **Remote Work Optimization**: Understand how distributed teams collaborate and identify potential isolation or over-collaboration issues
- **Leadership Development**: Identify informal leaders and understand influence patterns beyond formal hierarchy
- **Diversity & Inclusion**: Analyze collaboration patterns across different demographic groups to identify potential barriers or silos
- **Team Formation**: Use network insights to create more effective cross-functional teams based on existing collaboration patterns
- **Risk Management**: Identify over-dependencies on key individuals or potential knowledge silos that could impact business continuity

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

<div data-lang-block="r" markdown="1">
### R Environment
```r
install.packages(c("vivainsights", "igraph", "ggplot2", "dplyr", "visNetwork"))
```
</div>

<div data-lang-block="python" markdown="1">
### Python Environment
```bash
pip install vivainsights networkx matplotlib seaborn plotly pandas numpy
```
</div>

---

## Best Practices

1. **Data Quality**: Ensure clean, complete collaboration data
2. **Privacy**: Anonymize person-level data when appropriate
3. **Interpretation**: Focus on actionable insights rather than metrics alone
4. **Validation**: Cross-check network insights with qualitative feedback
5. **Temporal Analysis**: Track network changes over time for trends

---

## Related pages

- [Essentials]({{ site.baseurl }}/essentials/): core utilities and visualizations to prepare your data
- [Advanced Analytics]({{ site.baseurl }}/advanced/): predictive modeling and statistical testing
- [Copilot Analytics]({{ site.baseurl }}/copilot/): measure Microsoft Copilot adoption and impact
- [Joining People Skills Data]({{ site.baseurl }}/skills-data-join/): enrich network analysis with People Skills data
- [Getting Started]({{ site.baseurl }}/getting-started/): environment setup and first steps

---

## Need Help?

- **Network Analysis**: [NetworkX Documentation](https://networkx.org/documentation/stable/) | [igraph R Documentation](https://igraph.org/r/)
- **Visualization**: [Matplotlib](https://matplotlib.org/) | [ggplot2](https://ggplot2.tidyverse.org/)
- **Viva Insights**: [Package Documentation](https://microsoft.github.io/vivainsights/)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)

<script src="{{ '/assets/js/lang-switch.js' | relative_url }}"></script>
<script src="{{ '/assets/js/topic-switch.js' | relative_url }}"></script>
