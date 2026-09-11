---
layout: page
title: "Copilot Analytics"
eyebrow: "Copilot analytics"
description: "Explore Copilot adoption, ways of working, and impact with interactive demo dashboards, R and Python analyses, and Power BI segmentation."
permalink: /copilot/
css: "/assets/css/lang-switch.css"
---
# Copilot Analytics Scripts

This page contains specialized scripts for analyzing Microsoft Copilot usage data from Viva Insights.

Beyond covering key analyses around Copilot usage volume and breadth (range of actions and applications), these scripts also include a method for measuring Copilot habituality based on behavioral research. This approach determines whether a user can be considered a habitual Copilot user, enabling visualization through user segmentation that identifies **Power Users** and **Habitual Users** within an organization. This segmentation provides a framework for continuously tracking Copilot adoption success and measuring the effectiveness of your deployment strategy.

See our [DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/) page for detailed instructions on how to identify Copilot Usage Segments using Power BI templates and pre-built DAX formulas.

For more information on the Copilot Usage Segments, see this [introduction]({{ site.baseurl }}/copilot-usage-segments/).

For more inspiration on analyzing Copilot adoption and impact, have a look at our [advanced examples playbook](https://aka.ms/CopilotAdvancedAnalytics/).

Pick a card to jump straight to a section, or read on for the full picture.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="#interactive-dashboards">
    <span class="vi-card-icon">📊</span>
    <span class="vi-card-title">Interactive Dashboards</span>
    <span class="vi-card-desc">Two ready-to-view flexdashboard reports pairing the new Consumption and GitHub queries with a Person Query — see what a finished stakeholder report looks like.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#analysis-scripts">
    <span class="vi-card-icon">📈</span>
    <span class="vi-card-title">Analysis Scripts</span>
    <span class="vi-card-desc">Core R/Python scripts and a 12-week adoption-journey notebook covering usage volume, breadth, and Power/Habitual segmentation.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#usage-segment-trends">
    <span class="vi-card-icon">👥</span>
    <span class="vi-card-title">Usage Segment Trends</span>
    <span class="vi-card-desc">Track how the mix of Power, Habitual, Novice, Low, and Non-users evolves week by week.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#causal-analysis">
    <span class="vi-card-icon">🔬</span>
    <span class="vi-card-title">Causal Analysis</span>
    <span class="vi-card-desc">Difference-in-differences and event-study methods that isolate Copilot's effect from a company-wide trend.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
</div>

---

## Interactive Dashboards

Open a demo in your browser without installing anything. Both reports use **synthetic data** and are available as **R templates** to adapt.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" id="copilot-consumption-and-ways-of-working" href="{{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html">
    <img src="{{ site.baseurl }}/assets/images/reports/copilot-consumption-overview.png" alt="Consumption report overview preview" loading="lazy">
    <span class="vi-card-title">Copilot Consumption and Ways of Working</span>
    <span class="vi-card-desc">Explore token consumption, credit intensity, and their associations with collaboration patterns.</span>
    <span class="vi-card-more">View demo →</span>
  </a>
  <a class="vi-card" id="developer-experience-and-copilot" href="{{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html">
    <img src="{{ site.baseurl }}/assets/images/reports/github-copilot-devex-focus.png" alt="Developer experience report focus and coordination preview" loading="lazy">
    <span class="vi-card-title">Developer Experience and Copilot</span>
    <span class="vi-card-desc">Explore developer working conditions alongside GitHub and Microsoft 365 Copilot use.</span>
    <span class="vi-card-more">View demo →</span>
  </a>
</div>

**[Browse dashboard templates]({{ site.baseurl }}/copilot-dashboards/)** for previews, required queries, source code, and setup guidance.

---

## Analysis Scripts

<div class="lang-switch" role="group" aria-label="Choose code language for analysis scripts">
  <span class="lang-switch-label">Show code in:</span>
  <div class="lang-switch-group">
    <button type="button" class="lang-switch-btn" data-lang-btn="r">R</button>
    <button type="button" class="lang-switch-btn" data-lang-btn="python">Python</button>
  </div>
  <span class="lang-switch-note">Applies to scripts below, not dashboards. Remembers your choice on this device.</span>
</div>
<script>
(function () {
  var stored = null;
  try { stored = window.localStorage.getItem('vi-lang-pref'); } catch (e) {}
  document.documentElement.setAttribute('data-lang', stored === 'python' ? 'python' : 'r');
})();
</script>

### Copilot Advanced Analysis

<div data-lang-block="r" markdown="1">
**📄 [copilot-analytics-examples.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-analytics-examples.R)**
- **Purpose**: Comprehensive analysis of Copilot usage patterns and trends
- **Prerequisites**: vivainsights R package, Copilot usage data
- **Key Analysis**: Usage segmentation, trend analysis, adoption metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/copilot-analytics-examples.R)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [copilot-analytics-examples.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.py)**
- **Purpose**: Comprehensive analysis of Copilot usage patterns and trends
- **Prerequisites**: vivainsights Python package, Copilot usage data
- **Key Analysis**: Usage segmentation, trend analysis, adoption metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.py)**

**📓 [copilot-analytics-examples.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.ipynb)** (Jupyter Notebook)
- **Purpose**: Interactive analysis of Copilot usage with visualizations
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, Copilot usage data
- **Key Features**: Step-by-step analysis, interactive visualizations
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.ipynb)**
</div>

### Copilot Adoption Journey and Ways of Working

Where the scripts above give you a tour of the core Copilot metrics, this notebook is the
next step: an end-to-end assessment of how far adoption has travelled and what sustained
use is associated with. Reach for it once you have at least 12 weeks of Copilot data and
need to answer a leadership question rather than explore the data.

| Start with | When |
| --- | --- |
| **Copilot Advanced Analysis** (above) | You are getting oriented, and you want the core metrics, segmentation, and standard visuals. |
| **Copilot Adoption Journey** (below) | You have 12+ weeks of data and need cohorts, habit formation, conversion targeting, and associated ways of working. |

<div data-lang-block="python" markdown="1">
**📓 [copilot-adoption-journey-analysis.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-adoption-journey-analysis.ipynb)** (Jupyter Notebook)
- **Purpose**: End-to-end assessment of Copilot reach, habit formation, conversion opportunity, and associated ways of working
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights, pandas, NumPy, SciPy, statsmodels, matplotlib, and a Copilot-rich Person Query export with at least 12 weeks of data. Add pyarrow if you are loading a Parquet export
- **Key Features**: 12-week Power/Habitual/Novice/Low/Non-user segmentation, entry-cohort and function analysis, native Viva Insights scans, adjusted associations, and same-person diagnostics
- **Interpretation**: Observational associations only. Set `INPUT_FILE` to your own export before running; any metric or attribute your query lacks is reported as unavailable and skipped
- **Related**: [Copilot Usage Segments]({{ site.baseurl }}/copilot-usage-segments/) for the segment definitions, and [Causal Inference]({{ site.baseurl }}/causal-inference/) when you need an effect rather than an association
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-adoption-journey-analysis.ipynb)**
</div>

<div data-lang-block="r" markdown="1">
This example is currently available in Python only. Switch the toggle above to Python to
see the notebook details, or use the R scripts in the other sections on this page.
</div>

---

## Usage Segment Trends

Track how Copilot adoption matures over time — a running adoption-health check that complements, rather than replaces, the causal analyses below.

### Copilot Usage Segments Over Time

Sums individual Copilot-action columns, classifies each person-week with `identify_usage_segments(version = "12w")`, and visualises how the mix of Power, Habitual, Novice, Low, and Non-users evolves week by week.

<div data-lang-block="r" markdown="1">
**📄 [copilot-usage-segments-trend.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-usage-segments-trend.Rmd)**
- **Purpose**: Track how the mix of Copilot usage segments evolves week by week
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, dplyr, tidyr, ggplot2, scales
- **Key Features**: identify_usage_segments (12-week rolling), stacked-area segment mix, action trend
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/copilot-usage-segments-trend.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-usage-segments-trend.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [copilot-usage-segments-trend.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-usage-segments-trend.py)**
- **Purpose**: Track how the mix of Copilot usage segments evolves week by week
- **Prerequisites**: vivainsights Python package, pandas, numpy, matplotlib
- **Key Features**: identify_usage_segments (12-week rolling), stacked-area segment mix, action trend
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-usage-segments-trend.py)**
</div>

---

## Causal Analysis

These two examples build small, clearly labelled seeded simulations so that the models have something to recover; swap the simulation block for your own export before drawing conclusions.

### Difference-in-Differences Metric Scan

Runs a within-person DiD per metric across two both-licensed groups (Power vs Low Copilot users) and assembles the effects, confidence intervals, and significance into one sortable table plus a forest plot, honestly surfacing the metrics that do not move.

<div data-lang-block="r" markdown="1">
**📄 [did-metric-scan.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.Rmd)**
- **Purpose**: Run a within-person DiD per metric (Power vs Low Copilot users) into one sortable table
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, fixest, dplyr, tidyr, ggplot2, purrr, scales
- **Key Features**: Per-metric TWFE DiD, significance stars, forest plot, honest reporting of null effects
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/did-metric-scan.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [did-metric-scan.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/did-metric-scan.py)**
- **Purpose**: Run a within-person DiD per metric (Power vs Low Copilot users) into one sortable table
- **Prerequisites**: vivainsights Python package, linearmodels, pandas, numpy, matplotlib
- **Key Features**: Per-metric TWFE DiD, significance stars, forest plot, honest reporting of null effects
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/did-metric-scan.py)**
</div>

---

### Event-Study & Difference-in-Differences

Aligns each adopter on their own event time, checks the parallel-trends assumption before trusting a single headline number, and reads the within-person change net of a non-adopting control.

<div data-lang-block="r" markdown="1">
**📄 [event-study-did.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.Rmd)**
- **Purpose**: Measure within-person behaviour change around Copilot adoption with a TWFE event-study/DiD
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, fixest, dplyr, tidyr, ggplot2, scales
- **Key Features**: Event-time alignment, pre-trend check, person + week fixed effects, z-scored composite index
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/event-study-did.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [event-study-did.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/event-study-did.py)**
- **Purpose**: Measure within-person behaviour change around Copilot adoption with a TWFE event-study/DiD
- **Prerequisites**: vivainsights Python package, linearmodels, pandas, numpy, matplotlib
- **Key Features**: Event-time alignment, pre-trend check, person + week fixed effects, z-scored composite index
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/event-study-did.py)**
</div>

---

## Power BI Integration

### DAX Calculated Columns
**📁 [DAX Calculated Columns](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/dax/calculated-columns)**
- **Purpose**: Pre-built DAX formulas for Copilot usage segmentation in Power BI
- **Language**: DAX
- **Format**: Individual .dax files
- **Prerequisites**: Power BI Desktop, Copilot usage data

**Available Columns:**

#### 12-Week Rolling (RL12W) - Recommended for long-term analysis
- **[📄 _Total Copilot actions_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL12W.dax)**: Average weekly actions over 12 weeks
- **[📄 _IsHabit_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_IsHabit_RL12W.dax)**: Habit formation indicator (9+ weeks of usage)
- **[📄 _CopilotUsageSegment_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL12W.dax)**: User segmentation (Power/Habitual/Novice/Low/Non-users)

#### 4-Week Rolling (RL4W) - Recommended for short-term/pilot analysis
- **[📄 _Total Copilot actions_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL4W.dax)**: Average weekly actions over 4 weeks
- **[📄 _IsHabit_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_IsHabit_RL4W.dax)**: Habit formation indicator (4 weeks of usage)
- **[📄 _CopilotUsageSegment_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL4W.dax)**: User segmentation (Power/Habitual/Novice/Low/Non-users)

**[📖 DAX Documentation](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md)**

---

## Usage Segmentation

### User Segments Defined

These five segments form a **single mutually-exclusive ladder**, evaluated top-down so every user falls into exactly one tier (full definitions and decision tree on the [Copilot Usage Segments]({{ site.baseurl }}/copilot-usage-segments/#formal-definitions) page):

1. **Power Users**: Habitual **and** averaging 15+ weekly Copilot actions
2. **Habitual Users**: Habitual (9+ of 12 weeks in RL12W, all weeks in RL4W) but averaging < 15 weekly actions
3. **Novice Users**: Not habitual, averaging 1+ weekly Copilot actions
4. **Low Users**: Not habitual, some usage but averaging < 1 weekly action
5. **Non-users**: No Copilot usage in the measurement period

---

## Sample Data

### Example Datasets
**📁 [Example Data](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)**
- **[📄 copilot-metrics-taxonomy.csv](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/example-data/copilot-metrics-taxonomy.csv)**: Copilot metrics reference
- **[📄 viva-insights-org-data-sample.xlsx](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/example-data/viva-insights-org-data-sample.xlsx)**: Sample organizational data

---

## Analysis Workflow

1. **Export Copilot Usage Data** from Viva Insights
2. **Choose Your Analysis Method**:
   - R/Python scripts for detailed analysis
   - DAX columns for Power BI dashboards
3. **Select Time Frame**:
   - RL12W for long-term habit analysis
   - RL4W for pilot programs or short-term analysis
4. **Run Analysis** using the appropriate script

## Related pages

- [Copilot Usage Segments]({{ site.baseurl }}/copilot-usage-segments/): how Power, Habitual, and Novice segments are defined
- [DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/): ready-to-use Power BI formulas for segmentation
- [Copilot Causal Toolkit]({{ site.baseurl }}/copilot-causal-toolkit/): measure the causal impact of Copilot on business outcomes
- [Causal Inference in Copilot Analytics]({{ site.baseurl }}/causal-inference/): methods for isolating Copilot's true effect
- [Frontier Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/): generate Copilot reports and dashboards with coding agents
- [Advanced Analytics]({{ site.baseurl }}/advanced/): machine learning, regression, and statistical testing
- [Network Analysis]({{ site.baseurl }}/network/): organizational network analysis (ONA)
- [Essentials]({{ site.baseurl }}/essentials/): utilities and visualizations to prepare your data
- [Getting Started]({{ site.baseurl }}/getting-started/): environment setup and first steps

---

## Need Help?

- **Copilot Analytics Documentation**: [Viva Insights Copilot Guide](https://learn.microsoft.com/en-us/viva/insights/)
- **Power BI Integration**: [DAX Documentation](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)

<script src="{{ '/assets/js/lang-switch.js' | relative_url }}"></script>
