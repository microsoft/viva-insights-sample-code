---
layout: page
title: "Copilot Analytics"
eyebrow: "Copilot analytics"
description: "Analyze Microsoft 365 Copilot usage from Viva Insights — adoption metrics, Power User and Habitual User segmentation, habit-based behavioral models, usage-segment trends over time, and causal impact analysis with difference-in-differences and event-study methods in R, Python, and Power BI."
permalink: /copilot/
---
# Copilot Analytics Scripts

This page contains specialized scripts for analyzing Microsoft Copilot usage data from Viva Insights.

Beyond covering key analyses around Copilot usage volume and breadth (range of actions and applications), these scripts also include a method for measuring Copilot habituality based on behavioral research. This approach determines whether a user can be considered a habitual Copilot user, enabling visualization through user segmentation that identifies **Power Users** and **Habitual Users** within an organization. This segmentation provides a framework for continuously tracking Copilot adoption success and measuring the effectiveness of your deployment strategy.

See our [DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/) page for detailed instructions on how to identify Copilot Usage Segments using Power BI templates and pre-built DAX formulas.

For more information on the Copilot Usage Segments, see this [introduction]({{ site.baseurl }}/copilot-usage-segments/).

For more inspiration on analyzing Copilot adoption and impact, have a look at our [advanced examples playbook](https://aka.ms/CopilotAdvancedAnalytics/).

---

## Advanced Analysis Scripts

### Copilot Advanced Analysis (R)
**📄 [copilot-analytics-examples.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-analytics-examples.R)**
- **Purpose**: Comprehensive analysis of Copilot usage patterns and trends
- **Language**: R
- **Prerequisites**: vivainsights R package, Copilot usage data
- **Key Analysis**: Usage segmentation, trend analysis, adoption metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/copilot-analytics-examples.R)**

### Copilot Advanced Analysis (Python)
**📄 [copilot-analytics-examples.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.py)**
- **Purpose**: Comprehensive analysis of Copilot usage patterns and trends
- **Language**: Python
- **Prerequisites**: vivainsights Python package, Copilot usage data
- **Key Analysis**: Usage segmentation, trend analysis, adoption metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.py)**

### Copilot Analytics (Jupyter Notebook)
**📓 [copilot-analytics-examples.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-analytics-examples.ipynb)**
- **Purpose**: Interactive analysis of Copilot usage with visualizations
- **Language**: Python
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, Copilot usage data
- **Key Features**: Step-by-step analysis, interactive visualizations
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-analytics-examples.ipynb)**

---

## Adoption Tracking & Causal Analysis

The examples in this section focus on measuring Copilot adoption credibly over time. The *Copilot usage segments over time* scripts sum individual Copilot-action columns, classify each person-week with `identify_usage_segments(version = "12w")`, and visualise how the mix of Power, Habitual, Novice, Low, and Non-users evolves week by week. The *difference-in-differences metric scan* runs a within-person DiD per metric across two both-licensed groups (Power vs Low Copilot users) and assembles the effects, confidence intervals, and significance into one sortable table plus a forest plot, honestly surfacing the metrics that do not move. The *event-study and difference-in-differences* example aligns each adopter on their own event time, checks the parallel-trends assumption before trusting a single headline number, and reads the within-person change net of a non-adopting control. The two causal examples build small, clearly labelled seeded simulations so that the models have something to recover; swap the simulation block for your own export before drawing conclusions.

### Copilot Usage Segments Over Time (Python)
**📄 [copilot-usage-segments-trend.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/copilot-usage-segments-trend.py)**
- **Purpose**: Track how the mix of Copilot usage segments evolves week by week
- **Language**: Python
- **Prerequisites**: vivainsights Python package, pandas, numpy, matplotlib
- **Key Features**: identify_usage_segments (12-week rolling), stacked-area segment mix, action trend
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/copilot-usage-segments-trend.py)**

### Copilot Usage Segments Over Time (R)
**📄 [copilot-usage-segments-trend.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-usage-segments-trend.Rmd)**
- **Purpose**: Track how the mix of Copilot usage segments evolves week by week
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, dplyr, tidyr, ggplot2, scales
- **Key Features**: identify_usage_segments (12-week rolling), stacked-area segment mix, action trend
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/copilot-usage-segments-trend.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-usage-segments-trend.html)**

---

### Difference-in-Differences Metric Scan (Python)
**📄 [did-metric-scan.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/did-metric-scan.py)**
- **Purpose**: Run a within-person DiD per metric (Power vs Low Copilot users) into one sortable table
- **Language**: Python
- **Prerequisites**: vivainsights Python package, linearmodels, pandas, numpy, matplotlib
- **Key Features**: Per-metric TWFE DiD, significance stars, forest plot, honest reporting of null effects
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/did-metric-scan.py)**

### Difference-in-Differences Metric Scan (R)
**📄 [did-metric-scan.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.Rmd)**
- **Purpose**: Run a within-person DiD per metric (Power vs Low Copilot users) into one sortable table
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, fixest, dplyr, tidyr, ggplot2, purrr, scales
- **Key Features**: Per-metric TWFE DiD, significance stars, forest plot, honest reporting of null effects
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/did-metric-scan.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.html)**

---

### Event-Study & Difference-in-Differences (Python)
**📄 [event-study-did.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/event-study-did.py)**
- **Purpose**: Measure within-person behaviour change around Copilot adoption with a TWFE event-study/DiD
- **Language**: Python
- **Prerequisites**: vivainsights Python package, linearmodels, pandas, numpy, matplotlib
- **Key Features**: Event-time alignment, pre-trend check, person + week fixed effects, z-scored composite index
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/event-study-did.py)**

### Event-Study & Difference-in-Differences (R)
**📄 [event-study-did.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.Rmd)**
- **Purpose**: Measure within-person behaviour change around Copilot adoption with a TWFE event-study/DiD
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, fixest, dplyr, tidyr, ggplot2, scales
- **Key Features**: Event-time alignment, pre-trend check, person + week fixed effects, z-scored composite index
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/event-study-did.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.html)**

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

## Getting Started

1. **Export Copilot Usage Data** from Viva Insights
2. **Choose Your Analysis Method**:
   - R/Python scripts for detailed analysis
   - DAX columns for Power BI dashboards
3. **Select Time Frame**:
   - RL12W for long-term habit analysis
   - RL4W for pilot programs or short-term analysis
4. **Run Analysis** using the appropriate script

## Related pages

- [Copilot Usage Segments]({{ site.baseurl }}/copilot-usage-segments/) — how Power, Habitual, and Novice segments are defined
- [DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/) — ready-to-use Power BI formulas for segmentation
- [Copilot Causal Toolkit]({{ site.baseurl }}/copilot-causal-toolkit/) — measure the causal impact of Copilot on business outcomes
- [Causal Inference in Copilot Analytics]({{ site.baseurl }}/causal-inference/) — methods for isolating Copilot's true effect
- [Frontier Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) — generate Copilot reports and dashboards with coding agents

---

## Need Help?

- **Copilot Analytics Documentation**: [Viva Insights Copilot Guide](https://docs.microsoft.com/en-us/viva/insights/)
- **Power BI Integration**: [DAX Documentation](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)
