---
layout: page
title: "Copilot Analytics"
permalink: /copilot/
---

# Copilot Analytics Scripts

Specialized scripts for analyzing Microsoft Copilot usage data from Viva Insights.

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

**Power Users**: 15+ average weekly Copilot actions + habitual usage
**Habitual Users**: Consistent usage patterns (9+ weeks in RL12W, all weeks in RL4W)
**Novice Users**: 1+ average weekly Copilot actions
**Low Users**: Some Copilot usage but below novice threshold
**Non-users**: No Copilot usage in the measurement period

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

## Need Help?

- **Copilot Analytics Documentation**: [Viva Insights Copilot Guide](https://docs.microsoft.com/en-us/viva/insights/)
- **Power BI Integration**: [DAX Documentation](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)
