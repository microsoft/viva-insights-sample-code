---
layout: default
title: "Calculated Columns for Copilot Usage Segmentation"
permalink: /dax-calculated-columns/
---

# Calculated Columns for Copilot Usage Segmentation

{% include custom-navigation.html %}

<style>
/* Hide default Minima navigation to prevent duplicates */
.site-header .site-nav,
.site-header .trigger,
.site-header .page-link {
  display: none !important;
}
</style>

This page provides DAX scripts for calculated columns used to segment users by their Copilot usage patterns in Power BI.

---

## RL12W (12-week rolling) Approach

**Recommended if the adoption timeframe is longer than 12 weeks, and the goal is to evaluate long term habit formation on Copilot usage.**

- **[_Total Copilot actions_RL12W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL12W.dax)**:  
  Calculates the average weekly Copilot actions for each user over the last 12 weeks.

- **[_IsHabit_RL12W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_IsHabit_RL12W.dax)**:  
  Flags users as "habitual" if they used Copilot in at least 9 of the last 12 weeks.

- **[_CopilotUsageSegment_RL12W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL12W.dax)**:  
  Categorizes users as Power, Habitual, Novice, Low, or Non-users based on their calculated usage and habit status.

### Definitions

- **Power User**: average 15+ weekly total Copilot actions and any use of Copilot in 9 of the last 12 weeks.
- **Habitual User**: any use of Copilot in 9 of the last 12 weeks.
- **Novice User**: average 1+ weekly Copilot actions over the past 12 weeks.
- **Low User**: at least one Copilot action in the past 12 weeks.
- **Non−user**: zero Copilot actions in the past 12 weeks.

---

## RL4W (4-week rolling) Approach

**Recommended if the adoption period is shorter than 12 weeks, or adoption is being evaluated as part of an agile pilot program.**

- **[_Total Copilot actions_RL4W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL4W.dax)**:  
  Calculates the average weekly Copilot actions for each user over the last 4 weeks.

- **[_IsHabit_RL4W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_IsHabit_RL4W.dax)**:  
  Flags users as "habitual" if they used Copilot in all of the last 4 weeks.

- **[_CopilotUsageSegment_RL4W.dax](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL4W.dax)**:  
  Categorizes users as Power, Habitual, Novice, Low, or Non-users based on their calculated usage and habit status.

### Definitions

- **Power User**: average 15+ weekly total Copilot actions and any use of Copilot in all of the last 4 weeks.
- **Habitual User**: any use of Copilot in all of the last 4 weeks.
- **Novice User**: average 1+ weekly Copilot actions over the past 4 weeks.
- **Low User**: at least one Copilot action in the past 4 weeks.
- **Non−user**: zero Copilot actions in the past 4 weeks.

---

## Usage Instructions

1. **Open Power BI Desktop** and select your table in Data view.
2. **Download the DAX file** you want to use from the links above.
3. **Copy the DAX expression** and paste it into Power BI as a 'New Column' formula.
4. **Use the resulting columns** to analyze and visualize Copilot adoption and habit formation.

### Available DAX Files

**12-Week Rolling (RL12W) Files:**
- [📄 _Total Copilot actions_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL12W.dax)
- [📄 _IsHabit_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_IsHabit_RL12W.dax)
- [📄 _CopilotUsageSegment_RL12W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL12W.dax)

**4-Week Rolling (RL4W) Files:**
- [📄 _Total Copilot actions_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_Total%20Copilot%20actions_RL4W.dax)
- [📄 _IsHabit_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_IsHabit_RL4W.dax)
- [📄 _CopilotUsageSegment_RL4W.dax](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/dax/calculated-columns/_CopilotUsageSegment_RL4W.dax)

---

## Creating Total Copilot Actions Column

It may be necessary to create a custom "Total Copilot actions" column in the Power Query step under 'Transform Data'. Here's an example M (Power Query) script:

```m
= let
    Source = #"Changed Type", // Update with name of previous step as appropriate
    AddedCustomColumn = Table.AddColumn(
    Source, 
    "Total Copilot actions",
    each [#"Copilot actions taken in Teams"] + 
         [#"Copilot actions taken in Outlook"] +
         [#"Copilot actions taken in Word"] +
         [#"Copilot actions taken in Excel"] +
         [#"Copilot actions taken in Powerpoint"],
    Int64.Type
    )
in
    AddedCustomColumn
```

---

## Source Files

- **[📁 View on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/dax/calculated-columns)**: Complete source code and DAX files
- **[📖 Original README](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/dax/calculated-columns/README.md)**: Detailed documentation

---

## Need Help?

For detailed usage instructions and additional context, see our [Usage Segmentation Guide]({{ site.baseurl }}/copilot/) or visit the [source folder on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/dax/calculated-columns).
