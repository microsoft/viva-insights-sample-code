

---
layout: page
title: "Copilot Usage Segments"
permalink: /copilot-usage-segments/
---

{% include custom-navigation.html %}

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

# Copilot Usage Segments

## Why Copilot Usage Segments?

The goal of this segmentation technique is to identify segments of users in an organization who are using Copilot more effectively than others, which can provide insights on how organizations can increase overall Copilot adoption and drive AI transformation.

Furthermore, consistency is a critical factor in successful AI adoption. Consistent users are less likely to revert to non-usage, making this an important concept for measuring how well an organization has successfully evolved its AI culture and embedded Copilot into daily workflows.

## Definitions - What Are the Segments?

There are five user segments that represent different stages of Copilot adoption:

### 🚀 Power Users

**Power Users** represent the ideal user who maximizes the potential of Copilot and are both consistent and high-volume users. This group is likely to be a minority of users but can be seen as an aspirational group for deploying an AI adoption strategy.

### 🔄 Habitual Users  
**Habitual Users** are consistent users of Copilot, with lower volume than Power Users. They represent 'everyday' users who have successfully adopted Copilot into their routine.

> **Success Measure**: Power and Habitual Users represent a success measure. The higher the incidence of Power and Habitual Users in your organization, the more embedded Copilot is in your organization.

### 🌱 Novice Users
**Novice Users** are users with potential to become Habitual Users, but who may need additional support to avoid lapsing into lower usage.

### 📈 Low Users  
**Low Users** are either early in their adoption journey or require significant assistance with onboarding and utilizing Copilot.

### ⭕ Non-users
**Non-users** are individuals who are enabled on Copilot, but do not use it.

---

## Formal Definitions

| Segment | Definition |
|---------|------------|
| **Power Users** | Averaging 15+ weekly total Copilot actions **AND** any use of Copilot in at least 9 out of past 12 weeks |
| **Habitual Users** | Any use of Copilot in at least 9 out of past 12 weeks |
| **Novice Users** | Averaging at least one Copilot action over the last 12 weeks |
| **Low Users** | Having any Copilot action in the past 12 weeks |
| **Non-users** | Zero Copilot actions in the last 12 weeks |

---


## Frequently Asked Questions

### Why do we use a 9 out of 12-weeks rolling window?

The 9 out of 12 week rolling window is based out of research on habits that it takes an average of 66 days to form a habit. The 3-week gap allows for breaks in the habit, such as when someone is on leave. The literature supports that short breaks in behaviour do not necessarily inhibit the habit-forming process. 

### What do we recommend customers who only have 3-months of data?

In general, we recommend only performing an analysis on habit formation with a minimum of 3 months of data. However, there is an estimation method based on 4 out of 4 week consistent usage that identifies a similar segment to the main method. 
 
### The thresholds of Habitual Users seem very strict. Why not make them easier? 

The goal of the Power/Habitual User segments are to identify users who have achieved usage stickiness and are unlikely to revert to low or non-usage in the long term. They represent users who have adopted Copilot into their regular workflow. Reducing the thresholds would increase the incidence of this group, but will dilute the interpretation of the segments. 

### Why focus on consistency? Surely using more is better, even if it’s inconsistent usage? 

We found from the data that consistency predicts stickiness, and consistent users use a wider set of features than users who are high volume but inconsistent. Hence, consistency is a more important metric when interpreting the results of AI transformation.

---

## Implementation

To implement these segments in your analysis, you can use our pre-built DAX calculations or Python/R scripts:

- **[DAX Calculated Columns](/dax-calculated-columns/)** - Ready-to-use Power BI formulas
- **[Copilot Analytics Scripts](/copilot/)** - Python and R implementation examples

---

## Need Help?

- **Methodology Questions**: Review the FAQ section above for common implementation questions
- **Technical Implementation**: Visit our [Copilot Analytics](/copilot/) page for code examples
- **Power BI Integration**: Check out our [DAX Calculated Columns](/dax-calculated-columns/) for ready-to-use formulas  
