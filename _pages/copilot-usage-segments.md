---
layout: page
title: "Copilot Usage Segments"
eyebrow: "Copilot analytics"
permalink: /copilot-usage-segments/
---
## Why Copilot Usage Segments?

The goal of this segmentation technique is to identify segments of users in an organization who are using Copilot more effectively than others, which can provide insights on how organizations can increase overall Copilot adoption and drive AI transformation.

Furthermore, consistency is a critical factor in successful AI adoption. Consistent users are less likely to revert to non-usage, making this an important concept for measuring how well an organization has successfully evolved its AI culture and embedded Copilot into daily workflows.

## Definitions - What Are the Segments?

There are five user segments that represent different stages of Copilot adoption:

### 1. Power Users

**Power Users** represent the ideal user who maximizes the potential of Copilot and are both consistent and high-volume users. This group is likely to be a minority of users but can be seen as an aspirational group for deploying an AI adoption strategy.

### 2. Habitual Users  
**Habitual Users** are consistent users of Copilot, with lower volume than Power Users. They represent 'everyday' users who have successfully adopted Copilot into their routine.

Power and Habitual Users represent a success measure. The higher the incidence of Power and Habitual Users in your organization, the more embedded Copilot is in your organization.

### 3. Novice Users
**Novice Users** are users with potential to become Habitual Users, but who may need additional support to avoid lapsing into lower usage.

### 4. Low Users  
**Low Users** are either early in their adoption journey or require significant assistance with onboarding and utilizing Copilot.

### 5. Non-users
**Non-users** are individuals who are enabled on Copilot, but do not use it.

---

## Formal Definitions

The five segments form a **single mutually-exclusive ladder**. Every licensed user falls into exactly one segment. The rules are evaluated **top-down (highest tier first)**, and each lower tier therefore *excludes* everyone already captured above it. Two inputs are computed over a rolling 12-week window:

- **Average weekly actions** — average weekly `Total Copilot actions` over the window.
- **Habitual** — any use of Copilot in **at least 9 of the past 12 weeks** (see the rolling-window rationale below).

| Priority | Segment | Condition (first match wins) |
|:--------:|---------|------------------------------|
| 1 | **Power Users** | Habitual **AND** averaging **15+** weekly actions |
| 2 | **Habitual Users** | Habitual (and averaging **< 15** weekly actions) |
| 3 | **Novice Users** | **Not** habitual **AND** averaging **1+** weekly actions |
| 4 | **Low Users** | **Not** habitual **AND** averaging **>0 but <1** weekly actions |
| 5 | **Non-users** | **Zero** Copilot actions in the window |

Because the tiers are evaluated in order, the segments never overlap: a habitual, high-volume user is a Power User (not also counted as Habitual/Novice/Low), and Novice/Low users are by construction *not* habitual.

### Decision tree

```text
Any Copilot actions in the window?
├─ No  → Non-user
└─ Yes → Habitual? (used Copilot in 9 of the last 12 weeks)
         ├─ Yes → Averaging 15+ weekly actions?
         │        ├─ Yes → Power User
         │        └─ No  → Habitual User
         └─ No  → Averaging 1+ weekly actions?
                  ├─ Yes → Novice User
                  └─ No  → Low User
```

---

## Adoption Maturity Framework

Organizations can assess their Copilot adoption maturity by examining the percentage of Power Users in their workforce:

| Power User % | Adoption Stage | Description |
|--------------|----------------|-------------|
| **0 - 20%** | **Initial Rollout** | Early deployment phase with limited adoption |
| **20 - 40%** | **Ramping Up** | Growing user base with increasing engagement |
| **40 - 60%** | **Embedding** | Copilot becoming integral to daily workflows |
| **60%+** | **Full Integration** | Organization-wide AI adoption with mature usage patterns |

> **Note**: These benchmarks serve as general guidelines. Organizations should consider their specific context, rollout timeline, and industry when interpreting their adoption progress.

---

## Segment Variations

While the standard definitions above work well for most organizations, you may need to adjust the thresholds based on your specific deployment context and maturity stage. Here are two validated variations:

### 🌱 Early Rollout Variation (4-Week Window)

**When to Use**: Organizations that have recently deployed Copilot (within 3 months) and need to measure short-term adoption success with limited historical data.

**Key Differences**: 
- Uses 4 weeks of data instead of 12 weeks
- Maintains the same behavioral profiles but with shorter observation period
- Less precise for measuring long-term consistency and habit formation

**Trade-offs**: While this variation identifies similar user segments, it provides a less accurate measure of true habit formation since it doesn't observe the extended period typically required for behavioral consistency.

The same mutually-exclusive, top-down ladder applies — only the habit window changes (all 4 of the last 4 weeks instead of 9 of 12):

| Priority | Segment | Definition (4-Week Window) |
|:--------:|---------|------------|
| 1 | **Power Users** | Habitual (used Copilot in all 4 weeks) **AND** averaging 15+ weekly actions |
| 2 | **Habitual Users** | Habitual (all 4 weeks) and averaging < 15 weekly actions |
| 3 | **Novice Users** | Not habitual **AND** averaging 1+ weekly actions over the last 4 weeks |
| 4 | **Low Users** | Not habitual **AND** averaging >0 but <1 weekly actions |
| 5 | **Non-users** | Zero Copilot actions in the last 4 weeks |

---

### 🚀 Mature Organization Variation (Enhanced Thresholds)

**When to Use**: Organizations that have had Copilot deployed for over a year, achieved 60%+ Power Users, and need more granular measurement for advanced optimization.

**Key Differences**:
- Higher activity thresholds (50+ actions for Power Users, 25+ for Habitual Users)
- Equivalent to ~5-10 actions per workday in a standard 5-day work week
- Provides better differentiation in mature adoption environments

**Benefits**: This variation helps identify truly advanced users and provides more meaningful segmentation when basic adoption has already been achieved organization-wide.

The ladder is identical to the standard one; only the volume thresholds are raised. Tiers are still evaluated top-down and remain mutually exclusive:

| Priority | Segment | Definition (Enhanced Thresholds) |
|:--------:|---------|------------|
| 1 | **Power Users** | 25+ actions in at least 9 of the past 12 weeks **AND** averaging 50+ weekly actions |
| 2 | **Habitual Users** | 25+ actions in at least 9 of the past 12 weeks (and averaging < 50 weekly actions) |
| 3 | **Novice Users** | Below the habitual threshold **AND** averaging 1+ weekly actions over the last 12 weeks |
| 4 | **Low Users** | Below the habitual threshold **AND** averaging >0 but <1 weekly actions |
| 5 | **Non-users** | Zero Copilot actions in the last 12 weeks |

---

### 💡 Choosing the Right Variation

| Your Situation | Recommended Variation | Rationale |
|----------------|----------------------|-----------|
| **New Deployment** (< 3 months) | Early Rollout (4-week) | Provides quick insights with limited data |
| **Standard Deployment** (3-12 months) | Standard Definition | Balances accuracy with practical measurement |
| **Mature Deployment** (12+ months, 60%+ Power Users) | Enhanced Thresholds | Better differentiation for optimization |

> **Recommendation**: Start with the standard definition and consider variations only when your specific context clearly warrants the adjustment. Consistency in measurement approach over time is often more valuable than perfect threshold optimization.

---

## What about Super Users?

**Super Users** represent a segment of highly engaged Copilot users as identified in our [Super Users report](https://aka.ms/decodingsuperusage/). Super Users are defined as the top 10% of users based on weekly Copilot actions, calculated over a predetermined date period.

The Super Users report identifies five distinct usage groups based on activity volume: 

* **Super Users** (top 10%)
* **High usage** (top 25%)
* **Moderate usage** (top 50%)
* **Low usage** (bottom 50%)
* **Very low usage** (bottom 25%)

### When to Use Super Users vs. Power Users

The Super Users paradigm offers several advantages:
- **Simple calculation**: Straightforward percentile-based segmentation
- **Clear explanation**: Easy to communicate to stakeholders
- **Usage distribution insights**: Reveals how Copilot activity is distributed across your user population
- **Gap analysis**: Highlights opportunities to bridge usage differences between high and low users

However, the **Power Users framework is recommended for ongoing measurement** and goal-setting because:
- **Consistent definitions**: Segments remain stable over time, enabling trend analysis
- **Absolute thresholds**: Not dependent on population size or date range selection
- **Habit measurement**: Focuses on behavioral consistency, which predicts long-term adoption success

**Best Practice**: Use Super Users for initial usage distribution analysis and stakeholder communication, then transition to Power Users segmentation for continuous monitoring and improvement initiatives. 

---

## How to calculate Copilot Usage Segments

To calculate the Copilot Usage Segments, you can use the `identify_usage_segments()` function from the R or Python libraries. 

For an implementation in Power BI, have a look at [this guide]({{ site.baseurl }}/dax-calculated-columns/).

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

- **[DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/)** - Ready-to-use Power BI formulas
- **[Copilot Analytics Scripts]({{ site.baseurl }}/copilot/)** - Python and R implementation examples

---

## Need Help?

- **Methodology Questions**: Review the FAQ section above for common implementation questions
- **Technical Implementation**: Visit our [Copilot Analytics]({{ site.baseurl }}/copilot/) page for code examples
- **Power BI Integration**: Check out our [DAX Calculated Columns]({{ site.baseurl }}/dax-calculated-columns/) for ready-to-use formulas  
