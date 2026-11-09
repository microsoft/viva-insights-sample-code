---
layout: page
title: "Interactive Dashboards"
eyebrow: "Copilot analytics"
description: "Compare three synthetic AI usage dashboards and open a step-by-step guide with screenshots, query requirements and rendering commands."
permalink: /copilot-dashboards/
---

[Copilot Analytics]({{ site.baseurl }}/copilot/) / Interactive Dashboards

Explore three reports built with **synthetic data**. The demos open in a
browser without installation. Each report also has a standalone guide with
screenshots, expected outputs, required query exports and step-by-step commands.

Select **Start using** for the practical setup, a starter prompt and capability
limits, or open a report's **Full guide** for its complete walkthrough.

{% include dashboard-catalogue.html %}

## Build with your own data

The templates use **R**, not Python. Start by reproducing the synthetic
example in the relevant guide, then choose a supported real-data path.

| Report | Real-data support |
|---|---|
| [Business Leaders AI Usage]({{ site.baseurl }}/copilot-dashboards/business-leaders-ai-usage/) | Synthetic reference only. No verified real-export adapter. |
| [Consumption and Ways of Working]({{ site.baseurl }}/copilot-dashboards/copilot-consumption-and-ways-of-working/) | M365 credits-only build from Consumption activity and people metadata, with explicit mapping approval. |
| [Developer Experience and Copilot]({{ site.baseurl }}/copilot-dashboards/developer-experience-and-copilot/) | Input-readiness inspection only. No verified real GitHub dashboard adapter. |

For the two runner-supported demos, the
[Copilot query dashboards Skill]({{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-copilot-dashboards)
guides reproduction, inspection and customisation.
[Joining Viva Insights queries]({{ site.baseurl }}/query-join-guide/) explains
query setup and integration. It does not expand the runner's supported inputs.

**Before sharing:** apply your organisation's privacy rules, review embedded
data as well as visible charts, and retain coverage limitations. Credit
consumption is not monetary cost or business value. Descriptive comparisons
do not establish causal impact. Keep customer exports outside this repository.

[Copilot Analytics]({{ site.baseurl }}/copilot/) ·
[Getting Started]({{ site.baseurl }}/getting-started/) ·
[Causal Inference]({{ site.baseurl }}/causal-inference/)
