---
layout: page
title: "Interactive Dashboards"
eyebrow: "Copilot analytics"
description: "Explore two Copilot demo reports, then use the R templates to build an analysis with your own Viva Insights exports."
permalink: /copilot-dashboards/
---

[Copilot Analytics]({{ site.baseurl }}/copilot/) / Interactive Dashboards

**Just exploring?** Open either demo in your browser; no installation is required.

**Building your own?** Follow the [setup guide](#build-with-your-own-data).
For cross-query analysis, use [Joining Viva Insights queries]({{ site.baseurl }}/query-join-guide/)
for export setup, schemas and person-week joins.

Both reports use **synthetic data**. They illustrate analysis approaches, not evidence of Copilot's impact. Assumptions and simulation details are included in each report's methods or appendix.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="#copilot-consumption-and-ways-of-working">
    <span class="vi-card-title">Copilot Consumption and Ways of Working</span>
    <span class="vi-card-desc">How is Copilot consumed, and how does consumption relate to collaboration?</span>
    <span class="vi-card-more">Explore this template →</span>
  </a>
  <a class="vi-card" href="#developer-experience-and-copilot">
    <span class="vi-card-title">Developer Experience and Copilot</span>
    <span class="vi-card-desc">What do developer working conditions and joint Copilot use look like?</span>
    <span class="vi-card-more">Explore this template →</span>
  </a>
</div>

## Copilot Consumption and Ways of Working

**Synthetic data · R template · Seven-page HTML report**

**When to use it:** reach for this report when you need to answer questions such as —
*Are a small number of users or groups driving most of our credit spend?
Do certain user groups consume disproportionately more credits, not just more sessions?
How does cost per session vary by service, organisation or function?
Are our heaviest consumers consistently heavy, or do they spike occasionally?*
It is built around the volume-vs-mix distinction: total consumption differs from
*how expensive* each session of that consumption is.

**Scope:** "consumption" here refers to **Copilot credit consumption** as captured by
the Viva Insights Consumption Query. That query covers Microsoft 365 Copilot services
and also emits GitHub AI credits, so both appear in this report. The report additionally
reads the GitHub query's *activity* file to tell observed non-use apart from absent
observation. For the GitHub Copilot feature, model and language **breakdowns** — which
this report does not present — see the
[Developer Experience and Copilot](#developer-experience-and-copilot) report below.

**Real v1 capability:** the supplied runner builds a narrower **M365 credits-only**
report from Consumption activity and its matching people metadata, with explicit
mapping approval. Sessions, Person Query associations and GitHub panels are not
implemented by that real adapter. Reproducing the full synthetic demonstration on
governed inputs requires separately implemented and verified support; matching headers
alone does not make it a like-for-like substitution.

Separate consumption volume from consumption cost: credit concentration and percentile bands, heavy-user consumption patterns, cost per session, service mix, and function drill-downs. For the [Copilot Usage Segments]({{ site.baseurl }}/copilot-usage-segments/) ladder — Power, Habitual, Novice, Low and Non-users — use `identify_usage_segments()`, which needs a longer rolling window than this report's product feed provides.

**[View demo]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

**[Build or customise with AI]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/)**:
reproduce the demo or build a scoped credits report with an agent and reusable R code.

[![Consumption report overview]({{ site.baseurl }}/assets/images/reports/copilot-consumption-overview.png)]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)

**Demo inputs:** Five fixture files across Person, Consumption and GitHub queries.
The GitHub activity summary is used to distinguish recorded zero activity from
missing observation. Its four breakdown files are not used by this report.
See the [template input manifest](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md#which-queries-each-report-needs)
for the exact files, and the [join tutorial]({{ site.baseurl }}/query-join-guide/#queries)
for export schemas.

**Interpretation:** Associations only. Credit intensity is a cost measure, not a measure of value or quality. Concentration curves and banded percentiles help describe skewed consumption without relying on an average.

<details markdown="1">
<summary>See the product usage mix preview</summary>

![Consumption report product usage mix]({{ site.baseurl }}/assets/images/reports/copilot-consumption-product-usage-mix.png)

</details>

## Developer Experience and Copilot

**Synthetic data · R template · GitHub Copilot usage as the organising lens**

**When to use it:** reach for this report when you need to answer questions such as —
*Which developers are the heaviest GitHub Copilot users, and where are they
concentrated? Do heavy GitHub users show different meeting load or
uninterrupted-focus time than other observed developers? Do they collaborate with
broader or narrower internal networks? How does GitHub Copilot use vary by
team, model, or language? Are developers using GitHub Copilot and Microsoft 365
Copilot jointly, or are the two adopted independently? Is missing activity actually
zero usage, or a coverage/eligibility gap?* It defines a **heavy GitHub-use** group
on the GitHub-observed population, then compares developer working conditions across
the three GitHub-intensity groups, and keeps eligibility and coverage visible rather
than treating missing activity as zero.

**Scope:** this report adds **GitHub Copilot activity** and feature, model and language
breakdowns to a developer working-conditions baseline, alongside **Microsoft 365
Copilot** credit records. The Consumption demo above also contains both products'
credit measures, in separately named units; it does not include the GitHub activity
breakdowns. Neither demo establishes credit-unit equivalence. These synthetic-demo
capabilities do not expand the narrower M365 credits-only real-v1 adapter.

**Inputs for a real-data adaptation:** Person, GitHub and Consumption exports,
including GitHub breakdowns. Use the [join tutorial]({{ site.baseurl }}/query-join-guide/)
for query setup and integration, and the
[template input manifest](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md#which-queries-each-report-needs)
for this report's file requirements. The tutorial does not supply a verified real
GitHub dashboard adapter or establish eligibility and coverage for your exports.

Start with Overview, which defines the **heavy GitHub-use** group and shows where
it concentrates by team and role where disclosure permits, then explore **Collaboration**, **Focus**,
**After-hours** and **Network** — each leads with the three-group GitHub-intensity
comparison. The **More** menu contains GitHub breakdowns, GitHub coverage,
working-pattern comparisons and methods. Detailed tables expand in place, and
charts reflow for smaller screens.

**[View demo]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-copilot-developer-productivity-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

**[Build or customise with AI]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/)**:
reproduce the synthetic reference or assess real inputs. No verified real GitHub adapter is supplied yet.

[![Developer experience report heavy GitHub Copilot user overview]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-overview.png)]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)

The developer demo uses **synthetic values in confirmed export headers**, not a certified real-data adapter. Weekly M365 eligibility uses enabled days; ingestion completeness remains unknown without independent evidence. GitHub row presence is observation, not provisioning. Breakdown/model allocations are illustrative synthetic invariants, not verified real-export semantics. M365 Copilot credits and GitHub AI credits remain separately named units; no cross-product total or share is supported.

**Interpretation:** Associations only. Acceptance rate is not code quality, after-hours activity is not burnout, and calendar space does not establish coding time.

<details markdown="1">
<summary>See the recorded GitHub use and observation preview</summary>

![Developer experience report network breadth by GitHub intensity]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-network.png)

</details>

## Build with your own data

The demos need only a browser. To render or adapt the templates, use **R**, **Pandoc**, **R Markdown**, and **flexdashboard**. There is no Python version of these templates.

1. **Get the files.** Start with the report source linked above. For Developer Experience, also get [github-developer-experience-helpers.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-developer-experience-helpers.R) and [render-github-developer-experience.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/render-github-developer-experience.R), keeping the repository folder structure.
2. **Render the synthetic example first.** Dependencies include `vivainsights`, `dplyr`, `tidyr`, `ggplot2`, `ggrepel`, `scales`, `stringr`, `flexdashboard`, `knitr`, and `rmarkdown`. The AI-guided journeys copy the required fixtures and source into an isolated output folder before rendering.
3. **Choose the real-data path.** The [Consumption journey]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/) builds the scoped M365 credits-only report from activity and people metadata, without requiring a Person Query. The [Developer Experience journey]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/) assesses readiness only. For a separately implemented cross-query adaptation, follow the [join tutorial]({{ site.baseurl }}/query-join-guide/). Its person-week panel is not a drop-in input to either runner.
4. **Review before sharing.** Update simulation labels and methodology to describe your actual inputs. Apply your organization's privacy requirements and retain the distinction between association and causation. Do not put customer exports in this sample repository or in shareable HTML.

[Template setup and rendering commands](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md#rerun)
· [Query setup and joins]({{ site.baseurl }}/query-join-guide/)

## Continue exploring

[Copilot Analytics]({{ site.baseurl }}/copilot/) for scripts and segmentation ·
[Getting Started]({{ site.baseurl }}/getting-started/) for environment setup ·
[Causal Inference]({{ site.baseurl }}/causal-inference/) for impact-analysis methods
