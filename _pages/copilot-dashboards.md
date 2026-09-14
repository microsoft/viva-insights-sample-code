---
layout: page
title: "Interactive Dashboards"
eyebrow: "Copilot analytics"
description: "Explore two Copilot demo reports, then use the R templates to build an analysis with your own Viva Insights exports."
permalink: /copilot-dashboards/
---

[Copilot Analytics]({{ site.baseurl }}/copilot/) / Interactive Dashboards

**Just exploring?** Open either demo in your browser; no installation is required.

**Building your own?** Review the required queries, get the source, and follow the setup guide below.

Both seven-page reports use **synthetic data**. They illustrate analysis approaches, not evidence of Copilot's impact. Assumptions and simulation details are included in each report's appendix.

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
*Are a small number of users or groups driving most of our token/credit spend?
Do certain user groups consume disproportionately more tokens and credits, not just more "actions"?
How does credit intensity (cost per 1,000 tokens) vary by task type or function?
What does the delegated-task mix look like across usage segments?*
It is built around the volume-vs-mix distinction: total consumption differs from
*how efficiently* that consumption converts into credits.

**Scope:** "consumption" here refers to **Microsoft 365 Copilot** usage only (credits,
tokens, sessions), as captured by the Viva Insights Consumption Query. It does not
cover GitHub Copilot consumption — for GitHub Copilot activity, see the
[Developer Experience and Copilot](#developer-experience-and-copilot) report below.

**Prerequisites for real data:** to run this on your own tenant's data you need both a
**Person Query** export (for organisational attributes and HR groupings) and a
**[Consumption Query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query)**
export (for credits and sessions). The demo additionally simulates token and
delegated-task-type fields that are not part of the public Consumption schema, so the
real-data path (see below) produces a narrower, credits-only report.

Separate consumption volume from consumption mix: token concentration and percentile bands, credit intensity per 1,000 tokens, delegated task types, usage segments, and function drill-downs.

**[View demo]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

**[Build or customise with AI]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/)**:
reproduce the demo or build a scoped credits report with an agent and reusable R code.

[![Consumption report overview]({{ site.baseurl }}/assets/images/reports/copilot-consumption-overview.png)]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)

**Demo inputs:** Synthetic credit, token and task-type data alongside Person Query and network fixtures. The [public Consumption schema](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query) establishes credits and sessions, but not this demo's token/task-type fields. The AI-guided real-data path therefore builds a narrower credits-only report and labels unsupported panels.

**Interpretation:** Associations only. Credit intensity is a cost-mix measure, not a measure of value or quality. Concentration curves and banded percentiles help describe skewed consumption without relying on an average.

<details markdown="1">
<summary>See the token distribution preview</summary>

![Consumption report token distribution]({{ site.baseurl }}/assets/images/reports/copilot-consumption-token-distribution.png)

</details>

## Developer Experience and Copilot

**Synthetic data · R template · Seven-page HTML report**

**When to use it:** reach for this report when you need to answer questions such as —
*Do developers who also use Microsoft 365 Copilot show different meeting load or
uninterrupted-focus time than those who don't? How does GitHub Copilot use vary by
team, model, or language? Are developers using GitHub Copilot and Microsoft 365
Copilot jointly, or are the two adopted independently? Is missing activity actually
zero usage, or a coverage/eligibility gap?* It establishes a baseline of developer
working conditions alongside GitHub **and** Microsoft 365 Copilot use, and keeps
eligibility and coverage visible rather than treating missing activity as zero.

**Scope:** this report is the one place on this page that covers **both** products —
**GitHub Copilot** (query activity, model mix, language mix) and **Microsoft 365
Copilot** (feature actions, for the joint-use analysis only). This is distinct from
the Consumption report above, which covers Microsoft 365 Copilot credits/tokens
exclusively and does not touch GitHub Copilot.

**Prerequisites for real data:** you need a **Person Query** export (working-pattern
metrics and organisational attributes) plus GitHub query activity, model mix and
language mix per the [GitHub data contract](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/_data/github/README.md),
and Microsoft 365 Copilot feature-action data for the joint-use panels. There is no
verified real-data adapter for the GitHub inputs yet (v1); treat the real-data path as
a readiness assessment rather than a drop-in export, and confirm field definitions
before adapting the demo's extended illustrative schema.

Explore meeting and uninterrupted hours, team comparisons, joint product-use patterns, model and language mix, and trends.

**[View demo]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-copilot-developer-productivity-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

**[Build or customise with AI]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/)**:
reproduce the synthetic reference or assess real inputs. No verified real GitHub adapter is supplied yet.

[![Developer experience report focus and coordination]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-focus.png)]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)

The developer demo uses an **extended illustrative schema**, not a drop-in flexible-query export. Its M365 feature actions, eligibility flags, and completeness reference are demonstration contracts; confirm equivalent sources and definitions before adapting it.

**Interpretation:** Associations only. Acceptance rate is not code quality, after-hours activity is not burnout, and calendar space does not establish coding time.

<details markdown="1">
<summary>See the joint AI-use preview</summary>

![Developer experience report joint AI use]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-ai-use.png)

</details>

## Build with your own data

The demos need only a browser. To render or adapt the templates, use **R**, **Pandoc**, **R Markdown**, and **flexdashboard**. There is no Python version of these templates.

1. **Get the files.** Start with the report source linked above. For Developer Experience, also get [github-developer-experience-helpers.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-developer-experience-helpers.R) and [render-github-developer-experience.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/render-github-developer-experience.R), keeping the repository folder structure.
2. **Render the synthetic example first.** Dependencies include `vivainsights`, `dplyr`, `tidyr`, `ggplot2`, `ggrepel`, `scales`, `stringr`, `flexdashboard`, `knitr`, and `rmarkdown`. The AI-guided journeys copy the required fixtures and source into an isolated output folder before rendering.
3. **Adapt only supported inputs.** Use the [Consumption journey]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/) for the scoped credits adapter. Use the [Developer Experience journey]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/) for a real-input readiness assessment. Verify contracts before replacing any simulation; do not assume a CSV with similar headers is equivalent.
4. **Review before sharing.** Update simulation labels and methodology to describe your actual inputs. Apply your organization's privacy requirements and retain the distinction between association and causation. Do not put customer exports in this sample repository or in shareable HTML.

[Read the full template setup guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md)

## Continue exploring

[Copilot Analytics]({{ site.baseurl }}/copilot/) for scripts and segmentation ·
[Getting Started]({{ site.baseurl }}/getting-started/) for environment setup ·
[Causal Inference]({{ site.baseurl }}/causal-inference/) for impact-analysis methods
