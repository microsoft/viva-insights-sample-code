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

Both seven-page reports use **synthetic data**, not real organizational results. They illustrate analysis approaches, not evidence of Copilot's impact. Assumptions and simulation details are included in each report's appendix.

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

Separate consumption volume from consumption mix: token concentration and percentile bands, credit intensity per 1,000 tokens, delegated task types, usage segments, and function drill-downs.

**[View demo]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

[![Consumption report overview]({{ site.baseurl }}/assets/images/reports/copilot-consumption-overview.png)]({{ site.baseurl }}/examples/utility-r/copilot-consumption-ways-of-working-simulation.html)

**Required queries:** Consumption query for credits, tokens, and delegated task types, joined with a Person Query for collaboration and network metrics. See the [AI cost query documentation](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query) for schema and access details.

**Interpretation:** Associations only. Credit intensity is a cost-mix measure, not a measure of value or quality. Concentration curves and banded percentiles help describe skewed consumption without relying on an average.

<details markdown="1">
<summary>See the token distribution preview</summary>

![Consumption report token distribution]({{ site.baseurl }}/assets/images/reports/copilot-consumption-token-distribution.png)

</details>

## Developer Experience and Copilot

**Synthetic data · R template · Seven-page HTML report**

Establish a baseline of developer working conditions alongside GitHub and Microsoft 365 Copilot use. Explore meeting and uninterrupted hours, team comparisons, joint product-use patterns, model and language mix, and trends. Eligibility and coverage remain visible rather than treating missing activity as zero.

**[View demo]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)** ·
[Get source](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-copilot-developer-productivity-simulation.Rmd) ·
[Setup guide](#build-with-your-own-data)

[![Developer experience report focus and coordination]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-focus.png)]({{ site.baseurl }}/examples/utility-r/github-copilot-developer-productivity-simulation.html)

**Required data:** GitHub query activity, model mix, and language mix; Person Query working-pattern metrics; and Microsoft 365 Copilot activity for the joint-use analysis. The [data contract](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/_data/github/README.md) documents coverage and reference inputs.

The developer demo uses an **extended illustrative schema**, not a drop-in flexible-query export. Its M365 feature actions, eligibility flags, and completeness reference are demonstration contracts; confirm equivalent sources and definitions before adapting it.

**Interpretation:** Associations only. Acceptance rate is not code quality, after-hours activity is not burnout, and calendar space does not establish coding time.

<details markdown="1">
<summary>See the joint AI-use preview</summary>

![Developer experience report joint AI use]({{ site.baseurl }}/assets/images/reports/github-copilot-devex-ai-use.png)

</details>

## Build with your own data

The demos need only a browser. To render or adapt the templates, use **R**, **Pandoc**, **R Markdown**, and **flexdashboard**. There is no Python version of these templates.

1. **Get the files.** Start with the report source linked above. For Developer Experience, also get [github-developer-experience-helpers.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/github-developer-experience-helpers.R) and [render-github-developer-experience.R](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/render-github-developer-experience.R), keeping the repository folder structure.
2. **Render the synthetic example first.** Dependencies include `vivainsights`, `dplyr`, `tidyr`, `ggplot2`, `scales`, `stringr`, `flexdashboard`, `knitr`, and `rmarkdown`. Follow the [template setup guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md) for rendering instructions and simulated input contracts.
3. **Adapt the inputs.** Use the guide to replace the simulation block with your query exports. Check query schemas, join keys, date ranges, eligibility, and coverage before interpreting results.
4. **Review before sharing.** Update simulation labels and methodology to describe your actual inputs. Apply your organization's privacy requirements and retain the distinction between association and causation. Do not put customer exports in this sample repository or in shareable HTML.

[Read the full template setup guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md)

## Continue exploring

[Copilot Analytics]({{ site.baseurl }}/copilot/) for scripts and segmentation ·
[Getting Started]({{ site.baseurl }}/getting-started/) for environment setup ·
[Causal Inference]({{ site.baseurl }}/causal-inference/) for impact-analysis methods
