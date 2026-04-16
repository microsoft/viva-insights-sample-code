---
layout: page
title: "Frontier Analytics — Prompt Library"
permalink: /frontier-analytics-prompts/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}

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

# Prompt Card Library

This page contains **prompt cards** — ready-to-use prompts that you can paste directly into a coding agent (such as [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or similar tools) to generate analytics outputs from Viva Insights data.

Each prompt card includes the purpose, required inputs, assumptions, the full prompt text, adaptation notes, and common failure modes.

## How to use a prompt card

1. **Prepare your data.** Export the required query from the Viva Insights Analyst portal.
2. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or your preferred AI assistant.
3. **Copy the prompt.** Find the relevant card below and copy the full text from the **Prompt** section.
4. **Paste and run.** Paste the prompt into the agent. Point it at your data file.
5. **Review and adapt.** Check the output against the **Common failure modes** section. Use the **Adaptation notes** to customize.

> **Tip:** You can prepend context to any prompt. For example: _"My CSV is at `./data/person-query.csv`. The Organization column is called `Org`."_ followed by the full prompt text.

## Available prompts

### Copilot Adoption

| Prompt Card | Description |
|---|---|
| [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompt-dashboard/) | Generate a comprehensive static HTML dashboard showing Copilot adoption trends, usage patterns, and organizational breakdowns. |
| [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompt-executive-summary/) | Produce a concise executive memo summarizing Copilot adoption metrics for VP/C-suite audiences. |
| [Segmentation & Churn]({{ site.baseurl }}/frontier-analytics-prompt-segmentation/) | Classify users into usage segments, track transitions, and calculate churn rates. |
| [ROI Estimation]({{ site.baseurl }}/frontier-analytics-prompt-roi/) | Estimate return on investment for Copilot by quantifying time savings and license costs. |
| [Executive PowerPoint Deck]({{ site.baseurl }}/frontier-analytics-prompt-powerpoint/) | Generate an exec-ready 10–15 page PowerPoint deck with editable native charts. |

### Purview Augmentation

| Prompt Card | Description |
|---|---|
| [Agent Usage Analysis]({{ site.baseurl }}/frontier-analytics-prompt-agent-usage/) | Analyze Copilot agent and extension usage patterns from Purview audit logs. |
| [Audit Log Parsing]({{ site.baseurl }}/frontier-analytics-prompt-audit-parsing/) | Parse and clean raw Purview audit log exports into analysis-ready flat tables. |

### Causal Inference

| Prompt Card | Description |
|---|---|
| [Copilot Causal Toolkit]({{ site.baseurl }}/frontier-analytics-prompt-causal-toolkit/) | Run a causal inference analysis using the Copilot Causal Toolkit, then interpret results for senior leadership. Two prompts: one to run the analysis, one to interpret outputs. |

---

## Tips for adapting prompts

- **Column names vary between tenants.** Always verify your actual column names against what the prompt expects. Prepend a note like _"In my data, the Organization column is called `Org`."_
- **Granularity matters.** Most prompts assume person-week data. If your export is person-day, instruct the agent to aggregate first.
- **Privacy thresholds.** For smaller organizations, add a note requesting minimum group sizes (e.g., suppress segments with fewer than 5 people).
- **Language preference.** Prompts support both R and Python. Choose whichever is already installed in your environment to minimize setup. If you have a strong preference, add _"Use R"_ or _"Use Python"_ before the prompt.
- **Intermediary files.** Prompts that produce HTML output instruct the agent to create an RMarkdown (.Rmd) or Jupyter notebook (.ipynb) first, then export to HTML. Keep these intermediary files — they make troubleshooting and iteration much easier.
- **Package availability.** Prompts reference the [vivainsights R package](https://microsoft.github.io/vivainsights/) and [vivainsights Python package](https://microsoft.github.io/vivainsights-py/). Install them beforehand.

## Related resources

- [Frontier Analytics Overview]({{ site.baseurl }}/frontier-analytics/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
