---
layout: page
title: "Viva Insights Prompt Library"
eyebrow: "Frontier"
permalink: /frontier-analytics-prompts/
css: /assets/css/prompt-catalogue.css
---
<span id="prompt-card-library"></span>

Use these prompt cards with a coding agent such as [GitHub Copilot](https://github.com/features/copilot) or [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) to analyse Viva Insights data.

Each card opens a copyable quick prompt. The full prompt cards retain the detailed instructions, required inputs, assumptions, adaptation notes, and common failure modes.

## How to use a prompt card

1. **Prepare your data.** Export the required query from the Viva Insights Analyst portal.
2. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or your preferred AI assistant.
3. **Choose a starting point.** Copy the quick prompt from a catalogue dialog, or open the full prompt card for detailed instructions.
4. **Paste and run.** Paste the prompt into the agent. Point it at your data file.
5. **Review and adapt.** Check the output against the **Common failure modes** section. Use the **Adaptation notes** to customise.

> **Tip:** You can prepend context to any prompt. For example: _"My CSV is at `./data/person-query.csv`. The Organization column is called `Org`."_ followed by the full prompt text.

> **Review the schema first.** Before running any prompt, check the [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) to confirm your export's columns, granularity and Copilot/agent action definitions. These vary between tenants. Verify them before asking an agent to calculate metrics.

{% include responsible-use.html %}

## Available prompts

Start with the existing R reports and let an agent guide the next step. These journeys work with a pasted prompt or the optional dashboard Skill.

The [dashboard catalogue]({{ site.baseurl }}/copilot-dashboards/) has three
synthetic demos, including **Business Leaders AI Usage**, with source and input
requirements. Viewing them needs only a browser. The two query-dashboard prompts
below cover Consumption and Developer Experience. Business Leaders AI Usage does
not yet have a dedicated prompt journey or verified real-data adapter.

{% include prompt-catalogue.html %}

---

## Tips for adapting prompts

- **Column names vary between tenants.** Always verify your actual column names against what the prompt expects. Prepend a note like _"In my data, the Organization column is called `Org`."_
- **Granularity matters.** Most prompts assume person-week data. If your export is person-day, instruct the agent to aggregate first.
- **Privacy thresholds.** Use the stricter of your organisation's requirement and the workflow's stated minimum group size. Keep complementary cells and embedded data protected.
- **Language preference.** Check the selected card's runtime requirements. Some workflows use R or Python, while the query-dashboard journeys require R. If both are supported, specify your preference.
- **Reproducible source.** Keep the R or Python scripts, RMarkdown (.Rmd) files or Jupyter notebooks (.ipynb) used to produce an HTML report. The selected card specifies its output files.
- **Package availability.** Prompts reference the [vivainsights R package](https://microsoft.github.io/vivainsights/) and [vivainsights Python package](https://microsoft.github.io/vivainsights-py/). Install them beforehand.

## Related resources

- [vivainsights-context.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) provides reusable context for coding agents. Add it to your project or paste it before a prompt to avoid repeating setup instructions.
- [Frontier Overview]({{ site.baseurl }}/frontier-analytics/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)

<script src="{{ '/assets/js/prompt-dialog.js' | relative_url }}"></script>
