---
layout: page
title: "Frontier — Prompt Library"
eyebrow: "Frontier"
permalink: /frontier-analytics-prompts/
css: /assets/css/prompt-catalogue.css
---
# Prompt Card Library

This page contains **prompt cards** — ready-to-use prompts that you can paste directly into a coding agent (such as [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or similar tools) to generate analytics outputs from Viva Insights data.

Each card opens a copyable quick prompt. The full prompt cards retain the detailed instructions, required inputs, assumptions, adaptation notes, and common failure modes.

## How to use a prompt card

1. **Prepare your data.** Export the required query from the Viva Insights Analyst portal.
2. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or your preferred AI assistant.
3. **Choose a starting point.** Copy the quick prompt from a catalogue dialog, or open the full prompt card for detailed instructions.
4. **Paste and run.** Paste the prompt into the agent. Point it at your data file.
5. **Review and adapt.** Check the output against the **Common failure modes** section. Use the **Adaptation notes** to customize.

> **Tip:** You can prepend context to any prompt. For example: _"My CSV is at `./data/person-query.csv`. The Organization column is called `Org`."_ followed by the full prompt text.

> **📋 Review the schema first.** Before running any prompt, check the **[Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)** to confirm which columns exist in your export and what they mean. Column names, granularity, and the Copilot/agent action taxonomy vary between tenants — validating the schema first prevents the most common failure mode, where a prompt silently runs against the wrong or missing columns.

{% include responsible-use.html %}

## Available prompts

Start with the existing R reports and let an agent guide the next step. These journeys work with a pasted prompt or the optional dashboard Skill.

{% include prompt-catalogue.html %}

---

## Tips for adapting prompts

- **Column names vary between tenants.** Always verify your actual column names against what the prompt expects. Prepend a note like _"In my data, the Organization column is called `Org`."_
- **Granularity matters.** Most prompts assume person-week data. If your export is person-day, instruct the agent to aggregate first.
- **Privacy thresholds.** For smaller organizations, add a note requesting minimum group sizes (e.g., suppress segments with fewer than 5 people).
- **Language preference.** Prompts support both R and Python. Choose whichever is already installed in your environment to minimize setup. If you have a strong preference, add _"Use R"_ or _"Use Python"_ before the prompt.
- **Intermediary files.** Prompts that produce HTML output instruct the agent to create an RMarkdown (.Rmd) or Jupyter notebook (.ipynb) first, then export to HTML. Keep these intermediary files — they make troubleshooting and iteration much easier.
- **Package availability.** Prompts reference the [vivainsights R package](https://microsoft.github.io/vivainsights/) and [vivainsights Python package](https://microsoft.github.io/vivainsights-py/). Install them beforehand.

## Related resources

- [vivainsights-context.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) — reusable context file for AI coding agents (GitHub Copilot, Claude Code, etc.). Drop it into your project or paste it as context before a prompt so the agent understands your data without repeating setup instructions.
- [Frontier Overview]({{ site.baseurl }}/frontier-analytics/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)

<script src="{{ '/assets/js/prompt-dialog.js' | relative_url }}"></script>
