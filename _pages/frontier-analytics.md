---
layout: page
title: "Frontier"
eyebrow: "Frontier"
description: "Explore synthetic Viva Insights dashboards, or use prompts, Skills and schema guides to build and review your own analysis."
permalink: /frontier-analytics/
---
<span id="frontier"></span>

Frontier brings together dashboard examples, prompts, agent Skills and schema
guides for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/).
Explore a synthetic report in your browser, or use a coding agent such as
[GitHub Copilot](https://github.com/features/copilot) or
[Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
to build an analysis from your exports. Review the inputs, calculations and
limitations before sharing the output.

> **Note:** Everything in this section is sample code and starter assets. It is not production software. Outputs require review, validation, and adaptation to your environment before use.

---

## Quick start with a coding agent

Follow these steps to validate a Person Query, then choose a report to build.
Use a coding agent and data-handling environment approved by your organisation.
If you want to explore an example first, [open the synthetic dashboard demos](#explore-the-dashboard-demos)
without installing anything.

### 1. Prepare a private working folder and your export

Create a folder outside this public sample repository. For example, run the
following in **PowerShell** from an approved location for your analysis:

```powershell
New-Item -ItemType Directory -Path viva-insights-workspace
Set-Location viva-insights-workspace
New-Item -ItemType Directory -Path data, output
```

In the Viva Insights Analyst portal, [create and download a Person Query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query).
Save its CSV as `data\person-query.csv`, or use your own filename in the prompt
below. Keep the query's population, date grouping, week start and metric settings
alongside the export. Start with the [Person Query schema]({{ site.baseurl }}/frontier-analytics-schemas/#person-query-key-concepts)
to check the required columns. For several exports, also use the
[query join guide]({{ site.baseurl }}/query-join-guide/).

No export yet? The [sample-data examples]({{ site.baseurl }}/getting-started/#load-and-explore-your-data)
show how to load `pq_data` in R or `load_pq_data()` in Python. Tell the agent to
use that sample instead of a CSV and label every output as sample data.

### 2. Set up R or Python

Choose one language for general analysis. The reference dashboard templates
require R and their additional rendering dependencies.

<details markdown="1">
<summary>Python: install the package in an isolated environment</summary>

Install [Python](https://www.python.org/downloads/) if needed, then run these
commands in PowerShell from your working folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install vivainsights
.\.venv\Scripts\python.exe -c "from importlib.metadata import version; import vivainsights; print(version('vivainsights'))"
```

The last command should print the installed version without an import error.
Tell the agent to use `.\.venv\Scripts\python.exe` so it uses this environment.
For other operating systems, see the [environment setup guide]({{ site.baseurl }}/getting-started/#install-the-vivainsights-package).
The [Python function guide](https://microsoft.github.io/vivainsights-py/function-discovery.html)
lists supported analyses.

</details>

<details markdown="1">
<summary>R: install the package and check it loads</summary>

Install [R](https://cran.r-project.org/) and, optionally,
[RStudio](https://posit.co/download/rstudio-desktop/). Run this in the R console:

```r
install.packages("vivainsights")
library(vivainsights)
packageVersion("vivainsights")
```

The last command should print the installed version without an error.
The [R function guide](https://microsoft.github.io/vivainsights/articles/function-discovery.html)
lists supported analyses. See [Getting Started]({{ site.baseurl }}/getting-started/)
for import and plotting examples.

</details>

### 3. Install the analysis Skill and open your agent

For **GitHub Copilot CLI**, install [GitHub CLI](https://cli.github.com/) and
[Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli)
if needed. Run this in your working folder:

```powershell
gh skill install microsoft/viva-insights-sample-code frontier-analytics/skills/viva-insights-analysis --agent github-copilot --scope user
copilot skill list
copilot
```

Check that `viva-insights-analysis` appears in the Skill list, then sign in if
prompted when opening Copilot. The install command copies the whole Skill,
including its references and examples, into your personal Skills directory.
Review the files before use. The Skill does not install R, Python or packages.

For **Claude Code**, use the same install command with `--agent claude-code`,
then open Claude Code in your working folder. Its
[setup guide](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
covers installation and launch. If `gh skill` is unavailable, update GitHub CLI
or follow the [whole-folder installation instructions](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#installing-a-skill).
Review existing customised Skills before replacing them.

If your agent does not support Skills, use the
[context file](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md)
and a [prompt card]({{ site.baseurl }}/frontier-analytics-prompts/) instead.

### 4. Run a validation pass before building

Paste this into the agent. Change the filename if needed. For R, replace the
Python sentence with "Use R and the vivainsights package."

```text
Use the installed viva-insights-analysis Skill and its shared analysis policy.
Confirm that the Skill is available before starting.
Use Python and the vivainsights package through .\.venv\Scripts\python.exe.
My Person Query CSV is data\person-query.csv, relative to this working folder.
Save generated files under output and leave the source export unchanged.

Inspect the actual headers, installed package version, row count, distinct
people, date range, duplicate person-period keys and missingness. Ask me for
any missing query settings, including grouping, week start and metric windows.
Do not infer cadence from gaps between dates.

Use existing package functions after checking their installed signatures.
Keep individual records out of the report. Apply the stricter organisational
minimum or 10 distinct people to every published group and its complement.
Summarise coverage and data-quality issues, separating executed checks from
checks you could not run. Save the validation summary and reproducible code.

Propose a supported analysis of Copilot usage by available organisational
attributes. If suitable attributes are absent, propose overall results.
Stop for my review before building the report or making productivity claims.
```

Check the saved summary against your query settings. Resolve duplicate keys,
coverage gaps and uncertain metric definitions before approving the analysis.
The [data-pitfalls reference](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/viva-insights-analysis/reference/data-pitfalls.md)
explains common import problems. The [shared analysis policy]({{ site.baseurl }}/frontier-analytics-policy/)
sets out privacy, licence and aggregation rules.

### 5. Choose the next prompt

Open a card below, copy its quick or detailed prompt and add your file location
and preferred language. Use the same agent session so it retains the validation
results. Check the card's inputs and limitations before asking it to build.

| What you want to produce | Start here |
|---|---|
| A Copilot adoption dashboard from a Person Query | [Dashboard Overview]({{ site.baseurl }}/frontier-analytics-prompt-dashboard/) |
| A short leadership briefing | [Executive Summary]({{ site.baseurl }}/frontier-analytics-prompt-executive-summary/) |
| AI adoption across Person, Consumption and Agent queries | [AI Adoption Across Tools]({{ site.baseurl }}/frontier-analytics-prompt-ai-adoption/) |
| Descriptive workload and connected-time signals | [Employee Wellbeing and Workload Signals]({{ site.baseurl }}/frontier-analytics-prompt-employee-wellbeing/) |
| External participation and collaboration hours | [External Collaboration Patterns]({{ site.baseurl }}/frontier-analytics-prompt-external-collaboration/) |

The AI adoption, wellbeing and external collaboration cards are guided prompts
without completed end-to-end data trials. Their outputs require validation.
For the R reference templates, use the [Consumption journey]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/)
or [Developer Experience journey]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/).
Each links its rendering requirements and supported real-data scope. Repeated
dashboard work can use the [paired dashboard Skill installation]({{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-copilot-dashboards).
Business Leaders has a separate [rendering guide]({{ site.baseurl }}/copilot-dashboards/business-leaders-ai-usage/)
and no verified real-export adapter.

Before sharing, rerun the saved code, reconcile headline counts and denominators,
check privacy suppression in charts and embedded data, and open any self-contained
HTML report offline. Keep the source and validation log with the output.

---

## Which asset should I use?

Frontier offers three ways to bring this work into a coding agent. Pick a card to jump straight to what you need, or use the comparison below.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="{{ site.baseurl }}/frontier-analytics-prompts/">
    <span class="vi-card-icon">📋</span>
    <span class="vi-card-title">Prompt Library</span>
    <span class="vi-card-desc">Copy a ready-made prompt into your agent for a one-off analysis. No setup required.</span>
    <span class="vi-card-more">Browse prompts →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/frontier-analytics-skills/">
    <span class="vi-card-icon">🧠</span>
    <span class="vi-card-title">Skills</span>
    <span class="vi-card-desc">Your agent loads the right conventions automatically for ongoing work. Needs a Skill-compatible agent, for example GitHub Copilot CLI or Claude Code.</span>
    <span class="vi-card-more">Explore Skills →</span>
  </a>
  <a class="vi-card" href="https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md">
    <span class="vi-card-icon">📄</span>
    <span class="vi-card-title">Context file</span>
    <span class="vi-card-desc">Paste once at the start of a session. A lighter fallback when your agent doesn't support Skills.</span>
    <span class="vi-card-more">View on GitHub →</span>
  </a>
</div>

### Full comparison

| Mechanism | What it is | Status | Use it when |
|-----------|------------|--------|-------------|
| [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) | Structured text you copy and paste into any coding agent for a single analysis task. | Available | You want a one-off analysis without installing a Skill. |
| [Agent Skills]({{ site.baseurl }}/frontier-analytics-skills/) | A packaged capability that a compatible coding agent loads automatically, so it applies the right conventions without you pasting anything. | Available | Your coding agent supports the Skill format (for example GitHub Copilot CLI or Claude Code) and you want ongoing, repeated Viva Insights work to follow consistent conventions. |
| [vivainsights-context.md](https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md) | A single context file you paste once at the start of a session. | Available | Your agent does not support Skills, but you still want to avoid repeating setup instructions in every prompt. |

---

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

---

## What's inside

### Explore the dashboard demos

Open any of these three reports in your browser without installing anything.
They use synthetic data and include R Markdown source and rendering requirements.
Select **Start using** for the links and capability limits.

**Business Leaders AI Usage** brings together M365 Copilot, Cowork, Agent,
Consumption and GitHub Copilot fixtures. Explore reach, usage and trends over
four, eight or thirteen weeks by organisation, function and team. It is a synthetic
reference dashboard, with no verified real-export adapter or dedicated prompt
journey. Descriptive comparisons do not establish causal effects.

{% include dashboard-catalogue.html %}

### Build or customise the query dashboards

These two guided journeys reproduce the existing Consumption and Developer
Experience R demos. The agent inspects real inputs and asks you to approve
supported analyses before building. Missing metrics and coverage remain explicit.

| Journey | Demo | Real-data support |
|---|---|---|
| [Consumption Dashboard]({{ site.baseurl }}/frontier-analytics-prompt-consumption-dashboard/) | Full synthetic reference | Documented credit consumption; token/task-type and Person Query panels excluded |
| [Developer Experience Dashboard]({{ site.baseurl }}/frontier-analytics-prompt-developer-experience-dashboard/) | Full synthetic reference | Input-readiness assessment only; real adapter not yet verified |

[Browse the dashboard examples]({{ site.baseurl }}/copilot-dashboards/) or
[install the optional dashboard skill](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/README.md#dashboard-skill-installation)
for repeated work. Prompt-only use requires no skill installation.

| Section | Description |
|---------|-------------|
| [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) | Data dictionaries for person query exports, Purview audit logs, join patterns, and common data pitfalls. |

Additional resources available on GitHub:

| Folder | Description |
|--------|-------------|
| [examples/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/examples) | Sample output specifications describing what a finished deliverable looks like. |
| [templates/](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/templates) | Templates for contributing new prompt cards, skills, and schema docs. |

---

## Getting started

### Prerequisites

Viewing the synthetic demos above needs only a browser. To run a prompt or
render a report, check its specific requirements. The general analysis prompts
use R or Python. The dashboard templates require R, Pandoc, R Markdown,
flexdashboard and the packages listed in their rendering requirements.

For analysis of your own exports, prepare:

1. **Exported Viva Insights data.** Use the query types required by the selected workflow. A Person Query has rows keyed by `PersonId` and `MetricDate` (person-week or person-day granularity). HR attributes such as organisation, function, geography and level are included as columns. Cross-query dashboards may need several exports.

2. **An R or Python environment.** You need one of the following set up locally:
   - **R** (4.1+) with [vivainsights](https://microsoft.github.io/vivainsights/) installed: `install.packages("vivainsights")`
   - **Python** (3.9+) with [vivainsights](https://microsoft.github.io/vivainsights-py/) installed: `pip install vivainsights`

3. **A coding agent.** Any AI-assisted coding tool that can run R or Python:
   - [GitHub Copilot](https://github.com/features/copilot) (in VS Code, JetBrains, or CLI)
   - [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)
   - Other coding agents with code execution capabilities

### Don't have a coding agent yet?

- **GitHub Copilot** is free to start with a personal GitHub account. Install the Copilot extension in [VS Code](https://code.visualstudio.com/), or install the [GitHub Copilot CLI](https://github.com/features/copilot) and sign in from a terminal.
- **Claude Code** is a terminal-based coding agent from Anthropic. Its [getting started guide](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) covers installation.
- Use your organisation's approved coding agent and data-handling environment for real HR data.

Once installed, open the agent in a folder that has R or Python available, then continue with the workflow below.

### Workflow

1. **Export your data.** Run a person query (or other query type) from the Viva Insights Analyst portal and download the CSV.
2. **Pick a prompt card.** Browse the [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/) for the analysis task that matches your use case.
3. **Review the schema docs.** Check [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/) to understand the structure of your exported data: column definitions, expected granularity, and common pitfalls.
4. **Open your coding agent.** Launch GitHub Copilot, Claude Code, or a similar tool in a workspace with R or Python available.
5. **Paste the prompt.** Copy the prompt text from the card, point it at your data file, and let the agent generate the output.
6. **Review and iterate.** Check the output against the documented failure modes and adaptation notes. Refine as needed.

### Tips for working with coding agents

1. **Be specific about your data.** Tell the agent the file name, column names, and date range. The more context you give, the better the output.
2. **Iterate in small steps.** If the output is not right, ask the agent to fix one thing at a time rather than re-generating everything.
3. **Validate the output.** Spot-check row counts, date ranges, and aggregation logic. Coding agents can make plausible-looking mistakes.
4. **Use the vivainsights packages.** The R and Python packages handle common data validation and visualization tasks. Prompts that reference these functions tend to produce cleaner code.

See "Responsible use & data privacy" below before you work with real HR data.

---

{% include responsible-use.html %}

---

## Quick links

- [Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)
- [Schema Documentation]({{ site.baseurl }}/frontier-analytics-schemas/)
- [Agent Skills]({{ site.baseurl }}/frontier-analytics-skills/)
- [Browse on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics)
- [Contributing](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md)
- [Changelog](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CHANGELOG.md)

---

## Contributing

See the [Contributing guide](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/CONTRIBUTING.md) for how to add prompt cards and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.
