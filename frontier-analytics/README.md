# Frontier Analytics

Frontier Analytics is an **export-first, self-service analytics toolkit** for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/). It provides reusable prompts, starter kits, schema documentation, and example specifications that you can combine with a coding agent to produce analytics outputs from exported Viva Insights data.

> **Note:** Everything in this folder is sample code and starter assets. It is not production software. Outputs require review, validation, and adaptation to your environment before use.

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

## Which asset should I use?

Frontier Analytics offers four ways to bring this work into a coding agent. They solve overlapping problems, so use this table to pick the right one before browsing the folders below.

| Mechanism | What it is | Status | Use it when |
|-----------|------------|--------|-------------|
| [Prompt cards](prompts/) | Structured text you copy and paste into any coding agent for a single analysis task. | Available | Your agent does not support Skills or MCP, or you want a one-off analysis without any setup. |
| [Skills](skills/) | A packaged capability that a compatible coding agent loads automatically, so it applies the right conventions without you pasting anything. | Available | Your coding agent supports the Skill format (for example GitHub Copilot CLI or Claude Code) and you want ongoing, repeated Viva Insights work to follow consistent conventions. |
| [vivainsights-context.md](../vivainsights-context.md) | A single context file you paste once at the start of a session. | Available | Your agent does not support Skills, but you still want to avoid repeating setup instructions in every prompt. |
| [mcp/](mcp/) | A protocol-level integration that would let an agent query prompts, schemas, and tools directly from a server. | Concept only, no server implemented | Not yet. Read the folder for the design intent and to track progress. |

If you are not sure where to start, use a [starter kit](starter-kits/) for an end-to-end walkthrough, or a [prompt card](prompts/) for a single task.

## Don't have a coding agent yet?

Frontier Analytics assumes you have a coding agent available. If you don't yet, here is the fastest path to get one:

- **GitHub Copilot** is free to start with a personal GitHub account. Install the Copilot extension in [VS Code](https://code.visualstudio.com/) (or open the [GitHub Copilot CLI](https://github.com/features/copilot) in a terminal) and sign in with your GitHub account.
- **Claude Code** is a terminal-based coding agent from Anthropic. See its [getting started guide](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) for installation.
- If your organization provides an enterprise-hosted coding agent, use that instead, especially when working with real HR data, since it keeps your data inside your organization's boundary.

Once installed, open the agent in a folder that has R or Python available, then continue with the workflow below.

## What's inside

| Folder | Description |
|--------|-------------|
| [prompts/](prompts/) | Prompt cards. Structured, ready-to-paste prompts for coding agents. Covers Copilot adoption tracking, user segmentation, ROI estimation, and Purview audit log analysis. |
| [starter-kits/](starter-kits/) | Bundled workflows that combine a use case, required inputs, prompts, and expected outputs into a single package. Start here if you want an end-to-end walkthrough. |
| [skills/](skills/) | Agent Skills. Packaged capabilities that a compatible coding agent loads automatically for ongoing Viva Insights work. |
| [schemas/](schemas/) | Data dictionaries and documentation for person query exports, Purview audit logs, join patterns, and common data pitfalls. |
| [examples/](examples/) | Sample output specifications that describe what a finished deliverable should look like. |
| [templates/](templates/) | Templates for contributing new prompt cards, skills, starter kits, and schema documentation. |
| [mcp/](mcp/) | Concept and sample configuration for Model Context Protocol (MCP) integration with Viva Insights workflows. Not yet implemented: there is no MCP server today, see the folder's README for the design intent. |

## How to use this with a coding agent

### Prerequisites

- **Exported Viva Insights data.** Typically a person query CSV from the Viva Insights Analyst portal. Person query data has a panel structure with rows keyed by `PersonId` and `MetricDate` (person-week or person-day granularity), with HR attributes such as organization, function, geography, and level as columns.
- **An R or Python environment**, with the [vivainsights R package](https://microsoft.github.io/vivainsights/) or the [vivainsights Python package](https://microsoft.github.io/vivainsights-py/) installed. See "Recommended packages" below.
- **A coding agent.** See "Don't have a coding agent yet?" above if you need one.

### Workflow

1. **Export your data.** Run a person query (or other query type) from the Viva Insights Analyst portal and download the CSV.
2. **Pick a starter kit or prompt card.** Browse [starter-kits/](starter-kits/) for end-to-end workflows or [prompts/](prompts/) for individual analysis tasks.
3. **Review the schema docs.** Check [schemas/](schemas/) to understand the structure of your exported data, including column definitions, expected granularity, and common pitfalls.
4. **Open your coding agent.** Launch [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar tool in a workspace with R or Python available.
5. **Paste the prompt.** Copy the prompt text from the card, point it at your data file, and let the agent generate the output. If a card offers a quick prompt and a detailed prompt, start with whichever matches how exact you need the output to be.
6. **Review and iterate.** Check the output against the documented failure modes and adaptation notes. Refine as needed.

### What to expect as output

Depending on the starter kit or prompt card you use, you may get:

- **HTML dashboards**: self-contained interactive dashboards you can open in a browser or share
- **Markdown or HTML reports**: formatted summary documents suitable for leadership review
- **Jupyter notebooks or R Markdown**: reproducible analysis documents with code and commentary
- **PowerPoint decks**: exec-ready slides with native, editable charts
- **Data tables and charts**: intermediate outputs for further analysis

### Tips for working with coding agents

1. **Be specific about your data.** Tell the agent the file name, column names, and date range. The more context you give, the better the output.
2. **Iterate in small steps.** If the output is not right, ask the agent to fix one thing at a time rather than regenerating everything.
3. **Validate the output.** Spot-check row counts, date ranges, and aggregation logic. Coding agents can make plausible-looking mistakes.
4. **Use the vivainsights packages.** They handle common data validation and visualization tasks. Prompts that reference these functions tend to produce cleaner code.
5. **Keep your data private.** Do not paste raw data into cloud-based agents unless your organization's data policies allow it. Use local or enterprise-hosted agents when working with sensitive HR data.

### Recommended packages

These prompts are designed to work with the open-source Viva Insights packages:

- **R:** [vivainsights](https://microsoft.github.io/vivainsights/), installed with `install.packages("vivainsights")`
- **Python:** [vivainsights](https://microsoft.github.io/vivainsights-py/), installed with `pip install vivainsights`

The packages provide helper functions for reading, validating, and visualizing Viva Insights data. Prompts in this toolkit may reference package functions where appropriate.

## Quick links

- [Starter kits overview](starter-kits/README.md)
- [Skills library](skills/)
- [Contributing to Frontier Analytics](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Prompt card library](prompts/)
- [Main repository README](../README.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add prompt cards, starter kits, and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.

## License

This project is licensed under the [MIT License](../LICENSE).
