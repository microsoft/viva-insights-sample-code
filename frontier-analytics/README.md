# Frontier Analytics

Frontier Analytics is an **export-first, self-service analytics toolkit** for [Viva Insights](https://learn.microsoft.com/en-us/viva/insights/). It provides reusable prompts, starter kits, schema documentation, and example specifications that you can combine with a coding agent to produce analytics outputs from exported Viva Insights data.

> **Note:** Everything in this folder is sample code and starter assets — not production software. Outputs require review, validation, and adaptation to your environment before use.

## Who is this for?

- **People analytics leads** building dashboards and reports from Viva Insights exports
- **HR analysts** who need repeatable, transparent analysis workflows
- **Analytics consultants** delivering Copilot adoption or workplace analytics engagements
- **Technically capable users** comfortable with R or Python and willing to work with a coding agent

You do not need to be a software engineer. If you can export a CSV from Viva Insights and paste a prompt into a coding agent, you can use these assets.

## What's inside

| Folder | Description |
|--------|-------------|
| [prompts/](prompts/) | Prompt cards — structured, ready-to-paste prompts for coding agents. Covers Copilot adoption tracking, user segmentation, ROI estimation, and Purview audit log analysis. |
| [starter-kits/](starter-kits/) | Bundled workflows that combine a use case, required inputs, prompts, and expected outputs into a single package. Start here if you want an end-to-end walkthrough. |
| [schemas/](schemas/) | Data dictionaries and documentation for person query exports, Purview audit logs, join patterns, and common data pitfalls. |
| [examples/](examples/) | Sample output specifications that describe what a finished deliverable should look like. |
| [templates/](templates/) | Templates for contributing new prompt cards, starter kits, and schema documentation. |
| [mcp/](mcp/) | Concepts and sample configuration for Model Context Protocol (MCP) integration with Viva Insights workflows. |

## How to use this with a coding agent

The intended workflow:

1. **Export your data.** Run a person query (or other query type) from the Viva Insights Analyst portal and download the CSV.
2. **Pick a starter kit or prompt card.** Browse [starter-kits/](starter-kits/) for end-to-end workflows or [prompts/](prompts/) for individual analysis tasks.
3. **Review the schema docs.** Check [schemas/](schemas/) to understand the structure of your exported data — column definitions, expected granularity, and common pitfalls.
4. **Open your coding agent.** Launch [GitHub Copilot](https://github.com/features/copilot), [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview), or a similar tool in a workspace with R or Python available.
5. **Paste the prompt.** Copy the prompt text from the card, point it at your data file, and let the agent generate the output.
6. **Review and iterate.** Check the output against the documented failure modes and adaptation notes. Refine as needed.

### Recommended packages

These prompts are designed to work with the open-source Viva Insights packages:

- **R:** [vivainsights](https://microsoft.github.io/vivainsights/) — `install.packages("vivainsights")`
- **Python:** [vivainsights](https://microsoft.github.io/vivainsights-py/) — `pip install vivainsights`

The packages provide helper functions for reading, validating, and visualizing Viva Insights data. Prompts in this toolkit may reference package functions where appropriate.

## Quick links

- [Quickstart guide](QUICKSTART.md)
- [Starter kits overview](STARTER_KITS.md)
- [Contributing to Frontier Analytics](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Prompt card library](prompts/)
- [Main repository README](../README.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add prompt cards, starter kits, and schema documentation.

This project uses the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and requires a [Contributor License Agreement](https://cla.opensource.microsoft.com) for all contributions.

## License

This project is licensed under the [MIT License](../LICENSE).
