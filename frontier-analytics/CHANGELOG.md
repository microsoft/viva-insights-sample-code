# Changelog

All notable changes to Frontier Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Skills library** at [skills/](skills/), with the first skill, [viva-insights-analysis](skills/viva-insights-analysis/), covering the open-source vivainsights R and Python packages.
- [skills/README.md](skills/README.md) explaining what a Skill is, how it differs from a prompt card and from `vivainsights-context.md`, and which agents currently support the format.
- **Skill Template** at [templates/skill-template.md](templates/skill-template.md), and an "Adding a new skill" section in [CONTRIBUTING.md](CONTRIBUTING.md).
- A "Which asset should I use" comparison table on the Frontier Analytics overview, covering prompt cards, skills, `vivainsights-context.md`, and MCP.
- Two prompt cards that previously existed only on the website: [Executive PowerPoint Deck](prompts/copilot-adoption/executive-powerpoint-deck.md) and [Copilot Causal Toolkit](prompts/copilot-adoption/copilot-causal-toolkit.md). Both are now listed in [prompts/README.md](prompts/README.md).

### Changed

- The "What's inside" table now flags [mcp/](mcp/) as a concept with no server implemented yet, so this is visible before opening the folder.
- [vivainsights-context.md](../vivainsights-context.md) now points to the `viva-insights-analysis` skill for agents that support the Skill format, positioning itself as the fallback for agents that do not.

## [0.1.0] - 2025-04-15

### Added

- **Prompt card library** with 6 prompt cards:
  - Copilot adoption: [Dashboard Overview](prompts/copilot-adoption/dashboard-overview.md), [Executive Summary](prompts/copilot-adoption/executive-summary.md), [ROI Estimation](prompts/copilot-adoption/roi-estimation.md), [Segmentation and Churn](prompts/copilot-adoption/segmentation-and-churn.md)
  - Purview augmentation: [Audit Log Parsing](prompts/purview-augmentation/audit-log-parsing.md), [Agent Usage Analysis](prompts/purview-augmentation/agent-usage-analysis.md)
- **2 starter kits:**
  - [Copilot Adoption Dashboard](starter-kits/copilot-adoption-dashboard/) — intermediate complexity, HTML dashboard output
  - [Executive Summary Report](starter-kits/executive-summary-report/) — beginner complexity, Markdown/HTML memo output
- **Schema documentation** covering person query structure, Purview audit log fields, join patterns, and common data pitfalls
- **Example output specifications** (3 specs) describing expected deliverable formats
- **Templates** for creating new prompt cards, starter kits, and schema documentation
- **MCP concepts documentation** with sample configuration for Model Context Protocol integration
- [README](README.md), [Quickstart guide](QUICKSTART.md), [Starter Kits overview](STARTER_KITS.md), and [Contributing guide](CONTRIBUTING.md)
