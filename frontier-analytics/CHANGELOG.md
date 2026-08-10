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
- A "Quick prompt (short version)" on every prompt card: a short, general prompt that asks the agent to inspect the actual data and adapt column names, positioned as a faster first pass before the existing detailed, step-by-step prompt.
- A "Don't have a coding agent yet?" section on the Frontier Analytics overview (site page and repo README), pointing to GitHub Copilot and Claude Code setup for people who have not started using a coding agent at all.
- `reference/deliverables.md` in the `viva-insights-analysis` skill, distilling the same dashboard, executive summary, ROI, and segmentation conventions used in the prompt cards, so a Skill-driven session can produce those deliverables without a prompt card being pasted.

### Changed

- The "What's inside" table now flags [mcp/](mcp/) as a concept with no server implemented yet, so this is visible before opening the folder.
- [vivainsights-context.md](../vivainsights-context.md) now points to the `viva-insights-analysis` skill for agents that support the Skill format, positioning itself as the fallback for agents that do not.
- Consolidated `QUICKSTART.md` and `STARTER_KITS.md` into [README.md](README.md) and [starter-kits/README.md](starter-kits/README.md) respectively, since the three files repeated the same workflow steps with small wording differences. Removed the two standalone files.

## [0.1.0] - 2025-04-15

### Added

- **Prompt card library** with 6 prompt cards:
  - Copilot adoption: [Dashboard Overview](prompts/copilot-adoption/dashboard-overview.md), [Executive Summary](prompts/copilot-adoption/executive-summary.md), [ROI Estimation](prompts/copilot-adoption/roi-estimation.md), [Segmentation and Churn](prompts/copilot-adoption/segmentation-and-churn.md)
  - Purview augmentation: [Audit Log Parsing](prompts/purview-augmentation/audit-log-parsing.md), [Agent Usage Analysis](prompts/purview-augmentation/agent-usage-analysis.md)
- **2 starter kits:**
  - [Copilot Adoption Dashboard](starter-kits/copilot-adoption-dashboard/): intermediate complexity, HTML dashboard output
  - [Executive Summary Report](starter-kits/executive-summary-report/): beginner complexity, Markdown/HTML memo output
- **Schema documentation** covering person query structure, Purview audit log fields, join patterns, and common data pitfalls
- **Example output specifications** (3 specs) describing expected deliverable formats
- **Templates** for creating new prompt cards, starter kits, and schema documentation
- **MCP concepts documentation** with sample configuration for Model Context Protocol integration
- [README](README.md), Quickstart guide, Starter Kits overview, and [Contributing guide](CONTRIBUTING.md). The Quickstart guide and Starter Kits overview were later consolidated into [README.md](README.md) and [starter-kits/README.md](starter-kits/README.md), so those two links are intentionally not included here since the original files no longer exist.
