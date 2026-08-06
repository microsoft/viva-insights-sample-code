# Skills

This directory contains **agent Skills** for Viva Insights analytics. A Skill is a packaged capability that a compatible coding agent loads automatically when a task matches its description, without a person copying and pasting anything.

## What is a Skill, and how is it different from the rest of Frontier Analytics

Frontier Analytics offers a few distinct mechanisms for bringing this work into a coding agent, and each solves a different part of the problem:

- **Prompt cards** (see [prompts/](../prompts/)) are text that a person copies and pastes into an agent for a single analysis task. They require no setup, but they only apply to the one task being pasted, and they rely on the person remembering to paste the right card.
- **Skills** (this folder) are a small set of files that a compatible agent discovers and loads on its own, based on the task at hand. Once installed, the agent applies the conventions in the Skill to every relevant task in a session, not just one pasted prompt.
- **[vivainsights-context.md](../../vivainsights-context.md)** is a single file a person pastes once at the start of a session, as a fallback for agents that do not support the Skill format.
- **MCP** (see [mcp/](../mcp/)) is a protocol-level integration that would let an agent query prompts, schemas, and tools directly from a server. This is a concept only today. There is no MCP server implemented yet.

Use the comparison table on the [Frontier Analytics overview](../README.md#which-asset-should-i-use) to decide which of these fits your situation.

## Which agents support Skills

The Skill format used here follows the convention introduced by Anthropic's Claude and adopted by GitHub Copilot CLI: a `SKILL.md` file with a YAML frontmatter block (`name` and `description`) followed by a body the agent reads once it decides the Skill is relevant. As of this writing, this format is supported by GitHub Copilot CLI and Claude Code and Claude Desktop. Check your agent's own documentation to confirm current support and installation steps, since this is an actively evolving area.

## Available skills

| Skill | Description |
|---|---|
| [viva-insights-analysis/](viva-insights-analysis/) | Analyzing Viva Insights data with the open-source vivainsights R and Python packages: importing and validating a query, computing and visualizing metrics, segmenting Copilot usage, running network or information-value analysis, and avoiding well-known export data pitfalls. |

## Installing a skill

Copy the skill's folder (for example `viva-insights-analysis/`, including its `reference/` and `examples/` subfolders) into the skills directory your agent reads from. Consult your agent's documentation for the exact location, since this varies by tool.

## Contributing a new skill

See the [Skill Template](../templates/skill-template.md) for the expected structure, and [CONTRIBUTING.md](../CONTRIBUTING.md) for the full process. In short, a Skill in this repository must be customer-agnostic: it should contain no organization-specific paths, scopes, dates, or data, since it is intended for any Viva Insights user, not a single engagement.
