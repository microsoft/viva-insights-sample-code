# MCP — Model Context Protocol for Frontier Analytics

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol for connecting AI models to external data sources and tools. It provides a standardized way for coding agents — such as GitHub Copilot, Claude Code, and other AI assistants — to discover and interact with structured context beyond what is in their training data.

MCP defines three core primitives:

- **Prompts** — Reusable prompt templates that a server can serve to an agent on demand.
- **Resources** — Structured data (documents, schemas, data dictionaries) that an agent can query.
- **Tools** — Executable capabilities (validation, computation, generation) that an agent can invoke.

For more details, see the [MCP specification](https://modelcontextprotocol.io/) and the [MCP GitHub repository](https://github.com/modelcontextprotocol).

## Why MCP is relevant to Frontier Analytics

Frontier Analytics is built around prompt cards, data schemas, and structured workflows that analysts paste into coding agents. Today, this is a manual process: you open a prompt card, copy the text, and paste it into your agent along with context about your data.

MCP could change this by enabling coding agents to **directly access** the Frontier Analytics library:

- **Prompt cards served on demand.** Instead of copying a prompt card from a Markdown file, a coding agent could request the appropriate prompt from an MCP server and receive it pre-formatted and ready to execute.
- **Schema documentation as queryable resources.** A coding agent could ask "What columns are in the person query export?" and receive a structured answer from the schema docs — no manual lookup needed.
- **Validation and computation as tools.** A coding agent could validate a CSV against the expected schema, compute standard metrics, or generate a dashboard template by calling tools exposed through MCP.

This would reduce friction, improve consistency, and allow the prompt library to scale without requiring analysts to manually navigate the repository.

## Current status

> **This is forward-looking, not current functionality.**

MCP integration for Frontier Analytics is **conceptual**. No MCP server has been implemented. The contents of this folder are documentation and sample configurations that explore how MCP *could* enhance the Frontier Analytics workflow in the future.

The current file-based structure of prompts, schemas, and starter kits is designed to be **MCP-compatible** — the structured, consistent format of prompt cards and schema docs means they could be served through an MCP server with minimal transformation when the time comes.

## What's in this folder

| File | Description |
|------|-------------|
| [frontier-mcp-concepts.md](frontier-mcp-concepts.md) | Conceptual document explaining how MCP primitives map to Frontier Analytics components, potential use cases, and implementation considerations |
| [sample-config/example.mcp.json](sample-config/example.mcp.json) | A sample MCP client configuration file showing what a Frontier Analytics MCP server integration might look like |

### About the sample configuration

The `example.mcp.json` file in `sample-config/` is a **conceptual example** of an MCP client configuration. Since JSON does not support comments, the field names and values are written to be self-documenting. It shows how a coding agent's MCP client might be configured to connect to a hypothetical Frontier Analytics MCP server.

This file is not functional — there is no server to connect to. It is provided as a reference for what an MCP integration could look like.

## How MCP could enhance the workflow

| Today (manual) | Future with MCP |
|-----------------|-----------------|
| Analyst browses the prompts directory to find the right card | Agent queries the MCP server for prompts matching the analyst's goal |
| Analyst copies prompt text and pastes it into the agent | Agent receives the prompt directly from the server, pre-formatted |
| Analyst manually checks column names against schema docs | Agent queries the schema resource to validate columns automatically |
| Analyst reviews common failure modes after the fact | Agent checks failure modes proactively before executing |
| Analyst manually combines outputs from multiple prompt cards | Agent orchestrates a multi-step workflow using the starter kit as a plan |

## Links

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Frontier Analytics Prompt Library](../prompts/)
- [Frontier Analytics Templates](../templates/)
