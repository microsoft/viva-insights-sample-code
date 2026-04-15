# MCP Concepts for Frontier Analytics

> **Status: Conceptual / Future-facing.** Nothing described in this document is implemented. This is an exploration of how the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) could integrate with the Frontier Analytics prompt library and schema documentation.

## MCP primitives in Frontier Analytics terms

MCP defines three core primitives. Here is how each one maps to Frontier Analytics:

### Prompts

**MCP definition:** Reusable prompt templates that an MCP server serves to a connected coding agent.

**Frontier Analytics mapping:** The prompt cards in `frontier-analytics/prompts/` are already structured as self-contained prompt templates. Today, analysts manually copy them. With MCP, these same cards could be served dynamically:

- A coding agent connects to the Frontier Analytics MCP server.
- The analyst says: *"I need to analyze Copilot adoption."*
- The agent queries the server's prompt catalog and receives the `dashboard-overview` prompt, pre-formatted with the correct structure.
- The agent can also request adaptation notes and common failure modes as supplementary context.

Each prompt card's metadata (Purpose, Audience, When to use) would serve as the prompt's description in the MCP catalog, helping the agent select the right prompt for the analyst's goal.

### Resources

**MCP definition:** Data context that an MCP server exposes as read-only structured content.

**Frontier Analytics mapping:** Schema documentation, data dictionaries, and reference materials could be exposed as MCP resources:

- **Schema docs** (`frontier-analytics/schemas/`) — A coding agent could query *"What columns are in the person query export?"* and receive the column dictionary with types, descriptions, and edge cases.
- **Data dictionaries** — Metric definitions, HR attribute descriptions, and valid value ranges could be served as structured resources.
- **Starter kit metadata** — The workflow steps, required inputs, and expected outputs from starter kits could help an agent plan a multi-step analysis.

Resources are read-only and contextual. They give the agent knowledge about the data it is working with, reducing errors caused by incorrect assumptions about column names, data types, or missing values.

### Tools

**MCP definition:** Executable capabilities that an MCP server provides for a coding agent to invoke.

**Frontier Analytics mapping:** Common operations that analysts perform repeatedly could be exposed as MCP tools:

- **Schema validation** — Given a CSV file path, validate that it contains the expected columns for a specific data source (e.g., person query). Return a report of missing columns, unexpected columns, and type mismatches.
- **Metric computation** — Compute standard metrics (adoption rate, active users, licensed users) from a validated dataset, using the canonical definitions from the prompt library.
- **Dashboard generation** — Generate a static HTML dashboard from a standard metrics output, using a consistent template.
- **Privacy check** — Scan an output for groups with fewer than *N* users and flag them for suppression.

Tools are executable: the agent calls them with parameters and receives results. This moves common logic out of the prompt text and into reusable, tested code.

---

## Potential MCP server for Frontier Analytics

### What it would do

A hypothetical Frontier Analytics MCP server would:

1. **Serve prompt cards as MCP prompts.** The server reads the Markdown files from `frontier-analytics/prompts/`, parses their metadata, and exposes them through the MCP prompts API. A coding agent can list available prompts, filter by category or audience, and retrieve the full prompt text.

2. **Expose schema documentation as MCP resources.** The server reads schema docs from `frontier-analytics/schemas/` and serves them as structured resources. An agent can query for a specific schema by name and receive the column dictionary, example rows, and validation rules.

3. **Provide validation and computation tools.** The server implements a set of tools that encode the analytical logic described in the prompt cards — schema validation, metric computation, privacy threshold checks — so that agents can invoke them directly instead of re-implementing the logic each time.

### Concrete use cases

1. **Guided prompt selection.** An analyst tells their coding agent: *"I need to prepare a Copilot adoption report for my VP."* The agent queries the MCP server for prompts matching "executive" and "Copilot adoption", receives the `executive-summary` prompt card, and executes it — all without the analyst navigating the repository.

2. **Automatic data validation.** Before running an analysis, the agent calls the `validate-schema` tool with the analyst's CSV file. The tool returns: *"Missing column: `FunctionType`. Found unexpected column: `Function_Type` — did you mean `FunctionType`?"* The agent auto-corrects and proceeds.

3. **Multi-step workflow orchestration.** An analyst says: *"Run the full Copilot adoption starter kit."* The agent retrieves the starter kit's workflow from the server's resources, then executes each prompt card in sequence, passing outputs between steps.

4. **Context-aware error recovery.** During execution, the agent encounters a data issue. It queries the MCP server for the relevant prompt card's Common Failure Modes resource and applies the documented resolution automatically.

5. **Cross-prompt consistency.** Multiple prompt cards define "adoption rate" slightly differently depending on context. The MCP server's metric computation tool provides a single canonical implementation, ensuring consistent results across analyses.

### Implementation considerations

- **Runtime:** The server could be implemented in Node.js (using the [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)) or Python (using the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)).
- **Deployment:** For individual analysts, a local server running on the same machine as the coding agent is simplest. For teams, a shared server could be deployed as a container or cloud function.
- **Content source:** The server would read prompt cards and schema docs directly from the repository's file structure. No database is needed — the Markdown files are the source of truth.
- **Maintenance:** When new prompt cards or schemas are added to the repository, the server automatically picks them up (if reading from the local clone) or can be refreshed with a `git pull`.
- **Authentication:** For a local server, no authentication is needed. For a shared deployment, standard API authentication (e.g., Azure AD tokens) would be appropriate.

> **This is a future possibility, not a current feature.** The Frontier Analytics team is monitoring MCP ecosystem maturity and will consider implementation when the protocol and tooling are more widely adopted.

---

## How this relates to the current prompt library

### Today: manual workflow

1. Analyst opens the Frontier Analytics prompt library in the repository.
2. Analyst finds the relevant prompt card by browsing or searching.
3. Analyst reads the Required Inputs and Assumptions sections to prepare data.
4. Analyst copies the Prompt section text.
5. Analyst opens a coding agent and pastes the prompt with a note about their data file.
6. Analyst reviews the output against Common Failure Modes.
7. Analyst applies Adaptation Notes for their specific environment.

### Future with MCP: automated workflow

1. Analyst describes their goal to the coding agent.
2. The agent queries the MCP server to find matching prompts.
3. The agent retrieves the prompt and its metadata (assumptions, failure modes, adaptation notes).
4. The agent calls the schema validation tool to check the analyst's data.
5. The agent executes the prompt, using resources for column definitions and tool calls for metric computation.
6. The agent proactively checks its output against the documented failure modes.
7. The agent delivers the final output with a summary of any adaptations applied.

### Design for compatibility

The current file-based structure is intentionally designed to be MCP-compatible:

- **Consistent section headers** in prompt cards (Purpose, Audience, When to use, Required inputs, etc.) make it straightforward to parse metadata programmatically.
- **Structured column dictionaries** in schema docs can be converted to JSON schema resources with minimal transformation.
- **Separation of concerns** — prompt text, metadata, adaptation notes, and failure modes are in distinct sections that map cleanly to MCP prompt parameters and resource attachments.
- **Relative linking** between prompts, schemas, and starter kits mirrors the relationships that an MCP server would express through its API.

When the time comes to implement an MCP server, the existing content can be served with a thin parsing layer — no restructuring required.
