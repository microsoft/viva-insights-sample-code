# Changelog

All notable changes to Frontier Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Optional `viva-insights-process-analysis` Skill for collaboration around
  business processes, with approved scope contracts, exact event decisions,
  comparable windows, privacy and evidence-led reporting. Includes independently
  fabricated audit fixtures, a Python rehearsal, matched R classifier checks,
  routing/evaluation scenarios and paired-install catalogue guidance.

- Evidence-led reporting reference covering claim traceability, compact
  comparison layouts, separate numerical/privacy/visual verification and
  preservation-first publication edits. The executive deck prompt retains
  supporting evidence and avoids performance claims from adoption rankings.

- Shared seven-rule Meeting Query quality helpers in R and Python, with
  explicit flag parsing, duration checks and synthetic boundary regressions.
  Expanded refresh reconciliation, organiser-attribution, weighting, exact
  volume decomposition and inference guidance. Core analysis Skill 1.2.0.

- Standalone guides for Business Leaders AI Usage, Consumption and Ways of
  Working, and Developer Experience, with query requirements, screenshots,
  expected outputs and rendering commands. Dashboard navigation now opens
  these guides, and the shorter catalogue retains direct demo access.
- A coding-agent quick start recommending the core analysis Skill, and fuller
  dashboard setup dialogs with report-specific commands, starter prompts and
  explicit real-data support limits. Playwright captures the Business Leaders
  overview and collaboration previews.
- Meeting Query and cross-query diagnostic references reconciled with the
  installed analysis Skill, preserving the repository's stronger shared policy.
  Includes an audited, optional overlapping-export merge example with synthetic
  regression tests, corrected attendee-hours guidance and reviewed missing-HR
  diagnostics.

- Deterministic generation of prompt site pages and catalogue from canonical
  prompt cards, with shared routing metadata, preserved URLs/anchors and literal
  fenced prompts. Shared policy and workflow-support references also generate
  site pages and Skill catalogue limits, with bundled offline prompt guardrails.
- Regression gates for generation, rendered links/code at a non-root base URL,
  installed Skill references and synthetic installed-package recipe checks.

- Agent Skills catalogue on the Frontier Skills page. Each Skill opens a
  details panel with tested `gh skill install` steps, a copyable starter prompt,
  the questions it helps answer and the results to expect, including labelled
  synthetic dashboard previews. Content stays readable inline without
  JavaScript. The Skills README now documents `gh skill install`.
- Prompt Library catalogue cards with responsive details dialogs, copyable quick
  prompts generated from canonical Markdown cards, and direct links to the full
  prompt pages. The cards remain readable inline without JavaScript.
- Prompt catalogue source checks now cover all generated fields and full prompt
  blocks. Segmentation guidance separates weekly inactivity from rolling usage
  segments, and dialog section links retain browser history and keyboard focus.
- Synced the detailed prompt blocks in the canonical prompt Markdown files with
  their published prompt pages. The dashboard prompts were already in sync.
- Product query icons and exact Viva Insights descriptions in the join tutorial,
  with schema jump cards and matching icons on query section headings.
- Native query join tutorial under Advanced > Data enrichment, with Person,
  Consumption, GitHub and Agent schemas, a downloadable JSON contract and a
  reproducible page builder. Dashboard setup and Frontier journeys link to the
  shared guidance while preserving existing real-adapter limits. Related schema
  references now defer to this guide for current joins, with older layouts and
  fixture manifests explicitly scoped. All queries should be exported weekly
  where possible, with cadence confirmed from query configuration. Supplied
  Viva Insights data are assumed complete within the configured scope, and omitted
  inactive activity rows contribute zero. Panel aggregation preserves detail
  dimensions and conditional session additivity.
- Experimental AI usage archetype guidance for the `viva-insights-analysis`
  skill: three descriptive dimensions (Consistency, Surface mix, Metered use),
  a versioned reference rule with explicit measurement units, evidence states,
  sensitivity diagnostics and publication rules. Includes matched synthetic
  R and Python examples, and a link from the Copilot Usage Segments page. The
  existing usage ladder is unchanged.
- Agent query rules in the shared query contract, covering the activity layout
  observed in real exports, optional agent metadata and the earlier
  credits/responses layout as a dated variant.
- Optional policy-aware Consumption key in the dashboard runner (`policy_id`),
  with an explicit `policy_additivity` confirmation before multiple positive
  spending-policy rows for the same person, service and date are summed.
  Includes synthetic regression tests and four new dashboard skill evaluation cases.
- New data pitfalls and causal-design checks for repeated network stocks and
  their measurement windows, effective controls and cluster support.
- Skill versions: `viva-insights-analysis` now records `metadata.version`
  1.1.0, and `viva-insights-copilot-dashboards` moves from 1.0.0 to 1.1.0.

- Optional `viva-insights-causal-analysis` skill and Person Query diagnostic
  prompt. Adds explicit measurement-frequency gates for organisational outcomes,
  design/inference and privacy contracts, a report specification, and matched
  synthetic R/Python readiness and balanced-panel examples.
- Separates general exposure-outcome diagnostics from the existing prescribed
  Copilot Causal Toolkit, without changing its helpers or notebooks.

- Two AI-guided query-dashboard journeys (Consumption and Developer Experience),
  with prompt cards, site pages and starter kits sharing a reusable R runner.
- Optional `viva-insights-copilot-dashboards` skill with reproduce, adapt and
  customise modes; explicit paired installation with the general analysis skill.
- Shared Consumption/GitHub evidence contract distinguishing documented fields
  from synthetic-only inputs. Real Consumption support is scoped to credits;
  real GitHub input supports inspection, not an unverified report adapter.
- Full-prompt drift checks for the new dashboard prompt/page pairs.

- **Skills library** at [skills/](skills/), with the first skill, [viva-insights-analysis](skills/viva-insights-analysis/), covering the open-source vivainsights R and Python packages.
- [skills/README.md](skills/README.md) explaining what a Skill is, how it differs from a prompt card and from `vivainsights-context.md`, and which agents currently support the format.
- **Skill Template** at [templates/skill-template.md](templates/skill-template.md), and an "Adding a new skill" section in [CONTRIBUTING.md](CONTRIBUTING.md).
- A "Which asset should I use" comparison table on the Frontier Analytics overview, covering prompt cards, skills, `vivainsights-context.md`, and MCP.
- Two prompt cards that previously existed only on the website: [Executive PowerPoint Deck](prompts/copilot-adoption/executive-powerpoint-deck.md) and [Copilot Causal Toolkit](prompts/copilot-adoption/copilot-causal-toolkit.md). Both are now listed in [prompts/README.md](prompts/README.md).
- A "Quick prompt (short version)" on every prompt card: a short, general prompt that asks the agent to inspect the actual data and adapt column names, positioned as a faster first pass before the existing detailed, step-by-step prompt.
- A "Don't have a coding agent yet?" section on the Frontier Analytics overview (site page and repo README), pointing to GitHub Copilot and Claude Code setup for people who have not started using a coding agent at all.
- `reference/deliverables.md` in the `viva-insights-analysis` skill, distilling the same dashboard, executive summary, ROI, and segmentation conventions used in the prompt cards, so a Skill-driven session can produce those deliverables without a prompt card being pasted.

### Fixed

- M365 Copilot licensing follows Microsoft's documented native null/zero
  semantics, with Copilot Enabled User or Total Copilot enabled days preferred.
  Native licensed-only nulls mean unlicensed, while zero activity means licensed
  but inactive. Failed joins and known coverage failures remain unresolved.
  Unlicensed Copilot Chat and Consumption credits retain separate contracts.
  Absent in-scope activity can be zero. Configured cadence and metric windows
  govern aggregation.
- Adoption examples retain licensed zero-activity people and exclude native
  unlicensed nulls. Paired segmentation recipes withhold affected rolling windows
  without aborting other people's valid histories.
- Publication guidance applies the organisation requirement and workflow floor,
  removes individual leaderboards, and distinguishes heuristic flags from
  verified holidays or workforce classifications. Causal methods require a
  supported design; ROI remains an assumption-based scenario.
- Python holiday/inactive recipes request DataFrame return modes. The installed
  package's transfer-cardinality limitation is documented, not patched here.

### Changed

- The Consumption contract now records the four-file layout and
  `SpendingPolicyId` native key observed in real exports, keeps the earlier
  documented layout as a dated variant, and separates key uniqueness from
  additivity. Organisational attributes come from the Person Query when the
  people metadata has no HR attributes.
- The dashboard skill's adapt and Consumption references describe the
  policy-aware key, the additivity gate and `group: false` for metadata without
  HR attributes. The example real Consumption configuration follows the
  observed layout.

- Existing dashboard guidance now requires licence/coverage evidence, stops on
  duplicate keys and uses privacy-protected groups rather than individual ranks.
- Copilot dashboard examples link to the Frontier journeys. General skill
  references route dashboard orchestration to the optional skill.
- Corrected the adoption quickstart's plotting recommendation to match its
  static-image prompt, and removed its unmeasured completion-time promise.

- The "What's inside" table now flags `mcp/` as a concept with no server implemented yet, so this is visible before opening the folder.
- [vivainsights-context.md](../vivainsights-context.md) now points to the `viva-insights-analysis` skill for agents that support the Skill format, positioning itself as the fallback for agents that do not.
- Consolidated `QUICKSTART.md` and `STARTER_KITS.md` into [README.md](README.md) and [starter-kits/README.md](starter-kits/README.md) respectively, since the three files repeated the same workflow steps with small wording differences. Removed the two standalone files.

### Removed

- Removed the unimplemented MCP concept documents, non-functional sample
  configuration and navigation links. Frontier now presents prompt cards,
  Skills and the context file as its three agent entry points. The workflow
  support reference retains the unsupported-capability boundary.

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
