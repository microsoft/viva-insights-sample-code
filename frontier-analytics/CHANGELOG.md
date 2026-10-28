# Changelog

All notable changes to Frontier Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
