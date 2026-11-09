---
layout: page
title: "Workflow support"
permalink: /frontier-analytics-support/
---
<!-- Generated from the core Skill reference. Do not edit. -->

# Workflow support

This table owns cross-workflow capability summaries. Workflow READMEs and
references own exact input/configuration contracts and commands. Implementation
support is not proof of a real-data trial or an executed agent evaluation.

| Workflow | Supported path and runtime | Inputs and measures | Outputs and limitations | Verification entry point |
|---|---|---|---|---|
| Core analysis skill | Agent-guided R/Python using installed `vivainsights` APIs | Documented query exports with verified grain and metric semantics | Generated scripts, descriptive reports and readiness audits; not a universal export adapter or causal engine | Bundled `examples/README.md` and installed API inspection |
| Adoption prompts and starter | Copyable agent-guided R/Python workflow | Person Query, explicit eligibility for adoption denominators, verified activity and optional HR/assisted-hours metrics | Static HTML, memo, segmentation or editable deck; output code and numbers require verification; ROI is an assumption-based scenario | Prompt instructions and starter input/output specifications; validation metadata alone is not an executed test |
| Consumption dashboard runner | Maintained local R runner; synthetic demo plus scoped real-export build | Documented Consumption layouts, explicit column mappings, historical metadata and additive-credit decisions | Real builds are credits-only supported panels; no invented tokens, currency, Person Query adapter or ways-of-working panels | Runner README, `tests/`, and dashboard `evals/README.md` |
| GitHub developer dashboard | Synthetic R reproduction; real exports inspect/readiness only | Synthetic activity, Person Query and roster fixture; real headers inspected against the contract | No implemented real GitHub build adapter; no productivity or causal interpretation | Runner README and dashboard GitHub manifest |
| Person and Agent integration | Schema/join guidance, not a packaged dashboard adapter | Current selected-query join contract, historical identity and weekly Person spine | No implemented real Person/Agent dashboard adapters; documented joins do not extend runner support | Current join contract and its deterministic tests |
| Person Query causal skill | Agent-guided design and readiness; bounded synthetic Python/base-R runners | Verified exposure/outcome windows; fabricated balanced continuous-outcome panels in bundled runners | Synthetic readiness and limited linear comparisons, not general real-data PPML, DML, bootstrap or quasi-experimental execution | Causal `examples/README.md` and `reference/evaluation.md`; historical evidence is bounded there |
| Copilot Causal Toolkit | Separate prescribed notebook/DML workflow | Selected toolkit notebook, its fixed treatment, dependencies and design-specific input contract | Do not replace with or claim equivalence to the synthetic causal runner; inspect identification and actual notebook preprocessing before causal interpretation | Toolkit prompt and selected notebook's own diagnostics |
| External-collaboration integration | Standalone schemas/examples only where documented; no general cross-query adapter | Source-specific keys, grain and privacy must be established | No inferred identity bridge or assumed compatible join | Relevant source schema and example, not another query's contract |
| Live product / MCP access | Unsupported by this repository | Local exported files only in maintained workflows | No MCP server, live-product connector or credentials workflow is supplied | Not applicable |

## Resolving references

For an installed core skill, `examples/README.md` is bundled locally. Optional
workflow assets live in a retained sample-code checkout:

- Dashboard: `frontier-analytics/starter-kits/copilot-query-dashboard/README.md`,
  its `tests/`, and the dashboard skill's report manifests and `evals/README.md`.
- Causal: the paired causal skill's `examples/README.md` and
  `reference/evaluation.md`.
- DML: `frontier-analytics/prompts/copilot-adoption/copilot-causal-toolkit.md`
  and the notebook it selects.
- Joins: [current guide and downloadable contract](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/).

Do not treat a missing optional checkout as missing core analysis functionality.
Ask for the specific asset only when that workflow is requested. Keep the
analysis, dashboard, causal and DML workflows separate.

## Catalogue summaries

These marked plain-text lists are authored here for generated skill cards.

<!-- catalogue:viva-insights-analysis:start -->
- Agent-guided local R/Python analysis. Export layouts and installed APIs require verification for each workflow.
- Verify installed APIs and generated outputs. Built-in and synthetic examples are not real-export validation.
- Prefer enabled-user/enabled-days fields. Documented native licensed-only M365 Copilot nulls mean unlicensed. Failed joins remain unresolved. These licence semantics do not apply to unlicensed Copilot Chat or Consumption credits.
- Use max(organisation requirement, workflow floor): generic 5, joined-query/dashboard/causal 10 distinct people. Protect complementary cells and never publish individual rankings.
<!-- catalogue:viva-insights-analysis:end -->

<!-- catalogue:viva-insights-copilot-dashboards:start -->
- Requires a retained sample-code checkout and R; reference demo rendering also requires Pandoc. Skill installation does not install dependencies.
- Consumption real builds support scoped credits-only panels. GitHub real data is inspect/readiness only; real Person/Agent dashboard adapters are not implemented.
- Reference reproductions are synthetic. Do not invent missing panels, tokens, currency, productivity, ROI or causal effects from activity.
- Use max(organisation requirement, 10 distinct people) across published and embedded output, protect complementary cells and never publish individual rankings.
<!-- catalogue:viva-insights-copilot-dashboards:end -->

<!-- catalogue:viva-insights-causal-analysis:start -->
- Guided design/readiness workflow, not a general real-data causal engine. Bundled R/Python runners use fabricated balanced panels and limited linear comparisons.
- Choose a supported identifying design and compatible diagnostics first; readiness-only and unavailable checks are valid. No general PPML, DML, bootstrap or quasi-experimental runner is implemented here.
- The prescribed Copilot Causal Toolkit remains a separate notebook/DML workflow; do not silently substitute treatments or methods.
- Install alongside the core analysis skill. Use max(organisation requirement, 10 distinct people), protect complementary cells and never publish individual rankings or person-specific effects.
<!-- catalogue:viva-insights-causal-analysis:end -->

[Back to Skills]({{ site.baseurl }}/frontier-analytics-skills/)
