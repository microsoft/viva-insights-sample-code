# Developer Experience and Copilot

Paths below are relative to `repo_root`.

| Asset | Role |
|---|---|
| `examples/utility-r/github-copilot-developer-productivity-simulation.Rmd` | Canonical seven-page demo; historical filename retained |
| `examples/utility-r/github-developer-experience-helpers.R` | Synthetic generation, coverage-aware aggregation, assertions and plot helpers |
| `examples/utility-r/render-github-developer-experience.R` | Existing rendering entry point |
| `examples/utility-r/_data/github/README.md` | Exact synthetic schema, periods, keys, units and missingness rules |
| `frontier-analytics/starter-kits/copilot-query-dashboard/README.md` | Isolated runner and readiness interface |

Reproduce the demo only in the runner's copied workspace. The helper writes
synthetic exports and must never be sourced as a real-data loader.

## Capability boundary

The demo establishes a working-conditions baseline for its whole developer
roster, not just active or licensed Copilot users. It uses separate GitHub
and M365 feeds and explicit synthetic eligibility/completeness records.

In v1, real GitHub files are inspectable but **no verified real build adapter
is supplied**. Ask for the actual query contract before proposing an adapter.
Do not infer the absence of a product, licensing or ingestion completeness from
a sparse activity feed. Do not rename public GitHub API fields into an assumed
Viva Insights contract.

No delivery, survey, code-quality or DORA outcomes are supplied. Acceptance
rate is descriptive tool activity; calendar availability is not coding time.
Model/language mixes cover the stated full window, not every weekly slice.
