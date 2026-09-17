# Developer Experience and Copilot

Paths below are relative to `repo_root`.

| Asset | Role |
|---|---|
| `examples/utility-r/github-copilot-developer-productivity-simulation.Rmd` | Canonical seven-page demo; historical filename retained |
| `examples/utility-r/github-developer-experience-helpers.R` | Fixture loading, coverage derivation, aggregation, assertions and plot helpers |
| `examples/utility-r/render-github-developer-experience.R` | Existing rendering entry point |
| `examples/utility-r/simulate-query-exports.R` | Single deterministic generator for every fixture under `_data` |
| `examples/utility-r/_data/README.md` | Authoritative schema manifest: real export file names, keys, grains, units and join rules |
| `frontier-analytics/starter-kits/copilot-query-dashboard/README.md` | Isolated runner and readiness interface |

Reproduce the demo only in the runner's copied workspace. The helper reads the
committed fixtures and must never be treated as a real-data loader; fixture
generation belongs to `simulate-query-exports.R`.

## Capability boundary

The demo establishes a working-conditions baseline for its whole developer
roster, not just active or licensed Copilot users. It uses the GitHub query
activity and breakdown files alongside Consumption query credit records, and
uses weekly enabled days for M365 eligibility. M365 completeness stays unknown
without independent ingestion evidence; GitHub weekday row counts describe observed
coverage, not provisioning. No query produces an eligibility or completeness file.
Breakdown equality/model attribution are synthetic-only invariants, not real-data
validation rules. M365 Copilot credits and GitHub AI credits remain separate units.

In v1, real GitHub files are inspectable but **no verified real build adapter
is supplied**. Ask for the actual query contract before proposing an adapter.
Do not infer the absence of a product, licensing or ingestion completeness from
a sparse activity feed. Do not rename public GitHub API fields into an assumed
Viva Insights contract.

No delivery, survey, code-quality or DORA outcomes are supplied. Acceptance
rate is descriptive tool activity; calendar availability is not coding time.
Model/language mixes cover the stated full window, not every weekly slice.
