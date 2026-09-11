# Consumption and Ways of Working

Paths below are relative to `repo_root`. Read only the needed sections.

| Asset | Role |
|---|---|
| `examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd` | Canonical seven-page demo; reads supplied fixture CSVs |
| `examples/utility-r/_data/consumption/` | Synthetic consumption, task-type, person-query, network and roster fixtures |
| `examples/utility-r/copilot-consumption-github-demo-reports.md` | Background documentation; actual source and fixture manifest determine runtime inputs |
| `frontier-analytics/starter-kits/copilot-query-dashboard/README.md` | Runner configuration, dependencies and output contracts |

The existing Rmd reads fixtures; do not assume it generates them on render.
Use the runner's exact asset list, not a recursive copy of all examples.
Dependencies are determined from source and checked by the runner; do not
assume a short documentation list is exhaustive (the demo uses `ggrepel`).

## Capability boundary

| Mode | Result |
|---|---|
| Demo | Full synthetic reference, including tokens, task types, credit intensity, segments and working-pattern associations |
| Real v1 | Documented credit consumption by configured period, service and group, subject to privacy checks |
| Not supported automatically | Tokens, task-type mix, usage-segment translation, Person Query joins, network snapshots, value/ROI |

Credits cannot be substituted for Copilot actions in habituality segmentation.
Input/output/reasoning tokens require a verified non-overlapping definition
before summing; do not assume the demo formula applies to a new provider.
Reread the shared contract for source granularity and limits.
