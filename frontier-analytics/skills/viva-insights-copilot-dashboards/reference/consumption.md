# Consumption and Ways of Working

Paths below are relative to `repo_root`. Read only the needed sections.

| Asset | Role |
|---|---|
| `examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd` | Canonical seven-page demo; reads supplied fixture CSVs |
| `examples/utility-r/_data/consumption-query/` | Consumption query fixtures: `PeopleMetaData`, `PersonM365CreditsMetrics`, `PersonGitHubCreditsMetrics` |
| `examples/utility-r/_data/person-query/PersonQuery.csv` | Person query fixture supplying collaboration metrics and HR attributes |
| `examples/utility-r/_data/README.md` | Authoritative schema manifest: keys, grains, units and join rules |
| `examples/utility-r/copilot-consumption-github-demo-reports.md` | Background documentation; actual source and fixture manifest determine runtime inputs |
| `frontier-analytics/starter-kits/copilot-query-dashboard/README.md` | Runner configuration, dependencies and output contracts |

The existing Rmd reads fixtures; do not assume it generates them on render.
Use the runner's exact asset list, not a recursive copy of all examples.
Dependencies are determined from source and checked by the runner; do not
assume a short documentation list is exhaustive (the demo uses `ggrepel`).

## Capability boundary

| Mode | Result |
|---|---|
| Demo | Full synthetic reference in the real Consumption schema: credits, sessions, service mix, cost per session, segments and working-pattern associations |
| Real v1 | Documented credit consumption by configured period, service and group, subject to privacy checks |
| Not supported automatically | Usage-segment translation, Person Query joins, network snapshots, value/ROI. Tokens and task-type mix are not supported at all — no Consumption export contains them |

Credits cannot be substituted for Copilot actions in habituality segmentation.
Never derive tokens from credits, and do not reintroduce token fields into the
fixtures. Reread the shared contract for source granularity and limits.
