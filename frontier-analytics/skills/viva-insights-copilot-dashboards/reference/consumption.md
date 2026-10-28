# Consumption and Ways of Working

Paths below are relative to `repo_root`. Read only the needed sections.

| Asset | Role |
|---|---|
| `examples/utility-r/copilot-consumption-ways-of-working-simulation.Rmd` | Canonical seven-page demo; reads supplied fixture CSVs |
| `examples/utility-r/_data/consumption-query/` | Consumption query fixtures: `PeopleMetaData`, `PersonM365CreditsMetrics`, `PersonGitHubCreditsMetrics` |
| `examples/utility-r/_data/person-query/PersonQuery.csv` | Person query fixture supplying collaboration metrics and HR attributes |
| `examples/utility-r/_data/github-query/PersonGitHubActivityMetrics.csv` | GitHub activity fixture; its explicit weekday zero rows separate observed non-use from absent observation in the product usage mix. The four GitHub breakdown exports are **not** read by this report |
| `examples/utility-r/_data/README.md` | Authoritative manifest for committed fixtures only: keys, grains, units and synthetic assumptions |
| `examples/utility-r/copilot-consumption-github-demo-reports.md` | Background documentation; actual source and fixture manifest determine runtime inputs |
| `frontier-analytics/starter-kits/copilot-query-dashboard/README.md` | Runner configuration, dependencies and output contracts |

The existing Rmd reads fixtures; do not assume it generates them on render.
For current export schemas and cross-query joins, defer to
[Joining Viva Insights queries](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/).
Its complete-data default does not change the runner's mappings or capabilities.
Use the runner's exact asset list, not a recursive copy of all examples.
Dependencies are determined from source and checked by the runner; do not
assume a short documentation list is exhaustive (the demo uses `ggrepel`).

## Capability boundary

| Mode | Result |
|---|---|
| Demo | Full synthetic reference in the real Consumption schema: credits, sessions, service mix, cost per session, segments and working-pattern associations |
| Real v1 | Documented credit consumption by configured period, service and group, subject to privacy checks. Optional policy-aware key with an additivity gate |
| Not supported automatically | Usage-segment translation, Person Query joins, network snapshots, value/ROI. Tokens and task-type mix are not supported at all — no Consumption export contains them |

Real exports vary. Where the M365 activity file has one row per
`SpendingPolicyId`, map it as the optional activity field `policy_id`. The
runner checks uniqueness on person, service, policy and date, and blocks the
build when any person-service-date key has more than one positive-credit policy
row anywhere in the supplied input, before the `dates` filter, unless the user
has confirmed disjoint policy rows and the config records
`policy_additivity: {"confirmed": true, "evidence": "<reference>"}`. Keys with
one positive row and zero rows pass. Group labels come only from the people
metadata mapping, so when that file has no HR attributes use `group: false`.
Grouping by Person Query attributes stays unsupported. See
[Adapt](adapt.md) and the runner README for the exact fields.

Credits cannot be substituted for Copilot actions in habituality segmentation.
Never derive tokens from credits, and do not reintroduce token fields into the
fixtures. Reread the shared contract for source granularity and limits.
