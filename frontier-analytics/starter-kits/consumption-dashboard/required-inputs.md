# Required inputs

## Reproduce the synthetic demo

- Repository checkout containing the Rmd and `_data/consumption` fixtures.
- R, Pandoc and packages checked by the shared runner (including `ggrepel`).
- A new output workspace; do not point it at the repository or a data directory.

## Build the documented credits-only report

- Extracted Consumption activity CSV and people-metadata CSV.
- Query grain (day/week/month), selected date window and approved group column.
- Confirmed mappings for person/service/date, historical people key and credits.
- Unique source keys, valid credit values and metadata matches.
- Explicit user approval of source evidence, mapping and scoped output.
- Minimum privacy group size of 10 distinct people, or a stricter policy.

See the [shared contract](../../skills/viva-insights-analysis/reference/copilot-query-contracts.md)
and [runner config schema](../copilot-query-dashboard/README.md).
There is no required token field for the credits-only path. Person Query
associations, task types and policy-limit analyses are not part of that adapter.
Do not assume every person is observed just because they appear in metadata.
