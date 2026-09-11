# Required inputs

## Reproduce the demo

- Repository checkout with the developer Rmd, helper and renderer.
- R, Pandoc and the packages checked by the shared runner.
- A new output workspace. No customer data or real GitHub credentials needed.

The helper creates the fixed synthetic roster, activity and coverage records.
Read the [synthetic manifest](../../../examples/utility-r/_data/github/README.md)
for their exact grain and meaning.

## Assess real-data readiness

- Approved local export CSVs and source documentation.
- Evidence for identifiers, period grain, date window and population scope.
- Metric definitions and units, including acceptance numerator/denominator.
- Independent product eligibility and coverage evidence before interpreting
  missing activity as non-use.
- Actual period definitions for any model/language allocations.
- Person Query and M365 source contracts if proposing those joins.

The [shared contract](../../skills/viva-insights-analysis/reference/copilot-query-contracts.md)
distinguishes illustrative fields from verified inputs. The real adapter is
not implemented in v1; supplying headers alone does not enable it.
