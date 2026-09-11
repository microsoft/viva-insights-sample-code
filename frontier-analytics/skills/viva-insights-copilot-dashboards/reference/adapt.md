# Adapt to real data

## Inspect before proposing

Read the shared contract and selected report manifest. Ask for approved local
files, query setup/grain, audience, period, grouping and privacy requirements.
Do not request customer rows in chat. Extract archives locally after inspecting
their paths; reject traversal entries. The runner takes extracted CSV paths.

Use `dashboard.R inspect <config.json>` with a real config. Review bounded
schema summaries, counts and readiness checks. A header match alone is not
semantic confirmation. Inspect source documentation only for the relevant
metrics and record its date.

## Approve the supported scope

Present a short supported/excluded/reason table. In v1:

- Documented Consumption inputs can support a descriptive credits report.
  Confirm the activity and historical-metadata keys, grain, group mapping
  and units. Show the runner's required approval/evidence fields to the user.
- Token intensity, task types, Person Query associations and policy analysis
  are not automatically supported by that credits-only adapter.
- Real GitHub input can be inspected, but its build adapter is not verified.
  Stop with the required source contract. Do not run the synthetic helper on
  real data or claim the demo matches those files.

Ask for approval of the mappings and scope through prompt mode. Only then
record approval in the configuration and run
`dashboard.R build <config.json>` for a supported path. Never set an approval
flag on behalf of a user who has not approved it.

## Validate and explain

Require unique keys, valid units/dates, join reconciliation, observed-population
denominators and privacy suppression. Unsupported or missing data produces a
clear failure or an explicitly excluded panel, not an imputed result.

Return HTML, configuration, provenance and limitations. Keep raw files and
private operational summaries outside the repository. Show source window and
excluded panels in the HTML itself. Do not infer currency savings, burnout,
productivity or causation.

## When a user supplies new schema evidence

Record the new contract and propose a separately reviewed adapter extension.
Confirm keys, coverage and expected metrics using non-sensitive fixtures first.
Do not silently enable undocumented branches in the existing adapter. A
readiness assessment is the correct result while required evidence is absent.
