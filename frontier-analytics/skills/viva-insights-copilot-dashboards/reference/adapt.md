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
- Some real Consumption exports carry one M365 row per `SpendingPolicyId` for
  the same person, service and date. Map that column as the optional activity
  field `policy_id`. The runner then checks key uniqueness on person, service,
  policy and date, and reports the grain with a bounded count of
  person-service-date keys that have more than one positive-credit policy row.
  A positive row alongside zero rows passes. If any key has multiple positive
  rows, build is blocked until the user confirms that the policy rows record
  disjoint consumption and the config holds
  `policy_additivity: {"confirmed": true, "evidence": "<reference>"}`. Without
  the mapping, duplicated person-service-date keys fail with a suggestion to map
  `SpendingPolicyId`. Never record that confirmation on the user's behalf, or
  carry one over from another export.
- Group labels come only from the people metadata `group` mapping. When the
  metadata file has only `PeopleHistoricalId` and `IsCopilotLicensed`, set
  `group: false` and report all people together. Grouping by Person Query
  attributes is not supported by this adapter.
- Person Query associations and policy analysis are not automatically supported
  by that credits-only adapter. Token intensity and task-type mix are not
  supported at all, because no Consumption export contains those fields.
- The runner has no Agent query adapter. For a real Agent export, return a
  readiness assessment against the shared contract's Agent rules. A missing
  optional `AgentMetadata.csv` is reported as unavailable enrichment rather
  than a blocker for fact-level checks.
- Real GitHub input can be inspected, but its build adapter is not verified.
  Stop with the required source contract. The demo fixtures match the real
  export schema, but schema parity is not semantic parity.

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
