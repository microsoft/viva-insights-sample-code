# Sample query exports

Every CSV in this folder is **fully synthetic**. No real person, tenant or usage
record appears anywhere in it. What is real is the shape: column names, file
splits, row grains and join keys all match what a Microsoft Viva Insights query
export actually produces, so code written against these files should run against
a genuine download with minimal change.

## What is here

| Query | Folder | Files | Grain | What it measures |
|---|---|---|---|---|
| **Person** | `person-query/` | 1 | Person × week | Collaboration, focus, after-hours and network metrics, plus Copilot eligibility |
| **Consumption** | `consumption-query/` | 3 | Person × day, plus a person-level metadata file | M365 Copilot credits and sessions by service, GitHub AI credits, and HR attributes |
| **GitHub Copilot** | `github-query/` | 5 | Person × day, plus dimensional breakdowns | Code completions, chat requests and agent adoption, split by feature, language and model |
| **Agent** (Private Preview) | `github/agent-query/` | 3 | Person × agent × week, plus an agent metadata file | Agent responses, agent credits and returning-user indicators |

A standalone teaching dataset, `Top_Performers_Dataset_v2.csv`, also sits at the
root of this folder. It belongs to `top-performers-rf.Rmd` rather than to any
query export, and it is described further down.

```text
_data/
  person-query/
    PersonQuery.csv
  consumption-query/
    PeopleMetaData.csv
    PersonM365CreditsMetrics.csv
    PersonGitHubCreditsMetrics.csv
  github-query/
    PersonGitHubActivityMetrics.csv
    GitHubActivityBreakdownByFeatureMetrics.csv
    GitHubActivityBreakdownByLanguageFeatureMetrics.csv
    GitHubActivityBreakdownByLanguageModelMetrics.csv
    GitHubActivityBreakdownByModelFeatureMetrics.csv
  github/
    agent-query/
      AgentMetadata.csv
      PersonAgentCreditsRetentionMetrics.csv
      PersonAgentResponsesMetrics.csv
  Top_Performers_Dataset_v2.csv
```

## Joining the files

Three keys do all the work.

| Key | Appears in | Joins |
|---|---|---|
| `PersonId` | Person query, both Consumption activity files, all GitHub files, both Agent fact files | The person dimension across every query |
| `PeopleHistoricalId` | `PeopleMetaData.csv` and both Consumption activity files | Consumption activity to its HR attribute metadata |
| `AgentId` | `AgentMetadata.csv` and both Agent fact files | Agent facts to agent descriptive fields, many-to-one |

`MetricDate` aligns them in time. The Person query and the Agent query are both
weekly with a Sunday week start, while the Consumption and GitHub feeds are
daily, so joining a daily feed to a weekly one means aggregating up to the week
first.

Four things are worth knowing before you write the join.

1. **`PeopleMetaData.csv` has no `PersonId` column.** Its only link to the rest
   of the folder is `PeopleHistoricalId`, which the two Consumption activity
   files carry alongside `PersonId`. Read that crosswalk out of the activity
   files. `PeopleHistoricalId` is opaque, and no string operation on `PersonId`
   reproduces it, in these fixtures or in a real tenant.
2. **The four GitHub breakdown files do not share the activity file's grain.**
   Each adds a `Feature`, `Language` or `Model` dimension, so joining all five
   wide on `PersonId` and `MetricDate` multiplies rows. Keep the breakdowns as
   separate fact tables, or aggregate each one to a single row per person-period
   before joining.
3. **Both Agent fact files sit at person × agent × week**, so confirm uniqueness
   at that grain before combining them, or aggregate each one separately.
4. **The Agent sample uses its own `PersonId` namespace**, with values such as
   `synthetic-person-00000` over a different date window, so it will not join to
   the other folders here. Only the Person, Consumption and GitHub folders
   describe one shared population, being 480 people of whom 360 are developers
   across six teams of 60.

Bear in mind too that M365 Copilot credits and GitHub AI credits have no
verified common unit. Do not add them together, calculate cross-product shares,
or put GitHub AI credits into an M365 credits-per-session denominator.

## Person query

Source: [Person query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query).

`person-query/PersonQuery.csv` holds one row per `PersonId` × `MetricDate`,
covering 26 Sunday-start weeks from `2026-01-04` to `2026-07-04`.

| Group | Columns |
|---|---|
| Collaboration volume | `Collaboration_hours`, `Meeting_hours`, `Email_hours`, `Chat_hours`, `Scheduled_call_hours`, `Unscheduled_call_hours`, `Channel_message_hours`, `Active_connected_hours`, `External_collaboration_hours` |
| Focus and availability | `Available_to_focus_hours`, `Uninterrupted_hours`, `Interrupted_hours`, `Open_1_hour_block` |
| Meeting characteristics | `Recurring_meeting_hours`, `Conflicting_meeting_hours`, `Meeting_hours_with_six_or_fewer_hours_of_advanced_notice` |
| Counts | `Emails_sent`, `Chats_sent`, `Meetings`, `Calls` |
| After hours and weekend | `After_hours_collaboration_hours`, `After_hours_meeting_hours`, `After_hours_email_hours`, `After_hours_chat_hours`, `Weekend_collaboration_hours`, `Collaboration_span` |
| Network | `Internal_network_size`, `External_network_size`, `Strong_ties`, `Diverse_ties`, `Network_outside_organization` |
| Copilot eligibility | `Total_Copilot_enabled_days` |
| HR attributes | `Organization`, `FunctionType`, `LevelDesignation`, `IsManager`, `Team`, `Role`, `Seniority`, `Tenure`, `IsDeveloper` |

**Joins with:** any Consumption or GitHub file on `PersonId`, once the daily
feed has been aggregated to the matching week.

The Person query is a custom template, so the column set legitimately varies by
tenant and by analyst. Every metric selected here was checked against the
[metric reference](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics).
Two interpretation notes carry over to real data. `Available_to_focus_hours` is
the working-hours time left after meetings and scheduled calls, partitioned by
`Uninterrupted_hours` and `Interrupted_hours`, and it measures calendar
availability rather than coding or psychological flow. The three
meeting-characteristic metrics overlap each other and must not be stacked
additively.

## Consumption query

Source: [Create a custom consumption query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query).

A real run downloads a single `.zip` containing these three files. The sample
covers the final 13 weeks of the Person query window, from `2026-04-05` onward,
at daily grain and weekdays only.

### `PeopleMetaData.csv`

One row per person. `PeopleHistoricalId` and `IsCopilotLicensed` are always
present and cannot be removed. Every other column is an HR attribute chosen at
query setup, so this file's width varies from query to query.

`PeopleHistoricalId, IsCopilotLicensed, Organization, FunctionType, LevelDesignation, IsManager, Team, Role, Seniority, Tenure, IsDeveloper`

**Joins with:** `PersonM365CreditsMetrics.csv` and
`PersonGitHubCreditsMetrics.csv` on `PeopleHistoricalId`, one-to-many.

### `PersonM365CreditsMetrics.csv`

One row per `PersonId` × `ServiceId` × `MetricDate`. Someone who used three
services on one day has three rows for that day, so code assuming one row per
person per day will silently over-count.

`PersonId, ServiceId, ServiceName, SpendingPolicyId, MetricDate, Session count, Spending policy limit, Total Copilot Credits used, User limit, PeopleHistoricalId`

**Joins with:** `PeopleMetaData.csv` on `PeopleHistoricalId`, and the Person
query on `PersonId` plus an aggregated `MetricDate`.

In this fixture `SpendingPolicyId` is the all-zero GUID when no policy applies,
and `Spending policy limit` and `User limit` are empty. Validate real null
semantics against the query contract you were supplied rather than inferring
them from this recipe.

### `PersonGitHubCreditsMetrics.csv`

One row per `PersonId` × `MetricDate`.

`PersonId, MetricDate, Total GitHub AI Credits used, PeopleHistoricalId`

**Joins with:** `PeopleMetaData.csv` on `PeopleHistoricalId`, and the Person
query on `PersonId` plus an aggregated `MetricDate`.

This file is **sparse**, carrying a row only for days with billable usage,
unlike `PersonGitHubActivityMetrics.csv` which carries explicit weekday zeros.
Activity coverage and credit coverage are therefore two different questions. Do
not require one credit row per weekday in order to accept an observed activity
week, because that discards every fully observed but partly inactive week.
Resolve credit coverage on its own terms, checking that the set of credit dates
equals the set of observed billable dates, and treating a week with no billable
day as a measured zero. Compare the dates rather than their counts, since a
missing credit row on one billable day and a spurious row on another leave the
counts equal while the week is incomplete.

## GitHub Copilot query

Five files and no metadata file. HR attributes for this population come from the
Person query or from the Consumption query's `PeopleMetaData.csv`.

### `PersonGitHubActivityMetrics.csv`

One row per `PersonId` × `MetricDate`.

`PersonId, MetricDate, Agent adoption, Code completions accepted, Code completions suggested, User-initiated chat requests`

**Joins with:** the Person query on `PersonId` plus an aggregated `MetricDate`,
and the breakdown files once those have been aggregated to person-day.

The fixture includes explicit zero rows for provisioned users on days with no
activity. A measured zero is observed inactivity, whereas a missing row is
unknown rather than proof of no provisioning, and the presence of a row
demonstrates observation rather than a licence record. `Code completions
accepted` never exceeds `Code completions suggested`, so acceptance rate is
accepted divided by suggested with an explicit zero-denominator rule. `Agent
adoption` is an adoption indicator or percentage rather than a count of agent
actions, and it excludes Copilot code review activity.

### Breakdown files

| File | Grain | Measure |
|---|---|---|
| `GitHubActivityBreakdownByFeatureMetrics.csv` | `PersonId` × `MetricDate` × `Feature` | `Feature Usage Count` |
| `GitHubActivityBreakdownByLanguageFeatureMetrics.csv` | `PersonId` × `MetricDate` × `Language` × `Feature` | `Usage count by language and feature` |
| `GitHubActivityBreakdownByLanguageModelMetrics.csv` | `PersonId` × `MetricDate` × `Language` × `Model` | `Usage count by language and model` |
| `GitHubActivityBreakdownByModelFeatureMetrics.csv` | `PersonId` × `MetricDate` × `Model` × `Feature` | `Usage count by model and feature` |

**Joins with:** each other and with the activity file, but only after
aggregating to a shared grain. See the many-to-many warning in
[Joining the files](#joining-the-files).

`Feature` uses the real export vocabulary: `code_completion`, `chat_inline`,
`chat_panel_ask_mode`, `chat_panel_edit_mode`, `chat_panel_agent_mode`,
`chat_panel_plan_mode`, `chat_panel_custom_mode`, `chat_panel_unknown_mode`,
`agent_edit`, `copilot_cli` and `copilot_app`.

There is no published production schema for the GitHub query. These files
establish a useful demo shape, but identifiers, units, aggregation periods,
eligibility and coverage should all be checked against the actual customer
export before a report is refreshed against real data.

## Agent query

The Agent query is in Private Preview, so its schema and availability vary by
tenant and may change. Record the export date, preserve the original headers
when adapting a report, and keep `AgentId` as the stable analytical key, because
agent names and creator metadata change over time.

The sample sits under `github/agent-query/` and covers 53 Sunday-start weeks
from `2025-09-07` to `2026-09-06`, across 300 people and 914 agents.

### `AgentMetadata.csv`

One row per `AgentId`, covering 3,000 agents. It is a superset of the agents
that appear in the fact files, so most rows have no matching activity, which is
normal for a dimension table.

`AgentId, AgentName, CreatorType, PublishingSource`

`CreatorType` takes values such as `Microsoft`, `User`, `Your org` and `Third
party`. `PublishingSource` takes values such as `Microsoft Copilot Studio`,
`Microsoft other`, `Agent Builder`, `SharePoint` and `Other`. The `AgentName`
values in this sample are invented placeholders.

**Joins with:** both Agent fact files on `AgentId`, one-to-many.

### `PersonAgentResponsesMetrics.csv`

One row per `PersonId` × `AgentId` × `MetricDate`, where `MetricDate` is the
Sunday start of the week.

`PersonId, AgentId, MetricDate, Agent responses generated`

**Joins with:** `AgentMetadata.csv` on `AgentId`, and a Person query on
`PersonId` and `MetricDate` where both share a `PersonId` namespace.

### `PersonAgentCreditsRetentionMetrics.csv`

One row per `PersonId` × `AgentId` × `MetricDate`, on the same key set as
`PersonAgentResponsesMetrics.csv`.

`PersonId, AgentId, MetricDate, Copilot credits used for agents, Returning agent user (most recent and prior 28 days), Returning agent user (most recent and prior 7 days)`

Both returning-user columns are `0` or `1` indicators.

**Joins with:** `AgentMetadata.csv` on `AgentId`, and a Person query on
`PersonId` and `MetricDate` where both share a `PersonId` namespace.

The measures answer different questions and should stay distinct. Credits
measure consumption rather than agent quality or monetary cost. Responses
measure how often an agent responded rather than whether the response was useful
or completed a task. The two returning-user fields compare activity across
defined 28-day and 7-day windows, and neither should be read as daily retention.

## `Top_Performers_Dataset_v2.csv`

A small standalone teaching dataset used by `top-performers-rf.Rmd` and by the
Python `top-performers-rf.ipynb` notebook. It is not a query export and does not
join to the folders above.

`PersonId, Internal_network_size, Collaboration_hours, weekend_collaboration_hours, After_hours_call_hours, performance`

One row per person, with `performance` as the binary outcome label for the
random forest example.

---

## Reference notes

Everything below is background rather than anything you need in order to use the
files.

### Regenerating the fixtures

The Person, Consumption and GitHub fixtures are generated. The Agent sample is
not, and it is checked in as supplied.

```bash
cd examples/utility-r
Rscript simulate-query-exports.R
```

The script is deterministic, seeded at `20260916`, and it fails rather than
writing output if any ordered fixture schema or synthetic allocation invariant
breaks. Header vectors in `query-export-contracts.R` specify the nine generated
exports. Seven have a fully fixed header, while `PeopleMetaData.csv` and
`PersonQuery.csv` are driven by the selected HR attributes and the custom Person
query recipe. Changing the selected custom metrics or HR attributes requires an
explicit update to that contract.

### Formatting conventions

Dates are ISO `YYYY-MM-DD` on a UTC convention and do not record personal time
zones. Numbers use a decimal point with no thousands separators. Empty policy
and limit cells mean no policy in this simulation, although a real empty value
does not by itself establish policy absence.

The Person query covers 26 weeks while the product feeds cover the final 13. A
shorter product history than collaboration history is the normal real-world
case, so do not assume the windows always align.

### Category membership is persistent rather than a daily draw

Which features, languages, models and M365 services a simulated person appears
under is a stable property of their cohort, and only the volumes move day to
day. Three features are used by every adopting developer and each team adds two
more of its own, languages follow team ownership at three per team plus
`unknown`, and the enabled model list is organisation-wide because model
availability is a tenant-level setting. A developer revisits their whole profile
at least once every four weeks, so any four consecutive weeks contain the
complete cohort profile.

This matters for disclosure rather than realism alone. Real developers do not
re-roll their languages and models every morning, and a simulation that does
gives almost every person a unique pattern of cell membership, which means
linked cross-tabs built on it can never clear a group-size floor. The generator
asserts the floor over every cell, every complement and every membership
signature before writing, so an unpublishable fixture fails the build. Service
adoption in `PersonM365CreditsMetrics.csv` follows the same rule, and GitHub
provisioning, GitHub adoption and the Team × Role mix are fixed per team so that
published team cells and their complements clear the floor as well.

### Reconciliation properties of the breakdown files

In this simulation only, the four GitHub breakdown files are exact margins of
one underlying Feature × Language × Model allocation, so they reconcile with
each other, and the feature margin reconciles to `Code completions accepted`
plus `User-initiated chat requests` in the activity file. Model attribution,
including attribution to completion counts, is illustrative. Confirmed headers
do not establish these equalities or attribution rules for real exports, and the
report loader does not reject replacement data for failing them.

### Eligibility and coverage

No export file states eligibility or coverage, so keep these evidence classes
separate.

| Question | Real signal |
|---|---|
| Was this person eligible for M365 in this week? | `Total_Copilot_enabled_days > 0`. A static `IsCopilotLicensed` snapshot cannot override a zero-enabled week |
| Was this person provisioned for GitHub Copilot? | Independent provisioning evidence. Missing activity means unknown, and M365 licensing is not evidence of GitHub licensing |
| Was this person observed in GitHub? | Presence of a measured row. The demo checks observed weekday rows rather than verified ingestion completeness |
| Are this person-week's GitHub credits resolved? | The credit dates match the observed billable dates exactly, with no missing and no unexpected credit date |
| Is the sparse M365 export complete? | Unknown by default, even where positive rows are observed. Independent ingestion evidence is required |
| Did this person use Copilot on this day? | A row with a non-zero measure, rather than the absence of a row |

Fill an absent activity record with zero only once period eligibility and
independent ingestion completeness are both established. The demo helper accepts
an analyst-side evidence object through `derive_m365_coverage`, which is an
analyst input rather than a query-export column. Without it, M365 complete-week
and joint-product panels remain unavailable, and sparse source totals should be
labelled as observed records rather than full-window totals.

### Reporting guardrails

1. Confirm the row grain and uniqueness of every input before joining.
2. Align `MetricDate` to the same day, week, month or defined 28-day window, and
   never manufacture daily records from a weekly total.
3. Aggregate additive facts before joining them to person-period behavioural
   metrics.
4. Reconcile row counts and additive totals before and after each join, and
   report unmatched key counts without exposing identifiers.
5. Keep product eligibility and data coverage separate from observed zero usage,
   because a blank or missing record does not necessarily mean non-use.
6. Apply a minimum reporting group size of 10 and avoid individual rankings.
7. Treat relationships with collaboration or organisational measures as
   descriptive associations rather than causal evidence of productivity,
   wellbeing or agent effectiveness.

### Privacy and reuse

Identifiers are synthetic and exist only to demonstrate joins.
`PeopleHistoricalId` is minted independently of `PersonId`, so the crosswalk has
to be read rather than reconstructed. HR attribute names are defined by each
customer's own people data, so the specific attributes here are illustrative
even though the mechanism of appending analyst-selected attributes to each
output row is real. Every non-attribute column in this folder is a genuine
export column. Published aggregates in the accompanying reports require at least
10 distinct people. Before adapting any of this against real data, validate the
source contract and your organisation's privacy requirements.
