# Viva Insights query export schemas

Viva Insights "flexible queries" are configured in the Analyst portal and
downloaded as CSV. Each query **type** has a distinct grain (what one row means)
and column set. Knowing the grain first prevents most analysis mistakes.

## Query types and grain

| Query type | One row = | Key id column(s) | Typical use |
|---|---|---|---|
| **Person Query** | one person x one period (usually a week) | `PersonId`, `MetricDate` | Almost all metric, trend, segmentation, and driver analysis |
| **Meeting Query** | one meeting | `MeetingId` | Meeting quality, cost, subject-line text mining |
| **Person-to-Person (P2P)** | a tie between two people (edge) | `PrimaryCollaborator*`, `SecondaryCollaborator*` | `network_p2p()` |
| **Group-to-Group (G2G)** | collaboration between two groups | group-pair columns | `network_g2g()` |
| **Person-to-Group (P2G)** | a person's collaboration with a group | person + group columns | `network_summary()`, P2G views |

Always confirm the grain before aggregating. A Person Query is already
person-level. A Meeting Query is meeting-level and must be filtered and
aggregated before it means anything at the person or group level.

## Column naming: raw export vs imported

Raw exports use human-readable headers with spaces (e.g. `Collaboration hours`,
`Number of attendees`, `After hours collaboration hours`). After `import_query()`
these become underscore-joined tokens matching what package functions expect:

| Raw header (example) | After import |
|---|---|
| `Collaboration hours` | `Collaboration_hours` |
| `Meeting and call hours` | `Meeting_and_call_hours` |
| `After hours collaboration hours` | `After_hours_collaboration_hours` |
| `Number of attendees` | `Number_of_attendees` |
| `Intended participant count` | `Intended_participant_count` |

**Always feed package functions the imported (underscore) names.** If you loaded
the CSV with a plain reader instead of `import_query()`, the columns will still
have spaces and functions like `create_rank(metric = "Collaboration_hours")` will
fail to find the column.

## Person Query anatomy

Three column families:

1. **Identifiers / time:** `PersonId`, `MetricDate` (weekly by default).
2. **HR / organisational attributes:** e.g. `Organization`, `LevelDesignation`,
   `FunctionType`, `SupervisorIndicator`, `Region`, `IsManager`, hire date. Use
   `extract_hr()` to list them. These are the `hrvar=` grouping columns.
3. **Metrics:** collaboration, meeting, email, chat, call, focus, after-hours,
   network-size, and Copilot columns. Common examples: `Collaboration_hours`,
   `Meeting_and_call_hours`, `Email_hours`, `Chat_hours`, `Call_hours`,
   `After_hours_collaboration_hours`, `Internal_network_size`,
   `External_network_size`, `Copilot_actions_taken_in_Teams`.

Notes:
- Metrics are **per period** (per week). They are not cumulative.
- Many network / tie metrics are computed over a **trailing multi-week window**,
  which produces step plateaus. See `data-pitfalls.md`.
- `IsManager` comes through as `"Yes"`/`"No"` strings instead of booleans. See
  `data-pitfalls.md`.

## Meeting Query anatomy and the quality filter

A Meeting Query row is a calendar item, and roughly **half of raw rows are not
real collaborative meetings** (personal blocks, holds, all-day markers,
cancelled items, huge broadcasts). Apply a quality filter before any analysis.

Key columns (imported names): `MeetingId`, `Subject`, `Number_of_attendees`,
`Intended_participant_count`, `Attendee_meeting_hours`, `Duration` (where
present), `Cancelled`, `All_Day_Meeting`, `Recurring`.

**Generalised meeting quality filter** (adapt thresholds to the engagement):

```r
# R
mq <- mt_data %>%
  dplyr::filter(
    Number_of_attendees >= 2,            # drop solo holds / personal blocks
    Intended_participant_count > 1,
    Intended_participant_count <= 150,   # drop broadcasts / all-hands
    !as.logical(Cancelled),
    !as.logical(All_Day_Meeting)
    # if a Duration column exists (hours): Duration >= 1/60, Duration <= 8
  )
```

```python
# Python
mt = vi.load_mt_data()
mq = mt[(mt["Number_of_attendees"] >= 2)
        & (mt["Intended_participant_count"] > 1)
        & (mt["Intended_participant_count"] <= 150)
        & (~mt["Cancelled"].astype(bool))
        & (~mt["All_Day_Meeting"].astype(bool))]
```

**Critical unit gotcha:** where a `Duration` column is present it is measured in
**hours rather than minutes**. Multiply by 60 for a minutes display. `Meeting_hours`
and `Attendee_meeting_hours` are likewise in hours.

**Subject-line noise:** after the structural filter, non-meeting items still slip
through by subject (lunch, travel, OOO, personal holds). Exclude them with a
case-insensitive keyword/regex pass on `Subject`, and keep the exclusion list in
one place rather than inlining it repeatedly. Beware legitimate business terms
that look like noise in context (for example "flight" in a training programme
name, or a client name that matches a travel word), so review matches before
dropping.

## Network query anatomy

- **P2P** edges carry a primary and secondary collaborator plus tie strength.
  `network_p2p()` expects that edge shape and can compute centrality and detect
  communities.
- **G2G** rows carry a group pair and a collaboration weight. `network_g2g()`
  renders the group-level graph.
- Use the built-in `p2p_data` / `g2g_data` (and `p2p_data_sim()`) to prototype a
  network view before pointing at a real export.

## Quick schema check

```r
check_query(pq)          # variable types + structure
extract_hr(pq)           # which columns are HR attributes
extract_date_range(pq)   # period coverage
```
```python
vi.check_query(pq)
vi.extract_hr(pq)
vi.extract_date_range(pq)
```
