# Join Patterns — Viva Insights Data Sources

This document describes how to join different Viva Insights data sources together, including person query exports, Purview audit logs, external HR data, and license records. Joining data correctly is essential for enriched analysis but introduces several pitfalls around key matching, time alignment, and granularity differences.

> **Prerequisite reading:** Familiarize yourself with the [Person Query Data Dictionary](viva-person-query-data-dictionary.md) and [Purview Audit Data Dictionary](purview-audit-data-dictionary.md) before attempting cross-source joins.

---

## Overview of join scenarios

| Join | Left Source | Right Source | Join Keys | Difficulty |
|------|------------|-------------|-----------|------------|
| [Person query across time](#joining-person-query-data-across-time-periods) | Person query (period A) | Person query (period B) | `PersonId` | Low |
| [Person query ↔ Purview](#joining-person-query-with-purview-audit-data) | Person query | Purview audit logs | `PersonId` ↔ `UserId` | High |
| [Person query ↔ HR data](#joining-with-external-hr-data) | Person query | External HRIS export | `PersonId` ↔ `EmployeeId` | Medium |
| [Person query ↔ License data](#joining-with-license-assignment-data) | Person query | License assignment | `PersonId` ↔ `UserId`/`UPN` | Medium |

---

## Joining person query data across time periods

**Scenario:** You have two separate person query exports (e.g., Q1 and Q2) and want to combine them for longitudinal analysis.

**Join key:** `PersonId` (consistent across exports from the same tenant).

**Approach:** Stack (concatenate) the datasets vertically, then verify there are no duplicate `(PersonId, MetricDate)` pairs.

### Python

```python
import pandas as pd

q1 = pd.read_csv('person_query_q1.csv')
q2 = pd.read_csv('person_query_q2.csv')

# Stack the exports
combined = pd.concat([q1, q2], ignore_index=True)

# Check for duplicates
dupes = combined.duplicated(subset=['PersonId', 'MetricDate'], keep=False)
if dupes.any():
    raise ValueError(f"{dupes.sum()} duplicate person-period rows; resolve overlapping exports before joining")

print(f"Combined: {combined['PersonId'].nunique()} persons, "
      f"{combined['MetricDate'].nunique()} periods, "
      f"{len(combined)} total rows")
```

### R

```r
library(dplyr)
library(readr)

q1 <- read_csv("person_query_q1.csv")
q2 <- read_csv("person_query_q2.csv")

# Stack the exports
combined <- bind_rows(q1, q2)

# Check for duplicates
dupes <- combined |>
  group_by(PersonId, MetricDate) |>
  filter(n() > 1)

if (nrow(dupes) > 0) {
  stop(paste(nrow(dupes), "duplicate person-period rows; resolve overlapping exports before joining"))
}

message(paste("Combined:",
              n_distinct(combined$PersonId), "persons,",
              n_distinct(combined$MetricDate), "periods,",
              nrow(combined), "total rows"))
```

### Pitfalls

- **Column name mismatches:** If exports come from different time periods, column names may differ (e.g., a new metric was added). Use `bind_rows()` (R) or `pd.concat()` (Python) — both handle mismatched columns gracefully by inserting `NA`/`NaN`.
- **HR attribute changes:** A person's `Organization` or `LevelDesignation` may change between periods. This is expected — each row reflects their attributes at that `MetricDate`.
- **Overlapping date ranges:** Check duplicate keys and confirm an explicit reconciliation rule before proceeding. Do not retain an arbitrary first row.

---

## Joining person query with Purview audit data

**Scenario:** You want to enrich weekly person query summaries with event-level detail from Purview audit logs (e.g., which Copilot features each user used, what agents they invoked).

**Key challenge:** Person query data uses an anonymized `PersonId`, while Purview audit logs use the user's `UserId` (typically a UPN/email address). These are **not the same identifier**.

### Mapping strategy

There are two common approaches to resolve the `PersonId` ↔ `UserId` mapping:

#### Option A: Use a mapping table from the Viva Insights admin

If your Viva Insights admin can export a mapping table (`PersonId` → UPN), use it directly:

```python
# Python
mapping = pd.read_csv('person_id_mapping.csv')  # columns: PersonId, UPN
purview = pd.read_csv('purview_audit_cleaned.csv')

# Normalize the join key
mapping['UPN'] = mapping['UPN'].str.lower().str.strip()
purview['user_id'] = purview['user_id'].str.lower().str.strip()

# Join mapping to Purview, then join to person query
purview_mapped = purview.merge(mapping, left_on='user_id', right_on='UPN',
                               how='left', validate='many_to_one')
if purview_mapped['PersonId'].isna().any():
    raise ValueError("Unmatched identities; confirm the mapping before joining")
```

#### If no authorised mapping is available

Stop the person-level join and request a verified identity bridge. Shared HR
attributes are not unique identities; do not fuzzy-match employees. Separately
aggregated group comparisons may be possible, but are not linked-person evidence.

### Time alignment

Person query data is aggregated to **weekly** periods, while Purview audit logs are **event-level** (individual timestamps). To join them, you must align the time granularity:

```python
# Python — aggregate audit events to person-week
purview_mapped['event_week'] = purview_mapped['creation_time'].dt.to_period('W-SUN').apply(
    lambda x: x.start_time
)

purview_weekly = (
    purview_mapped
    .groupby(['PersonId', 'event_week'])
    .agg(
        copilot_events=('event_id', 'count'),
        unique_operations=('operation', 'nunique'),
        unique_apps=('copilot_app_host', 'nunique')
    )
    .reset_index()
)

# Now join with person query (after PersonId ↔ UserId mapping)
enriched = person_query.merge(
    purview_weekly,
    left_on=['PersonId', 'MetricDate'],
    right_on=['PersonId', 'event_week'],
    how='left',
    validate='one_to_one'
)
```

```r
# R — aggregate audit events to person-week
library(lubridate)

# Map Purview identities through the authorised identity bridge
purview_mapped <- purview |>
  left_join(mapping,
            by = c("user_id" = "UPN"),
            relationship = "many-to-one")
if (any(is.na(purview_mapped$PersonId))) {
  stop("Unmatched identities; confirm the mapping before joining")
}

purview_mapped <- purview_mapped |>
  mutate(event_week = floor_date(creation_time, unit = "week", week_start = 1))

purview_weekly <- purview_mapped |>
  group_by(PersonId, event_week) |>
  summarise(
    copilot_events = n(),
    unique_operations = n_distinct(operation),
    unique_apps = n_distinct(copilot_app_host),
    .groups = "drop"
  )

# Join with person query
enriched <- person_query |>
  left_join(purview_weekly,
            by = c("PersonId", "MetricDate" = "event_week"),
            relationship = "one-to-one")
```

### Pitfalls

- **Identifier mismatch is the #1 problem.** Never assume `PersonId == UserId`. They are different identifiers from different systems.
- **Week start alignment.** Viva Insights `MetricDate` is typically a Monday. Ensure your Purview weekly aggregation also uses Monday as the week start.
- **Many-to-one vs. many-to-many.** Each person should have one row per week in the person query. If your join produces more rows than the original person query, you have a many-to-many join — investigate duplicate keys.

---

## Joining with external HR data

**Scenario:** You want to add HR attributes not included in the Viva Insights export (e.g., cost center, hire cohort, performance rating) from an external HRIS.

**Common join keys:**

| Person Query Key | HR System Key | Notes |
|---|---|---|
| `PersonId` | `EmployeeId` | Requires a mapping table (PersonId is anonymized). |
| HR attributes (e.g., `Organization` + `LevelDesignation`) | Same attributes | Not a valid identity bridge; do not fuzzy-match people. |

### Python

```python
# Direct key join (with mapping table)
mapping = pd.read_csv('personid_to_employeeid.csv')
hr_data = pd.read_csv('hris_export.csv')

person_query_with_eid = person_query.merge(mapping, on='PersonId', how='left')
enriched = person_query_with_eid.merge(hr_data, on='EmployeeId', how='left')

# Check join success rate
match_rate = enriched['EmployeeId'].notna().mean()
print(f"Join match rate: {match_rate:.1%}")
```

### Pitfalls

- **Point-in-time HR data.** HR attributes change over time. If your HRIS export is a snapshot (e.g., "current state"), it will not reflect historical role changes. Ideally, use a versioned HR export with effective dates.
- **Key availability.** The `PersonId` in Viva Insights is anonymized — you need admin-provided mapping to link it to HRIS identifiers.
- **Duplicate matches.** If an employee has multiple records in the HRIS (e.g., one per role change), a join will produce duplicates. Filter the HRIS to the record effective during each `MetricDate` period.

---

## Joining with license assignment data

**Scenario:** You want to distinguish Copilot-licensed users from unlicensed users, or track when licenses were assigned/removed.

**Data source:** License assignment data typically comes from Azure AD / Entra ID exports or Microsoft 365 admin reports.

### Approach

```python
# Python
licenses = pd.read_csv('copilot_license_assignments.csv')
# Expected columns: UserId (UPN), LicenseAssignedDate, LicenseRemovedDate

# After mapping PersonId ↔ UserId:
person_query['is_licensed_by_assignment'] = person_query.apply(
    lambda row: (
        (row['LicenseAssignedDate'] <= row['MetricDate']) &
        ((pd.isna(row['LicenseRemovedDate'])) | (row['LicenseRemovedDate'] > row['MetricDate']))
    ), axis=1
)
```

### Validation

Cross-check the license data against the Copilot metric columns:

```python
# Users flagged as licensed by assignment data vs. by Copilot metric presence
copilot_cols = [c for c in df.columns if c.startswith('Copilot_')]
df['has_copilot_data'] = df[copilot_cols].notna().any(axis=1)

mismatch = df[df['is_licensed_by_assignment'] != df['has_copilot_data']]
print(f"Mismatches: {len(mismatch)} rows ({len(mismatch)/len(df):.1%})")
```

Small mismatches are expected (license activation delays, mid-week changes). Large mismatches indicate a mapping or data issue.

---

## Summary of time alignment considerations

| Data Source | Granularity | Time Key | Week Start |
|---|---|---|---|
| Person query | Weekly (or daily) | `MetricDate` | Typically Monday |
| Purview audit logs | Event-level | `CreationTime` | N/A (aggregate to match) |
| HR data | Snapshot or versioned | `EffectiveDate` (if available) | N/A |
| License data | Event-level (assign/remove) | `LicenseAssignedDate` | N/A |

**Rule of thumb:** Always aggregate event-level data to match the person query's granularity before joining. Never join event-level data directly to weekly summaries without aggregation — this produces row explosion.

---

## Checklist for coding agents

Before executing any join:

- [ ] Identify the join keys on both sides. Are they the same identifier type?
- [ ] Normalize only as the identity contract permits (for example case-insensitive UPNs). Preserve opaque PersonIds as strings; do not lowercase them or silently repair ambiguous identifiers.
- [ ] Check cardinality: is the join 1:1, 1:many, or many:many? Use `.merge()` with `validate='one_to_one'` or equivalent to catch surprises.
- [ ] Align time granularity: aggregate event-level data to the target period before joining.
- [ ] Check join match rate: what percentage of left-side rows matched? Low match rates indicate key problems.
- [ ] Verify row count: the output should not have significantly more rows than the left table (for left joins).
- [ ] Handle nulls: unmatched rows will have `NaN` in joined columns — decide whether to fill, flag, or drop.
