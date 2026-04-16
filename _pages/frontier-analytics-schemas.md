---
layout: page
title: "Frontier Analytics — Schema Documentation"
permalink: /frontier-analytics-schemas/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide any default Minima navigation that might appear */
.site-header .site-nav,
.trigger,
.page-link:not(.dropdown-toggle):not(.btn) {
  display: none !important;
}

/* Ensure our custom navigation is visible */
.custom-nav {
  display: block !important;
}
</style>

This page provides **practical schema guidance** for working with Viva Insights data exports. It focuses on structural concepts, join patterns, and common pitfalls — things that are not obvious from the official documentation alone.

For authoritative and up-to-date metric definitions and column references, see:

- [Viva Insights metric definitions (Microsoft Learn)](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
- [Copilot metrics taxonomy](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv)

> **On column names:** Column names can vary by tenant and localization settings (e.g., US vs UK English). Using `import_query()` from the `vivainsights` library (R or Python) is recommended because it replaces spaces and special characters with underscores, eliminating common issues when referring to column names in code. Use `extract_hr()` to programmatically identify the HR / organizational attribute columns available in your data.

**On this page:**

- [Person Query — Key Concepts](#person-query-key-concepts) — Panel structure, licensing logic, and what to watch for.
- [Purview Audit Logs — Key Concepts](#purview-audit-log-key-concepts) — Structure, the AuditData column, and data quality.
- [Join Patterns](#join-patterns) — How to join person query data with Purview audit logs, external HR data, and license records.
- [Common Pitfalls](#common-pitfalls) — Data quality issues, analytical mistakes, and edge cases to watch for.

**Related resources:**

- [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/) — ready-to-use prompts for coding agents
- [Frontier Analytics Overview]({{ site.baseurl }}/frontier-analytics/) — overview and getting started
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)

---

## Person Query — Key Concepts {#person-query-key-concepts}

### Panel structure

Person query data follows a **person-time panel** format:

- **Each row** represents one person for one time period (typically one week).
- **Primary key** is the combination of `(PersonId, MetricDate)` — each person appears once per time period.
- **Granularity** is usually **person-week** (one row per person per week), but some exports use **person-day**.
- A typical export contains `N_persons × N_periods` rows. For example, 5,000 employees over 12 weeks produces approximately 60,000 rows.

To detect granularity, check the gap between consecutive `MetricDate` values for a single person: a 7-day gap indicates weekly data; a 1-day gap indicates daily.

### Column categories

Person query exports typically contain these categories of columns. Exact column names vary by tenant — use `extract_hr()` and pattern matching (columns containing "Copilot") to identify them programmatically rather than hard-coding names.

| Category | Description |
|----------|-------------|
| **Identifiers** | `PersonId` (anonymized), `MetricDate` (period start date) |
| **HR attributes** | Organizational metadata (org, function, level, region, etc.) — configurable per tenant |
| **Collaboration metrics** | Meeting hours, email hours, chat hours, focus hours, network size, etc. |
| **Copilot metrics** | Copilot actions, assisted hours, chat queries, etc. — only populated for licensed users |

### Licensed vs. unlicensed vs. active

This distinction is critical for any Copilot adoption analysis:

| Status | Meaning | How to detect |
|--------|---------|---------------|
| **Unlicensed** | User does not have a Copilot license | All Copilot columns are `null` / `NA` |
| **Licensed but inactive** | User has a license but did not use Copilot | Copilot columns contain `0` values |
| **Active** | User has a license and used Copilot | Copilot activity metric > 0 |

> **Critical:** `null` ≠ `0` in Copilot columns. Replacing `NA` with `0` conflates unlicensed users with inactive licensed users, which inflates denominators and produces misleading adoption rates.

---

## Purview Audit Logs — Key Concepts {#purview-audit-log-key-concepts}

Purview audit logs capture user activity events across Microsoft 365 services, including Copilot interactions. They supplement person query data with granular event-level detail.

For authoritative documentation, see [Microsoft Learn — Audit log activities](https://learn.microsoft.com/en-us/purview/audit-log-activities).

### Key characteristics

| Property | Details |
|----------|---------|
| **Granularity** | One row per event (not aggregated) |
| **Export format** | Typically CSV or JSON from the Purview compliance portal |
| **Timestamps** | Always in UTC |
| **User identifiers** | UPN / email address (not anonymized like Viva Insights `PersonId`) |
| **Key challenge** | The `AuditData` column contains nested JSON that varies by event type |

### Working with AuditData

The `AuditData` column is a JSON string with event-specific details. Its structure varies by `Operation` and `Workload` — do not assume a consistent schema across all rows.

**Best practices:**
- **Explore before you extract.** Inspect key frequency across a sample before writing extraction logic.
- **Flatten one level only.** Extract top-level and first-level nested fields; store deeper nesting as JSON strings.
- **Wrap every parse in error handling.** Some rows may contain truncated or malformed JSON.
- **Filter by Workload first, then Operation.** This is more efficient than scanning all rows.

### Data quality considerations

- **Malformed JSON:** Always wrap JSON parsing in error handling — a single bad row should not crash the pipeline.
- **Mixed schemas:** Different `Operation` types produce completely different `AuditData` structures.
- **Encoding:** Purview CSV exports frequently use UTF-8 BOM encoding (`utf-8-sig`). If the first column name appears garbled, re-read with the correct encoding.
- **Volume:** Audit logs can be extremely large. For exports exceeding 500MB, use chunked reading.
- **User identifiers:** `UserId` is typically a UPN (email address), not the anonymized `PersonId` used in Viva Insights. See [Join Patterns](#join-patterns) for how to handle this.

---

## Join Patterns — Viva Insights Data Sources {#join-patterns}

This section describes how to join different Viva Insights data sources together. Joining data correctly is essential but introduces pitfalls around key matching, time alignment, and granularity differences.

### Overview of join scenarios

| Join | Left Source | Right Source | Join Keys | Difficulty |
|------|------------|-------------|-----------|------------|
| Person query across time | Person query (period A) | Person query (period B) | `PersonId` | Low |
| Person query ↔ Purview | Person query | Purview audit logs | `PersonId` ↔ `UserId` | High |
| Person query ↔ HR data | Person query | External HRIS export | `PersonId` ↔ `EmployeeId` | Medium |
| Person query ↔ License data | Person query | License assignment | `PersonId` ↔ `UserId`/`UPN` | Medium |

### Person query across time periods

Stack (concatenate) exports vertically using `PersonId` as the key. Check for duplicate `(PersonId, MetricDate)` pairs after stacking, especially if exports have overlapping date ranges.

**Watch for:** Column name mismatches between exports from different time periods (new metrics may have been added). Use `bind_rows()` (R) or `pd.concat()` (Python) — both handle mismatched columns by inserting `NA`/`NaN`.

### Person query ↔ Purview audit data

The key challenge is that person query data uses an anonymized `PersonId` while Purview uses `UserId` (UPN/email). These are **not the same identifier**.

**Mapping options:**
- **Mapping table from Viva Insights admin** (preferred) — a direct `PersonId` → UPN lookup. Normalize keys (lowercase, strip whitespace) before joining.
- **Fuzzy join on HR attributes** (fallback) — match using shared organizational attributes. Less reliable and should be used with caution.

**Time alignment:** Person query data is weekly; Purview logs are event-level. Aggregate audit events to the person-week level before joining. Ensure both use the same week-start day (typically Monday).

### External HR data and license data

Both follow the same pattern: you need a mapping table to link the anonymized `PersonId` to the external system's identifier. Key considerations:

- **Point-in-time HR data:** If your HRIS export is a snapshot, it won't reflect historical role changes. Use versioned exports with effective dates where possible.
- **License assignment timing:** Small mismatches between license assignment dates and Copilot metric presence are expected (activation delays, mid-week changes). Large mismatches indicate mapping issues.

### Checklist for joins

- Identify and normalize join keys on both sides (lowercase, strip whitespace)
- Verify cardinality: is the join 1:1, 1:many, or many:many?
- Align time granularity: aggregate event-level data to the target period before joining
- Check join match rate: low match rates indicate key problems
- Verify row count: output should not have significantly more rows than the left table (for left joins)

---

## Common Pitfalls — Viva Insights Data Analysis {#common-pitfalls}

These are the most common analytical mistakes when working with Viva Insights exports. Each is subtle and can invalidate results silently.

### 1. Panel structure misunderstanding

Treating rows as independent observations when they are a person-time panel. This leads to overstated sample sizes, invalid statistical tests, and incorrect averages. Always aggregate to the person level first, then summarize across persons.

### 2. Missing value confusion (NA ≠ zero)

Replacing `null` with `0` in Copilot columns conflates unlicensed and inactive users. This inflates adoption rate denominators and produces misleading trends. Create explicit `is_licensed` and `is_active` flags instead. See [Licensed vs. unlicensed vs. active](#licensed-vs-unlicensed-vs-active).

### 3. Survivorship bias

Assuming the same people appear in every time period. People leave, join, or go on leave during the analysis window. Check panel completeness and decide whether to use a balanced panel (same people throughout) or unbalanced panel (all data, accounting for entry/exit).

### 4. Small group suppression

Reporting metrics for groups with very few people, violating privacy thresholds. Viva Insights applies minimum group size thresholds (typically 5–10 users). Apply the same thresholds in custom analytics to protect individual privacy and ensure statistical reliability.

### 5. Time zone effects

Ignoring time zone differences when comparing collaboration metrics across geographies. After-hours metrics are relative to each person's configured working hours. Purview timestamps are UTC while person query periods may use the tenant's time zone. Document time zone assumptions in your output.

### 6. HR attribute changes over time

Assuming a person's organizational attributes are static. People change roles, teams, and levels over time — each row reflects their attributes at that `MetricDate`. Decide whether to use point-in-time attribution (each row's attributes as-is) or fixed-period attribution (attributes from a specific date).

### 7. Metric definition changes

Assuming metric definitions are consistent across all time periods. Microsoft periodically updates how metrics are calculated. Check for columns that appear only in a subset of periods, and note any sudden uniform shifts in metric distributions that may reflect methodology changes rather than behavior changes.

### 8. Double counting in aggregation

Aggregating metrics without respecting the panel structure. Common errors: summing hours across all rows (double-counts people across weeks), counting rows instead of unique persons, computing averages across all rows without per-person aggregation first.

### 9. Column name variations

Hard-coding column names from one tenant's export. The same metric may appear under different names due to tenant configuration or localization. Use `import_query()` for name cleaning and `extract_hr()` for attribute discovery instead of hard-coding names.
