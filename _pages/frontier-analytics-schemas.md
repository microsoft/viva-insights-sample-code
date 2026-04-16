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

This page consolidates the **data dictionaries and schema references** for the data sources used in Viva Insights analytics workflows. These documents describe the structure, column definitions, and data patterns you will encounter when working with Viva Insights exports and related data.

**How to use these docs:**

- **Building prompts:** Reference the data dictionaries when writing or adapting prompt cards. Understanding column names, types, and edge cases will produce better prompts.
- **Sharing with coding agents:** Paste the relevant schema documentation into your coding agent's context window (or point it at the file) before running an analytics prompt. This gives the agent the structural knowledge it needs to handle your data correctly.
- **Data validation:** Use the example rows and column descriptions to verify that your export matches the expected format before running any analysis.

> **Tip for coding agents:** You can prepend schema context to any prompt. For example: _"Refer to the data dictionary below for column definitions and data patterns, then execute the following analysis…"_ followed by the relevant schema content and the prompt.

**On this page:**

- [Person Query Data Dictionary](#person-query-data-dictionary) — Column definitions, data types, and value patterns for the Viva Insights person query export.
- [Purview Audit Data Dictionary](#purview-audit-data-dictionary) — Column definitions and nested JSON structure for Microsoft Purview unified audit log exports.
- [Join Patterns](#join-patterns) — How to join person query data with Purview audit logs, external HR data, and license records.
- [Common Pitfalls](#common-pitfalls) — Data quality issues, analytical mistakes, and edge cases to watch for.

**Important notes:**

- **Column names vary by tenant.** The column names documented here are the most common defaults. Your organization's Viva Insights configuration may use different names for the same metrics. Always inspect your actual column headers before running any analysis.
- **Metric definitions evolve.** Microsoft periodically updates how Viva Insights metrics are calculated. The descriptions here reflect common patterns but may not match every version.
- **These are reference documents, not specifications.** For authoritative and up-to-date metric definitions, refer to the [official Microsoft Viva Insights documentation](https://learn.microsoft.com/en-us/viva/insights/).

**Related resources:**

- [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/) — ready-to-use prompts for coding agents
- [Frontier Analytics Overview]({{ site.baseurl }}/frontier-analytics/) — overview and getting started
- [Viva Insights metric definitions (Microsoft Learn)](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [Source files on GitHub](https://github.com/microsoft/viva-insights-sample-code/tree/main/frontier-analytics/schemas/)

---

## Viva Insights Person Query — Data Dictionary {#person-query-data-dictionary}

This section describes the structure, columns, and data patterns of the **Viva Insights person query** export. Person query data is the primary data source for most Copilot adoption and collaboration analytics workflows.

> **Note:** Column names and available metrics vary by tenant configuration and Viva Insights version. Always verify your actual column headers against this reference. For authoritative metric definitions, see [Microsoft Learn — Viva Insights metrics](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics).

### Panel structure

Person query data follows a **person-time panel** format:

- **Each row** represents one person for one time period (typically one week).
- **Primary key** is the combination of `(PersonId, MetricDate)` — each person appears once per time period.
- **Granularity** is usually **person-week** (one row per person per week), but some exports use **person-day** (one row per person per day).
- A typical export contains `N_persons × N_periods` rows. For example, 5,000 employees over 12 weeks produces approximately 60,000 rows.

```
┌──────────────────────────────────────────────────────┐
│  PersonId   │ MetricDate │ Metric_A │ Metric_B │ ... │
├──────────────────────────────────────────────────────┤
│  person_001 │ 2024-01-01 │   12.5   │    3.2   │     │
│  person_001 │ 2024-01-08 │   10.0   │    4.1   │     │
│  person_001 │ 2024-01-15 │   14.2   │    2.8   │     │
│  person_002 │ 2024-01-01 │    8.3   │    5.0   │     │
│  person_002 │ 2024-01-08 │    9.1   │    4.7   │     │
│  ...        │ ...        │   ...    │   ...    │     │
└──────────────────────────────────────────────────────┘
```

#### How to detect granularity

To determine whether your data is person-week or person-day:

1. **Check the date intervals.** Sort by `MetricDate` for a single person and look at the gap between consecutive dates. A 7-day gap indicates weekly; a 1-day gap indicates daily.
2. **Count distinct dates.** If a 3-month export has ~13 distinct `MetricDate` values, it is weekly. If it has ~90, it is daily.

### Core identifier columns

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `PersonId` | string | Anonymized unique identifier for each person. | Consistent across time periods within the same export. Format varies by tenant (may be a hash, GUID, or encoded string). |
| `MetricDate` | date | Start date of the measurement period. | For weekly data, this is typically the Monday of each week. Format is usually `YYYY-MM-DD` or `M/D/YYYY`. Parse carefully. |

### HR attribute columns

HR attributes are organizational metadata associated with each person. These columns are **configurable** — your tenant's admin determines which HR attributes are included in the export. The values for a given person may change across time periods if the person changes roles, teams, or locations.

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `Organization` | string | The organizational unit (department, division, business group). | Most commonly used segmentation field. Name varies: `Organization`, `Org`, `Department`. |
| `FunctionType` | string | The job function category (Engineering, Marketing, Sales, etc.). | Sometimes called `JobFunction`, `Function`, or `Job_Function`. |
| `LevelDesignation` | string | The job level or seniority band (e.g., IC, Manager, Director, VP). | May be numeric (e.g., `60`, `63`, `67`) or descriptive (e.g., `Senior IC`, `Director`). |
| `SupervisorIndicator` | string | Whether the person is a manager or individual contributor. | Typical values: `Manager`, `Individual Contributor` (or `IC`). Sometimes called `IsManager`. |
| `Region` | string | Geographic region (e.g., Americas, EMEA, APAC). | May not be present in all tenants. |
| `Country` | string | Country of the person's primary work location. | Values may be ISO codes or full names depending on tenant config. |
| `City` | string | City of the person's primary work location. | Often suppressed for privacy in smaller organizations. |
| `TimeZone` | string | The person's configured time zone. | Important for interpreting collaboration hours across geographies. |
| `HireDate` | date | The person's hire date. | Useful for tenure-based analysis. May be null for long-tenured employees if not loaded. |

> **Note:** Tenants may include additional custom HR attributes (e.g., `CostCenter`, `BusinessUnit`, `EmployeeType`). Always inspect your column headers to discover what is available.

### Collaboration metric columns

Collaboration metrics measure how a person spends their work time. Values are typically in **hours per week** (for weekly data) or **hours per day** (for daily data), unless otherwise noted.

| Column | Type | Description | Typical Range | Notes |
|--------|------|-------------|---------------|-------|
| `Collaboration_Hours` | float | Total hours spent in meetings, emails, chats, and calls. | 5–50 hrs/wk | Sum of meeting, email, chat, and call hours. |
| `Meeting_Hours` | float | Hours spent in scheduled meetings (calendar events with at least one other attendee). | 2–30 hrs/wk | Derived from calendar data. |
| `Email_Hours` | float | Hours spent sending and reading emails. | 1–10 hrs/wk | Estimated from email send/receive activity. |
| `Chat_Hours` | float | Hours spent in Teams chat messages. | 0.5–8 hrs/wk | Includes 1:1 and group chats. |
| `Call_Hours` | float | Hours spent in unscheduled Teams calls. | 0–5 hrs/wk | Does not include scheduled meetings. |
| `Focus_Hours` | float | Available time blocks of ≥2 hours with no meetings or collaboration. | 0–30 hrs/wk | Inverse relationship with collaboration hours. |
| `After_Hours_Collaboration_Hours` | float | Collaboration hours outside the person's configured working hours. | 0–15 hrs/wk | Depends on the tenant's working-hours definition. |
| `Workweek_Span` | float | Time between the first and last activity of each workday, averaged over the week. | 8–14 hrs | Indicates the length of the working day. |
| `Internal_Network_Size` | integer | Count of distinct internal colleagues the person had meaningful interactions with. | 5–200 | "Meaningful" defined by Viva Insights (≥2 interactions in the period). |
| `External_Network_Size` | integer | Count of distinct external contacts the person interacted with. | 0–50 | Depends on external collaboration patterns. |
| `Conflicting_Meeting_Hours` | float | Hours spent in meetings that overlap with other meetings on the calendar. | 0–10 hrs/wk | Indicator of meeting overload. |
| `Multitasking_Hours` | float | Meeting hours during which the person was also sending emails or chats. | 0–10 hrs/wk | Proxy for meeting disengagement. |
| `Manager_Coaching_Hours_1on1` | float | Hours spent in 1:1 meetings with direct reports (for managers). | 0–5 hrs/wk | Null or zero for individual contributors. |
| `Networking_Outside_Organization` | float | Hours spent collaborating with people in other organizations/departments. | 0–20 hrs/wk | Cross-functional collaboration indicator. |

> **Note:** The full set of available metrics depends on your Viva Insights license tier and tenant configuration. Some metrics (e.g., `Networking_Outside_Organization`) may not be present in all exports.

### Copilot metric columns

Copilot metrics measure Microsoft 365 Copilot usage. These columns are **only populated for licensed Copilot users**. For unlicensed users, values are `NA` / `null` / `NaN` — **not zero**.

| Column | Type | Description | Typical Range | Notes |
|--------|------|-------------|---------------|-------|
| `Copilot_Actions` | integer | Total number of Copilot actions (prompts, completions, suggestions accepted) in the period. | 0–500/wk | Primary activity volume metric. Zero means licensed but inactive. |
| `Copilot_Assisted_Hours` | float | Estimated hours where Copilot assisted the user's work. | 0–20 hrs/wk | Derived metric; methodology may vary by version. |
| `Copilot_Chat_Queries` | integer | Number of queries sent to Copilot chat interfaces (e.g., M365 Chat, in-app chat). | 0–200/wk | Subset of Copilot_Actions focused on conversational interactions. |
| `Copilot_Summarized_Hours` | float | Hours of meetings or content summarized by Copilot on the user's behalf. | 0–10 hrs/wk | Includes meeting summaries, email summaries, document summaries. |
| `Copilot_Assisted_Meeting_Hours` | float | Meeting hours where Copilot features were used (e.g., transcription summary, recap). | 0–15 hrs/wk | Subset of Meeting_Hours where Copilot was active. |
| `Copilot_Assisted_Email_Hours` | float | Email hours where Copilot assisted with drafting or summarizing. | 0–5 hrs/wk | |
| `Copilot_Assisted_Document_Hours` | float | Hours of document work where Copilot assisted (e.g., in Word, PowerPoint, Excel). | 0–10 hrs/wk | |

> **Important:** A `null`/`NA` value in a Copilot column means the user was **not licensed** for Copilot that week. A value of `0` means the user **was licensed but did not use Copilot** that week. This distinction is critical for adoption analysis.

### Example rows

The following table shows example rows with fake data to illustrate the structure. Only a subset of columns is shown for readability.

| PersonId | MetricDate | Organization | LevelDesignation | Collaboration_Hours | Meeting_Hours | Focus_Hours | Copilot_Actions | Copilot_Assisted_Hours |
|----------|------------|-------------|-----------------|--------------------|--------------|-----------|-----------------|-----------------------|
| `abc123` | 2024-09-02 | Engineering | Senior IC | 28.5 | 15.2 | 12.0 | 45 | 3.2 |
| `abc123` | 2024-09-09 | Engineering | Senior IC | 32.1 | 18.7 | 8.5 | 62 | 4.8 |
| `def456` | 2024-09-02 | Marketing | Manager | 35.0 | 22.3 | 5.0 | | |
| `def456` | 2024-09-09 | Marketing | Manager | 30.8 | 19.1 | 9.2 | | |
| `ghi789` | 2024-09-02 | Sales | IC | 22.4 | 10.5 | 18.0 | 12 | 0.8 |
| `ghi789` | 2024-09-09 | Sales | IC | 25.1 | 12.8 | 15.0 | 0 | 0.0 |

**Reading the example:**

- **`abc123`** (Engineering, Senior IC) is a licensed Copilot user who was active both weeks.
- **`def456`** (Marketing, Manager) is **not licensed** for Copilot — Copilot columns are blank (null).
- **`ghi789`** (Sales, IC) is a licensed user who was active week 1 but **inactive** week 2 (Copilot_Actions = 0, not null).

### Special notes for coding agents

#### Detecting time granularity

```python
# Python
dates = df['MetricDate'].sort_values().unique()
gap = (pd.to_datetime(dates[1]) - pd.to_datetime(dates[0])).days
granularity = 'weekly' if gap >= 7 else 'daily'
```

```r
# R
dates <- sort(unique(df$MetricDate))
gap <- as.numeric(difftime(dates[2], dates[1], units = "days"))
granularity <- if (gap >= 7) "weekly" else "daily"
```

#### Identifying licensed vs. unlicensed users

A user is **Copilot-licensed** in a given period if any Copilot metric column has a non-null value for that row. A user is **active** if they are licensed AND have `Copilot_Actions > 0`.

```python
# Python
copilot_cols = [c for c in df.columns if c.startswith('Copilot_')]
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)
df['is_active'] = df['is_licensed'] & (df['Copilot_Actions'] > 0)
```

```r
# R
copilot_cols <- grep("^Copilot_", names(df), value = TRUE)
df$is_licensed <- rowSums(!is.na(df[copilot_cols])) > 0
df$is_active <- df$is_licensed & !is.na(df$Copilot_Actions) & df$Copilot_Actions > 0
```

#### Validating panel completeness

Check that each person appears in every time period. Missing person-periods may indicate employees joining or leaving.

```python
# Python
expected = df['PersonId'].nunique() * df['MetricDate'].nunique()
actual = len(df)
completeness = actual / expected
print(f"Panel completeness: {completeness:.1%} ({actual}/{expected})")
```

#### Common column name variations

The same metric may appear under different column names depending on the tenant's Viva Insights configuration:

| Standard Name | Known Variations |
|---|---|
| `Organization` | `Org`, `Department`, `Business_Unit` |
| `FunctionType` | `JobFunction`, `Function`, `Job_Function` |
| `LevelDesignation` | `Level`, `JobLevel`, `Level_Designation`, `Career_Stage` |
| `SupervisorIndicator` | `IsManager`, `Manager_Indicator`, `Supervisor_Indicator` |
| `Collaboration_Hours` | `Total_Collaboration_Hours`, `Collab_Hours` |
| `Copilot_Actions` | `Copilot_Total_Actions`, `M365_Copilot_Actions` |
| `MetricDate` | `Date`, `Week_Start_Date`, `Period_Start` |
| `PersonId` | `Person_Id`, `HashId`, `EmployeeId` |

> **Tip for coding agents:** When loading data, auto-detect Copilot metric columns by searching for columns whose names contain `Copilot` (case-insensitive). Similarly, identify the date column by looking for columns containing `Date` or `Period` and validating that values parse as dates.

---

## Purview Audit Log — Data Dictionary {#purview-audit-data-dictionary}

This section describes the structure, columns, and data patterns of **Microsoft Purview unified audit log** exports. Purview audit logs capture user activity events across Microsoft 365 services, including Copilot interactions, and are used to supplement Viva Insights person query data with granular event-level detail.

> **Note:** Purview audit log schemas are **less standardized** than Viva Insights person query data. The available fields, nested structures, and operation names vary significantly across tenants and event types. Treat this document as a starting reference — always explore your actual data before building analysis pipelines. For authoritative documentation, see [Microsoft Learn — Audit log activities](https://learn.microsoft.com/en-us/purview/audit-log-activities).

### Overview

| Property | Details |
|----------|---------|
| **What it captures** | User and admin activity events across Microsoft 365 services (SharePoint, Exchange, Teams, Copilot, etc.). |
| **Granularity** | One row per event (not aggregated). |
| **Export format** | Typically CSV or JSON, exported from the Purview compliance portal or via PowerShell. |
| **Volume** | Can be very large — thousands to millions of rows per day depending on organization size and activity scope. |
| **Time zone** | Timestamps are in UTC. |
| **Key challenge** | The `AuditData` column contains a nested JSON string with event-specific details that must be parsed. |

### Top-level columns

These columns appear directly in the export (outside of the `AuditData` JSON).

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `CreationTime` | datetime | UTC timestamp when the event was recorded. | Primary timestamp. Format varies: `YYYY-MM-DDTHH:MM:SS`, `M/D/YYYY H:MM:SS AM`, etc. |
| `UserId` | string | The user principal name (UPN) or email address of the user who performed the action. | Not anonymized — this is typically the full email address. May need hashing for privacy compliance. |
| `Operation` | string | The name of the activity performed (e.g., `FileAccessed`, `CopilotInteraction`, `MailItemsAccessed`). | This is the primary field for filtering event types. Values are not always intuitive. |
| `Workload` | string | The Microsoft 365 service where the event occurred (e.g., `SharePoint`, `Exchange`, `MicrosoftTeams`, `Copilot`). | Useful for high-level filtering before looking at specific operations. |
| `ObjectId` | string | The resource involved in the event (e.g., a file URL, mailbox identifier, or Teams channel). | May be null for some event types. |
| `RecordType` | integer | Numeric code indicating the type of audit record. | Different record types have different AuditData schemas. See [Microsoft's record type reference](https://learn.microsoft.com/en-us/office/office-365-management-api/office-365-management-activity-api-schema#auditlogrecordtype). |
| `ResultStatus` | string | Whether the operation succeeded or failed. | Typical values: `Succeeded`, `Failed`, `PartiallySucceeded`. |
| `UserKey` | string | An alternative user identifier (may be a GUID or PUID). | Sometimes present alongside `UserId`. Useful when `UserId` is null. |
| `UserType` | integer | Numeric code for the type of user (e.g., 0 = regular user, 2 = admin, 4 = system). | |
| `ClientIP` | string | The IP address of the client that triggered the event. | May be IPv4 or IPv6. Consider privacy before using in analysis. |

### The AuditData column

The `AuditData` column is the most important and most complex part of Purview exports. It contains a **JSON string** with event-specific details. The structure of this JSON varies by `Operation` and `Workload`.

#### Parsing approach

```python
# Python — parse AuditData JSON
import json
import pandas as pd

df['audit_parsed'] = df['AuditData'].apply(
    lambda x: json.loads(x) if pd.notna(x) else {}
)

# Inspect top-level keys across a sample
from collections import Counter
key_counts = Counter()
for record in df['audit_parsed'].head(500):
    key_counts.update(record.keys())
print(key_counts.most_common(30))
```

```r
# R — parse AuditData JSON
library(jsonlite)
library(purrr)

df$audit_parsed <- map(df$AuditData, ~ tryCatch(
  fromJSON(.x, simplifyVector = FALSE),
  error = function(e) list()
))

# Inspect top-level keys across a sample
keys <- unlist(map(head(df$audit_parsed, 500), names))
sort(table(keys), decreasing = TRUE)[1:30]
```

#### Common AuditData fields

These fields are frequently present in the parsed JSON, though not all appear in every record:

| Field | Type | Description | Frequency |
|-------|------|-------------|-----------|
| `Id` | string | Unique event identifier (GUID). | Nearly all records |
| `CreationTime` | datetime | Event timestamp (may match the outer `CreationTime`). | Nearly all records |
| `Operation` | string | Activity name (may be more specific than the outer `Operation`). | Nearly all records |
| `OrganizationId` | string | GUID of the tenant. | Nearly all records |
| `UserKey` | string | Alternative user identifier. | Common |
| `Workload` | string | Service name. | Common |
| `ClientIP` | string | Client IP address. | Common |
| `ObjectId` | string | Resource identifier (URL, mailbox, etc.). | Varies by event type |
| `ItemType` | string | Type of object involved (e.g., `File`, `Folder`, `Page`). | SharePoint events |
| `AppAccessContext` | object | Contains `AADSessionId`, `CorrelationId`, and other auth context. | Common in Copilot events |
| `CopilotEventData` | object | Copilot-specific event details (see below). | Copilot events only |

### Copilot-specific fields

For Copilot-related events, the `AuditData` JSON typically contains a `CopilotEventData` object (or similar) with details about the Copilot interaction.

> **Important:** The exact field names and structure vary across tenants and Copilot versions. The table below represents commonly observed fields.

| Field Path | Type | Description |
|------------|------|-------------|
| `CopilotEventData.AppHost` | string | The application hosting Copilot (e.g., `Word`, `Teams`, `Outlook`, `M365Chat`). |
| `CopilotEventData.AgentName` | string | Name of the Copilot agent or extension used (if applicable). |
| `CopilotEventData.PluginName` | string | Name of the plugin invoked during the interaction. |
| `CopilotEventData.ExtensionId` | string | Identifier of the Copilot extension. |
| `CopilotEventData.InteractionType` | string | Type of interaction (e.g., `Chat`, `Summarize`, `Draft`, `Rewrite`). |
| `CopilotEventData.AccessedResources` | array | List of resources Copilot accessed during the interaction. |
| `CopilotEventData.Contexts` | array | Contextual information provided to Copilot. |
| `CopilotEventData.ThreadId` | string | Identifier for the conversation thread. |

#### Common Copilot-related operations

| Operation | Workload | Description |
|-----------|----------|-------------|
| `CopilotInteraction` | `Copilot` | A user interacted with Copilot in any M365 app. |
| `CopilotQuery` | `Copilot` | A query was sent to Copilot. |
| `MicrosoftCopilotForM365` | varies | Copilot activity within M365 apps. |

> **Note:** Operation names for Copilot events are not fully standardized. Some tenants may use different names. When filtering, use case-insensitive substring matching against keywords: `Copilot`, `AI`, `Agent`, `GPT`, `Assist`, `Summarize`.

### Example rows

#### Raw CSV row (simplified)

```
CreationTime,UserId,Operation,Workload,AuditData
2024-09-15T14:23:45,user@contoso.com,CopilotInteraction,Copilot,"{""Id"":""a1b2c3..."",""Operation"":""CopilotInteraction"",""UserId"":""user@contoso.com"",""CopilotEventData"":{""AppHost"":""Word"",""InteractionType"":""Draft"",""AgentName"":null}}"
2024-09-15T14:25:10,user@contoso.com,FileAccessed,SharePoint,"{""Id"":""d4e5f6..."",""Operation"":""FileAccessed"",""ObjectId"":""https://contoso.sharepoint.com/sites/Team/Doc.docx"",""ItemType"":""File""}"
2024-09-15T15:01:33,admin@contoso.com,UserLoggedIn,AzureActiveDirectory,"{""Id"":""g7h8i9..."",""Operation"":""UserLoggedIn"",""ClientIP"":""10.0.0.1"",""ResultStatus"":""Succeeded""}"
```

#### Parsed and flattened output

| event_id | creation_time | user_id | operation | workload | event_category | copilot_app_host | copilot_interaction_type | copilot_agent_name | object_id |
|----------|--------------|---------|-----------|----------|---------------|-----------------|------------------------|--------------------|-----------|
| a1b2c3… | 2024-09-15 14:23:45 | user@contoso.com | CopilotInteraction | Copilot | copilot | Word | Draft | | |
| d4e5f6… | 2024-09-15 14:25:10 | user@contoso.com | FileAccessed | SharePoint | user_activity | | | | https://contoso.sharepoint.com/… |
| g7h8i9… | 2024-09-15 15:01:33 | admin@contoso.com | UserLoggedIn | AzureActiveDirectory | user_activity | | | | |

### Data quality considerations

- **Malformed JSON:** Some `AuditData` rows may contain truncated or invalid JSON, especially in large exports. Always wrap JSON parsing in error handling.
- **Mixed schemas:** Different `Operation` types produce completely different `AuditData` structures. Do not assume a flat, consistent schema across all rows.
- **Timestamps:** `CreationTime` is always UTC. The format may vary within the same export (rare but possible). Parse with flexible datetime parsers.
- **User identifiers:** `UserId` is typically a UPN (email address), not the anonymized `PersonId` used in Viva Insights. See [Join Patterns](#join-patterns) for how to handle this mapping.
- **Volume:** Audit logs can be extremely large. For exports exceeding 500MB, use chunked reading:

```python
# Python — chunked reading
for chunk in pd.read_csv('audit_log.csv', chunksize=100_000, encoding='utf-8-sig'):
    # process each chunk
    pass
```

- **Encoding:** Purview CSV exports frequently use UTF-8 BOM encoding (`utf-8-sig`). If the first column name appears garbled, re-read with the correct encoding.

### Tips for coding agents

1. **Explore before you extract.** Purview schemas vary. Always inspect the data (column names, sample rows, AuditData key frequency) before writing extraction logic.
2. **Use error handling everywhere.** Wrap every JSON parse and type conversion in try-except blocks. A single malformed row should not crash the pipeline.
3. **Flatten one level only.** Extract top-level and first-level nested fields from `AuditData`. Store deeper nesting as JSON strings to avoid column explosion.
4. **Use snake_case for output columns.** Prefix AuditData-derived columns with their context (e.g., `copilot_agent_name` not just `agent_name`) to avoid collisions.
5. **Filter by Workload first, then Operation.** This is more efficient than scanning all rows for specific Operation values.
6. **Protect privacy.** Do not print raw `UserId` values in output. Use aggregates, truncated identifiers, or hashing when displaying user-level information.
7. **Generate a data dictionary.** After parsing, document the columns you extracted — see the [Prompt Card Library]({{ site.baseurl }}/frontier-analytics-prompts/) for a prompt that does this automatically.

---

## Join Patterns — Viva Insights Data Sources {#join-patterns}

This section describes how to join different Viva Insights data sources together, including person query exports, Purview audit logs, external HR data, and license records. Joining data correctly is essential for enriched analysis but introduces several pitfalls around key matching, time alignment, and granularity differences.

> **Prerequisite reading:** Familiarize yourself with the [Person Query Data Dictionary](#person-query-data-dictionary) and [Purview Audit Data Dictionary](#purview-audit-data-dictionary) before attempting cross-source joins.

### Overview of join scenarios

| Join | Left Source | Right Source | Join Keys | Difficulty |
|------|------------|-------------|-----------|------------|
| [Person query across time](#joining-person-query-data-across-time-periods) | Person query (period A) | Person query (period B) | `PersonId` | Low |
| [Person query ↔ Purview](#joining-person-query-with-purview-audit-data) | Person query | Purview audit logs | `PersonId` ↔ `UserId` | High |
| [Person query ↔ HR data](#joining-with-external-hr-data) | Person query | External HRIS export | `PersonId` ↔ `EmployeeId` | Medium |
| [Person query ↔ License data](#joining-with-license-assignment-data) | Person query | License assignment | `PersonId` ↔ `UserId`/`UPN` | Medium |

### Joining person query data across time periods

**Scenario:** You have two separate person query exports (e.g., Q1 and Q2) and want to combine them for longitudinal analysis.

**Join key:** `PersonId` (consistent across exports from the same tenant).

**Approach:** Stack (concatenate) the datasets vertically, then verify there are no duplicate `(PersonId, MetricDate)` pairs.

#### Python

```python
import pandas as pd

q1 = pd.read_csv('person_query_q1.csv')
q2 = pd.read_csv('person_query_q2.csv')

# Stack the exports
combined = pd.concat([q1, q2], ignore_index=True)

# Check for duplicates
dupes = combined.duplicated(subset=['PersonId', 'MetricDate'], keep=False)
if dupes.any():
    print(f"Warning: {dupes.sum()} duplicate person-periods found")
    combined = combined.drop_duplicates(subset=['PersonId', 'MetricDate'], keep='first')

print(f"Combined: {combined['PersonId'].nunique()} persons, "
      f"{combined['MetricDate'].nunique()} periods, "
      f"{len(combined)} total rows")
```

#### R

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
  message(paste("Warning:", nrow(dupes), "duplicate person-periods found"))
  combined <- combined |> distinct(PersonId, MetricDate, .keep_all = TRUE)
}

message(paste("Combined:",
              n_distinct(combined$PersonId), "persons,",
              n_distinct(combined$MetricDate), "periods,",
              nrow(combined), "total rows"))
```

#### Pitfalls

- **Column name mismatches:** If exports come from different time periods, column names may differ (e.g., a new metric was added). Use `bind_rows()` (R) or `pd.concat()` (Python) — both handle mismatched columns gracefully by inserting `NA`/`NaN`.
- **HR attribute changes:** A person's `Organization` or `LevelDesignation` may change between periods. This is expected — each row reflects their attributes at that `MetricDate`.
- **Overlapping date ranges:** If both exports cover the same weeks, you will get duplicate rows. Always check for and remove duplicates after stacking.

### Joining person query with Purview audit data

**Scenario:** You want to enrich weekly person query summaries with event-level detail from Purview audit logs (e.g., which Copilot features each user used, what agents they invoked).

**Key challenge:** Person query data uses an anonymized `PersonId`, while Purview audit logs use the user's `UserId` (typically a UPN/email address). These are **not the same identifier**.

#### Mapping strategy

There are two common approaches to resolve the `PersonId` ↔ `UserId` mapping:

**Option A: Use a mapping table from the Viva Insights admin**

If your Viva Insights admin can export a mapping table (`PersonId` → UPN), use it directly:

```python
# Python
mapping = pd.read_csv('person_id_mapping.csv')  # columns: PersonId, UPN
purview = pd.read_csv('purview_audit_cleaned.csv')

# Normalize the join key
mapping['UPN'] = mapping['UPN'].str.lower().str.strip()
purview['user_id'] = purview['user_id'].str.lower().str.strip()

# Join mapping to Purview, then join to person query
purview_mapped = purview.merge(mapping, left_on='user_id', right_on='UPN', how='left')
```

**Option B: Join on HR attributes as a fuzzy bridge**

If no direct mapping is available, you can attempt a fuzzy join using shared HR attributes (Organization, LevelDesignation, etc.) combined with time-based aggregation. This is **less reliable** and should be used with caution.

#### Time alignment

Person query data is aggregated to **weekly** periods, while Purview audit logs are **event-level** (individual timestamps). To join them, you must align the time granularity:

```python
# Python — aggregate audit events to person-week
purview['event_week'] = purview['creation_time'].dt.to_period('W-MON').apply(
    lambda x: x.start_time
)

purview_weekly = (
    purview
    .groupby(['user_id', 'event_week'])
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
    right_on=['user_id', 'event_week'],
    how='left'
)
```

```r
# R — aggregate audit events to person-week
library(lubridate)

purview <- purview |>
  mutate(event_week = floor_date(creation_time, unit = "week", week_start = 1))

purview_weekly <- purview |>
  group_by(user_id, event_week) |>
  summarise(
    copilot_events = n(),
    unique_operations = n_distinct(operation),
    unique_apps = n_distinct(copilot_app_host),
    .groups = "drop"
  )

# Join with person query (after PersonId - UserId mapping)
enriched <- person_query |>
  left_join(purview_weekly,
            by = c("PersonId" = "user_id", "MetricDate" = "event_week"))
```

#### Pitfalls

- **Identifier mismatch is the #1 problem.** Never assume `PersonId == UserId`. They are different identifiers from different systems.
- **Week start alignment.** Viva Insights `MetricDate` is typically a Monday. Ensure your Purview weekly aggregation also uses Monday as the week start.
- **Many-to-one vs. many-to-many.** Each person should have one row per week in the person query. If your join produces more rows than the original person query, you have a many-to-many join — investigate duplicate keys.

### Joining with external HR data

**Scenario:** You want to add HR attributes not included in the Viva Insights export (e.g., cost center, hire cohort, performance rating) from an external HRIS.

**Common join keys:**

| Person Query Key | HR System Key | Notes |
|---|---|---|
| `PersonId` | `EmployeeId` | Requires a mapping table (PersonId is anonymized). |
| HR attributes (e.g., `Organization` + `LevelDesignation`) | Same attributes | Fuzzy / probabilistic — not recommended as a primary join. |

#### Python

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

#### Pitfalls

- **Point-in-time HR data.** HR attributes change over time. If your HRIS export is a snapshot (e.g., "current state"), it will not reflect historical role changes. Ideally, use a versioned HR export with effective dates.
- **Key availability.** The `PersonId` in Viva Insights is anonymized — you need admin-provided mapping to link it to HRIS identifiers.
- **Duplicate matches.** If an employee has multiple records in the HRIS (e.g., one per role change), a join will produce duplicates. Filter the HRIS to the record effective during each `MetricDate` period.

### Joining with license assignment data

**Scenario:** You want to distinguish Copilot-licensed users from unlicensed users, or track when licenses were assigned/removed.

**Data source:** License assignment data typically comes from Azure AD / Entra ID exports or Microsoft 365 admin reports.

#### Approach

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

#### Validation

Cross-check the license data against the Copilot metric columns:

```python
# Users flagged as licensed by assignment data vs. by Copilot metric presence
copilot_cols = [c for c in df.columns if c.startswith('Copilot_')]
df['has_copilot_data'] = df[copilot_cols].notna().any(axis=1)

mismatch = df[df['is_licensed_by_assignment'] != df['has_copilot_data']]
print(f"Mismatches: {len(mismatch)} rows ({len(mismatch)/len(df):.1%})")
```

Small mismatches are expected (license activation delays, mid-week changes). Large mismatches indicate a mapping or data issue.

### Summary of time alignment considerations

| Data Source | Granularity | Time Key | Week Start |
|---|---|---|---|
| Person query | Weekly (or daily) | `MetricDate` | Typically Monday |
| Purview audit logs | Event-level | `CreationTime` | N/A (aggregate to match) |
| HR data | Snapshot or versioned | `EffectiveDate` (if available) | N/A |
| License data | Event-level (assign/remove) | `LicenseAssignedDate` | N/A |

**Rule of thumb:** Always aggregate event-level data to match the person query's granularity before joining. Never join event-level data directly to weekly summaries without aggregation — this produces row explosion.

### Checklist for coding agents

Before executing any join:

- [ ] Identify the join keys on both sides. Are they the same identifier type?
- [ ] Normalize keys: lowercase, strip whitespace, consistent format.
- [ ] Check cardinality: is the join 1:1, 1:many, or many:many? Use `.merge()` with `validate='one_to_one'` or equivalent to catch surprises.
- [ ] Align time granularity: aggregate event-level data to the target period before joining.
- [ ] Check join match rate: what percentage of left-side rows matched? Low match rates indicate key problems.
- [ ] Verify row count: the output should not have significantly more rows than the left table (for left joins).
- [ ] Handle nulls: unmatched rows will have `NaN` in joined columns — decide whether to fill, flag, or drop.

---

## Common Pitfalls — Viva Insights Data Analysis {#common-pitfalls}

This section catalogs the most common data quality issues, analytical mistakes, and edge cases that arise when working with Viva Insights exports. Each pitfall includes an explanation of why it happens, how to detect it, and how to fix or avoid it.

> **For coding agents:** If you are processing Viva Insights data, review this list before starting your analysis. Many of these pitfalls are subtle and can invalidate results silently.

### 1. Panel structure misunderstanding

**The mistake:** Treating rows as independent observations when they are a person-time panel.

**Why it matters:** Person query data has a `(PersonId, MetricDate)` panel structure — each person appears once per time period. Ignoring this leads to:
- Overstating sample sizes (counting person-weeks as "people")
- Invalid statistical tests (observations are not independent — the same person appears multiple times)
- Incorrect aggregation (e.g., computing an average across all rows instead of per-person first, then across persons)

**How to detect:**

```python
# Check: are there repeated PersonIds?
persons = df['PersonId'].nunique()
rows = len(df)
print(f"{rows} rows, {persons} unique persons → {rows/persons:.1f} periods per person on average")
```

**How to fix:** Always aggregate to the appropriate level before analysis. For cross-sectional summaries, compute per-person averages first, then summarize across persons:

```python
# Wrong: mean across all rows (mixes within-person and between-person variation)
wrong_avg = df['Collaboration_Hours'].mean()

# Right: person-level mean first, then grand mean
person_avg = df.groupby('PersonId')['Collaboration_Hours'].mean()
correct_avg = person_avg.mean()
```

### 2. Missing value confusion (NA ≠ zero)

**The mistake:** Treating `NA`/`null` values in Copilot metric columns as zero.

**Why it matters:** In Viva Insights person query data:
- `NA` / `null` in a Copilot column → the user was **not licensed** for Copilot that period
- `0` in a Copilot column → the user **was licensed but did not use** Copilot that period

Replacing `NA` with `0` conflates unlicensed users with inactive licensed users, which:
- Inflates the denominator for adoption rate calculations
- Makes "average usage per user" appear artificially low
- Produces misleading trends (e.g., "usage is declining" when actually licenses were just revoked)

**How to detect:**

```python
copilot_cols = [c for c in df.columns if c.startswith('Copilot_')]
for col in copilot_cols:
    null_count = df[col].isna().sum()
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {null_count} nulls, {zero_count} zeros")
```

**How to fix:** Never use `fillna(0)` on Copilot columns. Instead, create explicit flags:

```python
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)
df['is_active'] = df['is_licensed'] & (df['Copilot_Actions'] > 0)

# Compute adoption rate correctly
adoption = df[df['is_licensed']].groupby('MetricDate')['is_active'].mean()
```

### 3. Survivorship bias

**The mistake:** Assuming the same set of people appears in every time period.

**Why it matters:** People leave the organization, go on extended leave, or join during the analysis window. If you only look at people present at the end of the period, you miss:
- People who churned (left the org or lost their Copilot license)
- People who were hired or onboarded mid-period
- Seasonal patterns driven by workforce changes, not behavior changes

**How to detect:**

```python
# Count persons per period
period_counts = df.groupby('MetricDate')['PersonId'].nunique()
print(period_counts)

# Check who appears in all periods
periods_per_person = df.groupby('PersonId')['MetricDate'].nunique()
total_periods = df['MetricDate'].nunique()
balanced = (periods_per_person == total_periods).mean()
print(f"Balanced panel: {balanced:.1%} of persons appear in all {total_periods} periods")
```

**How to fix:** Decide on your analytical approach:
- **Balanced panel:** Restrict to persons who appear in all periods (simplifies longitudinal analysis but may introduce selection bias).
- **Unbalanced panel:** Keep all person-periods but account for entry/exit in your calculations (e.g., use person-period counts rather than person counts).
- **Always report** the panel completeness rate and any restrictions applied.

### 4. Small group suppression

**The mistake:** Reporting metrics for groups with very few people, violating privacy thresholds.

**Why it matters:** Viva Insights applies minimum group size thresholds (typically 5–10 users, configurable by tenant admin) to protect individual privacy. When doing custom analytics, you must apply the same thresholds — otherwise:
- Individual behavior may be identifiable from small-group aggregates
- Results may be noisy and unreliable for very small groups
- You may violate your organization's data governance policies

**How to detect:**

```python
# Check group sizes for each segmentation variable
for attr in ['Organization', 'FunctionType', 'LevelDesignation']:
    sizes = df.groupby([attr, 'MetricDate'])['PersonId'].nunique()
    small = sizes[sizes < 10]
    if len(small) > 0:
        print(f"{attr}: {len(small)} segment-periods have fewer than 10 people")
```

**How to fix:** Suppress or aggregate small groups:

```python
MIN_GROUP_SIZE = 10  # or your org's threshold

def suppress_small_groups(summary_df, group_col, count_col, threshold=MIN_GROUP_SIZE):
    """Replace values with NaN for groups below the privacy threshold."""
    mask = summary_df[count_col] < threshold
    suppressed = summary_df.copy()
    metric_cols = [c for c in suppressed.columns if c not in [group_col, count_col]]
    suppressed.loc[mask, metric_cols] = float('nan')
    return suppressed
```

### 5. Time zone effects

**The mistake:** Ignoring time zone differences when interpreting `MetricDate` and collaboration metrics.

**Why it matters:**
- `MetricDate` in Viva Insights typically represents the start of a measurement period based on the **tenant's configured time zone** or the **individual's time zone** (depending on configuration).
- Collaboration hours are calculated relative to the individual's configured working hours and time zone.
- Comparing "after-hours collaboration" across time zones is misleading if you don't account for different working-hours definitions.
- Purview audit log timestamps are always UTC, while person query periods may not be.

**How to detect:**

```python
# Check for time zone information in the data
if 'TimeZone' in df.columns:
    print(df['TimeZone'].value_counts())
    
# Compare metric distributions across time zones
if 'TimeZone' in df.columns:
    tz_stats = df.groupby('TimeZone')['After_Hours_Collaboration_Hours'].describe()
    print(tz_stats)
```

**How to fix:**
- When comparing across geographies, normalize after-hours metrics or exclude them.
- When joining person query data (weekly, tenant time zone) with Purview audit logs (event-level, UTC), convert both to a consistent reference before aggregation.
- Document the time zone assumptions in your analysis output.

### 6. HR attribute changes over time

**The mistake:** Assuming a person's HR attributes (Organization, LevelDesignation, etc.) are static.

**Why it matters:** In a person-time panel, a person's attributes may change across periods due to:
- Organizational restructuring
- Promotions or role changes
- Transfers between teams or geographies

If you group by `Organization` without accounting for this, a person who transferred mid-period will appear in two different groups, which may cause:
- Double counting in aggregate summaries
- Misleading segment trends (a segment appears to lose a member that another segment gains)

**How to detect:**

```python
# Check for attribute changes within persons
changes = df.groupby('PersonId').agg(
    n_orgs=('Organization', 'nunique'),
    n_levels=('LevelDesignation', 'nunique')
)
changers = changes[(changes['n_orgs'] > 1) | (changes['n_levels'] > 1)]
print(f"{len(changers)} persons ({len(changers)/len(changes):.1%}) changed attributes during the period")
```

**How to fix:**
- **Point-in-time attribution:** Use each row's attributes as-is (the attribute reflects status during that `MetricDate`). This is the default and usually correct.
- **Fixed-period attribution:** Assign each person the attributes from a specific date (e.g., the last period) for consistent grouping across the entire analysis. Document which approach you chose.

### 7. Metric definition changes

**The mistake:** Assuming metric definitions are consistent across all time periods in the export.

**Why it matters:** Microsoft periodically updates how Viva Insights metrics are calculated. For example:
- The definition of "meeting" (minimum duration, minimum attendees) may change
- New Copilot metrics may be added (appearing as new columns in later periods)
- Existing metric calculations may be refined

This can cause apparent "trend changes" that are actually methodology changes, not behavior changes.

**How to detect:**
- Check for columns that appear only in a subset of `MetricDate` values.
- Look for sudden, uniform shifts in metric distributions that align with known Viva Insights update dates.

```python
# Check for columns that are all-null in early periods but populated later
for col in df.select_dtypes(include='number').columns:
    null_by_period = df.groupby('MetricDate')[col].apply(lambda x: x.isna().all())
    if null_by_period.any() and not null_by_period.all():
        print(f"{col}: appears only in some periods")
```

**How to fix:**
- Document the date range of your analysis and note any known metric definition changes.
- If a metric was added partway through, only analyze it from its first appearance onward.
- Avoid making trend claims that span known methodology changes without a caveat.

### 8. Double counting in aggregation

**The mistake:** Aggregating metrics without respecting the panel structure, leading to inflated totals.

**Why it matters:** Common errors include:
- Summing `Collaboration_Hours` across all rows to get "total collaboration hours" (this double-counts people who appear in multiple weeks)
- Counting rows instead of unique `PersonId` values to report "number of users"
- Computing averages across all rows without first aggregating to the person level

**How to detect:**

```python
# Sanity check: does the "total user count" make sense?
row_count = len(df)
person_count = df['PersonId'].nunique()
period_count = df['MetricDate'].nunique()
print(f"Rows: {row_count}, Persons: {person_count}, Periods: {period_count}")
print(f"Expected rows (persons × periods): {person_count * period_count}")
```

**How to fix:** Be explicit about the aggregation level:

```python
# Per-week summary (cross-sectional)
weekly = df.groupby('MetricDate').agg(
    n_users=('PersonId', 'nunique'),
    avg_collab=('Collaboration_Hours', 'mean'),
    total_copilot_actions=('Copilot_Actions', 'sum')
)

# Per-person summary (longitudinal)
person_level = df.groupby('PersonId').agg(
    avg_collab=('Collaboration_Hours', 'mean'),
    total_copilot_actions=('Copilot_Actions', 'sum'),
    active_weeks=('Copilot_Actions', lambda x: (x > 0).sum())
)
```

### 9. Column name variations across tenants

**The mistake:** Hard-coding column names from one tenant's export and assuming they work universally.

**Why it matters:** The same metric may appear under different names in different tenants. Common variations are documented in the [Common column name variations](#common-column-name-variations) section above. Hard-coding column names causes code to fail silently (if the column is missing) or loudly (KeyError) when applied to a different tenant's data.

**How to detect:**

```python
# Check for expected columns
expected = ['PersonId', 'MetricDate', 'Collaboration_Hours', 'Copilot_Actions']
missing = [c for c in expected if c not in df.columns]
if missing:
    print(f"Missing expected columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
```

**How to fix:**
- **Auto-detect columns** using pattern matching (e.g., columns containing `Copilot`, `Collab`, `Date`).
- **Print column names at the start** of every analysis and ask for confirmation or mapping.
- **Use flexible column mapping** in code:

```python
# Flexible column mapping
COLUMN_MAP = {
    'person_id': ['PersonId', 'Person_Id', 'HashId', 'EmployeeId'],
    'metric_date': ['MetricDate', 'Date', 'Week_Start_Date', 'Period_Start'],
    'organization': ['Organization', 'Org', 'Department', 'Business_Unit'],
}

def find_column(df, standard_name):
    """Find the actual column name from known variants."""
    for variant in COLUMN_MAP.get(standard_name, []):
        if variant in df.columns:
            return variant
    raise KeyError(f"Could not find column for '{standard_name}'. "
                   f"Available: {list(df.columns)}")
```

### Data validation checklist

Run through this checklist at the start of any Viva Insights analysis:

#### Structure validation
- [ ] **Panel structure confirmed:** Each `(PersonId, MetricDate)` pair is unique
- [ ] **Granularity identified:** Weekly (7-day gaps) or daily (1-day gaps)
- [ ] **Date range reasonable:** Start and end dates match expected export period
- [ ] **Row count makes sense:** Approximately `n_persons × n_periods`

#### Column validation
- [ ] **Core columns present:** `PersonId` and `MetricDate` (or variants) exist
- [ ] **HR attributes identified:** At least `Organization` (or variant) is present
- [ ] **Copilot columns detected:** Columns matching `Copilot_*` pattern exist (if Copilot analysis)
- [ ] **Column names documented:** Printed and verified against expected schema

#### Data quality
- [ ] **Missing values understood:** Nulls in Copilot columns represent unlicensed users, not zero usage
- [ ] **Panel completeness assessed:** What percentage of persons appear in all periods?
- [ ] **Small groups identified:** Segments with < minimum group size flagged for suppression
- [ ] **No unexpected duplicates:** No duplicate `(PersonId, MetricDate)` rows

#### Analysis setup
- [ ] **Licensed vs. unlicensed flagged:** `is_licensed` column created based on Copilot metric presence
- [ ] **Active vs. inactive flagged:** `is_active` column created based on `Copilot_Actions > 0`
- [ ] **Time zone assumptions documented:** Noted whether analysis assumes tenant TZ or individual TZ
- [ ] **Privacy threshold applied:** Minimum group size set and enforced in all outputs
