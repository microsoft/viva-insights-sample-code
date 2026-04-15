# Purview Audit Log — Data Dictionary

This document describes the structure, columns, and data patterns of **Microsoft Purview unified audit log** exports. Purview audit logs capture user activity events across Microsoft 365 services, including Copilot interactions, and are used to supplement Viva Insights person query data with granular event-level detail.

> **Note:** Purview audit log schemas are **less standardized** than Viva Insights person query data. The available fields, nested structures, and operation names vary significantly across tenants and event types. Treat this document as a starting reference — always explore your actual data before building analysis pipelines. For authoritative documentation, see [Microsoft Learn — Audit log activities](https://learn.microsoft.com/en-us/purview/audit-log-activities).

---

## Overview

| Property | Details |
|----------|---------|
| **What it captures** | User and admin activity events across Microsoft 365 services (SharePoint, Exchange, Teams, Copilot, etc.). |
| **Granularity** | One row per event (not aggregated). |
| **Export format** | Typically CSV or JSON, exported from the Purview compliance portal or via PowerShell. |
| **Volume** | Can be very large — thousands to millions of rows per day depending on organization size and activity scope. |
| **Time zone** | Timestamps are in UTC. |
| **Key challenge** | The `AuditData` column contains a nested JSON string with event-specific details that must be parsed. |

---

## Top-level columns

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

---

## The AuditData column

The `AuditData` column is the most important and most complex part of Purview exports. It contains a **JSON string** with event-specific details. The structure of this JSON varies by `Operation` and `Workload`.

### Parsing approach

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

### Common AuditData fields

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

---

## Copilot-specific fields

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

### Common Copilot-related operations

| Operation | Workload | Description |
|-----------|----------|-------------|
| `CopilotInteraction` | `Copilot` | A user interacted with Copilot in any M365 app. |
| `CopilotQuery` | `Copilot` | A query was sent to Copilot. |
| `MicrosoftCopilotForM365` | varies | Copilot activity within M365 apps. |

> **Note:** Operation names for Copilot events are not fully standardized. Some tenants may use different names. When filtering, use case-insensitive substring matching against keywords: `Copilot`, `AI`, `Agent`, `GPT`, `Assist`, `Summarize`.

---

## Example rows

### Raw CSV row (simplified)

```
CreationTime,UserId,Operation,Workload,AuditData
2024-09-15T14:23:45,user@contoso.com,CopilotInteraction,Copilot,"{""Id"":""a1b2c3..."",""Operation"":""CopilotInteraction"",""UserId"":""user@contoso.com"",""CopilotEventData"":{""AppHost"":""Word"",""InteractionType"":""Draft"",""AgentName"":null}}"
2024-09-15T14:25:10,user@contoso.com,FileAccessed,SharePoint,"{""Id"":""d4e5f6..."",""Operation"":""FileAccessed"",""ObjectId"":""https://contoso.sharepoint.com/sites/Team/Doc.docx"",""ItemType"":""File""}"
2024-09-15T15:01:33,admin@contoso.com,UserLoggedIn,AzureActiveDirectory,"{""Id"":""g7h8i9..."",""Operation"":""UserLoggedIn"",""ClientIP"":""10.0.0.1"",""ResultStatus"":""Succeeded""}"
```

### Parsed and flattened output

| event_id | creation_time | user_id | operation | workload | event_category | copilot_app_host | copilot_interaction_type | copilot_agent_name | object_id |
|----------|--------------|---------|-----------|----------|---------------|-----------------|------------------------|--------------------|-----------|
| a1b2c3… | 2024-09-15 14:23:45 | user@contoso.com | CopilotInteraction | Copilot | copilot | Word | Draft | | |
| d4e5f6… | 2024-09-15 14:25:10 | user@contoso.com | FileAccessed | SharePoint | user_activity | | | | https://contoso.sharepoint.com/… |
| g7h8i9… | 2024-09-15 15:01:33 | admin@contoso.com | UserLoggedIn | AzureActiveDirectory | user_activity | | | | |

---

## Data quality considerations

- **Malformed JSON:** Some `AuditData` rows may contain truncated or invalid JSON, especially in large exports. Always wrap JSON parsing in error handling.
- **Mixed schemas:** Different `Operation` types produce completely different `AuditData` structures. Do not assume a flat, consistent schema across all rows.
- **Timestamps:** `CreationTime` is always UTC. The format may vary within the same export (rare but possible). Parse with flexible datetime parsers.
- **User identifiers:** `UserId` is typically a UPN (email address), not the anonymized `PersonId` used in Viva Insights. See [Join Patterns](join-patterns.md) for how to handle this mapping.
- **Volume:** Audit logs can be extremely large. For exports exceeding 500MB, use chunked reading:

```python
# Python — chunked reading
for chunk in pd.read_csv('audit_log.csv', chunksize=100_000, encoding='utf-8-sig'):
    # process each chunk
    pass
```

- **Encoding:** Purview CSV exports frequently use UTF-8 BOM encoding (`utf-8-sig`). If the first column name appears garbled, re-read with the correct encoding.

---

## Tips for coding agents

1. **Explore before you extract.** Purview schemas vary. Always inspect the data (column names, sample rows, AuditData key frequency) before writing extraction logic.
2. **Use error handling everywhere.** Wrap every JSON parse and type conversion in try-except blocks. A single malformed row should not crash the pipeline.
3. **Flatten one level only.** Extract top-level and first-level nested fields from `AuditData`. Store deeper nesting as JSON strings to avoid column explosion.
4. **Use snake_case for output columns.** Prefix AuditData-derived columns with their context (e.g., `copilot_agent_name` not just `agent_name`) to avoid collisions.
5. **Filter by Workload first, then Operation.** This is more efficient than scanning all rows for specific Operation values.
6. **Protect privacy.** Do not print raw `UserId` values in output. Use aggregates, truncated identifiers, or hashing when displaying user-level information.
7. **Generate a data dictionary.** After parsing, document the columns you extracted — see the [Audit Log Parsing prompt](../prompts/purview-augmentation/audit-log-parsing.md) for a prompt that does this automatically.
