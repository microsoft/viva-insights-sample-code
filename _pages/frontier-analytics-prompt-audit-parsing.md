---
layout: page
title: "Prompt — Audit Log Parsing"
permalink: /frontier-analytics-prompt-audit-parsing/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}
{% include prompt-styles.html %}

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

/* Prompt page navigation */
.prompt-nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid #e0e0e0;
}
.prompt-nav a {
  text-decoration: none;
  color: #0366d6;
  font-weight: 500;
}
.prompt-nav a:hover {
  text-decoration: underline;
}
.prompt-nav .nav-disabled {
  color: #999;
  pointer-events: none;
}
</style>

# Audit Log Parsing — Purview Audit Logs

[← Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)

## Purpose

Parse and clean raw Microsoft Purview audit log exports into a flat, analysis-ready dataset. This is a data engineering prompt that prepares audit log data for downstream analytics (such as the [Agent Usage Analysis]({{ site.baseurl }}/frontier-analytics-prompt-agent-usage/) prompt).

## Audience

Data engineers, people analytics teams, IT administrators preparing data for analysis

## When to use

Immediately after exporting raw audit logs from Purview, before running any analytical prompts. Purview exports often contain nested JSON fields, inconsistent event types, and mixed schemas that need to be normalized before meaningful analysis can be performed.

## Required inputs

- Raw Purview audit log export (CSV or JSON format)
- Expected raw fields include: `CreationTime`, `UserId`, `Operation`, `Workload`, `AuditData` (a JSON string containing event details)
- Optional: a list of Copilot-related operation names to filter for (if known for your tenant)

## Assumptions

- The export is from Microsoft Purview unified audit log
- `AuditData` is a JSON string column containing nested event details
- Event types and field structures may vary across different `Workload` and `Operation` values
- Some records may be malformed or have missing fields
- The `vivainsights` R or Python package is available but not required for this task

## Recommended output

A cleaned, flat CSV file (or DataFrame) with one row per event and consistently named columns, ready for analysis with any downstream prompt or tool.

## Prompt

```
You are a data engineer. Your task is to parse and clean a raw Microsoft Purview audit log export into a flat, analysis-ready dataset. Purview audit logs contain nested JSON fields and mixed event schemas, so this task requires careful exploration, parsing, and normalization.

IMPORTANT: Purview audit log schemas vary by tenant and event type. Do NOT assume specific field names in the AuditData JSON — explore the data first and adapt.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.

PHASE 1: INITIAL LOADING AND INSPECTION
1. Load the raw audit log file. Auto-detect the format:
   - If CSV: load with pandas (Python) or readr (R). Handle encoding issues (try utf-8, then utf-8-sig, then latin-1).
   - If JSON: detect whether it is a JSON array, newline-delimited JSON, or a single object. Load accordingly.
2. Print: column names, data types, row count, and the first 3 rows.
3. Identify the core columns:
   - Timestamp: typically "CreationTime" or "CreationDate"
   - User: typically "UserId" or "UserKey"
   - Operation: typically "Operation"
   - Workload: typically "Workload"
   - AuditData: typically "AuditData" — a JSON string with event details Print which columns were identified and which are missing.
4. Check for data quality issues:
   a. How many rows have null/empty values in each core column?
   b. How many rows have non-parseable JSON in AuditData (if present)?
   c. Are there duplicate rows (identical across all columns)? Print a summary of these findings.

PHASE 2: PARSING THE AUDITDATA JSON COLUMN
5. If an AuditData column exists:
   a. Parse the JSON string for each row into a dictionary/object. Wrap in a try-except to handle malformed JSON gracefully — log the count of unparseable rows and skip them.
   b. Inspect the parsed JSON structure for a sample of 50 rows. Print the unique top-level keys found and their frequency. Identify which keys are present in most records vs. rare.
   c. Identify common nested objects (keys whose values are dicts or lists). Print examples of the nested structures.

6. Extract top-level fields from AuditData into new columns. Prioritize these fields (but use whatever is actually present in your data):
   - "Id" or "UniqueId" → event_id
   - "CreationTime" → audit_creation_time (may differ from the outer CreationTime)
   - "Operation" → audit_operation (may be more specific than the outer Operation)
   - "UserId" → audit_user_id
   - "ClientIP" → client_ip
   - "Workload" → audit_workload
   - "ObjectId" → object_id (the resource being accessed)
   - "ItemType" → item_type
   - "AppAccessContext" → extract nested fields like "AADSessionId", "CorrelationId"
   - "CopilotEventData" or similar → extract Copilot-specific details (agent name, plugin name, interaction type, prompt length, etc.) Create a new column for each extracted field.

7. For nested objects, flatten one level deep. For example, if AuditData contains: {"CopilotEventData": {"AgentName": "MyAgent", "InteractionType": "Chat"}} Create columns: copilot_agent_name, copilot_interaction_type. Do NOT attempt to flatten deeply nested structures (3+ levels) — store them as JSON strings.

PHASE 3: EVENT TYPE CLASSIFICATION
8. Analyze the distinct values in Operation (and audit_operation if different). Print:
   a. All unique Operation values with their counts.
   b. Group operations into categories:
      - Copilot events: operations containing "Copilot", "AI", "Agent", "GPT", "Assist", "Summarize", "CopilotInteraction" (case-insensitive)
      - User activity events: sign-in, file access, mail read, etc.
      - Admin events: settings changes, policy updates, etc.
      - Other/unknown
   c. Create a new column `event_category` with these classifications.
9. Print the count of events per category.

PHASE 4: FILTERING FOR COPILOT EVENTS
10. Create a filtered dataset containing only Copilot-related events (event_category == "copilot").
11. If the Copilot filter produces zero results:
    a. Print the top 30 most common Operation values so the user can identify Copilot events manually.
    b. Check if any AuditData fields contain Copilot-related values even if the Operation name does not indicate it.
    c. Save the full (unfiltered) cleaned dataset and note that manual filtering is needed.
12. If Copilot events are found, print:
    a. Count of Copilot events.
    b. Unique Copilot operation types.
    c. Date range of Copilot events.
    d. Sample of 5 Copilot event rows (all extracted columns).

PHASE 5: DATA CLEANING AND NORMALIZATION
13. Apply the following cleaning steps to the full dataset (or the Copilot-filtered subset):
    a. Parse all timestamp columns to datetime with timezone handling (Purview typically uses UTC).
    b. Normalize UserId:
       - Strip whitespace and convert to lowercase.
       - If UPNs (email format), keep as-is for joining. Optionally create a hashed_user_id column for anonymized output.
    c. Remove exact duplicate rows.
    d. Remove rows where both Operation and UserId are null.
    e. Create derived columns:
       - event_date: date extracted from the timestamp (UTC date)
       - event_hour: hour of day (0-23)
       - event_weekday: day of week (Monday=0, Sunday=6)
       - event_week: ISO week start date (Monday)

14. Standardize column names:
    - Use snake_case for all columns.
    - Prefix AuditData-derived columns with their source context (e.g., copilot_agent_name rather than just agent_name).
    - Ensure no column name conflicts between outer columns and AuditData-extracted columns.

PHASE 6: OUTPUT
15. Save the cleaned dataset:
    a. Full cleaned dataset → "purview_audit_cleaned_YYYYMMDD.csv"
    b. Copilot events only → "purview_copilot_events_YYYYMMDD.csv"
    c. Print the final schema: column names, data types, non-null counts for each output file.
    d. Print the row counts for each output.

16. Generate a data dictionary as a separate file ("purview_data_dictionary_YYYYMMDD.md"):
    - For each column in the cleaned output, list:
      - Column name
      - Data type
      - Source (outer field, AuditData top-level, AuditData nested)
      - Description (based on observed values)
      - Example values (2-3 examples)
      - Null rate

17. Print a final summary:
    a. Total raw records loaded.
    b. Records with parse errors (skipped).
    c. Duplicate records removed.
    d. Final clean record count.
    e. Copilot event count.
    f. Date range.
    g. Unique users.

IMPORTANT NOTES
- Prioritize robustness over speed. Wrap every JSON parse and type conversion in error handling.
- Do NOT assume the AuditData JSON structure is consistent across all rows — different Operation types may have completely different AuditData schemas. Handle this gracefully.
- If the file is very large (>500MB), process in chunks rather than loading all at once.
- Do NOT expose raw UserIds/email addresses in printed output — show only the first few characters or use aggregate counts when printing summaries.
- The data dictionary is critical for downstream users. Invest time in making it accurate.
```

## Adaptation notes

- **Your AuditData fields will differ.** The field names listed in step 6 (e.g., `CopilotEventData`, `AppAccessContext`) are examples. The prompt instructs the agent to explore first and adapt — but you may need to guide it if your tenant uses unusual field names.
- **Filtering criteria:** If you know your tenant's Copilot operation names (e.g., from the Purview audit log documentation or prior inspection), prepend them to the prompt: _"In my tenant, Copilot events use Operation values: CopilotInteraction, CopilotQuery."_
- **Large files:** For exports exceeding 1GB, add: _"Process the file in chunks of 100,000 rows using pandas chunked reading."_
- **Multiple export files:** If your audit log is split across multiple files (e.g., one per day), add: _"Load all CSV files from the directory and concatenate them before processing."_
- **Anonymization requirements:** If your organization requires user anonymization before analysis, strengthen step 13b: _"Replace all UserId values with a consistent SHA-256 hash. Do not retain the original identifier."_

## Common failure modes

- **Agent fails on malformed JSON in AuditData.** Some rows may have truncated or malformed JSON. The prompt includes try-except handling, but verify the agent implements it — a single bad row should not crash the entire parse.
- **Agent creates inconsistent column names.** When extracting from AuditData, column naming can become inconsistent across event types. Verify that the final output has standardized snake_case names.
- **Agent loads the entire file into memory.** For very large exports, this may cause out-of-memory errors. Watch for memory warnings and switch to chunked processing if needed.
- **Agent flattens deeply nested JSON, creating hundreds of columns.** The prompt limits flattening to one level deep, but some agents may over-expand. Verify the output schema is manageable (ideally <50 columns).
- **Copilot event filtering is too narrow or too broad.** If zero events match, the prompt instructs the agent to show all Operation values for manual inspection. If too many match, review the filter criteria and narrow them.
- **Encoding issues.** Purview CSV exports may use UTF-8 BOM encoding. If the agent encounters parsing errors on the first column name, instruct it to use `utf-8-sig` encoding.

<div class="prompt-nav">
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-agent-usage/">← Previous: Agent Usage Analysis</a>
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-causal-toolkit/">Next: Copilot Causal Toolkit →</a>
</div>
