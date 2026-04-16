---
layout: page
title: "Prompt — Agent Usage Analysis"
permalink: /frontier-analytics-prompt-agent-usage/
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

# Agent Usage Analysis — Purview Audit Logs

[← Back to Prompt Library]({{ site.baseurl }}/frontier-analytics-prompts/)

## Purpose

Analyze Copilot agent and extension usage patterns from Microsoft Purview audit logs to understand which Copilot features and agents are being used, by whom, and how usage trends over time.

## Audience

IT administrators, Copilot program managers, security and compliance teams, people analytics leads

## When to use

When you have access to Purview audit log exports containing Copilot-related events and want to understand Copilot agent/extension adoption patterns beyond what Viva Insights person query data provides. This is especially useful for tracking specific Copilot interaction types (e.g., chat vs. summarization vs. agent invocations).

## Required inputs

- Purview audit log export (CSV or JSON format)
- Expected fields include: `UserId`, `Operation`, `Workload`, `CreationTime`, and optionally `AuditData` (a JSON field containing event details)
- At least 4 weeks of audit data recommended for trend analysis

## Assumptions

- The audit log export contains Copilot-related events (operations may include terms like "CopilotInteraction", "CopilotQuery", "AIAppInteraction", or similar — exact names vary by tenant and configuration)
- `UserId` identifies the user (may be a UPN/email or anonymized identifier)
- `CreationTime` is a timestamp for each event
- `Workload` indicates the Microsoft 365 application (e.g., "MicrosoftCopilot", "Teams", "Exchange", "SharePoint")
- `Operation` indicates the type of Copilot action
- The audit log schema may vary between tenants — field names and event types should be verified before analysis

## Recommended output

An exploratory HTML report or Jupyter/R notebook with usage trends, operation breakdowns, and user activity distributions.

## Prompt

```
You are a data analyst working with Microsoft Purview audit logs. Your task is to analyze Copilot
agent and extension usage patterns from a Purview audit log export. Because the Purview audit
schema can vary between tenants, this analysis should be exploratory — start by understanding
the data structure before computing metrics.

IMPORTANT CAVEAT: Purview audit log schemas are not standardized across tenants. Field names,
operation types, and event structures may differ from what is described below. The first phase
of this analysis must be data exploration and validation.

LANGUAGE CHOICE
Choose R or Python based on what is already installed in your environment to minimize setup.

DATA LOADING AND EXPLORATION
1. Load the audit log file. Support both CSV and JSON formats — detect the format automatically.
   If the file is JSON, it may be a JSON array or newline-delimited JSON (one object per line).
   If CSV, parse normally with pandas or readr.
2. Print the column names, data types, and the first 5 rows to understand the schema.
3. Print the number of total records and the date range (from CreationTime or the equivalent
   timestamp field).
4. If there is an "AuditData" column that contains JSON strings, note it but do NOT parse it
   yet — we will handle it in a later step if needed.

FIELD IDENTIFICATION
5. Identify key fields by searching column names for common patterns:
   - User identifier: look for "UserId", "UserKey", "User", "UPN"
   - Timestamp: look for "CreationTime", "CreationDate", "Timestamp", "EventTime"
   - Operation: look for "Operation", "Action", "EventType", "Activity"
   - Workload/Application: look for "Workload", "Application", "AppName", "Product"
   Print the identified field mappings and ask for confirmation if ambiguous.
6. Parse the timestamp field as a datetime type. Extract date (day) and week columns.

COPILOT EVENT FILTERING
7. Explore the unique values in the Operation and Workload columns. Print value counts for both.
8. Filter for Copilot-related events. Use a broad filter first:
   - Operation values containing "Copilot", "AI", "Agent", "GPT", "Assist", "Summarize" (case-insensitive)
   - Workload values containing "Copilot", "Microsoft365", "M365" (case-insensitive)
   Print the number of matching events and the operation/workload values that matched.
9. If the filtered dataset is empty, expand the filter or report that no Copilot events were
   found and list all unique Operation and Workload values for manual inspection.
10. If an AuditData column exists and the initial filtering is too broad, parse the JSON in
    AuditData for a sample of 100 rows and look for additional fields that indicate Copilot
    usage (e.g., "AppName", "CopilotEventType", "AgentName", "ExtensionName").

USAGE METRICS
11. Using the filtered Copilot events, compute:
    a. Total events per day and per week.
    b. Unique users per day and per week.
    c. Events per user per week (distribution: mean, median, p25, p75).
12. Create trend charts:
    a. Line chart: daily event count over time (with a 7-day rolling average overlay).
    b. Line chart: weekly unique users over time.
    c. Line chart: weekly events per user over time (mean).

OPERATION TYPE BREAKDOWN
13. For each unique Operation value in the Copilot-filtered data:
    a. Count total events.
    b. Count unique users.
    c. Compute events per user.
14. Create:
    a. A horizontal bar chart of event counts by Operation (top 15 operations).
    b. A horizontal bar chart of unique users by Operation (top 15).
15. If Workload is available and has multiple values, create:
    a. A grouped bar chart showing event counts by Workload.
    b. A heatmap of Operation × Workload (event counts).

USER ACTIVITY DISTRIBUTION
16. Compute a per-user activity summary over the entire period:
    a. Total events per user.
    b. Active days per user.
    c. Active weeks per user.
    d. Most common Operation per user.
17. Create a histogram of total events per user (log scale if distribution is highly skewed).
18. Classify users into activity tiers:
    - "Heavy": top 10% by total events
    - "Moderate": 10th-50th percentile
    - "Light": bottom 50%
    Print the count and percentage in each tier.
19. If any identifier for user department or group is available (from AuditData or a separate
    mapping file), break down activity tiers by group.

AGENT/EXTENSION ANALYSIS (if data is available)
20. If the AuditData or other fields contain information about specific Copilot agents or
    extensions (e.g., "AgentName", "ExtensionId", "PluginName"), extract and analyze:
    a. Top agents/extensions by usage (event count and unique users).
    b. Trend of agent/extension usage over time.
    c. Agent-specific user engagement (events per user per agent).
    If no agent/extension information is found, skip this section and note its absence.

REPORT GENERATION
21. Compile into an intermediary document first, then export to HTML:
    - R: Create an RMarkdown file (.Rmd), then knit to a self-contained HTML file.
    - Python: Create a Jupyter notebook (.ipynb), then export to a self-contained HTML file.
    Keep the intermediary file alongside the HTML output for troubleshooting.
    The report should contain these sections:
    a. "Data Overview" — schema summary, date range, total events, Copilot filter criteria used.
    b. "Usage Trends" — trend charts from step 12.
    c. "Operation Breakdown" — charts from steps 14-15.
    d. "User Activity Distribution" — histogram and tier summary from steps 17-18.
    e. "Agent/Extension Usage" — analysis from step 20 (if available).
    f. "Key Findings" — 3-5 bullet points summarizing the most notable patterns.
    g. "Data Notes" — document any field mapping decisions, filter criteria, and schema
       observations for reproducibility.

22. Use static charts (matplotlib/seaborn or ggplot2).
23. Save the report and intermediary file as
    "purview_copilot_agent_analysis_YYYYMMDD.html".

IMPORTANT NOTES
- This is an EXPLORATORY analysis. The Purview schema is not standardized — always start by
  inspecting the data rather than assuming specific field names or values.
- Print intermediate outputs (unique values, sample rows) so I can verify the field mappings.
- If the dataset is very large (>1M rows), sample for exploration but use the full data for
  final metrics.
- User identifiers in Purview logs may be email addresses/UPNs. Do not expose raw email
  addresses in the report — if possible, hash or truncate them, or use only aggregate statistics.
- Some operations may be system-generated rather than user-initiated. Look for patterns that
  distinguish user actions from system events.
```

## Adaptation notes

- **Field names will vary.** The most critical adaptation step is verifying your audit log's actual field names. Run the exploration steps first, then update the prompt with your specific field mappings.
- **Copilot event identification** depends on your tenant's audit configuration. The filter in step 8 casts a wide net — narrow it after inspecting the Operation and Workload values in your data.
- **Joining with Viva Insights data:** If you want to enrich the audit log analysis with HR attributes from Viva Insights, add a join step using `UserId` (after normalizing the identifier format between the two sources).
- **Privacy:** Purview logs may contain PII (email addresses). Ensure your analysis complies with your organization's data handling policies. Add anonymization steps if needed.
- **Large datasets:** For audit logs spanning months, consider filtering to a specific date range before loading the full file to reduce memory usage.

## Common failure modes

- **Agent assumes specific Purview field names that don't exist.** The prompt starts with data exploration, but some agents may skip ahead. Ensure the schema inspection step runs first.
- **Agent fails to parse nested JSON in AuditData.** This column often contains a JSON string with nested objects. The agent may need guidance on which nested fields to extract.
- **Agent exposes raw email addresses in the report.** Purview logs use UPNs. Instruct the agent to anonymize or aggregate to avoid PII exposure.
- **Copilot events are not clearly labeled.** In some tenants, Copilot interactions are logged under generic operation names. You may need to inspect AuditData contents to identify them.
- **Agent treats all events equally.** Some operations may be system-level events or duplicates. Review the Operation values to determine which represent genuine user interactions.

<div class="prompt-nav">
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-powerpoint/">← Previous: Executive PowerPoint Deck</a>
  <a href="{{ site.baseurl }}/frontier-analytics-prompt-audit-parsing/">Next: Audit Log Parsing →</a>
</div>
