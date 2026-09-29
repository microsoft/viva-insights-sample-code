# Viva Insights Person Query — Data Dictionary

This document describes the structure, columns, and data patterns of the **Viva Insights person query** export. Person query data is the primary data source for most Copilot adoption and collaboration analytics workflows.

> **Note:** Column names and available metrics vary by tenant configuration and Viva Insights version. Always verify your actual column headers against this reference. For authoritative metric definitions, see [Microsoft Learn — Viva Insights metrics](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics).

> **For inferential analysis:** this dictionary includes illustrative headers
> and conventions that are not a complete current export contract. Person Query
> supports daily, weekly and monthly grouping. Verify the actual period boundaries,
> metric definitions and coverage before modelling. Null action metrics alone
> do not establish licence status. An organisational attribute repeated across
> rows may be a static or lower-frequency outcome. Apply the
> [Person Query causal-analysis contract](../skills/viva-insights-causal-analysis/reference/person-query-contract.md)
> before treating it as longitudinal evidence.

---

## Panel structure

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

### How to detect granularity

To determine whether your data is person-week or person-day:

1. **Check the date intervals.** Sort by `MetricDate` for a single person and look at the gap between consecutive dates. A 7-day gap indicates weekly; a 1-day gap indicates daily.
2. **Count distinct dates.** If a 3-month export has ~13 distinct `MetricDate` values, it is weekly. If it has ~90, it is daily.

---

## Core identifier columns

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `PersonId` | string | Anonymized unique identifier for each person. | Consistent across time periods within the same export. Format varies by tenant (may be a hash, GUID, or encoded string). |
| `MetricDate` | date | Start date of the measurement period. | For weekly data, this is typically the Monday of each week. Format is usually `YYYY-MM-DD` or `M/D/YYYY`. Parse carefully. |

---

## HR attribute columns

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

---

## Collaboration metric columns

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

---

## Copilot metric columns

Copilot metrics measure Microsoft 365 Copilot usage. Availability depends on
the metric definition, licence eligibility and export coverage. A missing value
(`NA` / `null` / `NaN`) is not an observed zero and does not, by itself, identify
an unlicensed user.

| Column | Type | Description | Typical Range | Notes |
|--------|------|-------------|---------------|-------|
| `Copilot_Actions` | integer | Total number of Copilot actions (prompts, completions, suggestions accepted) in the period. | 0–500/wk | Primary activity volume metric. A recorded zero indicates no observed actions for this metric and period, not licence status. |
| `Copilot_Assisted_Hours` | float | Estimated hours where Copilot assisted the user's work. | 0–20 hrs/wk | Derived metric; methodology may vary by version. |
| `Copilot_Chat_Queries` | integer | Number of queries sent to Copilot chat interfaces (e.g., M365 Chat, in-app chat). | 0–200/wk | Subset of Copilot_Actions focused on conversational interactions. |
| `Copilot_Summarized_Hours` | float | Hours of meetings or content summarized by Copilot on the user's behalf. | 0–10 hrs/wk | Includes meeting summaries, email summaries, document summaries. |
| `Copilot_Assisted_Meeting_Hours` | float | Meeting hours where Copilot features were used (e.g., transcription summary, recap). | 0–15 hrs/wk | Subset of Meeting_Hours where Copilot was active. |
| `Copilot_Assisted_Email_Hours` | float | Email hours where Copilot assisted with drafting or summarizing. | 0–5 hrs/wk | |
| `Copilot_Assisted_Document_Hours` | float | Hours of document work where Copilot assisted (e.g., in Word, PowerPoint, Excel). | 0–10 hrs/wk | |

> **Important:** Keep missing values distinct from recorded zeros. Establish
> licence eligibility for the relevant period using documented licence or
> enabled-days evidence, and verify metric coverage before classifying a user
> as licensed but inactive or unlicensed. Do not infer either status from an
> action metric alone.

---

## Example rows

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

- **`abc123`** (Engineering, Senior IC) has observed Copilot actions in both weeks.
- **`def456`** (Marketing, Manager) has missing Copilot values. Licence and coverage evidence is needed before classifying either week.
- **`ghi789`** (Sales, IC) has observed actions in week 1 and a recorded zero in week 2. Check licence and coverage evidence before labelling week 2 as licensed but inactive.

---

## Special notes for coding agents

### Detecting time granularity

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

### Identifying licensed vs. unlicensed users

First confirm which fields document licence eligibility for each person and
period, and whether the selected metric was observed for that period. Some
Copilot metric families cover unlicensed activity, so a non-null value in a
column containing `Copilot` is not proof of a licence. Classify observed
activity from the selected metric separately from licence status. Only label
a recorded zero as **licensed but inactive** when eligibility and coverage
are confirmed. Keep unresolved cases as **unknown** rather than assigning
them to the unlicensed population.

### Validating panel completeness

Check that each person appears in every time period. Missing person-periods may indicate employees joining or leaving.

```python
# Python
expected = df['PersonId'].nunique() * df['MetricDate'].nunique()
actual = len(df)
completeness = actual / expected
print(f"Panel completeness: {completeness:.1%} ({actual}/{expected})")
```

### Common column name variations

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
