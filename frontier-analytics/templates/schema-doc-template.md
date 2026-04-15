<!-- ==========================================================================
     SCHEMA DOCUMENT TEMPLATE — How to fill this in
     ==========================================================================

     Schema documents describe the structure of a data source so that both
     analysts and coding agents know exactly what to expect when loading a file.

     Good schema docs:
     1. Explain the data source and how it is exported
     2. Define every column with its type, meaning, and edge cases
     3. Provide realistic example rows so the reader can see the data shape
     4. Call out common pitfalls that trip up analysts and agents alike
     5. Include agent-specific guidance for validation and transformation

     Save completed schema docs in frontier-analytics/schemas/ with a
     descriptive filename (e.g., person-query.md, purview-audit-log.md).

     Replace every [placeholder] below with your content, then delete this
     comment block before submitting your PR.
     ========================================================================== -->

# [Data Source Name] — Schema Documentation

## Overview

[2-3 sentences describing what this data source is, where it comes from, and what it is typically used for.]

- **Source system**: [e.g., "Viva Insights Analyst portal", "Microsoft Purview compliance portal"]
- **Export method**: [e.g., "Person query export via the Analyst workbench", "Audit log search and download"]
- **File format**: [e.g., "CSV, UTF-8 encoded, comma-delimited"]
- **Typical file size**: [e.g., "10-500 MB depending on population and date range"]

## Panel structure

[Describe the granularity of the data — what each row represents.]

- **Granularity**: [e.g., "One row per person per week"]
- **Primary key**: [e.g., "`PersonId` × `MetricDate`"]
- **Date range**: [e.g., "Determined by the query parameters; typically 4-52 weeks"]
- **Population**: [e.g., "All measured employees or a filtered subset based on query scope"]

## Column dictionary

### Required columns

These columns are always present in the export.

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `[ColumnName]` | [String / Date / Integer / Float / Boolean] | [Clear description of what this column represents] | [Edge cases, formatting notes, or common values] |
| `[ColumnName]` | | | |
| `[ColumnName]` | | | |

### Metric columns

These columns contain the quantitative measures. Available columns depend on the query configuration.

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `[MetricName]` | [Float / Integer] | [What this metric measures] | [Units, NULL handling, typical range] |
| `[MetricName]` | | | |
| `[MetricName]` | | | |

### HR attribute columns

These columns contain organizational and demographic attributes. Available columns depend on the organizational data uploaded to the system.

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `[AttributeName]` | [String] | [What this attribute represents] | [Common values, cardinality notes] |
| `[AttributeName]` | | | |

### Optional / conditional columns

These columns may or may not be present depending on export configuration or tenant setup.

| Column | Type | Description | When present |
|--------|------|-------------|--------------|
| `[ColumnName]` | | | [Condition under which this column appears] |
| `[ColumnName]` | | | |

## Example rows

Provide 3-5 rows of **realistic but synthetic** data. Do not use real tenant data.

| [Column1] | [Column2] | [Column3] | [Column4] | [Column5] |
|-----------|-----------|-----------|-----------|-----------|
| [value] | [value] | [value] | [value] | [value] |
| [value] | [value] | [value] | [value] | [value] |
| [value] | [value] | [value] | [value] | [value] |
| [value] | [value] | [value] | [value] | [value] |
| [value] | [value] | [value] | [value] | [value] |

> **Note:** All example data above is synthetic. Column names and values may vary between tenants.

## Common pitfalls

| Pitfall | Description | How to handle |
|---------|-------------|---------------|
| [Pitfall 1, e.g., "Duplicate rows"] | [Why this happens, e.g., "Some exports include duplicate rows when a person changes organizations mid-period"] | [How to detect and resolve, e.g., "Deduplicate on PersonId × MetricDate, keeping the latest row"] |
| [Pitfall 2, e.g., "Mixed date formats"] | [Description] | [Resolution] |
| [Pitfall 3, e.g., "NULL vs. zero"] | [Description] | [Resolution] |
| [Pitfall 4, e.g., "Column name variations"] | [Description] | [Resolution] |
| [Pitfall 5, e.g., "Encoding issues"] | [Description] | [Resolution] |

## Notes for coding agents

This section provides structured guidance that coding agents (GitHub Copilot, Claude Code, etc.) can use when working with this data source.

### Validation steps

When loading this data, a coding agent should:

1. **Verify file format**: [e.g., "Confirm the file is CSV with UTF-8 encoding. If reading fails, try `latin-1` encoding."]
2. **Check required columns**: [e.g., "Assert that `PersonId` and `MetricDate` are present. Print all column names for the analyst to verify."]
3. **Parse date columns**: [e.g., "Parse `MetricDate` as a date. Expected formats: `YYYY-MM-DD` or `M/D/YYYY`."]
4. **Check for duplicates**: [e.g., "Verify uniqueness of `PersonId` × `MetricDate`. If duplicates exist, flag them and keep the first occurrence."]
5. **Validate value ranges**: [e.g., "Copilot metric columns should be non-negative. Flag any negative values as potential data issues."]
6. **Report data shape**: [e.g., "Print row count, column count, date range, and unique person count."]

### Common transformations

| Transformation | When needed | How to implement |
|----------------|-------------|------------------|
| [e.g., "Aggregate day to week"] | [e.g., "When export is at person-day granularity"] | [e.g., "Group by PersonId and week-start date, sum metric columns, take first value of HR attributes"] |
| [e.g., "Flag licensed users"] | [e.g., "Before computing adoption metrics"] | [e.g., "Create boolean column: True if any Copilot metric is non-null and non-zero"] |
| [e.g., "Normalize column names"] | [e.g., "When column names contain spaces or special characters"] | [e.g., "Strip whitespace, replace spaces with underscores, convert to snake_case"] |

### Recommended packages

| Language | Package | Purpose |
|----------|---------|---------|
| Python | `pandas` | Data loading and manipulation |
| Python | `vivainsights` | Viva Insights-specific helpers |
| R | `readr` / `vroom` | Fast CSV reading |
| R | `vivainsights` | Viva Insights-specific helpers |
| [Add others as relevant] | | |

## Official documentation

- [Link to relevant Microsoft documentation]
- [Link to Viva Insights documentation, e.g., "https://learn.microsoft.com/en-us/viva/insights/"]
- [Link to related schema docs in this repository]

---

<!-- ==========================================================================
     QUALITY CHECKLIST — Verify before submitting your PR
     ==========================================================================
     [ ] Every column referenced in prompt cards that use this data source is
         documented in the column dictionary
     [ ] Example rows use realistic synthetic data (no real tenant data)
     [ ] At least 3 common pitfalls are documented with resolutions
     [ ] The "Notes for coding agents" section includes validation steps
     [ ] Data types are specific (String, Date, Float, Integer) not generic
     [ ] The file is saved in frontier-analytics/schemas/ with a descriptive name
     [ ] No proprietary data, tenant-specific identifiers, or PII
     ========================================================================== -->
