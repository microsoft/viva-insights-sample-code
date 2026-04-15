# Schema Documentation

This directory contains **data dictionaries and schema references** for the data sources used in Viva Insights analytics workflows. These documents describe the structure, column definitions, and data patterns you will encounter when working with Viva Insights exports and related data.

## How to use these docs

- **Building prompts:** Reference the data dictionaries when writing or adapting prompt cards. Understanding column names, types, and edge cases will produce better prompts.
- **Sharing with coding agents:** Paste the relevant schema documentation into your coding agent's context window (or point it at the file) before running an analytics prompt. This gives the agent the structural knowledge it needs to handle your data correctly.
- **Data validation:** Use the example rows and column descriptions to verify that your export matches the expected format before running any analysis.

> **Tip for coding agents:** You can prepend schema context to any prompt. For example: _"Refer to the data dictionary below for column definitions and data patterns, then execute the following analysis…"_ followed by the relevant schema content and the prompt.

## Available schema docs

| Document | Description |
|---|---|
| [Person Query Data Dictionary](viva-person-query-data-dictionary.md) | Column definitions, data types, and value patterns for the Viva Insights person query export (person-week or person-day panel). |
| [Purview Audit Data Dictionary](purview-audit-data-dictionary.md) | Column definitions and nested JSON structure for Microsoft Purview unified audit log exports, including Copilot event data. |
| [Join Patterns](join-patterns.md) | How to join person query data with Purview audit logs, external HR data, and license records — with code examples in R and Python. |
| [Common Pitfalls](common-pitfalls.md) | Data quality issues, analytical mistakes, and edge cases to watch for when working with Viva Insights exports. |

## Important notes

- **Column names vary by tenant.** The column names documented here are the most common defaults. Your organization's Viva Insights configuration may use different names for the same metrics. Always inspect your actual column headers before running any analysis.
- **Metric definitions evolve.** Microsoft periodically updates how Viva Insights metrics are calculated. The descriptions here reflect common patterns but may not match every version.
- **These are reference documents, not specifications.** For authoritative and up-to-date metric definitions, refer to the [official Microsoft Viva Insights documentation](https://learn.microsoft.com/en-us/viva/insights/).

## Related resources

- [Prompt Card Library](../prompts/) — ready-to-use prompts for coding agents
- [Example Output Specs](../examples/) — specifications for common output formats
- [Viva Insights metric definitions (Microsoft Learn)](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
