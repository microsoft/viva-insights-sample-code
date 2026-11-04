# Required Inputs — Copilot Adoption Dashboard

This document describes the data you need before running the Copilot Adoption Dashboard prompts.

## Person query CSV

The primary input is a **person query export** from the Viva Insights Analyst portal. Each row represents one person in one time period (typically one week).

### Expected columns

| Column | Type | Description | Required? |
|---|---|---|---|
| `PersonId` | String | Anonymized unique identifier for each person. Consistent across weeks. | **Yes** |
| `MetricDate` | Date | Start of the configured measurement period; confirm week start rather than assuming Monday. Format: `YYYY-MM-DD`. | **Yes** |
| `Copilot_Actions` | Numeric | Verified weekly actions, mapped to the actual primary metric such as `Total_Copilot_actions_taken`. Null is unknown, not licence evidence. | **Yes** |
| Enabled-days or dated licence evidence | Numeric/Boolean | Explicit product-specific eligibility for each person-week, independently of activity. | For adoption denominators |
| `Copilot_Assisted_Hours` | Numeric | Verified assisted time, not measured time saved. Null is unknown. | Recommended |
| `Copilot_Chat_Queries` | Numeric | Verified queries sent to Copilot Chat. Null is unknown. | Recommended |
| `Copilot_Summarized_Hours` | Numeric | Verified summarised time, not automatically additive with assisted time. Null is unknown. | Optional |
| `Organization` | String | HR attribute indicating the person's organizational unit (e.g., department or division). | **Yes** |
| `FunctionType` | String | HR attribute indicating the person's job function (e.g., Engineering, Sales, Finance). | Recommended |
| `LevelDesignation` | String | HR attribute indicating the person's seniority level (e.g., IC, Manager, Director). | Recommended |
| `SupervisorIndicator` | String | Indicates whether the person is a manager (`Manager`) or individual contributor (`Individual Contributor`). | Optional |
| `City` | String | Person's city (HR attribute). | Optional |
| `Country` | String | Person's country or region (HR attribute). | Optional |
| `Region` | String | Person's geographic region (HR attribute). | Optional |

> **Note:** Column names vary between tenants and query configurations. The prompts auto-detect columns starting with `Copilot_` as Copilot metrics. If your HR attribute columns have different names (e.g., `Org` instead of `Organization`), tell the coding agent when you paste the prompt.

### Additional standard person query columns

Your export may also include collaboration metrics that are not required for this dashboard but can add context:

- `Collaboration_Hours`, `Meeting_Hours`, `Email_Hours`, `Chat_Hours`
- `After_Hours_Collaboration_Hours`
- `Internal_Network_Size`, `External_Network_Size`
- `Manager_Coaching_Hours_1on1`

The prompts will ignore columns they don't need, so including extra columns is harmless.

## Minimum data requirements

| Dimension | Minimum | Recommended |
|---|---|---|
| **Time span** | 4 weeks | 8–12+ weeks |
| **Licensed users** | 10 | 50+ |
| **HR attributes** | 1 (e.g., `Organization`) | 3+ (Organization, FunctionType, LevelDesignation) |
| **Copilot metrics** | 1 (e.g., `Copilot_Actions`) | 3+ |

- **Time span**: Fewer than 4 weeks makes trend analysis unreliable. 12+ weeks is ideal for detecting meaningful adoption trajectories.
- **Licensed users**: Small populations limit useful breakdowns. Always enforce the dashboard floor of 10 distinct people or the stricter organisational requirement.
- **HR attributes**: More attributes enable richer slicing. At minimum, you need one organizational grouping.

## How to export from Viva Insights

1. Open the **Viva Insights Analyst** portal at [insights.cloud.microsoft](https://insights.cloud.microsoft/).
2. Navigate to **Analysis** > **Custom queries** > **Person query**.
3. Configure the query:
   - **Time period**: Select a range covering at least 8 weeks.
   - **Granularity**: Weekly for these prompts. Record configured grouping, week start and metric measurement windows; spacing is not cadence evidence.
   - **Metrics**: Add all available `Copilot_*` metrics.
   - **Organizational data**: Include `Organization`, `FunctionType`, `LevelDesignation`, and any other HR attributes you want to analyze.
4. Run the query and wait for it to complete.
5. Download the results as a **CSV** file.

For detailed guidance, see the [Microsoft documentation on person queries](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query-overview).

## Data format notes

- **Encoding**: CSV files should be UTF-8 encoded. If you encounter character issues, re-save as UTF-8 in Excel or a text editor.
- **Date format**: `MetricDate` should be in `YYYY-MM-DD` format (e.g., `2024-09-02`). If your export uses a different format (e.g., `MM/DD/YYYY`), tell the coding agent so it can parse correctly.
- **Decimal separator**: Use a period (`.`) as the decimal separator. Comma-separated decimals (common in some European locales) will cause parsing errors — convert before loading or tell the agent.
- **Missing values**: Existing null cells remain unknown and do not establish licensing. Absent activity within a supplied complete query's configured scope contributes zero; unavailable sources, out-of-scope periods and known coverage failures remain distinct. See the [shared policy](../../skills/viva-insights-analysis/reference/analysis-policy.md).
- **File size**: Person query exports can range from a few MB to several hundred MB depending on the population size and time span. Files over 100 MB may require chunked processing — mention the file size to the agent.

## Schema reference

For the full data dictionary, see the [schemas directory](../../schemas/).
