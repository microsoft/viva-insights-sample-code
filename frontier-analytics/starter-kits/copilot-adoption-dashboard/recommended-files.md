# Recommended Files — Copilot Adoption Dashboard

Gather these files and resources before starting. Items marked **required** are necessary to produce the dashboard; everything else improves the output but isn't blocking.

## Required

| File / Resource | Description | Where to get it |
|---|---|---|
| **Person query CSV** | The exported Viva Insights person query containing `PersonId`, `MetricDate`, Copilot metrics, and HR attributes. This is the sole data input. | Export from the [Viva Insights Analyst portal](https://insights.cloud.microsoft/). See [required-inputs.md](required-inputs.md) for column details. |
| **Prompt cards** | The structured prompts that you paste into your coding agent. Start with Dashboard Overview. | [Dashboard Overview](../../prompts/copilot-adoption/dashboard-overview.md) · [Executive Summary](../../prompts/copilot-adoption/executive-summary.md) · [Segmentation & Churn](../../prompts/copilot-adoption/segmentation-and-churn.md) · [ROI Estimation](../../prompts/copilot-adoption/roi-estimation.md) |

## Recommended

| File / Resource | Description | Why it helps |
|---|---|---|
| **Data dictionary** | Reference documentation describing each column in the person query export — data types, expected values, and definitions. | Helps you verify that your export matches what the prompts expect. Speeds up troubleshooting when column names differ. Available in the [schemas directory](../../schemas/). |
| **License assignment data** | A list of users who have been assigned a Copilot license, with assignment dates. Typically sourced from Microsoft 365 Admin Center or Microsoft Entra. | The prompts infer licensing from non-null metric values, which is a reasonable approximation. Explicit license data gives more accurate adoption rates, especially for recently assigned users who haven't yet appeared in metrics. |
| **Organizational hierarchy file** | A CSV or Excel file mapping people to custom organizational groupings (e.g., business units, cost centers, or project teams) not captured in standard HR attributes. | Enables custom breakdowns beyond the default `Organization` and `FunctionType` columns. Useful if your standard HR attributes are too broad or too granular. |

## Optional

| File / Resource | Description | Why it helps |
|---|---|---|
| **Previous dashboard outputs** | HTML files or screenshots from earlier runs of this kit (or similar dashboards). | Provides a comparison baseline. You can ask the agent: _"Compare this week's metrics to the previous dashboard."_ Also useful for validating that trend lines are consistent. |
| **Target / benchmark values** | Internal targets for adoption rate, usage intensity, or ROI thresholds set by leadership or the Copilot deployment team. | Lets you add target lines or conditional formatting to the dashboard (e.g., highlight organizations below the 50% adoption target). Tell the agent: _"Add a horizontal target line at 60% adoption rate."_ |
| **Stakeholder distribution list** | The list of people who will receive the dashboard. | Knowing the audience helps you tailor the level of detail and privacy thresholds. For example, if the dashboard goes to a broad audience, you may want stricter suppression of small groups. |
| **Custom branding assets** | Company logo, brand colors (hex codes), or a style guide. | Ask the agent to apply your branding: _"Use #0078D4 as the primary color and include the logo at `./assets/logo.png`."_ |

## Checklist

Use this checklist before you begin:

- [ ] Person query CSV is downloaded and accessible in the coding agent's workspace
- [ ] You've reviewed [required-inputs.md](required-inputs.md) and confirmed your columns match (or noted differences)
- [ ] You've opened the [Dashboard Overview prompt card](../../prompts/copilot-adoption/dashboard-overview.md) and are ready to copy the prompt
- [ ] Your coding agent has access to Python (with `pandas`, `plotly`) or R (with `tidyverse`, `htmlwidgets`)
- [ ] (Optional) You have license assignment data or org hierarchy files ready to reference
