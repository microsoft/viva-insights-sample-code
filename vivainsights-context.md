# Viva Insights — Agent Context

> **What is this file?** This is a reusable context file for AI coding agents (GitHub Copilot, Claude Code, or similar). Drop it into your project or paste it as context before a prompt. It provides domain knowledge about Viva Insights data and the `vivainsights` package so you don't have to repeat it in every prompt.

## Data structure

Viva Insights exports **person query** data as CSV files at **person-week** granularity:

- Each row represents one person in one week.
- `PersonId` — anonymized identifier (treat as string, not numeric).
- `MetricDate` — date marking the start of each week.
- A balanced panel means every person appears in every week. Verify by checking that `PersonId × MetricDate` combinations are unique.
- Super Users Report exports use `Date` instead of `MetricDate`.

**Note:** Not all Viva Insights queries produce person-week panel data. Some exports (e.g., meeting queries, group-to-group queries) have different structures. This context file assumes person query format. If the data has a different structure, adapt the guidance accordingly or use a task-specific context prompt instead.

## Loading data

Always use `import_query()` from the **vivainsights** package to load CSV data. This function:

- Cleans column names (replaces spaces and special characters with underscores).
- Handles localization differences (e.g., US vs UK English column names).
- Parses date columns automatically.

```python
import vivainsights as vi
df = vi.import_query("path/to/person_query.csv")
```

```r
library(vivainsights)
df <- import_query("path/to/person_query.csv")
```

Do **not** use `pd.read_csv()` or `read.csv()` directly — `import_query()` handles edge cases that raw CSV readers miss.

## Identifying organizational attributes

Use `extract_hr()` to dynamically identify HR / organizational attribute columns in the data:

```python
hr_attrs = vi.extract_hr(df)
```

```r
hr_attrs <- extract_hr(df)
```

Use the returned list for all segmentation and grouping operations instead of hard-coding column names like `Organization`, `FunctionType`, or `LevelDesignation` — exact names vary between tenants and localization settings.

## Copilot metrics

Not all analyses involve Copilot data. If the dataset does not contain Copilot metric columns, skip this section and the sections on licensed/active classification and usage segmentation below.

- Copilot metric columns usually contain the word **"Copilot"** in their name, but do not always start with `Copilot_`.
- Identify them dynamically: filter columns where the name contains "Copilot".
- The **primary activity metric** is `Total_Copilot_actions_taken` — it captures all Copilot usage across apps and is generally recommended.
- Reference the metrics taxonomy for classification and validation:
  [copilot-metrics-taxonomy.csv](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/example-data/copilot-metrics-taxonomy.csv)

## Licensed and active user classification

- **Licensed:** A person-week where at least one Copilot metric column has a non-null value. Null Copilot metrics indicate the user is not licensed, not that they are inactive.
- **Active:** A licensed person-week where `Total_Copilot_actions_taken > 0`.
- **Adoption rate** = active users / licensed users (as a percentage), computed per week.

## Usage segmentation

Use `identify_usage_segments()` to classify users into segments based on both volume **and** consistency of usage over 12 weeks:

```python
df = vi.identify_usage_segments(df)
```

```r
df <- identify_usage_segments(df)
```

This produces segments including **Power Users** and **Habitual Users**, which reflect sustained usage patterns rather than one-off spikes.

## Holiday weeks

Employee data is affected by holiday periods (e.g., summer breaks, Christmas, Lunar New Year) where collaboration drops significantly. These weeks can skew averages and trends if not handled.

Use `identify_holidayweeks()` from the **vivainsights** package to detect holiday weeks based on `Collaboration_hours`:

```python
holiday_weeks = vi.identify_holidayweeks(df)
```

```r
holiday_weeks <- identify_holidayweeks(df)
```

Offer the user the choice to exclude identified holiday weeks from the analysis — this can materially change results. If excluding, document which weeks were removed in the methodology or footnotes.

## Language choice

Choose R or Python based on what is already installed in the user's environment to minimize setup:

- **R:** [vivainsights R package](https://microsoft.github.io/vivainsights/)
- **Python:** [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)

Both packages provide the same core functions (`import_query`, `extract_hr`, `identify_usage_segments`).

## Output conventions

- For **HTML outputs**: create an RMarkdown (`.Rmd`) or Jupyter notebook (`.ipynb`) first, then knit/export to HTML. Keep the intermediary file for troubleshooting.
- For **PowerPoint outputs**: generate `.pptx` directly using specialized packages (`officer` + `mschart` in R, or `python-pptx` in Python). Do not use an intermediary notebook.
- For **self-contained HTML**: all CSS, JS, fonts, and chart images must be inline or base64-encoded. No external dependencies.
- **Always include base sizes (n) and date ranges** in captions, footnotes, or subtitles of all charts and tables. These are critical for interpretation — a trend based on 20 people reads very differently from one based on 2,000.

## Exploratory data analysis

For exploratory analysis, prefer built-in functions from the **vivainsights** package over writing custom code from scratch. Key functions include:

- `create_bar()` — bar charts of metric averages by organizational attribute
- `create_rank()` — ranked comparisons across groups
- `create_trend()` — trend lines over time
- `create_boxplot()` — distributions by group
- `create_dist()` — distribution analysis

These functions accept a `return` parameter (R: `return`, Python: `return_type`) that can be set to `"table"` or `"data"` to return a data frame instead of a plot, making it easy to reuse the output in further analysis.

## Privacy and minimum group sizes

- Suppress any segment or group with fewer than the organization's minimum reporting threshold (default: 5 users) in all charts and tables.
- Do not expose raw `PersonId` values or email addresses in printed output.

## Official documentation

Column definitions and metric descriptions are maintained at:
[Microsoft Learn — Viva Insights metrics reference](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)

Do not hard-code specific column names from documentation — actual names can vary by tenant and localization. Use `import_query()` and `extract_hr()` to handle this dynamically.

## Prompt library

Ready-to-use prompt cards for common analytics tasks are available at:
[Frontier Analytics — Prompt Library](https://microsoft.github.io/viva-insights-sample-code/frontier-analytics-prompts/)
