# vivainsights packages: function inventory

> **Live, machine-readable equivalent:** both packages now publish an
> auto-generated `llms.txt` (R: https://microsoft.github.io/vivainsights/llms.txt,
> Python: https://microsoft.github.io/vivainsights-py/llms.txt) that reflects the
> exact installed API and is more current than this file. Check it first. Use
> this file for the grouped context (wrapper families, sample-dataset shapes,
> export/theming) it doesn't cover. See `ecosystem.md`.

Grouped by purpose. Function names verified against **R `vivainsights` 0.7.0.9000**
(~156 exported objects) and **Python `vivainsights` 0.4.2** (~74 exported objects).
R is the fuller package. Where a Python equivalent exists it is noted, otherwise
assume R-only.

> **Return argument:** R functions take `return = "plot" | "table" | "data"`.
> Python functions take `return_type = "plot" | "table" | "data"`. This is the
> most common cross-language mistake.

## 1. Import, validation, and inspection

| Purpose | R | Python |
|---|---|---|
| Read a query export, clean names, parse dates | `import_query(x, pid, dateid, date_format = "%m/%d/%Y", convert_date = TRUE, encoding = "UTF-8")` | `import_query(x, encoding='utf-8')` |
| Validate query structure / variable types | `check_query(data)` | `check_query(data)` |
| Full validation report (HTML/console) | `validation_report()` | N/A |
| Extract the HR/organisational attribute columns | `extract_hr(data, ...)` | `extract_hr(data)` |
| Extract the date range of a query | `extract_date_range(data)` | `extract_date_range(data)` |
| Anonymise identifiers | `anonymise()` / `anonymize()` | N/A |

`import_query()` is the canonical entry point. It reads the CSV, standardises
column names, and (in R) converts the date column. Prefer it over a raw
`read.csv` / `pd.read_csv` so downstream functions recognise the schema. For
non-English exports see `data-pitfalls.md`.

## 2. The `identify_*` family (population and behaviour flags)

| Purpose | R | Python |
|---|---|---|
| Copilot / metric usage segments | `identify_usage_segments(data, metric, metric_str, version = "12w", power_thres = 15, return = "data")` | `identify_usage_segments(data, metric, metric_str, version='12w', return_type='data', power_thres=None)` |
| Habitual-use flag | `identify_habit(data, metric, threshold = 1, width, max_window, hrvar, return = "plot")` | `identify_habit(data, metric, threshold=1, width=1, max_window=4, return_type='plot')` |
| Holiday weeks (low-activity weeks) | `identify_holidayweeks(data, ...)` | `identify_holidayweeks(data, ...)` |
| Inactive weeks | `identify_inactiveweeks()` | `identify_inactiveweeks(data, ...)` |
| Non-knowledge-worker / low-signal | `identify_nkw()` | `identify_nkw(data, ...)` |
| Statistical outliers | `identify_outlier()` | `identify_outlier(data, ...)` |
| Tenure from a hire-date column | `identify_tenure()` | `identify_tenure(data, ...)` |
| Churn between two periods | N/A | `identify_churn()` |
| Privacy-threshold groups | `identify_privacythreshold(data, hrvar = extract_hr(data), mingroup = 5, return = "table")` | N/A |
| Structural shifts in a series | `identify_shifts()` | N/A |

The `identify_usage_segments()` classifier (Power / Habitual / Novice / Low /
Non-user) is central to Copilot adoption work. See `analysis-conventions.md` for
the segment definitions and the correct licensed-population denominator.

## 3. Metric-family wrappers (R)

R ships convenience wrappers for each collaboration signal, each with a
consistent set of suffixes. `X` is one of: `collaboration` (alias `collab`),
`afterhours`, `email`, `meeting`, `one2one`, `external`.

| Suffix | Output |
|---|---|
| `X_summary` / `X_sum` | Summary table or bar of the group means |
| `X_dist` | Distribution across bands |
| `X_fizz` | Fizzy (jittered) per-person scatter by group |
| `X_line` | Time trend line by group |
| `X_trend` | Heat-style week-by-group trend |
| `X_rank` | Ranked groups by the metric |

Example: `collaboration_summary()`, `afterhours_dist()`, `email_line()`,
`meeting_rank()`, `one2one_trend()`, `external_fizz()`. **Python does not ship
these wrappers.** Build the equivalent with the generic `create_*` viz functions
below plus a group aggregation.

## 4. Generic visualisations (`create_*`)

Both packages share the core creators. Pass `metric` and usually `hrvar`.

| Function | Purpose | Python? |
|---|---|---|
| `create_bar` / `create_bar_asis` | Group mean bar chart | yes |
| `create_line` | Time trend by group | yes |
| `create_trend` | Week x group heat trend | yes |
| `create_rank` | Ranked groups on a metric | yes |
| `create_boxplot` | Distribution by group | yes |
| `create_dist` (`create_stacked`) | Banded stacked distribution | `create_inc` variants |
| `create_fizz` | Jittered per-person scatter | via wrappers |
| `create_scatter` | Two-metric scatter | N/A |
| `create_bubble` | Two-metric bubble by group | yes |
| `create_sankey` | Flow between two categoricals | yes |
| `create_lorenz` | Lorenz curve / inequality | yes |
| `create_inc` / `create_incidence` | Incidence (threshold-crossing) charts | yes |
| `keymetrics_scan` | Multi-metric heatmap scan across groups | yes |

R signature example:
`create_rank(data, metric, hrvar = extract_hr(data), mingroup = 5, return = "table", mode = "simple")`
Python: `create_rank(data, metric, hrvar, mingroup=5, return_type='plot')`.

`keymetrics_scan()` defaults to a broad metric list (collaboration, meeting,
email, focus, network sizes). The exact default list differs slightly between R
and Python, so pass an explicit `metrics=` vector when you need reproducibility.

## 5. Analytical / modelling functions

| Purpose | R | Python |
|---|---|---|
| Information Value (predictor screening) | `create_IV(data, predictors, outcome, bins = 5, siglevel = 0.05, return = "plot")`, `calculate_IV`, `IV_report` | `create_IV(data, predictors, outcome, bins=5, return_type='plot')`, `calculate_IV`, `create_odds_ratios`, `plot_WOE` |
| Person-to-person network | `network_p2p(data, hrvar = "Organization", return = "plot", centrality, community, layout = "mds")` | `network_p2p(data, hrvar='Organization', return_type='plot', centrality, community, layout='mds')` |
| Group-to-group network | `network_g2g()` | `network_g2g()` |
| Network summary metrics | `network_summary()` | `network_summary()` |
| Pairwise counts / co-occurrence | `pairwise_count()` | N/A |
| Interrupted time-series (ITSA) | `create_itsa(data, metrics, before_start, before_end, after_start, after_end, return = "table")` | `plot_ts_us` (usage TS) |
| Survival / retention curves | `create_survival()` (+ `_prep`, `_calc`, `_viz`) | N/A |
| Inequality (Gini) | `compute_gini`, `create_lorenz` | `compute_gini`, `create_lorenz` |
| Favourability scoring | N/A | `compute_fav` |
| Correlation (Chatterjee xi) | `xicor()` | `xicor()` |

## 6. Text mining on Meeting Query subjects (R only)

`tm_clean()`, `tm_freq()`, `tm_cooc()` (co-occurrence), `tm_wordcloud()`, and the
one-shot `meeting_tm_report()`. Use these to categorise meeting subject lines.
There is no Python equivalent, so replicate with a standard NLP stack if needed.

## 7. Built-in sample datasets (use these, never customer data)

| Dataset | R | Python |
|---|---|---|
| Person Query | `pq_data` | `load_pq_data()` (also `pq_data`) |
| Meeting Query | `mt_data` | `load_mt_data()` (also `mt_data`) |
| Person-to-person network | `p2p_data`, `p2p_data_sim()` | `load_p2p_data()`, `p2p_data_sim()` |
| Group-to-group network | `g2g_data` | `load_g2g_data()`, `g2g_data` |
| Person-to-group network | N/A | `load_p2g_data()`, `p2g_data` |

Verified shapes (current versions): the Python `load_pq_data()` returns ~10,500
person-weeks x 73 cols and the R `pq_data` ~6,900 x 73 (the two packages ship
slightly different samples). `mt_data` is ~612 rows x 41 cols in both. Use them
for every worked example, doc snippet, and test.

## 8. Export and theming

- Export any object (data frame, ggplot/matplotlib figure) with `export()`.
  R: `export(x, method = ...)`. Python: `export(x, file_format='auto', path='insights export', timestamp=True)`.
- R plots use `theme_wpa()` / `theme_wpa_basic()`. Both packages expose a Viva
  Insights colour palette: `Colors` / `color_codes` / `COLOR_PALLET_ALT_1` in
  Python, and `rgb2hex` plus palette helpers in R.

## Finding the exact signature at runtime

Do not guess signatures. Confirm against the installed version:

```r
args(vivainsights::create_rank)          # or ?create_rank
```
```python
import inspect, vivainsights as vi
print(inspect.signature(vi.create_rank))
```
