# Worked examples

Small, self-contained examples that run against the packages' **built-in sample
datasets**, so they need no customer data and produce no privacy risk. Each
concept is provided as a matched pair: one Python script under `python/` and the
equivalent R script under `r/`.

All scripts were verified to run end-to-end (exit 0) against R `vivainsights`
0.7.0.9000 and Python `vivainsights` 0.4.2.

| # | Topic | Python | R |
|---|---|---|---|
| 1 | Import and validate a query | `python/01_import_and_validate.py` | `r/01_import_and_validate.R` |
| 2 | Metrics and group ranking | `python/02_metrics_and_ranking.py` | `r/02_metrics_and_ranking.R` |
| 3 | Copilot usage segmentation | `python/03_copilot_segmentation.py` | `r/03_copilot_segmentation.R` |
| 4 | Meeting Query quality filter | `python/04_meeting_quality.py` | `r/04_meeting_quality.R` |
| 5 | Person-to-person network | `python/05_network_analysis.py` | `r/05_network_analysis.R` |

## Running

```powershell
# Python (headless plotting)
$env:MPLBACKEND = "Agg"
python examples/python/01_import_and_validate.py

# R
Rscript examples/r/01_import_and_validate.R
```

Prerequisites: `pip install vivainsights` (Python) or
`install.packages("vivainsights")` (R). Example 2 also uses `ggplot2` in R and
`matplotlib` in Python for the saved plot.

## What each example shows

1. **Import and validate.** `check_query`, `extract_hr`, `extract_date_range`,
   and a privacy-threshold audit. In real work, swap the sample loader for
   `import_query("your_query.csv")`.
2. **Metrics and ranking.** `create_rank` (table and saved plot) and
   `keymetrics_scan` across several metrics. Plots are written to the system
   temp directory, so the skill folder stays clean.
3. **Copilot segmentation.** Builds a total-actions metric from the per-app
   action columns, fills NaNs with 0, then `identify_usage_segments`. Shows the
   full-panel segment mix; filter to one week for a point-in-time report.
4. **Meeting quality.** The generalised meeting quality filter plus a per-rule
   attrition table. The sample Meeting Query is synthetic, so few rows survive;
   the lesson is the filter mechanics, not the surviving counts.
5. **Network.** `network_p2p` builds the graph and `network_summary` computes
   node centrality. The R sample ships HR attributes so it runs directly; the
   Python sample is a bare edge list, so the script attaches a clearly labelled
   illustrative group that you replace with real HR attributes.

## Conventions on display in these examples

- Built-in sample data only, never customer data.
- Headless, file-based plotting so scripts never block.
- Association-not-causation framing and explicit units (hours, not minutes).
- Any fabricated field (the illustrative network group) is labelled as such.

Sample dataset sizes differ slightly between packages (for instance the R
`pq_data` has fewer person-weeks than the Python `load_pq_data()`), so exact
counts in the printed output will not match line for line across languages. The
shapes and conclusions do.
