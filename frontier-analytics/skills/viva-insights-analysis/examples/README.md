# Worked examples

Small, self-contained examples that run against the packages' **built-in sample
datasets** or, for example 6, a seeded synthetic panel generated in the script,
so they need no customer data and produce no privacy risk. Each
concept is provided as a matched pair: one Python script under `python/` and the
equivalent R script under `r/`.

The additional `merge_meeting_exports.py` helper is Python-only and uses
synthetic fixtures in `test_merge_meeting_exports.py`. It performs an optional
in-memory historical reconstruction, not a production file merge.

Examples 1 to 5 were originally verified end-to-end (exit 0) against R `vivainsights`
0.7.0.9000 and Python `vivainsights` 0.4.2. Example 6 was verified with R 4.6.1
(base R only) and Python 3.14 with `pandas` 3.0.5.

| # | Topic | Python | R |
|---|---|---|---|
| 1 | Import and validate a query | `python/01_import_and_validate.py` | `r/01_import_and_validate.R` |
| 2 | Metrics and group ranking | `python/02_metrics_and_ranking.py` | `r/02_metrics_and_ranking.R` |
| 3 | Copilot usage segmentation | `python/03_copilot_segmentation.py` | `r/03_copilot_segmentation.R` |
| 4 | Meeting Query quality filter | `python/04_meeting_quality.py` | `r/04_meeting_quality.R` |
| 5 | Person-to-person network | `python/05_network_analysis.py` | `r/05_network_analysis.R` |
| 6 | Usage archetypes (experimental) | `python/06_usage_archetypes.py` | `r/06_usage_archetypes.R` |

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
`matplotlib` in Python for the saved plot. Example 6 needs only base R, or
`pandas` in Python (installed with `vivainsights`).

## What each example shows

1. **Import and validate.** `check_query`, `extract_hr`, `extract_date_range`,
   and a privacy-threshold audit. In real work, swap the sample loader for
   `import_query("your_query.csv")`.
2. **Metrics and ranking.** `create_rank` (table and saved plot) and
   `keymetrics_scan` across several metrics. Plots are written to the system
   temp directory, so the skill folder stays clean.
3. **Copilot segmentation.** Builds a total-actions metric from the per-app
   action columns, retaining native unlicensed nulls. Runs `identify_usage_segments` on
   the intact weekly calendar, then withholds labels and derived rolling fields
   without a full contiguous eligible history (4 or 12 weeks). Unknown
   eligibility, unlicensed weeks and calendar gaps make the affected windows
   unavailable. Native null enabled days classify a person-week as unlicensed.
   Null actions among licensed people and unmatched licence-source joins leave
   affected histories unresolved. Other people's valid histories still run.
   The printed distribution counts person-weeks, not
   distinct people. Filter to one week for a point-in-time snapshot.
4. **Meeting quality.** The generalised meeting quality filter plus a per-rule
   attrition table. The sample Meeting Query is synthetic, so few rows survive.
   The lesson is the filter mechanics rather than the surviving counts.
5. **Network.** `network_p2p` builds the graph and `network_summary` computes
   node centrality. The R sample ships HR attributes so it runs directly. The
   Python sample is a bare edge list, so the script attaches a clearly labelled
   illustrative group that you replace with real HR attributes.
6. **Usage archetypes (experimental).** Generates a seeded, clearly labelled
   synthetic person-week panel in the script and applies reference rule v1
   from `reference/usage-archetypes.md`. It assigns a named evidence state to
   everyone, then Consistency, Surface mix (integer group-week rule) and
   Metered use (median split of positive M365 credits, ties Lean) to full
   profiles. It checks Surface mix boundary cases, then prints the context
   record, a profile summary that withholds groups under 10 (with further
   cells withheld so that no withheld cell can be recovered from printed or
   median-implied totals), and leave-one-week and split-half sensitivity
   with the same disclosure checks. The full-profile count reappears as the
   profile total and the sensitivity denominator. So whenever the
   evidence-state table withholds it, including whenever the other states
   together number fewer than 10, the profile summary and sensitivity are
   replaced by a "population too small for this breakdown" message. Both
   scripts share a small Park-Miller generator, so R and Python build the
   same panel. For the committed seed, byte-identical output and identical
   person-level classifications were checked manually. There is no automated
   parity test. The numbers describe synthetic data only.

## Conventions on display in these examples

- Built-in sample data or in-script synthetic data only, never customer data.
- Headless, file-based plotting so scripts never block.
- Framing that favors association over causation, with explicit units (for example hours rather than minutes).
- Any fabricated field (the illustrative network group) is labelled as such.

Sample dataset sizes differ slightly between packages (for instance the R
`pq_data` has fewer person-weeks than the Python `load_pq_data()`), so exact
counts in the printed output will not match line for line across languages. The
shapes and conclusions do. Example 6 is the exception: its synthetic panel is
generated identically in both languages, and its output was checked manually
to match exactly for the committed seed.

The corrected segmentation example is checked with Python `vivainsights` 0.4.3
and R `vivainsights` 0.7.2. From the repository root, run
`python -B -m unittest discover -s scripts -p test_analysis_recipes.py` and
`Rscript scripts/test_segmentation.R` for native unlicensed nulls, unknown-history,
calendar-gap and eligible-zero regression fixtures. These are sample-workflow tests, not a
general real-export calendar completion or join adapter.

## Optional Meeting Query merge

Read [Meeting Query specifics](../reference/meeting-query.md) before using
`merge_mq(old, new, attribute_columns=...)`. The caller must confirm comparable
export settings and keys and normalise the schemas explicitly. The helper keeps
the full raw ID union, prefers an older quality-passing row only when the newer
row fails, refuses populated HR conflicts and backfills blanks from the other
vintage. It returns an aggregate audit with selection and backfill sources.
This preference can change attendance totals, so compare a newest-only sensitivity.

From the skill directory:

```powershell
python -B examples\test_merge_meeting_exports.py
```

Tests cover selection, all seven quality rules, missing inputs, blank tokens,
both backfill directions, conflicts, schema/key/timestamp failures, empty inputs
and input preservation. No customer data or output files are used. The helper
and tests were executed with Python `vivainsights` 0.4.3 and `pandas` 3.0.5
installed. The helper itself uses only `pandas` and `numpy`.
The repository's `scripts/test_skill_references.py` also runs these regressions.
