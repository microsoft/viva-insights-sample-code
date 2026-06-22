---
layout: default
title: "Causal Toolkit — Running & Troubleshooting"
eyebrow: "Causal inference · Toolkit · Step 4"
description: "Run the Copilot Causal Toolkit notebooks cell-by-cell or all at once, resolve common errors, and read the FAQ."
permalink: /copilot-causal-toolkit-run/
css: "/assets/css/causal-toolkit.css"
---

<nav class="ct-series-nav" aria-label="Toolkit steps">
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit/">Overview</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/"><span class="ct-chip-step">1</span>Set up</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-data/"><span class="ct-chip-step">2</span>Data</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/"><span class="ct-chip-step">3</span>Configure</a>
  <a class="ct-chip is-current" href="{{ site.baseurl }}/copilot-causal-toolkit-run/"><span class="ct-chip-step">4</span>Run</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/"><span class="ct-chip-step">5</span>Interpret</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">How it works</a>
</nav>

# Running the analysis

With your data in `data/` and the [parameters configured]({{ site.baseurl }}/copilot-causal-toolkit-configure/), you can run the notebook either cell-by-cell or all at once. When it finishes, head to the [Interpretation Guide]({{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/) for a walkthrough of every output.

## Recommended: run cell-by-cell

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">First run</span>
Run **cell-by-cell** the first time. It lets you catch errors early (missing columns, wrong paths), review intermediate outputs, understand each step, and adjust parameters before continuing.
</div>

- **Jupyter:** click inside a cell and press `Shift + Enter` to run it and move to the next.
- **VS Code:** click the ▶️ button to the left of each cell, or press `Shift + Enter`.

## Running all cells at once

Once you're confident the notebook is configured correctly:

- **Jupyter:** `Cell → Run All`
- **VS Code:** `Run All` at the top of the notebook

**Expected runtime:** 10–30 minutes depending on data size.

## Troubleshooting common errors

<details class="ct-details">
<summary><code>FileNotFoundError: [Errno 2] No such file or directory</code></summary>
<div markdown="1">

**Cause:** the data file path is incorrect or the file doesn't exist.

1. Check the CSV is in the `copilot-causal-toolkit/data/` folder.
2. Verify the filename in `data_file_path` matches exactly, including capitalization.
3. Try a full absolute path: `data_file_path = r"C:\Users\YourName\...\data\file.csv"`.

</div>
</details>

<details class="ct-details">
<summary><code>KeyError: 'ColumnName'</code> or "Column not found"</summary>
<div markdown="1">

**Cause:** a variable in the configuration doesn't exist in your data.

1. Check the list of columns your notebook prints early on.
2. Update `SUBGROUP_VARS`, `NETWORK_VARS`, `COLLABORATION_VARS` to match your actual column names.
3. Remove any variables from the lists that don't exist in your dataset.

</div>
</details>

<details class="ct-details">
<summary><code>ValueError: could not convert string to float</code></summary>
<div markdown="1">

**Cause:** a data-type mismatch or missing values in numeric columns.

Check for non-numeric values in metrics like `Total_Copilot_actions_taken` or the outcome variable.

</div>
</details>

<details class="ct-details">
<summary><code>MemoryError</code> or the notebook becomes unresponsive</summary>
<div markdown="1">

**Cause:** the dataset is too large for available memory.

1. Filter to a smaller time period.
2. Reduce the number of subgroups analyzed.
3. Close other applications to free up memory.

</div>
</details>

<div class="ct-callout" markdown="1">
<span class="ct-callout-label">Use GitHub Copilot for help</span>
If you have **GitHub Copilot / Copilot Chat** in VS Code: select the error message or problematic code, open Copilot Chat (`Ctrl+Shift+I` / `Cmd+Shift+I`), and ask specific questions — e.g. *"Why am I getting this error?"*, *"How do I fix this KeyError for the column 'Organization'?"*, or *"How do I change this variable list to use different column names?"*
</div>

### Still stuck?

- Read the cell outputs carefully — error messages usually pinpoint the problem.
- Confirm all prerequisites are installed (`pip list` to verify).
- Ensure your data follows the expected Person Query or Super Users Report schema.
- Try running on a small sample of data first to isolate the issue.

## FAQ

<details class="ct-details">
<summary>How much data do I need?</summary>
<div markdown="1">

Aim for **6 months of weekly data** (roughly 12–26 weeks). More person-weeks generally produce narrower confidence intervals and more reliable subgroup estimates.

</div>
</details>

<details class="ct-details">
<summary>I get "no significant subgroups" — what now?</summary>
<div markdown="1">

This can mean there genuinely is no detectable effect, or that you have insufficient data / overlap to detect one. Check that both Copilot users and comparable non/low-users exist across covariate values, widen the time window, or reduce the number of subgroup attributes. See [What this analysis can and cannot prove]({{ site.baseurl }}/copilot-causal-toolkit/#what-this-analysis-can-and-cannot-prove).

</div>
</details>

<details class="ct-details">
<summary>Which notebook do I use for my question?</summary>
<div markdown="1">

Pick by **outcome**, not scenario label: external collaboration → `CI-DML_ExtCollabHours_*`, after-hours → `CI-DML_AftCollabHours_*`, survey engagement → `CI-DML_Engagement_PQ`. The [overview]({{ site.baseurl }}/copilot-causal-toolkit/#choose-your-scenario) has a card per scenario.

</div>
</details>

<details class="ct-details">
<summary>Can I use this with a Super Users Report instead of a Person Query?</summary>
<div markdown="1">

Yes — use the `_SUR.ipynb` notebooks, which handle the report's `Date` column. Note that some Person Query metrics aren't available in the report; if critical confounders are missing, prefer a Person Query. See [Preparing your data]({{ site.baseurl }}/copilot-causal-toolkit-data/#method-2--export-from-a-super-users-report).

</div>
</details>

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">Next</span>
Run complete? Continue to the [**Interpretation Guide**]({{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/) to read every output the toolkit produces.
</div>

<nav class="ct-pager" aria-label="Toolkit pagination">
  <a class="ct-pager-link" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/">
    <span class="ct-pager-dir">← Back</span>
    <span class="ct-pager-title">3 · Configuring the notebook</span>
  </a>
  <a class="ct-pager-link is-next" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/">
    <span class="ct-pager-dir">Next →</span>
    <span class="ct-pager-title">5 · Interpreting the outputs</span>
  </a>
</nav>
