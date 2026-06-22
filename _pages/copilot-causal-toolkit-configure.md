---
layout: default
title: "Causal Toolkit — Configuring the Notebook"
eyebrow: "Causal inference · Toolkit · Step 3"
description: "The handful of parameters to edit in the Copilot Causal Toolkit notebook cells before running — file paths, attributes, date range — plus a pre-flight checklist."
permalink: /copilot-causal-toolkit-configure/
css: "/assets/css/causal-toolkit.css"
---

<nav class="ct-series-nav" aria-label="Toolkit steps">
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit/">Overview</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/"><span class="ct-chip-step">1</span>Set up</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-data/"><span class="ct-chip-step">2</span>Data</a>
  <a class="ct-chip is-current" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/"><span class="ct-chip-step">3</span>Configure</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-run/"><span class="ct-chip-step">4</span>Run</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/"><span class="ct-chip-step">5</span>Interpret</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">How it works</a>
</nav>

# Configuring the notebook

Before running, you customize a few parameters **directly inside the notebook cells**. These are **not** terminal commands — you open the `.ipynb` file in VS Code or Jupyter, click into the relevant cell, and change the values in place.

<div class="ct-callout" markdown="1">
<span class="ct-callout-label">How to edit a cell</span>
1. Open your chosen notebook (e.g. `CI-DML_AftCollabHours_PQ.ipynb`).
2. Scroll to the cell noted below (cell numbers are indicated for each setting).
3. Click into the cell and change the value.
4. **Don't run the cell yet** — review all parameters first, then run from the top.
</div>

## 1 · File paths — Cell 3 (Setup and Imports)

Update the data file path to match your CSV filename:

```python
# Update this line to match your data file name
data_file_path = os.path.join(script_dir, '..', 'data', 'PersonQuery.csv')
# For example:
# data_file_path = os.path.join(script_dir, '..', 'data', 'MyCompany_PersonQuery_2025.csv')
```

In the same cell, update the output directory name:

```python
# Replace [YOUR COMPANY] with your organization name
output_base_dir = os.path.join(script_dir, '..', 'output', 'Subgroup Analysis - [YOUR COMPANY]')
# e.g.
# output_base_dir = os.path.join(script_dir, '..', 'output', 'Subgroup Analysis - Contoso')
```

## 2 · Effect direction — Cell 3 (After-Hours &amp; Engagement notebooks only)

In the After-Hours notebooks (`CI-DML_AftCollabHours_*.ipynb`) and the Engagement notebook, set which direction of effect to search for:

```python
# Find subgroups with NEGATIVE (reductions) or POSITIVE (increases) effects
FIND_NEGATIVE_EFFECTS = True   # True = reductions, False = increases
```

This toggle does **not** exist in the External Collaboration notebooks.

<details class="ct-details">
<summary>Engagement notebook only — outcome variable &amp; scale (Cell 4)</summary>
<div markdown="1">

The Engagement notebook needs the ordinal survey outcome and its scale:

```python
# Match OUTCOME_VAR to your Glint survey metric column name
OUTCOME_VAR = 'eSat'   # e.g. 'eSat', 'eNPS', etc.

# Match the scale to your survey's response range
OUTCOME_SCALE_MIN = 1   # Minimum value on the scale
OUTCOME_SCALE_MAX = 7   # Maximum value (e.g. 5, 7, 9, or 10)
```

`OUTCOME_VAR` must match the exact column name in your Person Query export. The scale parameters drive ceiling/floor diagnostics and the interpretation of effect magnitudes on the ordinal scale.

</div>
</details>

## 3 · Organizational attributes — Cell 7 (Variable Definitions)

Update these lists to match the column names in **your** data.

**`SUBGROUP_VARS`** — attributes for subgroup / heterogeneity analysis:

```python
SUBGROUP_VARS = [
    'FunctionType',      # e.g. 'Function', 'Department', 'Division'
    'IsManager',         # usually consistent across organizations
    'LevelDesignation',  # e.g. 'Level', 'Grade', 'Band'
    'Organization'       # update to match your data
]
```

These variables are used both for person-level aggregation and to identify subgroups with heterogeneous effects. Include **2–4** key attributes you want to analyze.

<details class="ct-details">
<summary>Network &amp; collaboration variable lists (usually consistent, but verify)</summary>
<div markdown="1">

```python
NETWORK_VARS = [
    'Internal_network_size',
    'External_network_size',
    'Strong_ties',
    'Diverse_ties'
]

COLLABORATION_VARS = [
    'Collaboration_hours',
    'Available_to_focus_hours',
    'Active_connected_hours',
    'Uninterrupted_hours'
]
```

</div>
</details>

## 4 · Date range filter — Cell 7

Match the period covered by your data:

```python
start_date_str = '2025-03-01'  # your data's start date
end_date_str   = '2025-06-30'  # your data's end date
```

## 5 · Treatment &amp; outcome — Cell 5 (usually no change)

These are typically standard, but verify they exist in your data:

- **Treatment:** `Total_Copilot_actions_taken`
- **Outcome:** `External_collaboration_hours`, `After_hours_collaboration_hours`, or your chosen survey metric (e.g. `eSat`).

For the Engagement notebook, the outcome is specified in the configuration cell (see section 2) rather than pre-defined.

## Pre-flight checklist

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">Before you run</span>

- [ ] Data file is in the `data/` folder
- [ ] `data_file_path` matches your CSV filename exactly (case-sensitive)
- [ ] `output_base_dir` has your organization name (optional but recommended)
- [ ] Every variable in `SUBGROUP_VARS` exists in your dataset
- [ ] `FIND_NEGATIVE_EFFECTS` is set appropriately *(after-hours &amp; engagement only)*
- [ ] *(Engagement only)* `OUTCOME_VAR` matches the survey column name
- [ ] *(Engagement only)* `OUTCOME_SCALE_MIN` / `OUTCOME_SCALE_MAX` match the survey scale
</div>

<nav class="ct-pager" aria-label="Toolkit pagination">
  <a class="ct-pager-link" href="{{ site.baseurl }}/copilot-causal-toolkit-data/">
    <span class="ct-pager-dir">← Back</span>
    <span class="ct-pager-title">2 · Preparing your data</span>
  </a>
  <a class="ct-pager-link is-next" href="{{ site.baseurl }}/copilot-causal-toolkit-run/">
    <span class="ct-pager-dir">Next →</span>
    <span class="ct-pager-title">4 · Running &amp; troubleshooting</span>
  </a>
</nav>
