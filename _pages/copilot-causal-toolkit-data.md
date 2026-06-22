---
layout: default
title: "Causal Toolkit — Preparing Your Data"
eyebrow: "Causal inference · Toolkit · Step 2"
description: "Export Viva Insights data for the Copilot Causal Toolkit — via a Person Query or a Super Users Report — and the exact columns each scenario needs."
permalink: /copilot-causal-toolkit-data/
css: "/assets/css/causal-toolkit.css"
---

<nav class="ct-series-nav" aria-label="Toolkit steps">
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit/">Overview</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/"><span class="ct-chip-step">1</span>Set up</a>
  <a class="ct-chip is-current" href="{{ site.baseurl }}/copilot-causal-toolkit-data/"><span class="ct-chip-step">2</span>Data</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/"><span class="ct-chip-step">3</span>Configure</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-run/"><span class="ct-chip-step">4</span>Run</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/"><span class="ct-chip-step">5</span>Interpret</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">How it works</a>
</nav>

# Preparing your data

There are two ways to obtain data for the analysis:

1. **Export a Person Query** as a CSV from Viva Insights — *recommended*.
2. **Export a CSV from an existing Super Users Report**, if you already have one.

<div class="ct-callout" markdown="1">
<span class="ct-callout-label">Which method?</span>
We generally recommend **Method 1 (Person Query)** — it guarantees comprehensive coverage of the covariates needed for a result you can be confident in. Method 2 is quicker if you already have a Super Users Report, but some metrics may be missing.
</div>

Whichever you choose, save the resulting CSV into the toolkit's **`data/`** folder.

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">Your data stays local</span>
The analysis runs entirely on your machine — no data is uploaded anywhere. The `data/` folder is covered by a `.gitignore` rule (`*.csv`), so your exports won't be committed if you're working inside a Git clone. Keep filenames ending in `.csv` (lower-case) so that rule reliably applies on every operating system.
</div>


## Method 1 — Export a Person Query from Viva Insights

1. Open the [Viva Insights analysis page](https://analysis.insights.viva.office.com/Analysis/CreateAnalysis).
2. Select **Person Query → ‘Set up analysis’**.
3. Configure:
   * **Time period:** Last 6 months (rolling)
   * **Group by:** Week
   * **Metrics:** Include the columns listed under *Columns to include* below for your scenario.
   * **Filter:** `Is Active = True` (if available) — you can validate the number of employees here.
   * **Attributes:** Include `Organization` and `Function Type` (others optional) — this is the last box on the page.
4. **Save and Run** the query. Wait until **Status = Completed**, then export the CSV into `copilot-causal-toolkit/data/`.

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">Employee Engagement scenario</span>
The Person Query must include Glint survey data as the outcome. This needs a prior setup step to import survey data into Viva Insights — see [Import survey data from Viva Glint](https://learn.microsoft.com/en-us/viva/insights/advanced/admin/import-survey-glint). Once imported, metrics such as `eSat` appear as columns in the export.
</div>

## Method 2 — Export from a Super Users Report

<details class="ct-details">
<summary>Show Super Users Report export steps</summary>
<div markdown="1">

This assumes you already have a Super Users Report (`.pbit`/`.pbix`) populated with Viva Insights data. It contains pre-aggregated data that can be exported without setting up a new Person Query.

**From Power BI Desktop:**

1. **Open your Super Users Report.** If you don't have the file, ask your Viva Insights admin or check your organization's shared workspace.
2. **Open the Table view** — click the **Table** icon on the left sidebar.
3. **Find the data table** — look for person-level data with columns like `PersonId`, `Date`, `Total_Copilot_actions_taken`, collaboration metrics, and organizational attributes. It's usually called `Table`, but may have been renamed.
4. **Export to CSV:**
   * **Option A:** Right-click the table name → **Copy table** → paste into Excel → save as CSV.
   * **Option B:** Select the table → **Home → Transform data** to open Power Query Editor → right-click the table → **Export** → CSV.
5. **Verify the export** contains multiple weeks (ideally 12–26), all required columns, and no excessive blank rows.
6. **Save** the CSV into `copilot-causal-toolkit/data/` with a descriptive name (e.g. `SuperUsersReport_Export_2025.csv`).

**Important notes for Super Users Report data:**

- The report uses `Date` instead of `MetricDate` as the date column.
- The SUR notebooks (ending in `_SUR.ipynb`) are designed to handle this schema difference.
- Some Person Query metrics may be unavailable in the report (e.g. `Available_to_focus_hours`, `Weekend_collaboration_hours`). If critical metrics are missing, use Method 1 instead.

**Alternative: export from Power BI Service (online).** Open the report at powerbi.com → navigate to the page with the data table → click **…** (More options) on a visual → **Export data → Underlying data → .csv**. Some organizations restrict export; contact your Power BI admin if needed.

</div>
</details>

## Columns to include

The treatment variable is always **`Total_Copilot_actions_taken`**. Select the scenario you're running for its outcome and recommended confounders. **Organizational attributes** (e.g. `Organization`, `Function`, `Level`, `IsManager`, `Area`) are used for heterogeneity analysis in every scenario — include as many as you can; exact names vary by organization, so update them in the [notebook configuration]({{ site.baseurl }}/copilot-causal-toolkit-configure/).

<div class="ct-tabs" data-ct-tabs>
  <div class="ct-tablist" role="tablist" aria-label="Scenario">
    <button class="ct-tab" type="button">Seller Productivity</button>
    <button class="ct-tab" type="button">Burnout Prevention</button>
    <button class="ct-tab" type="button">Employee Engagement</button>
  </div>

  <div class="ct-panel" markdown="1">
**Goal:** understand how Copilot usage changes time spent collaborating with external stakeholders and customers.

* **Outcome:** `External_collaboration_hours` — hours in meetings, emails, chats, and calls with people outside the organization.
* **Treatment:** `Total_Copilot_actions_taken`.
* **Confounders (time-varying controls):**
  - `Meeting_hours`, `Email_hours`, `Chat_hours`, `Collaboration_hours`
  - `Internal_network_size`, `Networking_outside_organization`
  - `Total_focus_hours`
  - Other relevant behavioral and network metrics from your Person Query.
  </div>

  <div class="ct-panel" markdown="1">
**Goal:** understand how Copilot usage changes after-hours work patterns, which affect wellbeing and burnout risk.

* **Outcome:** `After_hours_collaboration_hours` — work-related activity outside standard business hours.
* **Treatment:** `Total_Copilot_actions_taken`.
* **Confounders (time-varying controls):**
  - `Meeting_hours`, `Email_hours`, `Chat_hours`, `Collaboration_hours`
  - `Internal_network_size`, `Networking_outside_organization`
  - `Total_focus_hours`, `Workweek_span`
  - Other relevant behavioral and network metrics from your Person Query.
  </div>

  <div class="ct-panel" markdown="1">
**Goal:** understand how Copilot usage influences employee engagement, measured by an **ordinal survey outcome** (e.g. a Glint metric). Unlike the other scenarios, the outcome is survey-based rather than a continuous Viva Insights metric.

<div class="ct-callout is-important" markdown="1">
<span class="ct-callout-label">Template notebook</span>
Glint metrics vary across organizations and may be custom-defined. Review and update the outcome variable, its scale, and the confounders to match your data before running.
</div>

* **Outcome:** a Glint metric such as `eSat` (Employee Satisfaction) — set this to whichever ordinal survey outcome you intend to evaluate.
* **Treatment:** `Total_Copilot_actions_taken`.
* **Outcome scale configuration:** because the outcome is ordinal, set `OUTCOME_SCALE_MIN` / `OUTCOME_SCALE_MAX` to match the survey scale (e.g. 1–5, 1–7, 1–10). These drive ceiling/floor diagnostics and interpretation.
* **Confounders (starting point — revise per outcome):**
  - `Collaboration_hours`, `Available_to_focus_hours`, `Active_connected_hours`, `Uninterrupted_hours`
  - `After_hours_collaboration_hours`, `Collaboration_span`
  - `Meeting_and_call_hours_with_manager_1_1`
  - Other relevant behavioral and network metrics from your Person Query.

**Data requirement:** the Person Query must include Glint survey data — see [Import survey data from Viva Glint](https://learn.microsoft.com/en-us/viva/insights/advanced/admin/import-survey-glint). Only the Person Query (PQ) notebook exists for this scenario; there is no SUR version.
  </div>
</div>

<details class="ct-details">
<summary>Explore which HR / organizational attributes are in your dataset</summary>
<div markdown="1">

Run this to list every HR attribute and its value counts:

```python
hrvar_str = vi.extract_hr(data, return_type = 'vars').columns

for hr_var in hrvar_str:
    hrvar_table = vi.hrvar_count(data, hrvar = hr_var, return_type = 'table')
    print(f"\nValue counts for {hr_var}:")
    print(hrvar_table)

for hr_var in hrvar_str:
    vi.hrvar_count(data = data, hrvar = hr_var, return_type = 'plot')
```

</div>
</details>

<nav class="ct-pager" aria-label="Toolkit pagination">
  <a class="ct-pager-link" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/">
    <span class="ct-pager-dir">← Back</span>
    <span class="ct-pager-title">1 · Setup &amp; installation</span>
  </a>
  <a class="ct-pager-link is-next" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/">
    <span class="ct-pager-dir">Next →</span>
    <span class="ct-pager-title">3 · Configuring the notebook</span>
  </a>
</nav>

<script src="{{ '/assets/js/causal-toolkit.js' | relative_url }}"></script>
