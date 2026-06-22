---
layout: default
title: "Copilot Causal Toolkit"
eyebrow: "Causal inference · Toolkit"
description: "Run causal inference on Viva Insights data to estimate the effect of Copilot usage on seller productivity, burnout risk, and employee engagement — using double machine learning."
permalink: /copilot-causal-toolkit/
css: "/assets/css/causal-toolkit.css"
---

# Copilot Causal Toolkit

<div class="ct-callout is-important" markdown="1">
<span class="ct-callout-label">In short</span>
The [Copilot Causal Toolkit](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/copilot-causal-toolkit) is a set of Python / Jupyter notebooks that estimate the **causal effect of Copilot usage** on a workplace outcome using **double machine learning (DML)**. Pick a scenario below, then follow the five-step path to a result.
</div>

It helps you move beyond correlation and answer questions such as:

* Does Copilot usage increase the time our sellers spend with customers?
* Does Copilot usage reduce after-hours work and burnout risk?
* Does Copilot usage influence employee engagement?

## Choose your scenario

Each scenario maps to one outcome variable and one or two notebooks. The notebook file names describe the **outcome** (e.g. `AftCollabHours`), not the scenario label — use these cards to pick the right one.

<div class="ct-scenario-grid" markdown="0">
  <div class="ct-scenario-card">
    <span class="ct-scenario-icon">🤝</span>
    <h3>Seller Productivity</h3>
    <p class="ct-scenario-outcome">Outcome: External collaboration hours</p>
    <p><code>CI-DML_ExtCollabHours_PQ.ipynb</code> (Person Query)<br>
    <code>CI-DML_ExtCollabHours_SUR.ipynb</code> (Super Users Report)</p>
  </div>
  <div class="ct-scenario-card">
    <span class="ct-scenario-icon">🌙</span>
    <h3>Burnout Prevention</h3>
    <p class="ct-scenario-outcome">Outcome: After-hours collaboration hours</p>
    <p><code>CI-DML_AftCollabHours_PQ.ipynb</code> (Person Query)<br>
    <code>CI-DML_AftCollabHours_SUR.ipynb</code> (Super Users Report)</p>
  </div>
  <div class="ct-scenario-card">
    <span class="ct-scenario-icon">💬</span>
    <h3>Employee Engagement</h3>
    <p class="ct-scenario-outcome">Outcome: Ordinal survey metric (e.g. <code>eSat</code>)</p>
    <p><code>CI-DML_Engagement_PQ.ipynb</code> (Person Query only)</p>
  </div>
</div>

All three use Copilot usage (`Total_Copilot_actions_taken`) as the **treatment** variable. The Employee Engagement notebook is a **template** — because Glint survey metrics vary by organization, you update the outcome name, scale, and confounders to match your survey before running. See [Preparing your data]({{ site.baseurl }}/copilot-causal-toolkit-data/) for the columns each scenario needs.

## The path to a result

<ol class="ct-steps">
  <li markdown="1"><span class="ct-step-title">Set up your environment</span>
  Download the toolkit, install Python + the required packages, and open the folder in your editor. → [Setup &amp; installation]({{ site.baseurl }}/copilot-causal-toolkit-setup/)</li>
  <li markdown="1"><span class="ct-step-title">Prepare your data</span>
  Export a Person Query (recommended) or a Super Users Report, and include the columns your scenario needs. → [Preparing your data]({{ site.baseurl }}/copilot-causal-toolkit-data/)</li>
  <li markdown="1"><span class="ct-step-title">Configure the notebook</span>
  Edit a handful of parameters in the notebook cells — file paths, attributes, and date range. → [Configuring the notebook]({{ site.baseurl }}/copilot-causal-toolkit-configure/)</li>
  <li markdown="1"><span class="ct-step-title">Run the analysis</span>
  Run cell-by-cell the first time, then all at once. Troubleshoot common errors. → [Running &amp; troubleshooting]({{ site.baseurl }}/copilot-causal-toolkit-run/)</li>
  <li markdown="1"><span class="ct-step-title">Interpret the outputs</span>
  Read every plot, table, and sensitivity metric the toolkit produces. → [Interpretation Guide]({{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/)</li>
</ol>

## Continue reading

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/">
    <span class="vi-card-icon">🧰</span>
    <span class="vi-card-title">1 · Setup &amp; installation</span>
    <span class="vi-card-desc">Download the toolkit, install Python and the required packages, and open the project.</span>
    <span class="vi-card-more">Start →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-data/">
    <span class="vi-card-icon">📤</span>
    <span class="vi-card-title">2 · Preparing your data</span>
    <span class="vi-card-desc">Export from Viva Insights or a Super Users Report, and the columns each scenario needs.</span>
    <span class="vi-card-more">Explore →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/">
    <span class="vi-card-icon">⚙️</span>
    <span class="vi-card-title">3 · Configuring the notebook</span>
    <span class="vi-card-desc">The handful of parameters to edit before running, plus a pre-flight checklist.</span>
    <span class="vi-card-more">Explore →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-run/">
    <span class="vi-card-icon">▶️</span>
    <span class="vi-card-title">4 · Running &amp; troubleshooting</span>
    <span class="vi-card-desc">Run cell-by-cell or all at once, fix common errors, and an FAQ.</span>
    <span class="vi-card-more">Explore →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/">
    <span class="vi-card-icon">📈</span>
    <span class="vi-card-title">5 · Interpreting the outputs</span>
    <span class="vi-card-desc">A file-by-file walkthrough of every plot, table, and sensitivity metric.</span>
    <span class="vi-card-more">Explore →</span>
  </a>
  <a class="vi-card" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">
    <span class="vi-card-icon">🔬</span>
    <span class="vi-card-title">How it works</span>
    <span class="vi-card-desc">The DML and causal-forest methodology, assumptions, and limitations.</span>
    <span class="vi-card-more">Explore →</span>
  </a>
</div>

## What this analysis can and cannot prove

<div class="ct-callout is-warning" markdown="1">
<span class="ct-callout-label">Read before you act on results</span>
This toolkit estimates the causal effect of Copilot usage **under the assumptions of double machine learning** — chiefly *unconfoundedness* (all relevant confounders are measured and included), *overlap* (both Copilot users and comparable non/low-users exist across covariate values), and a correctly handled treatment definition.

When those hold, it can support statements like "for comparable employees, higher Copilot usage is associated with a change of X in the outcome that is plausibly causal." It **cannot** prove causation if important confounders are unobserved (e.g. unmeasured motivation or role changes), if there is no overlap between treated and untreated groups, or if Copilot usage is itself driven by the outcome (reverse causality).

Treat the estimates as **decision-support evidence** to be triangulated with experiments and domain knowledge, not as definitive proof. The [Interpretation Guide]({{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/) discusses these caveats per output.
</div>

New to causal inference more broadly? Start with the conceptual [Causal Inference in Copilot Analytics]({{ site.baseurl }}/causal-inference/) guide.
