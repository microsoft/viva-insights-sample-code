---
layout: page
title: "Getting Started"
eyebrow: "Essentials"
description: "Set up R or Python for Viva Insights analytics. Install the vivainsights package, export your first query, and avoid the most common data pitfalls."
permalink: /getting-started/
css: "/assets/css/lang-switch.css"
last_validated: 2026-08-10
---

# Getting Started with Viva Insights Analytics

This page gets you from a blank machine to your first successful Viva Insights analysis in R or Python. For custom KPIs, multi-query joins, and automated reporting patterns, see [Essentials]({{ site.baseurl }}/essentials/) and [Advanced Analytics]({{ site.baseurl }}/advanced/) once you are set up.

{% include last-validated.html %}

<div class="lang-switch" role="group" aria-label="Choose code language for this page">
  <span class="lang-switch-label">Show code in:</span>
  <div class="lang-switch-group">
    <button type="button" class="lang-switch-btn" data-lang-btn="r">R</button>
    <button type="button" class="lang-switch-btn" data-lang-btn="python">Python</button>
  </div>
  <span class="lang-switch-note">Remembers your choice on this device.</span>
</div>
<script>
(function () {
  var stored = null;
  try { stored = window.localStorage.getItem('vi-lang-pref'); } catch (e) {}
  document.documentElement.setAttribute('data-lang', stored === 'python' ? 'python' : 'r');
})();
</script>

---

## Prerequisites

<div data-lang-block="r" markdown="1">
### R

- **R 4.1 or higher.** [Download R](https://cran.r-project.org/).
- **RStudio** (recommended). [Download RStudio](https://www.rstudio.com/products/rstudio/download/).
</div>

<div data-lang-block="python" markdown="1">
### Python

- **Python 3.9 or higher.** [Download Python](https://www.python.org/downloads/). Python 3.7 and 3.8 have reached end-of-life and are not supported.
- **pip**, usually included with Python.
- **A virtual environment** (recommended), so this project's packages stay isolated from others.
</div>

## Install the vivainsights package

<div data-lang-block="r" markdown="1">
```r
install.packages("vivainsights")
```

Add other packages only as your analysis needs them. `tidyverse` is a common companion for data wrangling, and `igraph` / `visNetwork` are useful if you plan to render network graphs yourself instead of using the package's built-in `network_p2p()` / `network_g2g()` plots.
</div>

<div data-lang-block="python" markdown="1">
```bash
python -m venv viva-insights-env

# Activate the virtual environment
# Windows:
viva-insights-env\Scripts\activate
# macOS/Linux:
source viva-insights-env/bin/activate

pip install vivainsights
```

`vivainsights` installs `pandas` as a dependency. Add other packages such as `jupyter`, `matplotlib`, or `networkx` only as your analysis needs them.
</div>

### Verify your installation

<div data-lang-block="r" markdown="1">
```r
library(vivainsights)
packageVersion("vivainsights")
help(package = "vivainsights")
```
</div>

<div data-lang-block="python" markdown="1">
```python
import vivainsights as vi
print(vi.__version__)
```
</div>

---

## Export your data from Viva Insights

Person query data is the most common starting point: one row per person per period, with HR attributes like organization, function, and level as columns. Other query types (Meeting Query, Person-to-Person, Group-to-Group, Person-to-Group) have different grains and are used for more specific analyses.

To export a query, use the query designer in the Viva Insights Analyst portal. Since the exact navigation can change between product releases, follow the current steps in [Microsoft's own Analyst portal documentation](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/query-designer) rather than a screenshot that may go stale.

## Load and explore your data

<div data-lang-block="r" markdown="1">
```r
library(vivainsights)

# import_query() standardizes column names and cleans special characters,
# which read.csv() does not do for you.
person_data <- import_query("path/to/your/person_query.csv")

check_query(person_data)
create_bar(person_data, metric = "Collaboration_hours")
```

No query export handy yet? Use the package's built-in sample dataset instead:

```r
person_data <- pq_data
```
</div>

<div data-lang-block="python" markdown="1">
```python
import vivainsights as vi

# import_query() standardizes column names and cleans special characters,
# which pd.read_csv() does not do for you.
person_data = vi.import_query("path/to/your/person_query.csv")

print(person_data.info())
vi.create_bar(person_data, metric="Collaboration_hours")
```

No query export handy yet? Use the package's built-in sample dataset instead:

```python
person_data = vi.load_pq_data()
```
</div>

## Avoid these common pitfalls

A handful of issues account for most of the confusing results people run into with a first export:

- **`IsManager` and similar flags often arrive as `"Yes"`/`"No"` text** rather than booleans, so a filter like `IsManager == TRUE` silently matches nothing.
- **`"#N/A"` sometimes arrives as a literal string** rather than a true null, which can inflate a group's row count with a bogus category.
- **Viva Insights suppresses small groups** (commonly under 5 people) in the portal. Reproduce that same threshold locally, or your numbers will not match what stakeholders see in the portal.
- **Holiday and low-activity weeks** shift population-level averages in ways that can look like a real behavioral change.

These are documented in full, with the fix for each, in the [Viva Insights Analysis skill's data pitfalls reference](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/skills/viva-insights-analysis/reference/data-pitfalls.md) and the [schema documentation]({{ site.baseurl }}/frontier-analytics-schemas/).

## Common analysis patterns

**Time trend:**

<div data-lang-block="r" markdown="1">
```r
create_trend(person_data, metric = "Collaboration_hours")
```
</div>
<div data-lang-block="python" markdown="1">
```python
vi.create_trend(person_data, metric="Collaboration_hours")
```
</div>

**Distribution by group:**

<div data-lang-block="r" markdown="1">
```r
create_boxplot(person_data, metric = "Meeting_hours", hrvar = "Organization")
```
</div>
<div data-lang-block="python" markdown="1">
```python
vi.create_boxplot(person_data, metric="Meeting_hours", hrvar="Organization")
```
</div>

For organizational network analysis (who collaborates with whom, across groups or individuals), see the dedicated [Network Analysis]({{ site.baseurl }}/network/) page.

---

## Next steps

Once your environment is set up:

1. **Explore the examples.** Browse the [utility scripts](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples) for runnable, real-world patterns.
2. **Read the data pitfalls guide** above before you draw conclusions from a first export.
3. **Try a Frontier Analytics prompt or Skill.** If you have a coding agent (GitHub Copilot, Claude Code, or similar), [Frontier]({{ site.baseurl }}/frontier-analytics/) turns an export into a finished dashboard, deck, or report.
4. **Move on to Essentials or Advanced Analytics** for custom KPIs, multi-query joins, machine learning, and statistical testing.

### Helpful resources

- [vivainsights R documentation](https://microsoft.github.io/vivainsights/)
- [vivainsights Python documentation](https://microsoft.github.io/vivainsights-py/)
- [Viva Insights documentation](https://learn.microsoft.com/en-us/viva/insights/)
- [Sample code repository](https://github.com/microsoft/viva-insights-sample-code)

### Need help?

- **Issues or bugs:** [open an issue](https://github.com/microsoft/viva-insights-sample-code/issues) on GitHub.
- **Feature requests or questions:** share them in [discussions](https://github.com/microsoft/viva-insights-sample-code/discussions).

---

## Related pages

- [Essentials]({{ site.baseurl }}/essentials/): core utility scripts, visualizations, and custom KPIs
- [Advanced Analytics]({{ site.baseurl }}/advanced/): machine learning, regression, and statistical testing
- [Network Analysis]({{ site.baseurl }}/network/): organizational network analysis (ONA)
- [Copilot Analytics]({{ site.baseurl }}/copilot/): measure Microsoft Copilot adoption and impact
- [Frontier]({{ site.baseurl }}/frontier-analytics/): self-service analytics with coding agents and prompt libraries

<script src="{{ '/assets/js/lang-switch.js' | relative_url }}"></script>
