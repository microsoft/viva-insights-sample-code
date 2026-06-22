---
layout: default
title: "Causal Toolkit — Setup & Installation"
eyebrow: "Causal inference · Toolkit · Step 1"
description: "Download the Copilot Causal Toolkit, install Python and the required packages, and open the project in your editor."
permalink: /copilot-causal-toolkit-setup/
css: "/assets/css/causal-toolkit.css"
---

<nav class="ct-series-nav" aria-label="Toolkit steps">
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit/">Overview</a>
  <a class="ct-chip is-current" href="{{ site.baseurl }}/copilot-causal-toolkit-setup/"><span class="ct-chip-step">1</span>Set up</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-data/"><span class="ct-chip-step">2</span>Data</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-configure/"><span class="ct-chip-step">3</span>Configure</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-run/"><span class="ct-chip-step">4</span>Run</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-interpretation-guide/"><span class="ct-chip-step">5</span>Interpret</a>
  <a class="ct-chip" href="{{ site.baseurl }}/copilot-causal-toolkit-methodology/">How it works</a>
</nav>

# Setup &amp; installation

This page gets your machine ready to run the toolkit: download the code, install **Python 3.8+** and the required packages, and open the project in your editor. Once done, continue to [Preparing your data]({{ site.baseurl }}/copilot-causal-toolkit-data/).

<div class="ct-callout is-warning" markdown="1">
<span class="ct-callout-label">One thing to get right</span>
The toolkit lives **several levels deep** in the repository, at `examples/utility-python/causal-inference/copilot-causal-toolkit`. Throughout setup, your **working directory** is that `copilot-causal-toolkit` folder — *not* the repository root. Opening the wrong folder is the most common cause of broken file paths later.
</div>

## Step 1 — Get a copy of the toolkit

<div class="ct-tabs" data-ct-tabs>
  <div class="ct-tablist" role="tablist" aria-label="Download method">
    <button class="ct-tab" type="button">Download ZIP (beginners)</button>
    <button class="ct-tab" type="button">Clone with Git</button>
  </div>
  <div class="ct-panel" markdown="1">
1. Click the green **Code** button at the top of the [repository](https://github.com/microsoft/viva-insights-sample-code/).
2. Select **Download ZIP**.
3. Extract the ZIP to a location on your computer, e.g. `C:\Users\YourName\Documents\viva-insights-sample-code`.
4. Navigate into the toolkit subdirectory:

   ```
   examples/utility-python/causal-inference/copilot-causal-toolkit
   ```

   For the example above, the full path would be:

   ```
   C:\Users\YourName\Documents\viva-insights-sample-code\examples\utility-python\causal-inference\copilot-causal-toolkit
   ```
  </div>
  <div class="ct-panel" markdown="1">
```bash
git clone https://github.com/microsoft/viva-insights-sample-code.git
cd viva-insights-sample-code/examples/utility-python/causal-inference/copilot-causal-toolkit
```
  </div>
</div>

## Step 2 — Open the project in your editor

We recommend **Visual Studio Code** with the Python and Jupyter extensions.

1. Open VS Code → **File → Open Folder…**
2. Select the **`copilot-causal-toolkit`** folder itself (the one containing `data/`, `script/`, and `output/`).
3. VS Code now shows the toolkit structure in the sidebar.

Opening the toolkit folder directly — rather than the parent repository — keeps all the relative file paths in the notebooks working.

<details class="ct-details">
<summary>What's inside the folder?</summary>
<div markdown="1">

```
copilot-causal-toolkit/
├── data/                 # Place your CSV exports here (Person Query or Super Users Report)
├── script/               # The five analysis notebooks
│   ├── CI-DML_AftCollabHours_PQ.ipynb / _SUR.ipynb
│   ├── CI-DML_ExtCollabHours_PQ.ipynb / _SUR.ipynb
│   ├── CI-DML_Engagement_PQ.ipynb
│   └── modules/          # Reusable helpers: data_processor, estimator,
│                         #   subgroup_analysis, sensitivity_analysis, visualizations …
├── output/               # Results, plots, and reports are written here
└── README.md
```

- **`data/`** — your Viva Insights exports (CSV).
- **`script/`** — the Jupyter notebooks you run.
- **`script/modules/`** — Python modules with the analysis functions.
- **`output/`** — all generated tables, plots, and HTML reports.

</div>
</details>

We recommend making a **copy** of the notebook you intend to use and renaming it to match your analysis (e.g. `CI-DML_AftCollabHours_PQ - Contoso.ipynb`).

## Step 3 — Install the prerequisites

### Python 3.8 or higher

Check whether Python is already installed:

```bash
python --version
```

If it's missing or too old, download it from [python.org](https://www.python.org/downloads/). During installation, tick **“Add Python to PATH”**.

### Jupyter

Jupyter runs the `.ipynb` notebooks interactively:

```bash
pip install jupyter
```

Alternatively, use **VS Code** with the Python and Jupyter extensions for a more integrated experience.

### Required Python packages

A `requirements.txt` is provided in the toolkit directory. From the `copilot-causal-toolkit` folder:

```bash
pip install -r requirements.txt
```

<details class="ct-details">
<summary>Prefer to install packages individually?</summary>
<div markdown="1">

```bash
pip install numpy pandas matplotlib scipy scikit-learn econml vivainsights jupyter
```

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data manipulation and numerical computing |
| `matplotlib` | Plots and visualizations |
| `scipy` | Statistical functions and tests |
| `scikit-learn` | ML models and preprocessing |
| `econml` | Causal inference (Double Machine Learning) |
| `vivainsights` | Viva Insights data loading and analysis |
| `jupyter` | Running notebooks interactively |

**Installation time:** roughly 5–10 minutes depending on your connection.

</div>
</details>

<details class="ct-details">
<summary>Installing for a specific Python environment</summary>
<div markdown="1">

If you have multiple Python versions installed, target a specific one:

```bash
# Using the py launcher (Windows)
py -3.11 -m pip install -r requirements.txt

# Using the full path to a specific Python executable
C:\Path\To\Python\python.exe -m pip install -r requirements.txt

# For a virtual environment (after creating it)
C:\Path\To\venv\Scripts\python.exe -m pip install -r requirements.txt

# For conda environments
conda run -n <env_name> pip install -r requirements.txt
```

To check which Python you're currently using:

```bash
python -c "import sys; print(sys.executable)"
```

</div>
</details>

### Verify the installation

```bash
python -c "import numpy, pandas, matplotlib, scipy, sklearn, econml, vivainsights"
```

If this runs without error, you're ready. You can run it in a terminal or a notebook cell.

<details class="ct-details">
<summary>Recommended: use a virtual environment</summary>
<div markdown="1">

A virtual environment avoids package conflicts. Run these from the `copilot-causal-toolkit` directory:

<div class="ct-tabs" data-ct-tabs>
  <div class="ct-tablist" role="tablist" aria-label="Operating system">
    <button class="ct-tab" type="button">Windows</button>
    <button class="ct-tab" type="button">macOS / Linux</button>
  </div>
  <div class="ct-panel" markdown="1">
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
  </div>
  <div class="ct-panel" markdown="1">
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
  </div>
</div>

</div>
</details>

<div class="ct-callout is-tip" markdown="1">
<span class="ct-callout-label">Next</span>
With your environment ready, head to [**Preparing your data**]({{ site.baseurl }}/copilot-causal-toolkit-data/) to export the right columns for your scenario.
</div>

<nav class="ct-pager" aria-label="Toolkit pagination">
  <a class="ct-pager-link" href="{{ site.baseurl }}/copilot-causal-toolkit/">
    <span class="ct-pager-dir">← Back</span>
    <span class="ct-pager-title">Overview</span>
  </a>
  <a class="ct-pager-link is-next" href="{{ site.baseurl }}/copilot-causal-toolkit-data/">
    <span class="ct-pager-dir">Next →</span>
    <span class="ct-pager-title">2 · Preparing your data</span>
  </a>
</nav>

<script src="{{ '/assets/js/causal-toolkit.js' | relative_url }}"></script>
