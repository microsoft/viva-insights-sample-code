# Copilot Causal Toolkit

Run **causal inference** on Microsoft Viva Insights data to estimate the effect of
**Copilot usage** on a workplace outcome, using **Double Machine Learning (DML)**.
All three scenarios use Copilot usage (`Total_Copilot_actions_taken`) as the treatment
variable and estimate its effect on an outcome.

> 📖 **Full step-by-step documentation lives on the website:**
> **<https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit/>**
> This README is a quickstart; the site has the detailed setup, data, configuration,
> running, interpretation, and methodology guides.

## Scenarios

| Scenario | Outcome variable | Notebook(s) |
|---|---|---|
| **Seller Productivity** | External collaboration hours | `CI-DML_ExtCollabHours_PQ.ipynb`, `CI-DML_ExtCollabHours_SUR.ipynb` |
| **Burnout Prevention** | After-hours collaboration hours | `CI-DML_AftCollabHours_PQ.ipynb`, `CI-DML_AftCollabHours_SUR.ipynb` |
| **Employee Engagement** | Ordinal survey metric (e.g. `eSat`) | `CI-DML_Engagement_PQ.ipynb` (template) |

Notebook names describe the **outcome**, not the scenario label. `PQ` = Person Query,
`SUR` = Super Users Report. The Engagement notebook is a template — update the outcome
variable, scale, and confounders to match your survey before running.

## Quickstart

```bash
# 1. From this directory (copilot-causal-toolkit), create an isolated environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the install
python -c "import numpy, pandas, matplotlib, scipy, sklearn, econml, vivainsights"
```

Then:

1. **Add your data** — export a Person Query (recommended) or Super Users Report from
   Viva Insights as a `.csv` and drop it in `data/`.
2. **Configure** — open the notebook for your scenario and edit the parameters near the
   top (data file path, organizational attributes, date range).
3. **Run** — run cell-by-cell the first time, then all cells. Results are written to
   `output/`.
4. **Interpret** — see the
   [Interpretation Guide](https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-interpretation-guide/).

> ⚠️ **Install note:** `econml` is **not** compatible with numpy 2.x, so `requirements.txt`
> pins `numpy<2`. If you hit a numpy/econml error, run `pip install "numpy<2.0"` then
> `pip install --force-reinstall econml`.

## Directory

```
copilot-causal-toolkit/
├── data/      # Your CSV exports go here (git-ignored — never committed)
├── script/    # The five analysis notebooks
│   └── modules/   # data_processor, data_filter, estimator, output_manager,
│                  # subgroup_analysis, sensitivity_analysis, visualizations,
│                  # custom_interpreter
├── output/    # Generated tables, plots, and reports
└── requirements.txt
```

## Your data stays local

The analysis runs entirely on your machine — nothing is uploaded. `data/` is git-ignored
(`*.csv`), so your exports won't be committed from a clone. Keep filenames ending in
lower-case `.csv` so the rule applies on every operating system.

## Documentation

| Step | Guide |
|---|---|
| Overview | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit/> |
| 1 · Setup & installation | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-setup/> |
| 2 · Preparing your data | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-data/> |
| 3 · Configuring the notebook | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-configure/> |
| 4 · Running & troubleshooting | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-run/> |
| 5 · Interpreting the outputs | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-interpretation-guide/> |
| How it works (methodology) | <https://microsoft.github.io/viva-insights-sample-code/copilot-causal-toolkit-methodology/> |

## What this can and cannot prove

DML estimates a causal effect **under assumptions** — chiefly *unconfoundedness*,
*overlap*, and a correctly handled treatment definition. Treat results as decision-support
evidence to triangulate with experiments and domain knowledge, not as definitive proof.
See the website for the full caveats per output.
