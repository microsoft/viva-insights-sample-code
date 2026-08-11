---
layout: page
title: "Advanced Analytics"
eyebrow: "Advanced analytics"
description: "Machine learning, regression, and statistical analysis for Viva Insights data, including random forest top-performer models, information value, pairwise chi-square tests, difference-in-differences intervention evaluation, meeting engagement drivers, and collaboration by time of day in R and Python."
permalink: /advanced/
css: "/assets/css/lang-switch.css"
---
# Advanced Analytics Scripts

This page covers machine learning, regression models, and statistical analysis techniques for Viva Insights data — from predicting top performance and selecting predictive features, to testing for significant associations, to measuring the real-world impact of a program or intervention.

Pick a card to jump straight to a technique, or read the short intro above each section for guidance on which one fits your question.

<div class="vi-card-grid" markdown="0">
  <a class="vi-card" href="#top-performers-modeling">
    <span class="vi-card-icon">🌳</span>
    <span class="vi-card-title">Top Performers Modeling</span>
    <span class="vi-card-desc">Random forest model identifying what characteristics drive top performance, or any other business outcome you upload.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#information-value-analysis">
    <span class="vi-card-icon">🔍</span>
    <span class="vi-card-title">Information Value Analysis</span>
    <span class="vi-card-desc">Rank which Viva Insights metrics are most predictive of a categorical outcome, for feature selection before a bigger model.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#pairwise-chi-square-tests">
    <span class="vi-card-icon">📐</span>
    <span class="vi-card-title">Pairwise Chi-Square Tests</span>
    <span class="vi-card-desc">Test whether organizational attributes are significantly associated with collaboration behaviors, with multiple-testing correction built in.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#collaboration-by-time-of-day">
    <span class="vi-card-icon">🕒</span>
    <span class="vi-card-title">Collaboration by Time of Day</span>
    <span class="vi-card-desc">Estimate a typical start and end of day from hourly collaboration metrics, and see how it shifts by weekday and role.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#evaluating-a-workplace-intervention">
    <span class="vi-card-icon">🧪</span>
    <span class="vi-card-title">Evaluating a Workplace Intervention</span>
    <span class="vi-card-desc">A treated-vs-control, difference-in-differences design that separates a genuine program effect from a company-wide trend.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
  <a class="vi-card" href="#meeting-engagement-drivers">
    <span class="vi-card-icon">📅</span>
    <span class="vi-card-title">Meeting Engagement Drivers</span>
    <span class="vi-card-desc">Rank the meeting characteristics that drive in-meeting messaging, used here as a proxy for disengagement.</span>
    <span class="vi-card-more">Jump to section →</span>
  </a>
</div>

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

## Machine Learning & Predictive Modeling

### Top Performers Modeling

Understand the drivers behind top performance, where top performance is usually a business outcome metric uploaded into Viva Insights. The scripts below use a random forest model, which handles non-linear relationships, provides feature importance rankings, and is robust to outliers and missing values. The same technique can predict other outcomes too, such as high engagement or likelihood to stay (using sentiment surveys). Random Forest is best when you have sufficient sample size (typically 100+ observations) and want robust predictions with feature importance rankings — for smaller samples or a first pass at feature selection, see Information Value below.

<div data-lang-block="r" markdown="1">
**📄 [top-performers-rf.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/top-performers-rf.Rmd)**
- **Purpose**: Identify characteristics of top performers using Random Forest
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, randomForest, dplyr
- **Key Features**: Feature importance analysis, model validation, performance metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/top-performers-rf.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/top-performers-rf.html)**
</div>

<div data-lang-block="python" markdown="1">
**📓 [top-performers-rf.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/top-performers-rf.ipynb)**
- **Purpose**: Identify characteristics of top performers using Random Forest
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, scikit-learn, pandas
- **Key Features**: Feature importance analysis, model validation, performance metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/top-performers-rf.ipynb)**
</div>

---

## Statistical Analysis

### Information Value Analysis

Feature selection: understand which Viva Insights metrics are most predictive of a categorical outcome variable, identifying variables with strong predictive power while avoiding overfitting. Use Information Value (IV) for initial feature selection, with limited sample sizes, or to understand the univariate predictive power of individual variables before building a more complex model like the Random Forest above — it's particularly valuable for preprocessing large numbers of potential predictors.

<div data-lang-block="r" markdown="1">
**📄 [information-value.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/information-value.Rmd)**
- **Purpose**: Calculate Information Value (IV) for feature selection and variable importance
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, Information, dplyr
- **Key Features**: IV calculation, binning strategies, feature ranking
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/information-value.Rmd)**
</div>

<div data-lang-block="python" markdown="1">
**📓 [information-value.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/information-value.ipynb)**
- **Purpose**: Calculate Information Value (IV) for feature selection and variable importance
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, pandas, numpy
- **Key Features**: IV calculation, binning strategies, feature ranking
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/information-value.ipynb)**
</div>

### Pairwise Chi-Square Tests

Statistical hypothesis testing for significant associations between categorical variables — typically organizational attributes (department, level, location) or survey attributes — and collaboration patterns or behaviors. Multiple testing corrections control for false discovery rates when running many simultaneous comparisons, keeping the conclusions reliable.

<div data-lang-block="r" markdown="1">
**📄 [pairwise_chisq.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/pairwise_chisq.Rmd)**
- **Purpose**: Perform pairwise chi-square tests for categorical variables
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, stats
- **Key Features**: Multiple testing correction, p-value adjustment, significance testing
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/pairwise_chisq.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/pairwise_chisq.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [pairwise-chisq.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/pairwise-chisq.py)**
- **Purpose**: Perform pairwise chi-square tests for categorical variables
- **Prerequisites**: vivainsights Python package, scipy, pandas
- **Key Features**: Multiple testing correction, p-value adjustment, significance testing
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/pairwise-chisq.py)**
</div>

---

## Behavioral & Program Analysis

These examples move from modelling attributes to answering practical workplace questions. Because the sample datasets don't contain the hourly buckets, a real intervention, or enough multi-person meetings needed below, each script generates a small, clearly labelled simulated dataset that shares the column names of a real query — so the same downstream code runs unchanged on your own export.

### Collaboration by Time of Day

Estimate a typical start and end of day from hourly collaboration metrics, and show how those hours shift by weekday and by role.

<div data-lang-block="r" markdown="1">
**📄 [collaboration-by-time-of-day.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/collaboration-by-time-of-day.Rmd)**
- **Purpose**: Estimate a typical start and end of day from hourly collaboration metrics
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, tidyverse, lubridate
- **Key Features**: Hourly activity matrix, two-stage aggregation, cuts by weekday and role
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/collaboration-by-time-of-day.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/collaboration-by-time-of-day.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [collaboration-by-time-of-day.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/collaboration-by-time-of-day.py)**
- **Purpose**: Estimate a typical start and end of day from hourly collaboration metrics
- **Prerequisites**: vivainsights Python package, pandas, numpy
- **Key Features**: Hourly activity matrix, two-stage aggregation, cuts by weekday and role
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/collaboration-by-time-of-day.py)**
</div>

---

### Evaluating a Workplace Intervention

Set up a treated-versus-control, difference-in-differences design so a genuine programme effect can be separated from a company-wide or seasonal trend — directly applicable to measuring the impact of a Microsoft 365 Copilot enablement wave.

<div data-lang-block="r" markdown="1">
**📄 [evaluate-intervention.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/evaluate-intervention.Rmd)**
- **Purpose**: Measure a workplace intervention with a treated-vs-control difference-in-differences design
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, tidyverse
- **Key Features**: Before/During/After windows, difference-in-differences, two-stage aggregation, displacement checks
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/evaluate-intervention.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/evaluate-intervention.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [evaluate-intervention.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/evaluate-intervention.py)**
- **Purpose**: Measure a workplace intervention with a treated-vs-control difference-in-differences design
- **Prerequisites**: vivainsights Python package, pandas, numpy
- **Key Features**: Before/During/After windows, difference-in-differences, two-stage aggregation, displacement checks
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/evaluate-intervention.py)**
</div>

---

### Meeting Engagement Drivers

Model in-meeting messaging as a proxy for disengagement and rank the meeting characteristics that drive it, then take a closer look at meeting duration to separate a real effect from simple exposure.

<div data-lang-block="r" markdown="1">
**📄 [meeting-engagement-drivers.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/meeting-engagement-drivers.Rmd)**
- **Purpose**: Rank the meeting characteristics that drive in-meeting messaging as a proxy for disengagement
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, tidyverse, randomForest
- **Key Features**: Meeting-level modelling, random forest permutation importance, rate-vs-exposure duration analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/meeting-engagement-drivers.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/meeting-engagement-drivers.html)**
</div>

<div data-lang-block="python" markdown="1">
**📄 [meeting-engagement-drivers.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/meeting-engagement-drivers.py)**
- **Purpose**: Rank the meeting characteristics that drive in-meeting messaging as a proxy for disengagement
- **Prerequisites**: vivainsights Python package, scikit-learn, pandas, numpy
- **Key Features**: Meeting-level modelling, random forest permutation importance, rate-vs-exposure duration analysis
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/meeting-engagement-drivers.py)**
</div>

---

## Sample Datasets

### Simulated Person Query
**📄 [simulated_person_query.csv](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/_data/simulated_person_query.csv)**
- **Purpose**: Simulated person-level data for analysis
- **Format**: CSV
- **Contents**: Weekly collaboration metrics, meeting data, email patterns

---

## Analysis Workflows

### 1. Feature Selection Workflow
1. **Load Data**: Import your Viva Insights query results
2. **Information Value**: Run IV analysis to identify important variables
3. **Statistical Testing**: Use chi-square tests for categorical relationships
4. **Model Building**: Apply selected features to predictive models

### 2. Top Performers Analysis Workflow
1. **Data Preparation**: Clean and prepare performance data
2. **Feature Engineering**: Create relevant collaboration metrics
3. **Model Training**: Train Random Forest model
4. **Interpretation**: Analyze feature importance and model results
5. **Validation**: Test model performance on holdout data

### 3. Statistical Analysis Workflow
1. **Exploratory Analysis**: Understand data distributions
2. **Hypothesis Testing**: Test relationships between variables
3. **Effect Size**: Calculate practical significance
4. **Reporting**: Generate analysis reports

---

## Prerequisites

<div data-lang-block="r" markdown="1">
### R Environment
```r
install.packages(c("vivainsights", "dplyr", "tidyr", "ggplot2", "scales", "purrr", "randomForest", "fixest", "Information", "rmarkdown"))
```
</div>

<div data-lang-block="python" markdown="1">
### Python Environment
```bash
pip install vivainsights pandas numpy scikit-learn linearmodels matplotlib seaborn jupyter
```
</div>

---

## Best Practices

1. **Data Quality**: Always validate your data before analysis
2. **Feature Selection**: Use IV analysis to identify meaningful variables
3. **Model Validation**: Always test models on holdout data
4. **Statistical Significance**: Consider both statistical and practical significance
5. **Documentation**: Document your analysis methodology and assumptions

---

## Related pages

- [Causal Inference in Copilot Analytics]({{ site.baseurl }}/causal-inference/): move beyond correlation to measure the true impact of an intervention
- [Network Analysis]({{ site.baseurl }}/network/): organizational network analysis (ONA) as a complementary advanced technique
- [Copilot Analytics]({{ site.baseurl }}/copilot/): adoption metrics and Power/Habitual user segmentation
- [Essentials]({{ site.baseurl }}/essentials/): utilities and visualizations to prepare your data
- [Getting Started]({{ site.baseurl }}/getting-started/): environment setup and first steps

---

## Need Help?

- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- **Statistical Analysis**: [R Stats Documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html)
- **Viva Insights**: [Package Documentation](https://microsoft.github.io/vivainsights/)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)

<script src="{{ '/assets/js/lang-switch.js' | relative_url }}"></script>
