---
layout: page
title: "Advanced Analytics"
permalink: /advanced/
---

{% include custom-navigation.html %}
{% include floating-toc.html %}

<style>
/* Hide any default Minima navigation that might appear */
.site-header .site-nav,
.trigger,
.page-link:not(.dropdown-toggle):not(.btn) {
  display: none !important;
}

/* Ensure our custom navigation is visible */
.custom-nav {
  display: block !important;
}
</style>

# Advanced Analytics Scripts

This page covers machine learning, regression models, and statistical analysis techniques for Viva Insights data.

The **top performers** use case is used for understanding the drivers behind top performance, in which top performance is usually provided by a business outcome metric uploaded into Viva Insights. The example scripts below make use of a random forest model, which has the benefits of handling non-linear relationships, providing feature importance rankings, and being robust to outliers and missing values. The same technique can be used for predicting other outcomes, such as high engagement or likelihood to stay (using sentiment surveys). 

The **information value** use case is used for feature selection and understanding which Viva Insights metrics are most predictive of a categorical outcome variable. Similar to the top performers use case, this allows you to identify the most important Viva Insights metrics that differentiate between different groups or categories for a categorical outcome variable. Information Value (IV) is particularly useful for identifying variables with strong predictive power while avoiding overfitting in your models.

When choosing which technique to use, consider if you need to build a predictive model that can handle complex, non-linear relationships and you want to make actual predictions on new data (e.g., predicting which employees are likely to be top performers). Random Forest is ideal when you have sufficient sample size (typically 100+ observations) and want robust predictions with feature importance rankings. Use information value when you need to perform initial feature selection, have limited sample sizes, or want to understand the univariate predictive power of individual variables before building more complex models. IV is particularly valuable for preprocessing large numbers of potential predictors and identifying which variables are worth including in downstream modeling efforts.

The **pairwise chi-square tests** use case is used for statistical hypothesis testing to determine if there are significant associations between categorical variables - typically organizational attributes or survey attributes - in your Viva Insights data. This technique is particularly valuable when you want to understand relationships between different organizational attributes (such as department, level, or location) and collaboration patterns or behaviors. The scripts include multiple testing corrections to control for false discovery rates when performing many simultaneous comparisons, ensuring reliable statistical conclusions. 

## Machine Learning & Predictive Modeling

### Top Performers Modeling (Python)
**📓 [top-performers-rf.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/top-performers-rf.ipynb)**
- **Purpose**: Identify characteristics of top performers using Random Forest
- **Language**: Python
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, scikit-learn, pandas
- **Key Features**: Feature importance analysis, model validation, performance metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/top-performers-rf.ipynb)**

### Top Performers Modeling (R)
**📄 [top-performers-rf.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/top-performers-rf.Rmd)**
- **Purpose**: Identify characteristics of top performers using Random Forest
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, randomForest, dplyr
- **Key Features**: Feature importance analysis, model validation, performance metrics
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/top-performers-rf.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/top-performers-rf.html)**

---

## Statistical Analysis

### Information Value Analysis (Python)
**📓 [information-value.ipynb](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/information-value.ipynb)**
- **Purpose**: Calculate Information Value (IV) for feature selection and variable importance
- **Language**: Python
- **Format**: Jupyter Notebook
- **Prerequisites**: vivainsights Python package, pandas, numpy
- **Key Features**: IV calculation, binning strategies, feature ranking
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/information-value.ipynb)**

### Information Value Analysis (R)
**📄 [information-value.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/information-value.Rmd)**
- **Purpose**: Calculate Information Value (IV) for feature selection and variable importance
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, Information, dplyr
- **Key Features**: IV calculation, binning strategies, feature ranking
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/information-value.Rmd)**

### Pairwise Chi-Square Tests (Python)
**📄 [pairwise-chisq.py](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/pairwise-chisq.py)**
- **Purpose**: Perform pairwise chi-square tests for categorical variables
- **Language**: Python
- **Prerequisites**: vivainsights Python package, scipy, pandas
- **Key Features**: Multiple testing correction, p-value adjustment, significance testing
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-python/pairwise-chisq.py)**

### Pairwise Chi-Square Tests (R)
**📄 [pairwise_chisq.Rmd](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/pairwise_chisq.Rmd)**
- **Purpose**: Perform pairwise chi-square tests for categorical variables
- **Language**: R
- **Format**: R Markdown
- **Prerequisites**: vivainsights R package, stats
- **Key Features**: Multiple testing correction, p-value adjustment, significance testing
- **[📥 Download](https://raw.githubusercontent.com/microsoft/viva-insights-sample-code/main/examples/utility-r/pairwise_chisq.Rmd)**
- **[🌐 View HTML Output](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/pairwise_chisq.html)**

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

### Python Environment
```bash
pip install vivainsights pandas numpy scikit-learn matplotlib seaborn jupyter
```

### R Environment
```r
install.packages(c("vivainsights", "dplyr", "ggplot2", "randomForest", "Information", "rmarkdown"))
```

---

## Best Practices

1. **Data Quality**: Always validate your data before analysis
2. **Feature Selection**: Use IV analysis to identify meaningful variables
3. **Model Validation**: Always test models on holdout data
4. **Statistical Significance**: Consider both statistical and practical significance
5. **Documentation**: Document your analysis methodology and assumptions

---

## Need Help?

- **Machine Learning**: [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- **Statistical Analysis**: [R Stats Documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html)
- **Viva Insights**: [Package Documentation](https://microsoft.github.io/vivainsights/)
- **Sample Data**: [Example datasets](https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/example-data)
