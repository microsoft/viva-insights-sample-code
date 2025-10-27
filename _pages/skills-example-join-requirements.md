---
layout: page
title: "Skills data join requirements"
permalink: /skills-data-join-requirements/
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

### Development Environment Setup

For performing data joins and analyses, this tutorial covers example scripts in both the R and Python analysis environments. Here are the pre-requisites for them respectively: 

#### Option 1: R Environment

To run the R code in this tutorial, you must have R and ideally a code editor installed: 

- Install R: Download from [https://cran.r-project.org/](https://cran.r-project.org/)
- Install RStudio (recommended): Download from [https://www.rstudio.com/](https://www.rstudio.com/). Alternatively, you may choose to use [Visual Studio Code](https://code.visualstudio.com/) as your IDE. 

**Required R Packages:**

The next step is to install the required R packages: 

- **[vivainsights](https://microsoft.github.io/vivainsights/)**: Microsoft's R package for Viva Insights analysis
- **[tidyverse](https://www.tidyverse.org/)**: Collection of R packages for data science
- **[here](https://here.r-lib.org/)**: Easy file path management

If these are not already installed, you can install them in R with: 
```r
install.packages(c("vivainsights", "tidyverse", "here"))
```

#### Option 2: Python Environment  

To run the Python code in this tutorial, you must have Python and ideally a code editor installed: 

- Install Python: Download from [https://www.python.org/downloads/](https://www.python.org/downloads/) or use [Anaconda](https://www.anaconda.com/products/distribution)
- Install Jupyter (recommended): Comes with Anaconda or install via `pip install jupyter`. Alternatively, you may choose to use [Visual Studio Code](https://code.visualstudio.com/) as your IDE. 

**Required Python Packages:**

Ensure that you have the following python packages installed: 

- **[vivainsights](https://microsoft.github.io/vivainsights-py/)**: Microsoft's Python package for Viva Insights analysis
- **[pandas](https://pandas.pydata.org/)**: Data manipulation and analysis library
- **[numpy](https://numpy.org/)**: Fundamental package for scientific computing
- **[plotly](https://plotly.com/)**: Interactive visualization library
- **[scipy](https://scipy.org/)**: Scientific computing tools

If not, you can install them in the Command Prompt with: 

```bash
pip install vivainsights pandas numpy plotly scipy
```

Once the developer pre-requisites are satisfied, see [how to load and join data]({{ site.baseurl }}/skills-data-join/#data-loading-and-joining). 