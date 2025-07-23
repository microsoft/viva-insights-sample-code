---
layout: home
title: "Viva Insights Sample Code Library"
---

{% include custom-navigation.html %}

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

This repository contains sample code for the Viva Insights open-source packages, including code that is used in tutorials or demos. The sample code shown here covers more elaborate scenarios and examples that leverage, but are not included in the R and Python packages themselves.

## Is this page for me?

**New to Viva Insights?** If you're just getting started with Viva Insights, we recommend beginning with the [Viva Insights Power BI templates](https://learn.microsoft.com/en-us/viva/insights/tutorials/power-bi-intro) and official documentation. These provide pre-built dashboards and reports that deliver immediate value without requiring any coding experience.

**Ready for Advanced Analysis?** This sample code library is designed for analysts, data scientists, and researchers who have some experience with Viva Insights and want to unlock deeper insights through custom analysis and advanced analytics. Whether you're looking to:

- Build predictive models to identify top performers or at-risk employees
- Create custom visualizations and dashboards beyond standard templates  
- Perform statistical analysis and hypothesis testing on collaboration patterns
- Develop organizational network analysis (ONA) to understand informal influence structures
- Measure and optimize Microsoft Copilot adoption across your organization
- Automate recurring analyses and build scalable analytics workflows

...this library provides the tools and examples you need to get started.

## Built for Data Scientists

Most scripts in this library are written in **R** and **Python** - the most popular data science languages for automation, experimentation, and advanced statistical analysis. To accelerate your work, we've developed dedicated packages that handle the complexities of Viva Insights data processing:

- **[vivainsights R package](https://microsoft.github.io/vivainsights/)** - Comprehensive toolkit for R users with 100+ functions for data manipulation and visualization
- **[vivainsights Python package](https://microsoft.github.io/vivainsights-py/)** - Full-featured Python library optimized for data science workflows

These packages allow analysts to hit the ground running without writing data processing code from scratch, so you can focus on analysis rather than data wrangling.

## Special Focus: Copilot Analytics

With the rapid adoption of Microsoft Copilot across organizations, understanding usage patterns and measuring impact has become critical. Our dedicated **[Copilot Analytics](/copilot/)** section provides specialized scripts and methodologies for:

- Measuring Copilot adoption rates and user segmentation
- Identifying power users and building habit-based usage models  
- Analyzing productivity impact and ROI of Copilot investments
- Creating executive dashboards for tracking deployment success

These analytics are based on behavioral research and provide actionable insights for maximizing your Copilot investment.

## Quick Links

- [Viva Insights official documentation](https://learn.microsoft.com/en-us/viva/insights/introduction)
- [vivainsights Python package](https://microsoft.github.io/vivainsights-py/)
- [vivainsights R package](https://microsoft.github.io/vivainsights/)
- [Copilot Analytics: Decoding Super Usage](https://github.com/microsoft/DecodingSuperUsage/)
- [Copilot Analytics: advanced examples playbook](https://aka.ms/CopilotAdvancedAnalytics/)

## How to Use This Library

Browse the categories below to find sample code that matches your needs. Each script includes:

- **Purpose**: What the script accomplishes
- **Prerequisites**: Required packages and data formats
- **Usage**: How to run and customize the code
- **Download**: Direct link to the raw script file

---

## Categories

- 🔧 [Essentials](essentials/)
Essential utility functions and visualizations for getting started with Viva Insights analysis.

- 📊 [Advanced Analytics](advanced/)
Regression models, machine learning, and statistical analysis techniques.

- 🔗 [Network Analysis](network/)
Collaboration network visualization and analysis scripts.

- 🤖 [Copilot Analytics](copilot/)
Analysis and visualization scripts specifically for Microsoft Copilot usage data.

---

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA). For details, visit [Microsoft CLA](https://cla.opensource.microsoft.com).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/microsoft/viva-insights-sample-code/blob/main/LICENSE) file for details.
