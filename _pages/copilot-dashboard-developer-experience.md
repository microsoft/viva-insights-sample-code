---
layout: page
title: "Developer Experience and Copilot"
eyebrow: "Copilot analytics"
description: "Reproduce the synthetic developer experience dashboard, inspect its usage groups and understand the limits of real GitHub export support."
permalink: /copilot-dashboards/developer-experience-and-copilot/
css: "/assets/css/dashboard-guide.css"
lang: en-GB
---
<div class="dashboard-guide" markdown="1">

[All dashboards]({{ '/copilot-dashboards/' | relative_url }}) · [Open the synthetic demo]({{ '/examples/utility-r/github-copilot-developer-productivity-simulation.html' | relative_url }})

## What this dashboard shows

This synthetic reference describes developers' collaboration, focus, after-hours activity and network breadth through a GitHub Copilot usage lens. Feature, model and language views add usage context.

Despite the historical `developer-productivity` source filename, the report does **not** measure overall developer productivity or establish a Copilot effect.

<figure>
  <a href="{{ '/assets/images/reports/github-copilot-devex-overview.png' | relative_url }}"><img src="{{ '/assets/images/reports/github-copilot-devex-overview.png' | relative_url }}" alt="Synthetic Developer Experience overview defining GitHub usage groups and linking to working-pattern pillars." loading="lazy"></a>
  <figcaption>Overview defines the heavy GitHub-use group. Open its pillar cards for Collaboration, Focus, After-hours and Network comparisons.</figcaption>
</figure>

Heavy use means accepted completions plus user-initiated chat requests at or above the 80th percentile among observed developers with recorded use. These are interaction counts, not equivalent units of value. Ties can make more than 20% qualify.

The headline heavy-use share uses **all GitHub-observed developers**, including observed people with no recorded use. Heavy use, other recorded use and no recorded use partition that observed population. Unobserved people remain separate.

<figure>
  <a href="{{ '/assets/images/reports/github-copilot-devex-network.png' | relative_url }}"><img src="{{ '/assets/images/reports/github-copilot-devex-network.png' | relative_url }}" alt="Synthetic Network page comparing relationship breadth across GitHub intensity groups." loading="lazy"></a>
  <figcaption>Network measures describe distinct people reached over trailing windows, not hours or weekly flows. A larger network is not inherently better.</figcaption>
</figure>

The **More** menu holds breakdowns, coverage, working-pattern comparisons and methods. Team and role views may be withheld even when the overall three-group comparison is publishable.

## You will need

**For the demo:** the full repository, R and rendering dependencies. No GitHub organisation access or customer exports are needed.

The complete synthetic report reads **nine files**:

{% include dashboard-query-card.html title="Person query" icon="person-query" anchor="person_query" description="Weekly collaboration, focus, after-hours and network metrics with organisational attributes." exports="PersonQuery.csv in the demo. Real Person Query filenames vary by user, date and locale." %}

{% include dashboard-query-card.html title="Consumption query" icon="consumption-query" anchor="consumption_query" description="Separate M365 and GitHub credits, plus historical M365 licensing context." exports="PersonM365CreditsMetrics.csv, PersonGitHubCreditsMetrics.csv and PeopleMetaData.csv." %}

{% include dashboard-query-card.html title="GitHub query" icon="github-query" anchor="github_query" description="Activity observations and feature, language and model detail." exports="PersonGitHubActivityMetrics.csv, GitHubActivityBreakdownByFeatureMetrics.csv, GitHubActivityBreakdownByLanguageFeatureMetrics.csv, GitHubActivityBreakdownByLanguageModelMetrics.csv and GitHubActivityBreakdownByModelFeatureMetrics.csv." %}

The report does not use Agent exports or `M365SpendingPolicyMetaData.csv`. For query setup and current schemas, see [Joining Viva Insights queries]({{ '/query-join-guide/' | relative_url }}).

**For real GitHub data, the runner supports inspection only.** The nine-file demo bundle is not a verified real-data contract.

### Software prerequisites

Install [R](https://cran.r-project.org/), [Git](https://git-scm.com/) and [Pandoc](https://pandoc.org/installing.html), or use Pandoc bundled with RStudio. Ensure `Rscript` and `git` are available in PowerShell.

Clone the whole repository to retain the source Rmd, `github-developer-experience-helpers.R`, `copilot-super-panel-helpers.R`, fixtures and runner:

```powershell
git clone https://github.com/microsoft/viva-insights-sample-code.git
Set-Location .\viva-insights-sample-code
```

Run all remaining commands from the **repository root**:

```powershell
Rscript -e "install.packages(c('jsonlite', 'rmarkdown', 'flexdashboard', 'dplyr', 'tidyr', 'ggplot2', 'scales', 'knitr', 'stringr', 'htmltools', 'vivainsights'), repos='https://cloud.r-project.org')"
Rscript -e "stopifnot(rmarkdown::pandoc_available(), capabilities('cairo'))"
```

The developer report's SVG device requires Cairo graphics support. If that check fails, use an R installation with Cairo support. If Pandoc is missing, set `$env:RSTUDIO_PANDOC` to its installed folder.

If `Rscript` is not on PATH, replace it with `& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'`, matching your installed version. Inspection needs only base R, `jsonlite` and Git.

## Run the demo

1. Inspect the supplied synthetic reference:

   ```powershell
   Rscript .\frontier-analytics\starter-kits\copilot-query-dashboard\dashboard.R inspect .\frontier-analytics\starter-kits\copilot-query-dashboard\configs\demo-github.json
   ```

2. Render it in the runner's isolated workspace:

   ```powershell
   Rscript .\frontier-analytics\starter-kits\copilot-query-dashboard\dashboard.R reproduce .\frontier-analytics\starter-kits\copilot-query-dashboard\configs\demo-github.json
   ```

3. Open the standalone HTML:

   ```powershell
   Start-Process ..\copilot-dashboard-work\github-demo\reproduce\dashboard.html
   ```

### Expected outputs

| Location beneath `..\copilot-dashboard-work\github-demo` | Contents |
|---|---|
| `inspect\readiness.json` | Reference readiness and provenance, without rendering. |
| `reproduce\dashboard.html` | Complete synthetic Developer Experience dashboard. |
| `reproduce\provenance.json` | Synthetic mode, repository revision, configuration, checksums, packages, Pandoc and execution details. |
| `reproduce\work` | Isolated source, helpers, fixtures and rendering intermediates. |

The runner requires tracked, unmodified references. It copies inputs into the work directory and checks source hashes afterwards. It does not rewrite committed reports or fixtures.

Each action directory must be new. For a rerun, choose a fresh `output_dir`. All configured paths resolve relative to the **JSON file**, not the shell directory. Outputs must be outside the repository and cannot contain inputs.

Demo mode preserves the reference's fixed privacy minimum of 10. It does not apply custom real-data mappings, approvals or filters.

## Use your own data

You can inspect a local GitHub export, but **no supported real GitHub dashboard build exists**.

1. Obtain authorised exports through **Create analysis > Customized query > GitHub query**. Keep the files outside the checkout.
2. Copy the inspection configuration into a private workspace:

   ```powershell
   New-Item -ItemType Directory -Force ..\copilot-dashboard-private
   Copy-Item .\frontier-analytics\starter-kits\copilot-query-dashboard\configs\real-github-inspect.example.json ..\copilot-dashboard-private\github-inspect.json
   notepad ..\copilot-dashboard-private\github-inspect.json
   ```

3. Set absolute `repo_root`, `output_dir` and `inputs.activity` paths. JSON requires escaped Windows backslashes, such as `C:\\private\\PersonGitHubActivityMetrics.csv`. The optional `people` input does not establish a Person Query adapter.
4. Run inspection:

   ```powershell
   Rscript .\frontier-analytics\starter-kits\copilot-query-dashboard\dashboard.R inspect ..\copilot-dashboard-private\github-inspect.json
   ```

5. Read `inspect\readiness.json` beneath your configured output directory. Expect a bounded schema and count summary plus a real-GitHub adapter blocker, not dashboard HTML.

A successful inspection exit means the assessment completed. It does not certify fields, joins, eligibility, completeness or business meaning. `build` remains blocked until an approved adapter establishes those contracts. There is no synthetic fallback.

Develop any full real-data adaptation separately, using the [source documentation](https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/copilot-consumption-github-demo-reports.md) and [runner contract](https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/starter-kits/copilot-query-dashboard/README.md). Never substitute customer files for committed fixtures.

## Optional agent-assisted route

Install both the shared analysis Skill and the dashboard add-on from this checkout:

```powershell
gh skill install . viva-insights-analysis --from-local --agent github-copilot --scope user
gh skill install . viva-insights-copilot-dashboards --from-local --agent github-copilot --scope user
```

Keep the checkout after installation. In Copilot CLI, run `/skills reload` then `/skills info viva-insights-copilot-dashboards`. Ask for the Developer Experience demo and supply your checkout's full path.

The Skill supports reproduction and real GitHub inspection, not a verified real GitHub adapter. It does not support Business Leaders or remove rendering dependencies.

## Privacy and interpretation

- Reference groups require at least 10 distinct people. Linked cells, complements and joint team-by-role-by-intensity comparisons can trigger wider withholding.
- Synthetic reports embed aggregate charts, not individual scatter points. These rules do not guarantee safe publication of a real adaptation.
- GitHub row presence proves observation, not provisioning. Missing activity is unknown, while an explicit zero is observed inactivity.
- Sparse credit records and activity coverage require separate checks. M365 licensing metadata does not establish GitHub eligibility.
- The fixture's breakdown reconciliation is a synthetic invariant, not verified real-export semantics.
- Network measures are trailing-window stocks. Do not sum them across weeks or interpret them as weekly flows.
- Group differences are unadjusted and may reflect team or role composition. They do not establish productivity, wellbeing, tool value or causal effects.
- Keep real exports, configurations, inspection artifacts and outputs private pending review.

</div>
