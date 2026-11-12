---
layout: article
title: "Causal Questions About AI Usage"
description: "A question-led guide to choosing a causal design for AI usage data from Viva Insights, covering staggered, function-led rollouts, proxy outcomes and the checks every analysis should run."
permalink: /articles/causal-questions-ai-usage/
eyebrow: "Copilot Analytics Lab · PANDAS Team · November 2026"
dek: "Most AI rollouts are staggered, product-specific and led by function. This guide asks a short series of questions about your data and points you to a design that fits, or says plainly when none does."
byline: "By the PANDAS team · A Copilot Analytics Lab methods guide"
read_time: "29 min read"
css: "/assets/css/article.css"
image: /assets/images/articles/causal-questions-ai-usage-card.png
keywords:
  - causal inference
  - difference-in-differences
  - staggered adoption
  - event study
  - parallel trends
  - fixed effects
  - instrumental variables
  - regression discontinuity
  - propensity score
  - matching
  - doubly robust
  - double machine learning
  - synthetic control
  - synthetic difference-in-differences
  - interrupted time series
  - ITSA
  - placebo test
  - negative control
  - clustering
  - Copilot
  - Cowork
  - Agents
  - GitHub Copilot
---

<nav class="article-contents reveal" aria-label="Article contents">
<ol>
  <li><a href="#choosing-outcome">Part 1 · Choose the outcome first</a></li>
  <li><a href="#decision-guide">Part 2 · Find your route</a></li>
  <li><a href="#routes">Part 3 · The routes</a></li>
  <li><a href="#joined-panel">Part 4 · Build the joined panel</a></li>
  <li><a href="#function-confounding">Part 5 · Function as a confounder</a></li>
  <li><a href="#checks">Part 6 · Standard checks</a></li>
  <li><a href="#communicating">Part 7 · Report what you can claim</a></li>
  <li><a href="#run-it">Part 8 · Run it yourself</a></li>
  <li><a href="#references">References</a></li>
</ol>
</nav>

<p class="lead">Leaders ask a plain question: did AI use change how work gets done? The honest answer depends on how the data is structured. AI use now spans Microsoft 365 Copilot, Cowork, Agents and GitHub Copilot, and each is measured in a different query, with different units, start dates and eligible people. Whether you can say anything causal depends on who was exposed, when, and what you can fairly compare them with.</p>


### The TL;DR

<ul class="tldr">
  <li>Choose the outcome before the method. The outcome decides what a credible result can say.</li>
  <li>Staggered, product-specific, function-led rollouts are the normal case. They need estimators built for them.</li>
  <li>Compare within function wherever timing varies inside functions.</li>
  <li>Run the standard checks on every route where they apply, and say where they cannot be run.</li>
  <li>Report results on Viva Insights proxies as behavioural outcomes or candidate leading indicators, and keep business impact claims for outcomes that measure it.</li>
</ul>

<h3 id="setting">The usual starting point</h3>

<p>Most analyses of AI use start from the same setting. Licences are product-specific, so some people hold a GitHub Copilot licence and others never will. Rollout tends to follow function, so function shapes both when people get access and how their outcomes move. That makes function a confounder from the start.</p>

<p>The data is typically a person-week panel joined onto the Person Query. Copilot and other AI features have now been available for over two years. Many early adopters therefore have no observed pre-period in the export, and any before-and-after comparison is limited to later cohorts and newer tools.</p>

<p>This article is for people analytics practitioners and data scientists who hold Viva Insights exports and know regression. Many will be new to modern difference-in-differences. It routes you to a design based on your data and does not catalogue every method. It replaces the site's earlier technique-by-technique guide to causal inference.</p>

<section id="choosing-outcome" class="part-head">
<h2>Choose the outcome first</h2>
</section>

<p>Pick the outcome before you pick a method. The outcome decides what a credible result can say, and a method cannot repair an outcome that sits too far from the exposure. In practice the choice is between a business KPI and a Viva Insights proxy. Each has different strengths and risks.</p>

<h3>Business KPIs</h3>

<p>Revenue, cases closed and cycle time are what leaders want to hear about. They are also the hardest outcomes to tie to AI use, for four reasons.</p>

<ul>
  <li><strong>Many functions contribute.</strong> A KPI is produced by several functions working together, so one person's or one function's AI exposure is a single contribution among many.</li>
  <li><strong>Processes are long.</strong> A KPI comes from a drawn-out, multi-step process. The lag between exposure and the KPI can exceed the window you have.</li>
  <li><strong>The grain does not match.</strong> KPIs are usually recorded for teams or business units, monthly or quarterly. Exposure is individual and weekly. Few units and few periods also leave little statistical power.</li>
  <li><strong>Attribution is rarely defensible.</strong> Pinning a KPI movement on one tool or one function is hard to justify when other tools, reorganisations and market conditions moved at the same time.</li>
</ul>

<h3>Viva Insights proxies</h3>

<p>Measures such as <code>External collaboration hours</code>, <code>After hours collaboration hours</code> and <code>Meeting hours</code> are recorded per person, per week, and line up with exposure dates. That supports panel designs with many units and many periods. They carry their own risks.</p>

<ul>
  <li><strong>They measure behaviour.</strong> A change says work changed. It does not say the business improved. More <code>External collaboration hours</code> may reflect more customer contact, but it is not necessarily more sales. It could equally reflect more rework or longer sales cycles.</li>
  <li><strong>Direction is ambiguous.</strong> Fewer meeting hours could mean efficiency or disengagement. Declare the expected direction before looking at results.</li>
  <li><strong>The tool can move the metric directly.</strong> Meeting recap features can change meeting attendance, for example. Prefer proxies the tool cannot generate itself, and flag those it can.</li>
  <li><strong>Targets change behaviour.</strong> Once a proxy becomes something people are asked to improve, they may change how they work to move the number, and it then measures something different from before.</li>
</ul>

<table>
  <thead>
    <tr>
      <th scope="col"></th>
      <th scope="col">Business KPI</th>
      <th scope="col">Viva Insights proxy (<code>External collaboration hours</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Grain</th>
      <td>Team or business unit</td>
      <td>Person</td>
    </tr>
    <tr>
      <th scope="row">Frequency</th>
      <td>Often monthly or quarterly</td>
      <td>Weekly</td>
    </tr>
    <tr>
      <th scope="row">Alignment with exposure</th>
      <td>Weak. Individual exposure meets team outcomes</td>
      <td>Strong. Same person, same week</td>
    </tr>
    <tr>
      <th scope="row">What a change means</th>
      <td>Business results moved, with many possible contributors</td>
      <td>Time spent with external contacts changed. The direction of benefit is open</td>
    </tr>
    <tr>
      <th scope="row">Main risk</th>
      <td>Lag, low power and weak attribution</td>
      <td>Reading a behaviour shift as business impact</td>
    </tr>
  </tbody>
</table>

<aside class="callout is-action reveal" markdown="1">
<span class="callout-label">What to do</span>

- Write the causal chain: exposure, behaviour, process measure, KPI. Say which link your analysis tests.
- Pre-declare one primary outcome and its expected direction. Treat everything else as secondary or exploratory.
- Label proxy results as behavioural outcomes or candidate leading indicators. Do not report a proxy shift as business impact.
- Where a KPI exists at team grain with enough teams and periods, analyse it separately at that grain and compare it with the proxy result.
- Check whether the proxy has tracked the KPI historically. That is predictive evidence only. Treating a proxy as a surrogate for the KPI needs much stronger assumptions <a class="cite" href="#ref-surrogate">[1]</a> <a class="cite" href="#ref-prentice1989">[2]</a>, so state which you are relying on, or say that surrogacy has not been established.
- Add a negative-control outcome that the tool should not plausibly affect (see [the negative-control check](#check-negative-control)).
- Confirm how often the outcome is really measured. An annual rating copied onto weekly rows is one observation a year. This matches the outcome-frequency gate in the causal Skill. Network-size metrics also repeat within a month, so they are not weekly outcomes.
</aside>

<p>None of this argues against using proxies. They are often the only outcomes with enough weekly variation to analyse. The task is to label them honestly. Once the outcome is settled, the next question is which design fits your data, and the <a href="#decision-guide">decision guide</a> takes you there.</p>

<section id="decision-guide" class="part-head">
<h2>Find your route</h2>
</section>

<p>Answer one question at a time. Each answer leads to the next question or to a route in Part 3, and you can step back to change an answer. With JavaScript turned off, the full set of questions appears as a nested list.</p>

{% include causal-decision-guide.html %}

<p>The guide covers the common cases. If your situation sits between two routes, read both and run the stricter set of checks. Every route also applies the <a href="#checks">standard checks</a> where they are available.</p>

<section id="routes" class="part-head"><h2>The routes</h2></section>

<p>Each route states when it fits, the assumption it rests on, the main threats to that assumption, and where to run it. Start with the route that matches your answers in the decision guide.</p>

<h3 id="route-descriptive">Describe use without a causal claim</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>The question is who uses what, how much, and how that changes over time.</dd>
  <dt>Key assumption</dt><dd>None about causes. The claim is a description of the data you hold.</dd>
  <dt>What would break it</dt><dd>Wording that drifts into "because of AI". Coverage gaps that are treated as zero use.</dd>
  <dt>Where to run it</dt><dd>The <a href="{{ site.baseurl }}/frontier-analytics-prompt-ai-adoption/">AI Adoption Across Tools prompt</a>.</dd>
</dl>

<p>If the question is about the spread of use across Microsoft 365 Copilot, Cowork, Agents, Consumption credits and GitHub Copilot, you do not need a causal design. Report counts, shares and trends by function and tool, keep unlike units such as actions, tasks, sessions and credits separate, and treat unresolved coverage as unknown.</p>

<h3 id="route-randomised">Randomised access</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Access was assigned at random, for example a pilot lottery or waves randomised within function.</dd>
  <dt>Key assumption</dt><dd>Assignment is as good as random. To estimate the effect of usage, assignment must affect the outcome only through use (exclusion) and must not push anyone to use less (monotonicity).</dd>
  <dt>What would break it</dt><dd>Assignment that was not truly random, unequal attrition between arms, or assignment that changes behaviour without any use, for example through training or announcements.</dd>
  <dt>Where to run it</dt><dd><code>fixest</code> (<code>feols</code> with its IV syntax), <code>pyfixest</code>, or <code>linearmodels</code> IV. Seek specialist review if the randomisation was partial.</dd>
</dl>

<p>Compare people by the group they were assigned to, whether or not they used the tool. This is the intention-to-treat comparison, and it answers the question of what giving access did. It is the cleanest estimate available in this setting, and waves randomised within function are the strongest design you are likely to meet, because function is balanced by construction.</p>

<p>For the effect of usage, assignment can serve as an instrument. For a binary measure such as any use, a relevant random assignment identifies a local effect for compliers, the people whose use changed because of assignment, provided exclusion and monotonicity hold <a class="cite" href="#ref-ia1994">[3]</a>. It says little about people who would have used the tool anyway, or never would. For continuous usage intensity, the instrumental-variable estimate is a weighted average of effects across the range of use and needs its own stated assumptions and interpretation.</p>

<p>Randomised access does not make chosen usage random. Among the people who were given access, those who use it heavily still differ from those who do not. Keep the assignment effect and the usage effect as separate results.</p>

<p>Instruments that were not randomised, such as a manager's early adoption or an office's rollout date, need a separate argument that they affect the outcome only through use. In workplace data that argument is rarely convincing, because managers and offices also shape outcomes directly. Treat such designs as specialist work and seek review before relying on them.</p>

<h3 id="route-rdd">An eligibility cut-off</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Access depended on a threshold, such as a score or a start date, and many people sit close to it on either side.</dd>
  <dt>Key assumption</dt><dd>Outcomes would have varied smoothly through the cut-off without treatment. Nothing else changes at the cut-off, and nobody could sort themselves precisely across it.</dd>
  <dt>What would break it</dt><dd>Too few people near the cut-off, or manipulation of the score or date.</dd>
  <dt>Where to run it</dt><dd>Rare here. Flag for specialist review.</dd>
</dl>

<p>Regression discontinuity compares people just either side of the threshold <a class="cite" href="#ref-ll2010">[4]</a>. Rollouts here rarely use a sharp rule, so expect to confirm first that one exists. If it does, check how many people fall near the cut-off and whether managers could adjust who sat on which side.</p>

<h3 id="route-staggered-did">Staggered rollout, compared within function</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Licences or access arrived in waves, several waves or both treated and not-yet-treated people exist inside the same function, and you hold repeated pre-adoption weeks.</dd>
  <dt>Key assumption</dt><dd>Within function, people treated in a wave would have followed the same path as comparable people not yet treated, had they not been treated (conditional parallel trends).</dd>
  <dt>What would break it</dt><dd>Licences going first to people already on a different path, comparison people taking up other AI tools at a different rate from the treated group during the window, behaviour changing before the licence date, or too few clusters.</dd>
  <dt>Where to run it</dt><dd><code>did</code> in R, <code>didimputation</code>, <code>did2s</code>, <code>fixest::sunab()</code> or <code>pyfixest</code>. In Python, <code>differences</code> or <code>csdid</code>. The repository notebook is a starting point only.</dd>
</dl>

<p>This is the usual case here, so it gets the most care. Build the comparison from the same population that could have received the tool. For GitHub Copilot, that means people who could hold a licence, such as developers without one yet. People who will never be licensed are not a comparison group. Within that population, label each person as treated with a dated start, not yet treated, or never treated in the window. Not-yet-treated people are valid comparisons only for the weeks before their own start, and only inside an estimator built for that. People never treated within the eligible population are the other valid comparison.</p>

<p>Avoid reading the effect from a single two-way fixed-effects coefficient. When start dates differ, that coefficient is a weighted average of many two-group comparisons, and some of them use people who were already treated as the comparison <a class="cite" href="#ref-gb2021">[5]</a>. If effects differ across cohorts or change over time, the coefficient can place negative weight on some of the underlying effects, so it may not represent any meaningful average <a class="cite" href="#ref-cdh2020">[6]</a> <a class="cite" href="#ref-sa2021">[7]</a>. The estimators below avoid this. For a wider orientation, see the overview in <a class="cite" href="#ref-rsbp2023">[8]</a>.</p>

<p>Pick one of these, with function in the set of things you condition on:</p>

<ul>
  <li><strong>Group-time estimates.</strong> Estimate an effect for each start cohort and week, then summarise <a class="cite" href="#ref-cs2021">[9]</a>. In R, use <code>did::att_gt()</code> with <code>xformla</code> to put function in the parallel-trends conditioning set and <code>control_group = "notyettreated"</code>. In Python, <code>differences</code> and <code>csdid</code> are available. Check their documentation for option parity with R before relying on them.</li>
  <li><strong>Imputation.</strong> Fit the untreated outcomes with person and function-by-period fixed effects, then compare treated people with their predictions <a class="cite" href="#ref-bjs2024">[10]</a>. Packages are <code>didimputation</code>, <code>did2s</code> and <code>pyfixest.did2s()</code>.</li>
  <li><strong>Sun and Abraham event studies.</strong> Use <code>fixest::sunab()</code> for an event study that keeps cohorts apart <a class="cite" href="#ref-sa2021">[7]</a>.</li>
  <li><strong>Stacked DiD.</strong> Build a separate data set for each cohort, with a clean comparison window and comparison people from the same function, then combine them with the weights needed for the average effect you want to report <a class="cite" href="#ref-wfh2024">[11]</a>. Stacking the data and fitting one pooled regression without those weights is not enough.</li>
</ul>

<p>Report an event study with pre-period leads, and for function-level estimates wherever cells meet the privacy threshold. Effects probably differ by function, so function-specific results are useful in their own right. Where a function falls below the threshold or lacks power, pool with a stated justification or report it as unidentified. Treat the leads as intervals to read, as described in <a href="#check-leads">the time-reversal placebo</a>.</p>

<p>Cluster at the level at which licences were assigned, which is often function or team and seldom the person <a class="cite" href="#ref-cluster2023">[12]</a> <a class="cite" href="#ref-cm2015">[13]</a>. Repeated weeks for the same person add dependence too. Count the effective clusters in each model sample. With few clusters, consider small-sample corrections or randomisation-based inference. If no method is defensible, say that uncertainty cannot be reliably quantified, and do not present the intervals as if it can.</p>

<p>Two data choices change the estimand. First, decide whether the start date is the licence date or the first week of use. Licence date supports an assignment effect. First use follows a chosen behaviour, so it carries selection and belongs with the usage questions. Second, these estimates cover only cohorts with an observed pre-period. See <a href="#check-pre-adoption">when there is no before</a>.</p>

<p>The repository's <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/did-itsa">DiD and ITSA notebook</a> is useful for learning the mechanics. It is not a staggered-robust estimator, and it has no clustered errors, so do not use it for the main estimate. For a published example of a staggered AI rollout, Brynjolfsson, Li and Raymond study an AI assistant introduced to customer-support agents at different times <a class="cite" href="#ref-brynjolfsson2025">[14]</a>.</p>

<h3 id="route-between-function">Whole functions treated at once</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Every person in a function started together, and timing differs only between functions.</dd>
  <dt>Key assumption</dt><dd>Functions treated at different times would have trended in parallel without treatment, and you can show pre-period evidence for it.</dd>
  <dt>What would break it</dt><dd>Function-specific shocks such as reorganisations, targets or seasonality that coincide with the start.</dd>
  <dt>Where to run it</dt><dd>The staggered estimators above with function as the unit of comparison, and <code>HonestDiD</code> for sensitivity. Seek specialist review.</dd>
</dl>

<p>When a whole function is treated at once, its effect is not separately identified from whatever else changed in that function at that time. Only cross-function comparisons remain. They need a defensible case that the functions would have moved together, supported by pre-period evidence, and a sensitivity analysis that shows how far the conclusion survives violations of parallel trends <a class="cite" href="#ref-rr2023">[15]</a>. <code>HonestDiD</code> takes the event-study coefficients and their covariance from any estimator.</p>

<p>If you cannot make that case, report the result as unidentified for that function. Before you do, check whether any sub-function, such as a team or region, has variation in timing. If it does, the within-function route above applies at that level.</p>

<h3 id="route-synthetic">A few treated groups with a long history</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>One or a few groups were treated, you have a long pre-period, and many untreated groups can serve as donors.</dd>
  <dt>Key assumption</dt><dd>A weighted mix of untreated groups reproduces the treated group's pre-period path and would have continued to track it.</dd>
  <dt>What would break it</dt><dd>A short or unstable pre-period, too few donor groups, or donors that were themselves affected by the tool.</dd>
  <dt>Where to run it</dt><dd><code>tidysynth</code> or <code>pysyncon</code> for synthetic control, <code>synthdid</code> for synthetic DiD, and segmented regression for controlled ITSA.</dd>
</dl>

<p>Synthetic control builds a comparison from a weighted set of untreated groups <a class="cite" href="#ref-adh2010">[16]</a>. Synthetic DiD combines that weighting with a DiD comparison <a class="cite" href="#ref-sdid2021">[17]</a>. Note that <code>synthdid</code> is available from GitHub only, is not actively maintained and assumes block adoption, where all treated groups start together. Controlled ITSA fits a pre-existing trend and a change at the start date for the treated group and for a comparison group <a class="cite" href="#ref-lb2018">[18]</a>.</p>

<p>Inference with few treated units is limited. Use placebo-in-space, which assigns the treatment to each donor in turn, to see whether the treated group's gap stands out. Report that, and the pre-period fit, alongside the estimate.</p>

<h3 id="route-itsa">Everyone treated at once: single-series ITSA</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>Everyone in scope was treated in the same week, there is no usable comparison group, and you have a long, stable pre-period with a sharp start date.</dd>
  <dt>Key assumption</dt><dd>Without the change, the pre-period trend would have continued unchanged through the post-period.</dd>
  <dt>What would break it</dt><dd>A concurrent event, seasonality, a reorganisation or another AI tool launching around the same date. Any of these can produce the same break.</dd>
  <dt>Where to run it</dt><dd>Segmented regression with <code>statsmodels</code> OLS and Newey-West (HAC) errors, or the single-series and controlled ITSA examples in the <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/did-itsa">repository notebook</a>.</dd>
</dl>
<p>Single-series interrupted time series fits a level and a slope before the start date, then asks whether the level or slope changes afterwards <a class="cite" href="#ref-lb2017">[19]</a>. It needs many weekly points before the change, so a short export rarely supports it. Treat the result as weak evidence, because anything else that happened that week is folded into the estimate.</p>
<p>Period fixed effects are unavailable here. When everyone is treated together, period effects absorb the treatment completely. Two checks remain. Run the same model on a negative-control outcome that the tool should not plausibly change. Then place several false breakpoints inside the pre-period and see how often a break of similar size appears. If it appears often, the series is too noisy for a single-series claim. Where another group was not treated, the controlled version is stronger, and the notebook covers it too.</p>

<h3 id="route-dose">Variation in how much people use AI</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>The question is about intensity of use, and you can observe usage repeatedly for the same people.</dd>
  <dt>Key assumption</dt><dd>Changes in a person's usage are unrelated to other changes in their work that also move the outcome.</dd>
  <dt>What would break it</dt><dd>Usage responding to workload, deadlines or projects (time-varying confounding), or the outcome driving usage (reverse causality).</dd>
  <dt>Where to run it</dt><dd>Within-person fixed effects (M2 in the <a href="{{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-causal-analysis">causal Skill</a>), <code>contdid</code> for continuous-treatment DiD (early release), or <code>DIDmultiplegtDYN</code> for non-binary treatment.</dd>
</dl>
<p>A within-person model compares the same person in heavier and lighter weeks. It removes stable differences between people, such as role or working style. It does not remove changes that move usage and the outcome together. Busy weeks may raise both Copilot use and meeting load, for example.</p>
<p>Continuous-treatment DiD compares groups by dose, and it needs stronger parallel-trends assumptions than the binary case <a class="cite" href="#ref-cgs2024">[20]</a>. Both packages are young or niche, so read their documentation and confirm the estimator supports doses that change from week to week. Timing checks are diagnostics. A causal reading also needs a justified strategy for time-varying confounding and assumptions that fit the observed pattern of use. Without them, report associations.</p>
<p>Avoid adding a lagged outcome to a short fixed-effects panel. It induces dynamic panel bias <a class="cite" href="#ref-nickell1981">[21]</a>. Keep unlike units separate. Actions, tasks, sessions and credits each get their own dose measure, and you never sum them into one AI score.</p>

<h3 id="route-selection">Selection on observables</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>You can credibly adjust for confounders measured before exposure, and comparable people exist across exposure levels within function. The outcome may be observed only once.</dd>
  <dt>Key assumption</dt><dd>After conditioning on the covariates, exposure is unrelated to the outcome you would have seen without it (unconfoundedness), and comparable people exist in both groups (overlap).</dd>
  <dt>What would break it</dt><dd>Unmeasured drivers of adoption such as motivation, manager support or project type, and heavy users who exist only in one function.</dd>
  <dt>Where to run it</dt><dd>Regression, matching, doubly robust estimation or double machine learning <a class="cite" href="#ref-dml2018">[22]</a> with <code>DoubleML</code> or <code>econml</code>. For Copilot actions, see the <a href="{{ site.baseurl }}/copilot-causal-toolkit/">Copilot Causal Toolkit</a>.</dd>
</dl>
<p>The covariates must be measured before exposure and must include function. Check overlap within function, using the distribution of estimated scores in each group. Do not rely on an AUC rule of thumb. If heavy users sit in one function only, the comparison extrapolates.</p>
<p>When access was neither randomised nor set by a defensible cut-off, this is also the route for an outcome observed once, with heavy caveats, because no fixed effect or timing check is available. The Toolkit implements double machine learning for <code>Total Copilot actions taken</code>. It is Copilot-specific and relies on unconfoundedness. No universal E-value threshold turns a sensitivity number into a pass, so report the sensitivity analysis and let readers judge it.</p>

<h3 id="route-no-estimate">When no design fits</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>No route above has its assumptions met.</dd>
  <dt>Key assumption</dt><dd>None. You make no causal claim.</dd>
  <dt>What would break it</dt><dd>Wording that slips an effect into a descriptive result.</dd>
  <dt>Where to run it</dt><dd>The <a href="{{ site.baseurl }}/frontier-analytics-prompt-ai-adoption/">AI Adoption Across Tools prompt</a> for descriptive reporting.</dd>
</dl>
<p>Say plainly that a causal estimate is unavailable, and report what you can describe. Then list what would change the answer. Useful additions are randomised waves, staggered waves inside a function, a longer pre-period, or an outcome that repeats weekly.</p>

<section id="joined-panel" class="part-head">
<h2>Build the joined panel</h2>
</section>
<p>Most routes start from the same exposure table, so build it once and carefully. The Person Query is the spine, at person-week grain. Other person-level sources join onto it after they have been reduced to the same grain. For a team or business-unit outcome, aggregate exposure to that outcome's own unit and period before joining, so the analysis runs at team-period level. Copying a team outcome onto every person creates repeated copies of one observation.</p>
<p>Agent data arrives at person-agent-week. Aggregate it to person-week before joining, or each person appears once per agent and the population multiplies. Consumption data may be daily, so aggregate it to the Person Query weeks at native grain first. The <a href="{{ site.baseurl }}/query-join-guide/">query join guide</a> lists the native keys, aligned week starts and scoped missing-row rules. After every join, confirm that the row count and the person-week keys match the spine. Log unmatched keys and stop if cardinality is ambiguous. Keep the raw header names, such as <code>Cowork tasks submitted</code> and <code>Total GitHub AI Credits used</code>, so every number traces back to its export.</p>

<h3 id="panel-status">Eligibility and treatment status, per tool</h3>
<p>Restrict each tool's analysis to the people who could have received it. Staff who will never hold a GitHub Copilot licence are not a comparison group for GitHub Copilot. Then classify every eligible person for that tool.</p>
<table>
  <thead>
    <tr><th>Status</th><th>Definition</th><th>Role in estimation</th></tr>
  </thead>
  <tbody>
    <tr><td>Treated in window</td><td>Exposure began on a dated week in the window</td><td>Treated cohort, defined by start week</td></tr>
    <tr><td>Already treated before the window</td><td>Access or adoption predates the export</td><td>Excluded from untreated comparison groups. No adoption effect can be estimated for them without a pre-period, but they can still support within-person intensity analyses</td></tr>
    <tr><td>Not yet treated</td><td>Eligible, exposure begins later in the window</td><td>Valid comparison only for weeks before their own start</td></tr>
    <tr><td>Never treated</td><td>Eligible, with licence or access records confirming no exposure through the window. Absent activity alone does not establish this</td><td>Comparison group, with a note that later exposure is possible</td></tr>
    <tr><td>Ineligible or unresolved</td><td>Not in the tool's population, or status cannot be established</td><td>Excluded from that tool's estimate</td></tr>
  </tbody>
</table>
<p>State the exposure definition for each tool, because the choices differ. A licence start date measures access. First observed use measures adoption. An intensity measure captures dose. These answer different questions, so say which one each estimate uses.</p>

<h3 id="panel-contamination">Contamination across tools</h3>
<p>Check every AI tool in the joined data, including those outside your study. Unlicensed Copilot Chat, Cowork and Agents can make a nominal comparison group partly treated. Decide first whether the estimand is this tool versus no AI, or this tool versus other AI use. Then build groups to match. Tools adopted together cannot be separated, so report a joint effect and do not attribute it to one tool.</p>
<p>Spillovers matter as well. Untreated colleagues can benefit from a treated colleague's drafts, summaries or meeting recaps, and within-function comparison people are the most exposed to this. If indirect exposure is likely, define direct and spillover effects separately, or move to a team-level design. A person's own non-use does not make them an unaffected comparison.</p>

<h3 id="panel-missing">Missing is not zero</h3>
<p>Unresolved coverage stays unknown. An omitted source or a failed join says nothing about use. Within a complete query's configured scope, absent activity rows do contribute zero. For documented licensed-only native Microsoft 365 Copilot metrics, a null means unlicensed and a zero means licensed but inactive. Do not extend that rule to unlicensed Copilot Chat or to Consumption credits, and do not infer a licence from credits or Agent sessions.</p>
<p>Publish only groups that meet the stricter of your organisation's minimum or 10 people. This applies to every function-level estimate in the sections that follow.</p>

<section id="function-confounding" class="part-head">
<h2>Function as a confounder</h2>
</section>
<p>Rollout usually follows function. An organisation licenses Engineering first, Sales later and Finance last, often for good business reasons. Function then shapes two things at once. It determines when people get the tool, and it determines how their outcomes move, through its own targets, seasons and reorganisations. A comparison that ignores function mixes the two.</p>
<p>The table below is a hypothetical illustration. It shows what each pattern allows, and contains no results.</p>
<table>
  <thead>
    <tr><th>Function</th><th>Rollout</th><th>Comparison available</th><th>What you can estimate</th></tr>
  </thead>
  <tbody>
    <tr><td>Engineering (hypothetical)</td><td>Licensed in three waves</td><td>Within function: earlier waves against later waves, before the later ones are treated</td><td>Wave-specific effects, valid under parallel trends conditional on function</td></tr>
    <tr><td>Sales (hypothetical)</td><td>Everyone licensed in one week</td><td>Between function only, against Engineering or Finance</td><td>A weak estimate that depends on Sales trending like the comparison function, with pre-period evidence</td></tr>
    <tr><td>Finance (hypothetical)</td><td>Not yet licensed</td><td>A between-function comparison for Sales, with the caveat in the Sales row</td><td>No effect for Finance itself</td></tr>
  </tbody>
</table>
<p>Prefer within-function comparisons where suitable untreated observations remain. Holding function fixed does not make them valid on its own. They still need conditional parallel trends, no anticipation and a clear treatment definition. Between-function comparisons rest on stronger assumptions. Sales and Engineering differ for many reasons, and a Sales-wide reorganisation looks identical to a Sales-wide AI effect. For a function treated all at once, the effect is not identified separately from function-specific shocks. Report it as unidentified unless you can show convincing pre-period parallel trends.</p>
<p>Function-specific estimates are useful in their own right, since effects probably differ by function. Report them where privacy thresholds allow.</p>

<h3 id="function-selection">Selection inside a function</h3>
<p>Within a function, licences may still go first to particular roles, levels or high performers. People may also be licensed after a dip in performance, which creates a pre-programme dip <a class="cite" href="#ref-ashenfelter1978">[23]</a> and makes recovery look like an effect. The check is to compare future adopters with later and never adopters before treatment, function by function. See the <a href="#check-pre-adoption">before-adoption comparison</a>.</p>
<p>Small cells are a real constraint. Within-function estimates may fall below the privacy threshold or lack the power to say anything. You can pool functions, but only with a stated justification, and you should say that you did.</p>

<section id="checks" class="part-head">
<h2>Standard checks</h2>
</section>

<p>Run these five checks on the observational routes where they apply. Each one says where it does not. Randomised and regression discontinuity designs have their own diagnostics, such as balance checks between arms and density tests at the cut-off, and are not covered here.</p>

<h3 id="check-fe-ladder">A fixed-effects ladder</h3>

<p>Report the estimate under increasingly demanding specifications, using the same sample, target population and aggregation weights each time. Start from a basic comparison with person and period adjustment, then add function-by-period fixed effects. Where useful, also show the pooled estimate without person effects. A large movement when person effects are added suggests that adopters differed from non-adopters from the start. A large movement when function-by-period effects are added suggests that function-specific shocks coincided with rollout. Show all three, so readers can see how much of the estimate depends on each layer.</p>

<p>Function-by-period effects are the default because rollout follows function. Plain period effects only absorb shocks that hit the whole organisation at once, such as a holiday week or a company-wide reorganisation. They do nothing about a function that was both licensed early and hit by its own targets or seasonality.</p>

<p>Three limits apply.</p>

<ul>
<li>Under staggered adoption, build this structure inside a staggered-adoption estimator (see <a href="#route-staggered-did">the staggered DiD route</a>). A single two-way fixed-effects coefficient can use already-treated people as comparisons and weight effects in misleading ways, so do not read the effect from one.</li>
<li>Person effects need repeated outcomes and some exposure variation in the sample. They do not apply to outcomes measured once. People whose exposure never changes, such as never-treated comparison people, still help estimate the period effects.</li>
<li>Period effects are unavailable without a comparison group. If everyone is treated in the same period, period effects absorb the treatment entirely. This rules them out on <a href="#route-itsa">the single-series ITSA route</a>.</li>
</ul>

<p>Fixed effects remove stable differences between people. They do not detect them, and they do nothing about differences that change over time, such as a person who is becoming more engaged and also more likely to be licensed. The checks below address that. Useful packages are <code>fixest</code> in R, and <code>pyfixest</code> or <code>linearmodels</code> in Python.</p>

<h3 id="check-leads">Time-reversal placebos</h3>

<p>Test whether future exposure "predicts" earlier outcomes. If it does, the usual explanations are licences targeted at people who were already improving, reverse causality, or allocation after a dip in performance, a pattern first documented as a pre-programme dip <a class="cite" href="#ref-ashenfelter1978">[23]</a>. In the causal Skill's labels this is the P1 timing placebo.</p>

<p>For dated licence events, the leads of an event study are this check. Plot the lead coefficients with their intervals. As a summary, run a joint Wald (F) test of the leads, excluding the reference period. For usage intensity, include current exposure alongside the lead. Usage persists from week to week, so a lead on its own partly measures current use.</p>

<p>People may change behaviour before access arrives, for example after training or an announcement. Specify an anticipation window. Keep those weeks out of the untreated comparison and out of the reference period, or move the treatment start to the announcement when that matches the question. Say which you did.</p>

<p>Report lead estimates with intervals and judge their size and precision. Pre-trend tests often have low power, and a non-significant result does not establish parallel trends <a class="cite" href="#ref-roth2022">[24]</a>. Where you can, show how far the conclusions survive bounded violations of parallel trends, using <code>HonestDiD</code> in R <a class="cite" href="#ref-rr2023">[15]</a>. It takes the event-study coefficients and covariance from any estimator.</p>

<p>Leads need dated exposure and several periods before it. They are not available on a route with a single series and no variation in timing across people.</p>

<h3 id="check-pre-adoption">Compare adopters before adoption</h3>

<p>Compare future adopters with later adopters and never adopters, within function, on outcomes and characteristics measured before treatment. Useful characteristics include level, tenure, role and earlier outcome trends. This addresses the stable selection that fixed effects remove without detecting. If future adopters already differed on the outcome or its direction of travel, the estimate needs a stronger argument or a different design.</p>

<aside class="callout reveal" markdown="1">
<span class="callout-label">When there is no 'before'</span>
Copilot and other AI features have been available for more than two years, so an export may start after many people had already adopted. Early adopters then have no observed pre-period. This check, and any DiD estimate, is limited to later cohorts and newer tools such as Cowork, Agents or recent licence waves. State plainly that the result describes those cohorts and says nothing direct about early adopters.
</aside>

<h3 id="check-negative-control">A negative-control outcome</h3>

<p>Estimate the same model on an outcome the tool should not plausibly change <a class="cite" href="#ref-negctrl2010">[25]</a>. An apparent effect there points to residual confounding or a shared shock, because the tool has no plausible route to it. A null result there is reassuring but does not rule out confounding that affects only your main outcome.</p>

<p>A useful negative control meets two conditions. The tool should have no plausible route to it, and it should share the confounding or measurement processes that threaten the main outcome, such as the same seasonal pattern or reorganisation. Write both reasons down before the analysis. AI tools can change meeting attendance and organisation, so meeting measures rarely qualify. If no candidate meets both conditions, say so in the write-up.</p>

<h3 id="check-leave-one-out">Leave one function out</h3>

<p>Re-estimate the model excluding each function in turn, and show the estimates side by side. Look at changes in the point estimate, its uncertainty and the weights. If one function dominates, report that concentration and estimate function-specific effects where the data support them. A loss of significance after dropping a function does not show that only that function benefited. Where cohorts are few, also leave one cohort out, since a single licence wave can dominate a pooled estimate. Small functions may fall below privacy thresholds, so the comparison can need pooling with a stated justification.</p>

<table>
<thead>
<tr><th>Check</th><th>Staggered DiD</th><th>Between-function</th><th>Synthetic or controlled ITSA</th><th>Single-series ITSA</th><th>Dose</th><th>Selection on observables</th></tr>
</thead>
<tbody>
<tr><td>Fixed-effects ladder</td><td>Partly: inside the estimator</td><td>Partly: person and period effects only, because function-by-period effects absorb a function-wide start</td><td>Partly: group and period terms, few units</td><td>No: no comparison group for period effects</td><td>Yes: person and function-by-period</td><td>Partly: needs repeated outcomes</td></tr>
<tr><td>Time-reversal placebo</td><td>Yes: event-study leads</td><td>Yes: pre-period trends by function</td><td>Partly: placebo dates or held-out pre-period fit</td><td>Partly: stable pre-trend only</td><td>Yes: lead plus current exposure</td><td>Partly: only with repeated outcomes before exposure</td></tr>
<tr><td>Compare adopters before adoption</td><td>Yes: within function</td><td>Partly: functions, not people</td><td>Partly: donor versus treated group</td><td>No: no untreated people</td><td>Yes: heavy versus light users</td><td>Yes: covariate balance and overlap, outcomes only if observed before exposure</td></tr>
<tr><td>Negative-control outcome</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes: an additional diagnostic</td><td>Yes</td><td>Yes</td></tr>
<tr><td>Leave one function out</td><td>Yes</td><td>Partly: few functions</td><td>Partly: leave one donor out</td><td>No: one series</td><td>Yes</td><td>Yes</td></tr>
</tbody>
</table>

<section id="communicating" class="part-head">
<h2>Report what you can claim</h2>
</section>

<p>Report four things separately: the effect, its precision, the identification argument and the limitations. A single sentence that blends them hides which part is weak. The identification argument says why the comparison might be credible. The limitations say what would break it.</p>

<p>Say whether the outcome is a proxy or a KPI. Viva Insights measures such as <code>External collaboration hours</code> or <code>After hours collaboration hours</code> describe behaviour, so label proxy results as behavioural outcomes or candidate leading indicators and do not present them as business impact. Say how the proxy relates to the KPI, or that this could not be checked.</p>

<p>Name the estimand. State what is compared, for which cohorts and which functions, and over which weeks. If early adopters have no observed pre-period, an adoption-effect estimate covers later cohorts only, and the text should say so. Within-person intensity results can still include early adopters.</p>

<p>Report the clustering level and the number of clusters in each model sample. Cluster at the level at which exposure was assigned or the sample was drawn, which is often function or team rather than person <a class="cite" href="#ref-cluster2023">[12]</a>. With few clusters, standard cluster-robust inference can be unreliable <a class="cite" href="#ref-cm2015">[13]</a>. With few clusters, consider small-sample corrections or randomisation-based inference. If no method is defensible, say that uncertainty cannot be reliably quantified.</p>

<p>Use association language by default. Causal wording needs the identification argument and the checks above to hold up together. A pair of hypothetical sentences shows the difference.</p>

<ul>
<li>Weak: "Copilot reduced after-hours work in Sales."</li>
<li>Careful: "Among later licence cohorts in Sales, weeks after licensing are associated with fewer <code>After hours collaboration hours</code>. The estimate is consistent with a reduction, and the leads show no clear pre-trend, but the interval is wide and function-level shocks cannot be ruled out."</li>
</ul>

<p>Decide the primary outcome and its expected direction before looking at results, and treat other outcomes as secondary or exploratory. If you test many outcomes, functions or cohorts, say how many, and adjust or flag the multiplicity. Pre-register the plan where your organisation allows it.</p>

<p>Apply privacy thresholds to every published cell, and check that complementary cells cannot be subtracted to reveal a small group. Within-function comparisons shrink cell sizes quickly. Pool with a stated justification, or report the function as not identified. Do not publish rankings of individuals.</p>

<p>Finally, include a short list of what you did not do. Examples are checks that were unavailable for the route, functions that were dropped for privacy reasons, and outcomes you examined but did not report. Readers can then judge the claim against the full set of choices. If a key check could not be run, say what it would have tested and why its absence matters for the claim you are making.</p>

<p>Keep the headline claim no stronger than the weakest link in the chain from exposure to outcome. If the exposure is a licence, the claim is about access. If it is chosen use, the claim is about people who chose to use the tool, and stronger assumptions apply.</p>

<section id="run-it" class="part-head">
<h2>Run it yourself</h2>
</section>

<p>These resources cover parts of the workflow. None of them runs every route in this article.</p>

<ul>
<li><a href="{{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-causal-analysis">The causal Skill</a> checks that a panel is ready and runs within-person diagnostics. It returns a proposed plan for DiD, event studies and instruments rather than running them.</li>
<li><a href="{{ site.baseurl }}/copilot-causal-toolkit/">The Copilot Causal Toolkit</a> runs a double machine learning workflow. It is specific to Copilot and suits the selection-on-observables route.</li>
<li>The event-study scripts in <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.Rmd">R</a> and <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/event-study-did.py">Python</a> fit a staggered-adoption event study on synthetic data, with <code>fixest::sunab()</code> in R and a saturated event study in <code>pyfixest</code>. They do not condition on function, so add function-by-period fixed effects for the within-function comparison described in the <a href="#route-staggered-did">staggered route</a>.</li>
<li>The difference-in-differences metric scans in <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.Rmd">R</a> and <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/did-metric-scan.py">Python</a> compare heavy and light users across many metrics. Treat them as exploratory. Groups defined by chosen usage carry selection, and testing many metrics needs a multiplicity adjustment.</li>
<li><a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/did-itsa">The DiD/ITSA notebook</a> implements 2x2 DiD, a two-way fixed-effects event study, and controlled and single-series ITSA. It is not a staggered-robust estimator and does not use clustered errors, so use it to learn the structure and not for a staggered rollout.</li>
<li><a href="{{ site.baseurl }}/query-join-guide/">The query join guide</a> explains how to join other sources onto the Person Query before you build the panel.</li>
</ul>

<p>For routes the repository does not implement, these maintained packages are the starting point. Check each package's documentation for current options.</p>

<table>
<thead>
<tr><th>Route</th><th>R</th><th>Python</th></tr>
</thead>
<tbody>
<tr><td>Staggered DiD</td><td><code>did</code></td><td><code>differences</code>, <code>csdid</code></td></tr>
<tr><td>Imputation or two-stage</td><td><code>didimputation</code>, <code>did2s</code></td><td><code>pyfixest</code></td></tr>
<tr><td>Event study with fixed effects</td><td><code>fixest</code></td><td><code>pyfixest</code></td></tr>
<tr><td>Non-binary or non-absorbing treatment</td><td><code>DIDmultiplegtDYN</code></td><td>None found</td></tr>
<tr><td>Continuous dose DiD</td><td><code>contdid</code> (early release)</td><td>None found</td></tr>
<tr><td>Pre-trend sensitivity</td><td><code>HonestDiD</code></td><td>No maintained port verified</td></tr>
<tr><td>Synthetic control</td><td><code>tidysynth</code>, <code>Synth</code></td><td><code>pysyncon</code></td></tr>
<tr><td>Synthetic DiD</td><td><code>synthdid</code> (not actively maintained)</td><td>Port is stale</td></tr>
<tr><td>Controlled or single-series ITSA</td><td>Base R or <code>fixest</code></td><td><code>statsmodels</code></td></tr>
<tr><td>Fixed-effects ladder</td><td><code>fixest</code></td><td><code>pyfixest</code>, <code>linearmodels</code></td></tr>
<tr><td>Double machine learning</td><td><code>DoubleML</code></td><td><code>DoubleML</code>, <code>econml</code></td></tr>
</tbody>
</table>

<section id="references" class="references">
<h2>References</h2>
<ol>
  <li id="ref-surrogate">Athey, S., Chetty, R., Imbens, G. W. and Kang, H. (2025). <a href="https://doi.org/10.1093/restud/rdaf087">The surrogate index: combining short-term proxies to estimate long-term treatment effects more rapidly and precisely</a>. <em>Review of Economic Studies</em>. Earlier version: NBER Working Paper 26463 (2019).</li>
  <li id="ref-prentice1989">Prentice, R. L. (1989). <a href="https://doi.org/10.1002/sim.4780080407">Surrogate endpoints in clinical trials: definition and operational criteria</a>. <em>Statistics in Medicine</em>, 8(4), 431–440.</li>
  <li id="ref-ia1994">Imbens, G. W. and Angrist, J. D. (1994). <a href="https://doi.org/10.2307/2951620">Identification and estimation of local average treatment effects</a>. <em>Econometrica</em>, 62(2), 467–475.</li>
  <li id="ref-ll2010">Lee, D. S. and Lemieux, T. (2010). <a href="https://doi.org/10.1257/jel.48.2.281">Regression discontinuity designs in economics</a>. <em>Journal of Economic Literature</em>, 48(2), 281–355.</li>
  <li id="ref-gb2021">Goodman-Bacon, A. (2021). <a href="https://doi.org/10.1016/j.jeconom.2021.03.014">Difference-in-differences with variation in treatment timing</a>. <em>Journal of Econometrics</em>, 225(2), 254–277.</li>
  <li id="ref-cdh2020">de Chaisemartin, C. and D'Haultfœuille, X. (2020). <a href="https://doi.org/10.1257/aer.20181169">Two-way fixed effects estimators with heterogeneous treatment effects</a>. <em>American Economic Review</em>, 110(9), 2964–2996.</li>
  <li id="ref-sa2021">Sun, L. and Abraham, S. (2021). <a href="https://doi.org/10.1016/j.jeconom.2020.09.006">Estimating dynamic treatment effects in event studies with heterogeneous treatment effects</a>. <em>Journal of Econometrics</em>, 225(2), 175–199.</li>
  <li id="ref-rsbp2023">Roth, J., Sant'Anna, P. H. C., Bilinski, A. and Poe, J. (2023). <a href="https://doi.org/10.1016/j.jeconom.2023.03.008">What's trending in difference-in-differences? A synthesis of the recent econometrics literature</a>. <em>Journal of Econometrics</em>, 235(2), 2218–2244.</li>
  <li id="ref-cs2021">Callaway, B. and Sant'Anna, P. H. C. (2021). <a href="https://doi.org/10.1016/j.jeconom.2020.12.001">Difference-in-differences with multiple time periods</a>. <em>Journal of Econometrics</em>, 225(2), 200–230.</li>
  <li id="ref-bjs2024">Borusyak, K., Jaravel, X. and Spiess, J. (2024). <a href="https://doi.org/10.1093/restud/rdae007">Revisiting event-study designs: robust and efficient estimation</a>. <em>Review of Economic Studies</em>, 91(6), 3253–3285.</li>
  <li id="ref-wfh2024">Wing, C., Freedman, S. M. and Hollingsworth, A. (2024). <a href="https://doi.org/10.3386/w32054">Stacked difference-in-differences</a>. NBER Working Paper 32054.</li>
  <li id="ref-cluster2023">Abadie, A., Athey, S., Imbens, G. W. and Wooldridge, J. M. (2023). <a href="https://doi.org/10.1093/qje/qjac038">When should you adjust standard errors for clustering?</a>. <em>The Quarterly Journal of Economics</em>, 138(1), 1–35.</li>
  <li id="ref-cm2015">Cameron, A. C. and Miller, D. L. (2015). <a href="https://doi.org/10.3368/jhr.50.2.317">A practitioner's guide to cluster-robust inference</a>. <em>Journal of Human Resources</em>, 50(2), 317–372.</li>
  <li id="ref-brynjolfsson2025">Brynjolfsson, E., Li, D. and Raymond, L. (2025). <a href="https://doi.org/10.1093/qje/qjae044">Generative AI at work</a>. <em>The Quarterly Journal of Economics</em>, 140(2), 889–942.</li>
  <li id="ref-rr2023">Rambachan, A. and Roth, J. (2023). <a href="https://doi.org/10.1093/restud/rdad018">A more credible approach to parallel trends</a>. <em>Review of Economic Studies</em>, 90(5), 2555–2591.</li>
  <li id="ref-adh2010">Abadie, A., Diamond, A. and Hainmueller, J. (2010). <a href="https://doi.org/10.1198/jasa.2009.ap08746">Synthetic control methods for comparative case studies: estimating the effect of California's tobacco control program</a>. <em>Journal of the American Statistical Association</em>, 105(490), 493–505.</li>
  <li id="ref-sdid2021">Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. and Wager, S. (2021). <a href="https://doi.org/10.1257/aer.20190159">Synthetic difference-in-differences</a>. <em>American Economic Review</em>, 111(12), 4088–4118.</li>
  <li id="ref-lb2018">Lopez Bernal, J., Cummins, S. and Gasparrini, A. (2018). <a href="https://doi.org/10.1093/ije/dyy135">The use of controls in interrupted time series studies of public health interventions</a>. <em>International Journal of Epidemiology</em>, 47(6), 2082–2093.</li>
  <li id="ref-lb2017">Lopez Bernal, J., Cummins, S. and Gasparrini, A. (2017). <a href="https://doi.org/10.1093/ije/dyw098">Interrupted time series regression for the evaluation of public health interventions: a tutorial</a>. <em>International Journal of Epidemiology</em>, 46(1), 348–355.</li>
  <li id="ref-cgs2024">Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024). <a href="https://doi.org/10.3386/w32117">Difference-in-differences with a continuous treatment</a>. NBER Working Paper 32117.</li>
  <li id="ref-nickell1981">Nickell, S. (1981). <a href="https://doi.org/10.2307/1911408">Biases in dynamic models with fixed effects</a>. <em>Econometrica</em>, 49(6), 1417–1426.</li>
  <li id="ref-dml2018">Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018). <a href="https://doi.org/10.1111/ectj.12097">Double/debiased machine learning for treatment and structural parameters</a>. <em>The Econometrics Journal</em>, 21(1), C1–C68.</li>
  <li id="ref-ashenfelter1978">Ashenfelter, O. (1978). <a href="https://doi.org/10.2307/1924332">Estimating the effect of training programs on earnings</a>. <em>The Review of Economics and Statistics</em>, 60(1), 47–57.</li>
  <li id="ref-roth2022">Roth, J. (2022). <a href="https://doi.org/10.1257/aeri.20210236">Pretest with caution: event-study estimates after testing for parallel trends</a>. <em>American Economic Review: Insights</em>, 4(3), 305–322.</li>
  <li id="ref-negctrl2010">Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010). <a href="https://doi.org/10.1097/EDE.0b013e3181d61eeb">Negative controls: a tool for detecting confounding and bias in observational studies</a>. <em>Epidemiology</em>, 21(3), 383–388.</li>
</ol>
</section>
