---
layout: article
title: "Causal Questions About AI Usage"
description: "A question-led guide to choosing a causal design for AI usage data from Viva Insights, covering staggered, function-led rollouts, proxy outcomes and the checks every analysis should run."
permalink: /articles/causal-questions-ai-usage/
eyebrow: "Copilot Analytics Lab · PANDAS Team · November 2026"
dek: "Most AI rollouts are staggered, product-specific and led by function. This guide asks a short series of questions about your data and points you to a design that fits, or says plainly when none does."
byline: "By the PANDAS team · A Copilot Analytics Lab methods guide"
read_time: "22 min read"
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

<p class="lead">Leaders ask a plain question: did AI use change how work gets done? The honest answer depends on how the data is structured. AI use now spans Microsoft 365 Copilot, Cowork, Agents and GitHub Copilot, each measured in a different query with different units, start dates and eligible people. Whether you can say anything causal depends on who was exposed, when, and what you can fairly compare them with.</p>


### The TL;DR

<ul class="tldr">
  <li>Choose the outcome before the method. The outcome decides what a credible result can say.</li>
  <li>Staggered, product-specific, function-led rollouts are the normal case. They need estimators built for them, and comparisons within function wherever timing varies inside functions.</li>
  <li>Run the standard checks on every route where they apply, and say where they cannot be run.</li>
  <li>Report results on Viva Insights proxies as behavioural outcomes or candidate leading indicators, and keep business impact claims for outcomes that measure it.</li>
</ul>

<h3 id="setting">The usual starting point</h3>

<p>Most analyses of AI use share a setting. Licences are product-specific, and rollout tends to follow function, so function shapes both when people get access and how their outcomes move. The data is typically a person-week panel joined onto the Person Query. Copilot has been available for over two years, so many early adopters have no observed pre-period in the export, and before-and-after comparisons are limited to later cohorts and newer tools.</p>

<p>This article is for people analytics practitioners who hold Viva Insights exports and know regression, and who may be new to modern difference-in-differences. It routes you to a design and does not catalogue every method. It replaces the site's earlier technique-by-technique guide.</p>

<h3 id="key-terms">Key terms</h3>

<p>Each term is also underlined where it first appears in the text.</p>

<dl class="key-terms">
  <dt id="term-anticipation">Anticipation</dt>
  <dd>People changing behaviour before access starts, for example after an announcement, which makes pre-period weeks partly treated.</dd>
  <dt id="term-clustering">Clustering</dt>
  <dd>Allowing for correlated outcomes within a group, such as a function or team. Standard errors are usually clustered at least at the level at which exposure was assigned.</dd>
  <dt id="term-cohort">Cohort and not-yet-treated</dt>
  <dd>A cohort is the group of people whose exposure started in the same week. Not-yet-treated people start later, so they are valid comparisons only for weeks before their own start.</dd>
  <dt id="term-conditional-parallel-trends">Conditional parallel trends</dt>
  <dd>The assumption that, among comparable people (here, within the same function), treated and comparison groups would have moved in parallel without treatment.</dd>
  <dt id="term-complier">Compliers (local average treatment effect)</dt>
  <dd>Compliers are the people whose use changes because of their assignment. Their effect, using assignment as an instrument, is the local average treatment effect.</dd>
  <dt id="term-estimand">Estimand</dt>
  <dd>The exact quantity you want to estimate, such as the average effect of access for later licence cohorts in one function over stated weeks.</dd>
  <dt id="term-event-study">Event study and leads</dt>
  <dd>A model that estimates effects for each week relative to the start date. Leads are the estimates for weeks before the start, and should be close to zero if the comparison is fair.</dd>
  <dt id="term-exclusion-monotonicity">Exclusion restriction and monotonicity</dt>
  <dd>Two assumptions for using assignment as an instrument. Exclusion says assignment affects the outcome only through use. Monotonicity says assignment never pushes anyone to use less.</dd>
  <dt id="term-exposure">Exposure and treatment</dt>
  <dd>Exposure is a person's contact with an AI tool, such as a licence, first use or an amount of use. Treatment is the exposure whose effect you are estimating.</dd>
  <dt id="term-itt">Intention to treat</dt>
  <dd>Comparing people by assigned group, whether or not they used the tool. It estimates the effect of giving access.</dd>
  <dt id="term-negative-control">Negative control</dt>
  <dd>An outcome the tool should not plausibly change. An apparent effect on it signals confounding or a shared shock.</dd>
  <dt id="term-surrogate">Surrogate</dt>
  <dd>A short-term measure standing in for a longer-term outcome. It is valid only if it captures all of the effect on that outcome.</dd>
  <dt id="term-twfe">Two-way fixed effects (TWFE)</dt>
  <dd>A regression with person and period fixed effects. With staggered start dates, one TWFE coefficient can use already-treated people as comparisons and give misleading weights.</dd>
  <dt id="term-unconfoundedness">Unconfoundedness and overlap</dt>
  <dd>Unconfoundedness says that, given the covariates, exposure is unrelated to the outcome a person would have had without it. Overlap says comparable people exist at each exposure level.</dd>
</dl>

<section id="choosing-outcome" class="part-head">
<h2>Choose the outcome first</h2>
</section>

<p>Pick the outcome before you pick a method. A method cannot repair an outcome that sits too far from the exposure. In practice the choice is between a business KPI and a Viva Insights proxy.</p>

<h3>Business KPIs</h3>

<p>Revenue, cases closed and cycle time are what leaders want to hear about, and they are the hardest outcomes to tie to AI use. Many functions contribute to a KPI, so one person's or function's AI <a class="term" href="#term-exposure" data-def="Contact with an AI tool, such as a licence, first use or amount of use. Treatment is the exposure being studied.">exposure</a> is a single contribution among many. The lag between exposure and the KPI can exceed the window you have. KPIs are usually recorded for teams or business units, monthly or quarterly, while exposure is individual and weekly, which leaves little statistical power. Attribution is rarely defensible when other tools, reorganisations and market conditions moved at the same time.</p>

<h3>Viva Insights proxies</h3>

<p>Measures such as <code>External collaboration hours</code>, <code>After hours collaboration hours</code> and <code>Meeting hours</code> are recorded per person, per week, and line up with exposure dates. That supports panel designs with many units and many periods. They carry their own risks.</p>

<ul>
  <li><strong>They measure behaviour.</strong> More <code>External collaboration hours</code> may reflect more customer contact, but it is not necessarily more sales. It could equally reflect more rework or longer sales cycles.</li>
  <li><strong>Direction is ambiguous.</strong> Fewer meeting hours could mean efficiency or disengagement. Declare the expected direction before looking at results.</li>
  <li><strong>The tool can move the metric directly.</strong> Meeting recap features can change meeting attendance, for example. Prefer proxies the tool cannot generate itself.</li>
  <li><strong>Targets change behaviour.</strong> Once people are asked to improve a proxy, it can start to measure something different.</li>
</ul>

<aside class="callout is-action reveal" markdown="1">
<span class="callout-label">What to do</span>

- Write the causal chain: exposure, behaviour, process measure, KPI. Say which link your analysis tests.
- Pre-declare one primary outcome and its expected direction. Treat everything else as secondary or exploratory.
- Label proxy results as behavioural outcomes or candidate leading indicators, never as business impact.
- Where a KPI exists at team grain with enough teams and periods, analyse it separately and compare it with the proxy result.
- Check whether the proxy has tracked the KPI historically. That is predictive evidence only, because treating a proxy as a <a class="term" href="#term-surrogate" data-def="A short-term measure standing in for a longer-term outcome. It needs strong assumptions to be valid.">surrogate</a> for the KPI needs much stronger assumptions <a class="cite" href="#ref-surrogate">[1]</a> <a class="cite" href="#ref-prentice1989">[2]</a>.
- Add a <a class="term" href="#term-negative-control" data-def="An outcome the tool should not plausibly change. An apparent effect on it signals confounding or a shared shock.">negative-control</a> outcome (see [the negative-control check](#check-negative-control)).
- Confirm how often the outcome is really measured. An annual rating copied onto weekly rows is one observation a year.
</aside>

<p>Proxies are often the only outcomes with enough weekly variation to analyse, so the task is to label them honestly. Once the outcome is settled, the <a href="#decision-guide">decision guide</a> helps you pick a design.</p>

<section id="decision-guide" class="part-head">
<h2>Find your route</h2>
</section>

<p>Answer one question at a time. Each answer leads to the next question or to a route in Part 3, and you can step back to change an answer. With JavaScript turned off, the questions appear as a nested list.</p>

{% include causal-decision-guide.html %}

<p>If your situation sits between two routes, read both and run the stricter set of checks.</p>

<section id="routes" class="part-head"><h2>The routes</h2></section>

<p>Each route states when it fits, the assumption it rests on, the main threats to that assumption, and where to run it.</p>

<h3 id="route-descriptive">Describe use without a causal claim</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>The question is who uses what, how much, and how that changes over time.</dd>
  <dt>Key assumption</dt><dd>None about causes. The claim is a description of the data you hold.</dd>
  <dt>What would break it</dt><dd>Wording that drifts into "because of AI". Coverage gaps that are treated as zero use.</dd>
  <dt>Where to run it</dt><dd>The <a href="{{ site.baseurl }}/frontier-analytics-prompt-ai-adoption/">AI Adoption Across Tools prompt</a>.</dd>
</dl>

<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Describe use without a causal claim</span></summary>

<p>If the question is about the spread of use across Microsoft 365 Copilot, Cowork, Agents, Consumption credits and GitHub Copilot, you do not need a causal design. Report counts, shares and trends by function and tool, keep unlike units separate, and treat unresolved coverage as unknown.</p>

</details>

<h3 id="route-randomised">Randomised access</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Access was assigned at random, for example a pilot lottery or waves randomised within function.</dd>
  <dt>Key assumption</dt><dd>Assignment is as good as random. To estimate the effect of usage, assignment must affect the outcome only through use (<a class="term" href="#term-exclusion-monotonicity" data-def="Exclusion: assignment affects the outcome only through use. Monotonicity: assignment never pushes anyone to use less.">exclusion</a>) and must not push anyone to use less (<a class="term" href="#term-exclusion-monotonicity" data-def="Monotonicity: assignment never pushes anyone to use less. Exclusion: assignment affects the outcome only through use.">monotonicity</a>).</dd>
  <dt>What would break it</dt><dd>Assignment that was not truly random, unequal attrition between arms, or assignment that changes behaviour without any use, for example through training or announcements.</dd>
  <dt>Where to run it</dt><dd><code>fixest</code> (<code>feols</code> with its IV syntax), <code>pyfixest</code>, or <code>linearmodels</code> IV. Seek specialist review if the randomisation was partial.</dd>
</dl>

<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Randomised access</span></summary>

<p>Compare people by the group they were assigned to, whether or not they used the tool. This <a class="term" href="#term-itt" data-def="Comparing people by the group they were assigned to, whether or not they used the tool. It estimates the effect of access.">intention-to-treat</a> comparison answers what giving access did. Waves randomised within function are the strongest design you are likely to meet, because function is balanced by construction.</p>

<p>For the effect of usage, assignment can serve as an instrument. For a binary measure such as any use, it identifies a local effect for <a class="term" href="#term-complier" data-def="People whose use changes because of their assignment. Their effect is the local average treatment effect.">compliers</a>, provided exclusion and monotonicity hold <a class="cite" href="#ref-ia1994">[3]</a>. For continuous usage intensity, the instrumental-variable estimate is a weighted average of effects across levels of use and needs its own stated assumptions. Randomised access does not make chosen usage random, so keep the assignment effect and the usage effect as separate results. Instruments that were not randomised, such as a manager's early adoption, need a separate argument that they are as good as random and affect the outcome only through use. That is rarely convincing in workplace data, so treat them as specialist work.</p>

</details>

<h3 id="route-rdd">An eligibility cut-off</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Access depended on a threshold, such as a score or a start date, and many people sit close to it on either side.</dd>
  <dt>Key assumption</dt><dd>Outcomes would have varied smoothly through the cut-off without treatment. Nothing else changes at the cut-off, and nobody could sort themselves precisely across it.</dd>
  <dt>What would break it</dt><dd>Too few people near the cut-off, or manipulation of the score or date.</dd>
  <dt>Where to run it</dt><dd>Rare here. Flag for specialist review.</dd>
</dl>

<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: An eligibility cut-off</span></summary>

<p>Regression discontinuity compares people just either side of the threshold <a class="cite" href="#ref-ll2010">[4]</a>. Rollouts here rarely use a sharp rule, so confirm first that one exists. If it does, check how many people fall near the cut-off and whether managers could adjust who sat on which side.</p>

</details>

<h3 id="route-staggered-did">Staggered rollout, compared within function</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Licences or access arrived in waves, several waves or both treated and <a class="term" href="#term-cohort" data-def="Eligible people whose exposure starts later. They are valid comparisons only for weeks before their own start.">not-yet-treated</a> people exist inside the same function, and you hold repeated pre-adoption weeks.</dd>
  <dt>Key assumption</dt><dd>Within function, people treated in a wave would have followed the same path as comparable people not yet treated, had they not been treated (<a class="term" href="#term-conditional-parallel-trends" data-def="Among comparable people, here within function, treated and comparison groups would have moved in parallel without treatment.">conditional parallel trends</a>).</dd>
  <dt>What would break it</dt><dd>Licences going first to people already on a different path, comparison people taking up other AI tools at a different rate from the treated group during the window, behaviour changing before the licence date, or too few <a class="term" href="#term-clustering" data-def="Groups, such as functions or teams, within which outcomes are correlated. Standard errors are usually clustered at least at the level of assignment.">clusters</a>.</dd>
  <dt>Where to run it</dt><dd><code>did</code> in R, <code>didimputation</code>, <code>did2s</code>, <code>fixest::sunab()</code> or <code>pyfixest</code>. In Python, <code>differences</code> or <code>csdid</code>. The repository notebook is a starting point only.</dd>
</dl>

<details class="route-more" open>
<summary>Read the full route<span class="vi-sr-only">: Staggered rollout, compared within function</span></summary>

<p>This is the usual case here. Build the comparison from the population that could have received the tool, such as developers without a GitHub Copilot licence yet. Label each person as treated with a dated start, not yet treated, or never treated in the window. Not-yet-treated people are valid comparisons only for the weeks before their own start, and only inside an estimator built for that.</p>

<p>Avoid reading the effect from a single <a class="term" href="#term-twfe" data-def="A regression with person and period fixed effects. With staggered starts it can use already-treated people as comparisons.">two-way fixed-effects</a> coefficient. When start dates differ, it is a weighted average of many two-group comparisons, and some use already-treated people as the comparison <a class="cite" href="#ref-gb2021">[5]</a>. If effects differ across <a class="term" href="#term-cohort" data-def="A group of people whose exposure started in the same week.">cohorts</a> or over time, it can place negative weight on some of the underlying effects <a class="cite" href="#ref-cdh2020">[6]</a> <a class="cite" href="#ref-sa2021">[7]</a>. For a wider orientation, see <a class="cite" href="#ref-rsbp2023">[8]</a>. Pick one of these estimators, with function in the conditioning set:</p>

<ul>
  <li><strong>Group-time estimates.</strong> Estimate an effect for each start cohort and week, then summarise <a class="cite" href="#ref-cs2021">[9]</a>. In R, use <code>did::att_gt()</code> with <code>xformla</code> to put function in the conditioning set and <code>control_group = "notyettreated"</code>.</li>
  <li><strong>Imputation.</strong> Fit the untreated outcomes with person and function-by-period fixed effects, then compare treated people with their predictions <a class="cite" href="#ref-bjs2024">[10]</a>.</li>
  <li><strong>Sun and Abraham event studies.</strong> Use <code>fixest::sunab()</code> for an <a class="term" href="#term-event-study" data-def="Effects estimated for each week relative to the start date. Leads are the estimates for weeks before the start.">event study</a> that keeps cohorts apart <a class="cite" href="#ref-sa2021">[7]</a>.</li>
  <li><strong>Stacked DiD.</strong> Build a data set for each cohort, with a clean comparison window and comparison people from the same function, then combine them with the weights needed for the average effect you want to report <a class="cite" href="#ref-wfh2024">[11]</a>.</li>
</ul>

<p>Report an event study with pre-period leads, by function wherever cells meet the privacy threshold. Where a function falls below the threshold or lacks power, pool with a stated justification or report it as unidentified. Cluster at the level at which licences were assigned, which is often function or team and seldom the person <a class="cite" href="#ref-cluster2023">[12]</a> <a class="cite" href="#ref-cm2015">[13]</a>. With few clusters, consider small-sample corrections or randomisation-based inference.</p>

<p>Two data choices change the <a class="term" href="#term-estimand" data-def="The exact quantity you want to estimate, for example the effect of access for later cohorts in one function over stated weeks.">estimand</a>. Licence date as the start supports an assignment effect, while first week of use follows a chosen behaviour and carries selection. These estimates also cover only cohorts with an observed pre-period (see <a href="#check-pre-adoption">when there is no before</a>). For a published example of a staggered AI rollout, see Brynjolfsson, Li and Raymond on an AI assistant introduced to customer-support agents at different times <a class="cite" href="#ref-brynjolfsson2025">[14]</a>.</p>

</details>

<h3 id="route-between-function">Whole functions treated at once</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>Every person in a function started together, and timing differs only between functions.</dd>
  <dt>Key assumption</dt><dd>Functions treated at different times would have trended in parallel without treatment, and you can show pre-period evidence for it.</dd>
  <dt>What would break it</dt><dd>Function-specific shocks such as reorganisations, targets or seasonality that coincide with the start.</dd>
  <dt>Where to run it</dt><dd>The staggered estimators above with function as the unit of comparison, and <code>HonestDiD</code> for sensitivity. Seek specialist review.</dd>
</dl>

<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Whole functions treated at once</span></summary>

<p>When a whole function is treated at once, its effect is not separately identified from whatever else changed in that function at that time. Only cross-function comparisons remain. They need pre-period evidence that the functions moved together, and a sensitivity analysis showing how far the conclusion survives violations of parallel trends <a class="cite" href="#ref-rr2023">[15]</a>. Otherwise report the result as unidentified. First check whether a sub-function, such as a team or region, has variation in timing, in which case the within-function route applies at that level.</p>

</details>

<h3 id="route-synthetic">A few treated groups with a long history</h3>

<dl class="route-facts">
  <dt>Use when</dt><dd>One or a few groups were treated, you have a long pre-period, and many untreated groups can serve as donors.</dd>
  <dt>Key assumption</dt><dd>A weighted mix of untreated groups reproduces the treated group's pre-period path and would have continued to track it.</dd>
  <dt>What would break it</dt><dd>A short or unstable pre-period, too few donor groups, or donors that were themselves affected by the tool.</dd>
  <dt>Where to run it</dt><dd><code>tidysynth</code> or <code>pysyncon</code> for synthetic control, <code>synthdid</code> for synthetic DiD, and segmented regression for controlled ITSA.</dd>
</dl>

<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: A few treated groups with a long history</span></summary>

<p>Synthetic control builds a comparison from a weighted set of untreated groups <a class="cite" href="#ref-adh2010">[16]</a>. Synthetic DiD combines that weighting with a DiD comparison <a class="cite" href="#ref-sdid2021">[17]</a>, but <code>synthdid</code> is available from GitHub only, is not actively maintained and assumes block adoption. Controlled ITSA fits a pre-existing trend and a change at the start date for the treated group and a comparison group <a class="cite" href="#ref-lb2018">[18]</a>. Inference with few treated units is limited, so use placebo-in-space, which assigns the treatment to each donor in turn, and report it with the pre-period fit.</p>

</details>

<h3 id="route-itsa">Everyone treated at once: single-series ITSA</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>Everyone in scope was treated in the same week, there is no usable comparison group, and you have a long, stable pre-period with a sharp start date.</dd>
  <dt>Key assumption</dt><dd>Without the change, the pre-period trend would have continued unchanged through the post-period.</dd>
  <dt>What would break it</dt><dd>A concurrent event, seasonality, a reorganisation or another AI tool launching around the same date. Any of these can produce the same break.</dd>
  <dt>Where to run it</dt><dd>Segmented regression with <code>statsmodels</code> OLS and Newey-West (HAC) errors, or the single-series and controlled ITSA examples in the <a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/did-itsa">repository notebook</a>.</dd>
</dl>
<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Everyone treated at once: single-series ITSA</span></summary>

<p>Single-series interrupted time series fits a level and a slope before the start date, then asks whether either changes afterwards <a class="cite" href="#ref-lb2017">[19]</a>. It needs many weekly points before the change, and anything else that happened that week is folded into the estimate, so treat the result as weak evidence.</p>
<p>Period fixed effects are unavailable here, because they absorb the treatment completely. Instead, run a negative-control outcome, and place several false breakpoints inside the pre-period. If breaks of similar size appear often, the series is too noisy for a single-series claim.</p>

</details>

<h3 id="route-dose">Variation in how much people use AI</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>The question is about intensity of use, and you can observe usage repeatedly for the same people.</dd>
  <dt>Key assumption</dt><dd>Changes in a person's usage are unrelated to other changes in their work that also move the outcome.</dd>
  <dt>What would break it</dt><dd>Usage responding to workload, deadlines or projects (time-varying confounding), or the outcome driving usage (reverse causality).</dd>
  <dt>Where to run it</dt><dd>Within-person fixed effects (M2 in the <a href="{{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-causal-analysis">causal Skill</a>), <code>contdid</code> for continuous-treatment DiD (early release), or <code>DIDmultiplegtDYN</code> for non-binary treatment.</dd>
</dl>
<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Variation in how much people use AI</span></summary>

<p>A within-person model compares the same person in heavier and lighter weeks. It removes stable differences such as role or working style, but not changes that move usage and the outcome together. Busy weeks may raise both Copilot use and meeting load, for example. Continuous-treatment DiD compares groups by dose and needs stronger parallel-trends assumptions than the binary case <a class="cite" href="#ref-cgs2024">[20]</a>. Both packages are young, so confirm the estimator supports doses that change week to week. Timing checks are diagnostics only, and passing them does not establish a causal dose effect. Without a strategy for time-varying confounding, report associations.</p>
<p>Avoid adding a lagged outcome to a short fixed-effects panel, because it induces dynamic panel bias <a class="cite" href="#ref-nickell1981">[21]</a>. Actions, tasks, sessions and credits each get their own dose measure, and are never summed into one AI score.</p>

</details>

<h3 id="route-selection">Selection on observables</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>You can credibly adjust for confounders measured before exposure, and comparable people exist across exposure levels within function. The outcome may be observed only once.</dd>
  <dt>Key assumption</dt><dd>After conditioning on the covariates, exposure is unrelated to the outcome you would have seen without it (<a class="term" href="#term-unconfoundedness" data-def="Given the covariates, exposure is unrelated to the outcome a person would have had without it. Overlap means comparable people exist at each exposure level.">unconfoundedness</a>), and comparable people exist in both groups (overlap).</dd>
  <dt>What would break it</dt><dd>Unmeasured drivers of adoption such as motivation, manager support or project type, and heavy users who exist only in one function.</dd>
  <dt>Where to run it</dt><dd>Regression, matching, doubly robust estimation or double machine learning <a class="cite" href="#ref-dml2018">[22]</a> with <code>DoubleML</code> or <code>econml</code>. For Copilot actions, see the <a href="{{ site.baseurl }}/copilot-causal-toolkit/">Copilot Causal Toolkit</a>.</dd>
</dl>
<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: Selection on observables</span></summary>

<p>The covariates must be measured before exposure and must include function. Check overlap within function using the distribution of estimated scores in each group. If heavy users sit in one function only, the comparison extrapolates. When access was neither randomised nor set by a defensible cut-off, this is also the route for an outcome observed once, with heavy caveats, because no fixed effect or timing check is available. No universal E-value threshold turns a sensitivity number into a pass, so report the sensitivity analysis and let readers judge it.</p>

</details>

<h3 id="route-no-estimate">When no design fits</h3>
<dl class="route-facts">
  <dt>Use when</dt><dd>No route above has its assumptions met.</dd>
  <dt>Key assumption</dt><dd>None. You make no causal claim.</dd>
  <dt>What would break it</dt><dd>Wording that slips an effect into a descriptive result.</dd>
  <dt>Where to run it</dt><dd>The <a href="{{ site.baseurl }}/frontier-analytics-prompt-ai-adoption/">AI Adoption Across Tools prompt</a> for descriptive reporting.</dd>
</dl>
<details class="route-more">
<summary>Read the full route<span class="vi-sr-only">: When no design fits</span></summary>

<p>Say plainly that a causal estimate is unavailable, and report what you can describe. Then list what would change the answer, such as randomised waves, staggered waves inside a function, a longer pre-period, or an outcome that repeats weekly.</p>

</details>

<section id="joined-panel" class="part-head">
<h2>Build the joined panel</h2>
</section>
<p>Most routes start from the same exposure table, so build it once and carefully. The Person Query is the spine, at person-week grain, and other sources join onto it after they have been reduced to the same grain. For a team or business-unit outcome, aggregate exposure to that unit and period before joining, because copying a team outcome onto every person creates repeated copies of one observation.</p>
<p>Agent data arrives at person-agent-week, and Consumption data may be daily, so aggregate both to person-week first. The <a href="{{ site.baseurl }}/query-join-guide/">query join guide</a> lists the native keys, aligned week starts and scoped missing-row rules. After every join, confirm that the row count and the person-week keys match the spine, and stop if cardinality is ambiguous. Keep the raw header names, such as <code>Cowork tasks submitted</code> and <code>Total GitHub AI Credits used</code>, so every number traces back to its export.</p>

<h3 id="panel-status">Eligibility and treatment status, per tool</h3>
<p>Restrict each tool's analysis to the people who could have received it, then classify every eligible person for that tool. Absent activity alone does not establish that someone was never treated.</p>
<table>
  <thead>
    <tr><th>Status</th><th>Definition</th><th>Role in estimation</th></tr>
  </thead>
  <tbody>
    <tr><td>Treated in window</td><td>Exposure began on a dated week in the window</td><td>Treated cohort, defined by start week</td></tr>
    <tr><td>Already treated before the window</td><td>Access or adoption predates the export</td><td>Excluded from untreated comparison groups. They can support within-person intensity analyses</td></tr>
    <tr><td>Not yet treated</td><td>Eligible, exposure begins later in the window</td><td>Valid comparison only for weeks before their own start</td></tr>
    <tr><td>Never treated</td><td>Eligible, with licence or access records confirming no exposure through the window</td><td>Comparison group, with a note that later exposure is possible</td></tr>
    <tr><td>Ineligible or unresolved</td><td>Not in the tool's population, or status cannot be established</td><td>Excluded from that tool's estimate</td></tr>
  </tbody>
</table>
<p>State the exposure definition for each tool. A licence start date measures access, first observed use measures adoption, and an intensity measure captures dose. These answer different questions.</p>

<h3 id="panel-contamination">Contamination across tools</h3>
<p>Check every AI tool in the joined data, including those outside your study. Unlicensed Copilot Chat, Cowork and Agents can make a nominal comparison group partly treated. Decide whether the estimand is this tool versus no AI, or versus other AI use, and build groups to match. Tools adopted together cannot be separated, so report a joint effect. Untreated colleagues can also benefit from a treated colleague's drafts or meeting recaps, so if spillovers are likely, define direct and spillover effects separately, or move to a team-level design.</p>

<h3 id="panel-missing">Missing is not zero</h3>
<p>Unresolved coverage stays unknown, and an omitted source or a failed join says nothing about use. Within a complete query's configured scope, absent activity rows do contribute zero. For documented licensed-only native Microsoft 365 Copilot metrics, a null means unlicensed and a zero means licensed but inactive. Do not extend that rule to unlicensed Copilot Chat or to Consumption credits, or infer a licence from credits or Agent sessions. Publish only groups that meet the stricter of your organisation's minimum or 10 people.</p>

<section id="function-confounding" class="part-head">
<h2>Function as a confounder</h2>
</section>
<p>Rollout usually follows function. An organisation might license Engineering first, Sales later and Finance last, often for good business reasons. Function then determines when people get the tool and how their outcomes move, so a comparison that ignores it mixes the two. In this hypothetical, Engineering licensed in three waves supports within-function comparisons of earlier against later waves. Sales licensed in a single week has only between-function comparisons, against Engineering or Finance, and they depend on Sales trending like the comparison function. Finance, not yet licensed, can serve as a comparison but has no effect of its own to estimate.</p>
<p>Prefer within-function comparisons where suitable untreated observations remain. They still need conditional parallel trends, no <a class="term" href="#term-anticipation" data-def="People changing behaviour before access starts, for example after an announcement. It makes pre-period weeks partly treated.">anticipation</a> and a clear treatment definition. Between-function comparisons rest on stronger assumptions, because a Sales-wide reorganisation looks identical to a Sales-wide AI effect.</p>

<h3 id="function-selection">Selection inside a function</h3>
<p>Within a function, licences may still go first to particular roles, levels or high performers, or after a dip in performance, which creates a pre-programme dip <a class="cite" href="#ref-ashenfelter1978">[23]</a> and makes recovery look like an effect. Compare future adopters with later and never adopters before treatment (see the <a href="#check-pre-adoption">before-adoption comparison</a>). Small cells may need pooling, with a stated justification.</p>

<section id="checks" class="part-head">
<h2>Standard checks</h2>
</section>

<p>Run these five checks on the observational routes where they apply. Each one says where it does not. Randomised and regression discontinuity designs have their own diagnostics, such as balance checks between arms and density tests at the cut-off.</p>

<h3 id="check-fe-ladder">A fixed-effects ladder</h3>

<p>Report the estimate under increasingly demanding specifications, using the same sample, target population and aggregation weights each time. Start from a basic comparison with person and period adjustment, then add function-by-period fixed effects, and where useful also show the pooled estimate without person effects. A large movement when person effects are added suggests that adopters differed from non-adopters from the start. A large movement when function-by-period effects are added suggests that function-specific shocks coincided with rollout. Plain period effects only absorb shocks that hit the whole organisation at once, so function-by-period effects are the default.</p>

<p>Three limits apply.</p>

<ul>
<li>Under staggered adoption, build this structure inside a staggered-adoption estimator (see <a href="#route-staggered-did">the staggered DiD route</a>).</li>
<li>Person effects need repeated outcomes and some exposure variation in the sample. Never-treated comparison people, whose exposure never changes, still help estimate the period effects.</li>
<li>Period effects are unavailable without a comparison group, which rules them out on <a href="#route-itsa">the single-series ITSA route</a>.</li>
</ul>

<p>Fixed effects remove stable differences between people but do not detect them, and they do nothing about differences that change over time, such as a person who is becoming more engaged and also more likely to be licensed. The checks below address that. Useful packages are <code>fixest</code> in R, and <code>pyfixest</code> or <code>linearmodels</code> in Python.</p>

<h3 id="check-leads">Time-reversal placebos</h3>

<p>Test whether future exposure "predicts" earlier outcomes. If it does, the usual explanations are licences targeted at people who were already improving, reverse causality, or allocation after a dip in performance <a class="cite" href="#ref-ashenfelter1978">[23]</a>. For dated licence events, the leads of an event study are this check. Plot the lead coefficients with their intervals, and as a summary run a joint Wald (F) test of the leads. For usage intensity, include current exposure alongside the lead, because usage persists from week to week. Leads need dated exposure and several periods before it, so they are not available with a single series.</p>

<p>People may change behaviour before access arrives. Specify an anticipation window and keep those weeks out of the untreated comparison and the reference period, or move the treatment start to the announcement when that matches the question.</p>

<p>Judge lead estimates by their size and precision. Pre-trend tests often have low power, and a non-significant result does not establish parallel trends <a class="cite" href="#ref-roth2022">[24]</a>. Where you can, show how far the conclusions survive bounded violations of parallel trends, using <code>HonestDiD</code> in R <a class="cite" href="#ref-rr2023">[15]</a>. It takes the event-study coefficients and covariance from any estimator.</p>

<h3 id="check-pre-adoption">Compare adopters before adoption</h3>

<p>Compare future adopters with later adopters and never adopters, within function, on outcomes and characteristics measured before treatment, such as level, tenure, role and earlier outcome trends. If future adopters already differed on the outcome or its direction of travel, the estimate needs a stronger argument or a different design.</p>

<aside class="callout reveal" markdown="1">
<span class="callout-label">When there is no 'before'</span>
Early adopters have no observed pre-period when an export starts after they adopted, so this check and any DiD estimate describe later cohorts and newer tools such as Cowork, Agents or recent licence waves. Say so in the write-up.
</aside>

<h3 id="check-negative-control">A negative-control outcome</h3>

<p>Estimate the same model on an outcome the tool should not plausibly change <a class="cite" href="#ref-negctrl2010">[25]</a>. An apparent effect there points to residual confounding or a shared shock. A null result is reassuring but does not rule out confounding that affects only your main outcome. A useful negative control has no plausible route from the tool and shares the confounding or measurement processes that threaten the main outcome, such as the same seasonal pattern or reorganisation. Write both reasons down before the analysis. Meeting measures rarely qualify, because AI tools can change meeting attendance. If no candidate qualifies, say so in the write-up.</p>

<h3 id="check-leave-one-out">Leave one function out</h3>

<p>Re-estimate the model excluding each function in turn, and show the estimates side by side. If one function dominates, report that concentration and estimate function-specific effects where the data support them. A loss of significance after dropping a function does not show that only that function benefited. Where cohorts are few, also leave one cohort out, since a single licence wave can dominate a pooled estimate.</p>

<p>Not every check applies to every route. The single-series ITSA route has no comparison group, so it cannot use the fixed-effects ladder, the before-adoption comparison or leave-one-out. A negative control still helps there. For synthetic and controlled ITSA designs, the comparisons run against donors, and leave-one-out means leaving out one donor. Between-function designs have few functions, so leave-one-out and the before-adoption comparison are coarse, and function-by-period fixed effects are unavailable because they absorb a start that covers a whole function. For selection on observables, the ladder needs repeated outcomes and the placebo needs outcomes observed before exposure, while covariate balance and overlap can be checked without earlier outcomes.</p>

<section id="communicating" class="part-head">
<h2>Report what you can claim</h2>
</section>

<p>Report four things separately: the effect, its precision, the identification argument and the limitations. A sentence that blends them hides which part is weak.</p>

<p>Say whether the outcome is a proxy or a KPI, and how the proxy relates to the KPI, or that this could not be checked. Name the estimand: what is compared, for which cohorts and functions, and over which weeks.</p>

<p>Report the clustering level and the number of clusters in each model sample, at the level at which exposure was assigned <a class="cite" href="#ref-cluster2023">[12]</a>. With few clusters, standard cluster-robust inference can be unreliable <a class="cite" href="#ref-cm2015">[13]</a>, so say which correction you used, or that uncertainty cannot be reliably quantified.</p>

<p>Causal wording needs the identification argument and the checks above to hold up together. A pair of hypothetical sentences shows the difference.</p>

<ul>
<li>Weak: "Copilot reduced after-hours work in Sales."</li>
<li>Careful: "Among later licence cohorts in Sales, weeks after licensing are associated with fewer <code>After hours collaboration hours</code>. The estimate is consistent with a reduction, and the leads show no clear pre-trend, but the interval is wide and function-level shocks cannot be ruled out."</li>
</ul>

<p>If you test many outcomes, functions or cohorts, say how many, and adjust or flag the multiplicity. Apply privacy thresholds to every published cell, including complementary cells that could be subtracted, and do not publish rankings of individuals.</p>

<p>Finally, list what you did not do, such as unavailable checks, functions dropped for privacy reasons and outcomes you examined but did not report. Keep the headline claim no stronger than the weakest link from exposure to outcome. A licence supports a claim about access, and chosen use supports a claim about people who chose to use the tool.</p>

<section id="run-it" class="part-head">
<h2>Run it yourself</h2>
</section>

<p>These resources cover parts of the workflow, and none runs every route in this article.</p>

<ul>
<li><a href="{{ site.baseurl }}/frontier-analytics-skills/#skill-viva-insights-causal-analysis">The causal Skill</a> checks that a panel is ready and runs within-person diagnostics. It returns a proposed plan for DiD, event studies and instruments rather than running them.</li>
<li><a href="{{ site.baseurl }}/copilot-causal-toolkit/">The Copilot Causal Toolkit</a> runs a double machine learning workflow for Copilot, and suits the selection-on-observables route.</li>
<li>The event-study scripts in <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/event-study-did.Rmd">R</a> and <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/event-study-did.py">Python</a> fit a staggered-adoption event study on synthetic data, with <code>fixest::sunab()</code> in R and a saturated event study in <code>pyfixest</code>. They include a synthetic function variable and function-by-week fixed effects, matching the within-function comparison in the <a href="#route-staggered-did">staggered route</a>.</li>
<li>The difference-in-differences metric scans in <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-r/did-metric-scan.Rmd">R</a> and <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/examples/utility-python/did-metric-scan.py">Python</a> compare heavy and light users across many metrics. They report Holm-adjusted p-values and remain exploratory, because groups defined by chosen usage carry selection.</li>
<li><a href="https://github.com/microsoft/viva-insights-sample-code/tree/main/examples/utility-python/causal-inference/did-itsa">The DiD/ITSA notebook</a> implements 2x2 DiD with pooled OLS, an event-study plot of week-by-week group-mean differences, and controlled and single-series ITSA. It is not a staggered-robust estimator and does not use clustered errors.</li>
<li><a href="{{ site.baseurl }}/query-join-guide/">The query join guide</a> explains how to join other sources onto the Person Query.</li>
</ul>

<p>For routes the repository does not implement, these maintained packages are the starting point.</p>

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
