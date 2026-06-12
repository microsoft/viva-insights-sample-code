---
layout: home
title: "Viva Insights Sample Code Library"
description: "Sample code, tutorials, and AI prompt libraries for Microsoft Viva Insights analytics in R and Python — from Copilot adoption and organizational network analysis to causal inference and AI-agent analytics."
---

{% include custom-navigation.html %}
{% include homepage-tab-rail.html %}

<header class="vi-home-hero">
  <div class="vi-home-hero-ribbon" aria-hidden="true">
    <svg viewBox="0 0 1440 520" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="vi-ribbon-1" cx="18%" cy="22%" r="55%">
          <stop offset="0" stop-color="#B4009E" stop-opacity="0.18"/>
          <stop offset="1" stop-color="#B4009E" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="vi-ribbon-2" cx="80%" cy="32%" r="60%">
          <stop offset="0" stop-color="#335CCC" stop-opacity="0.18"/>
          <stop offset="1" stop-color="#335CCC" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="vi-ribbon-3" cx="50%" cy="92%" r="55%">
          <stop offset="0" stop-color="#5C2D91" stop-opacity="0.14"/>
          <stop offset="1" stop-color="#5C2D91" stop-opacity="0"/>
        </radialGradient>
        <linearGradient id="vi-ribbon-line" x1="0" y1="0" x2="1" y2="0.4">
          <stop offset="0" stop-color="#764FF5" stop-opacity="0.18"/>
          <stop offset="0.5" stop-color="#3F6CE9" stop-opacity="0.18"/>
          <stop offset="1" stop-color="#20BBC6" stop-opacity="0.18"/>
        </linearGradient>
      </defs>
      <rect width="1440" height="520" fill="url(#vi-ribbon-1)"/>
      <rect width="1440" height="520" fill="url(#vi-ribbon-2)"/>
      <rect width="1440" height="520" fill="url(#vi-ribbon-3)"/>
      <path d="M-40 380 C 300 280, 600 460, 900 360 S 1380 240, 1480 320 L 1480 540 L -40 540 Z" fill="url(#vi-ribbon-line)"/>
    </svg>
  </div>

  <div class="vi-home-hero-inner">
    <h1>Sample code for Microsoft Viva Insights analytics</h1>
    <p class="vi-hero-lede">Templates, prompts, and tutorials in R and Python — from Copilot adoption and ONA to causal inference and AI-agent analytics.</p>
    <div class="vi-cta-row-v2">
      <a class="vi-cta-primary" href="{{ site.baseurl }}/getting-started/">Start here</a>
      <a class="vi-cta-on-light" href="#overview">Browse the library</a>
    </div>
  </div>

  <div class="vi-values">
    <div class="vi-values-panel">
      <div class="vi-value">
        <svg class="vi-value-icon" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <linearGradient id="vi-val-1" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#335CCC"/>
              <stop offset="1" stop-color="#764FF5"/>
            </linearGradient>
          </defs>
          <rect x="6" y="10" width="52" height="44" rx="6" fill="none" stroke="url(#vi-val-1)" stroke-width="2.5"/>
          <path d="M22 26 L14 32 L22 38" fill="none" stroke="url(#vi-val-1)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M42 26 L50 32 L42 38" fill="none" stroke="url(#vi-val-1)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M36 22 L28 42" stroke="url(#vi-val-1)" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
        <h3>Real code, ready to run</h3>
        <p>Working R and Python scripts built on the official Viva Insights packages.</p>
      </div>
      <div class="vi-value">
        <svg class="vi-value-icon" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <linearGradient id="vi-val-2" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#5C2D91"/>
              <stop offset="1" stop-color="#B4009E"/>
            </linearGradient>
          </defs>
          <path d="M10 14 H46 a6 6 0 0 1 6 6 V36 a6 6 0 0 1 -6 6 H26 L16 52 V42 H10 a4 4 0 0 1 -4 -4 V20 a6 6 0 0 1 4 -6 Z" fill="none" stroke="url(#vi-val-2)" stroke-width="2.5" stroke-linejoin="round"/>
          <path d="M52 12 L54 18 L60 20 L54 22 L52 28 L50 22 L44 20 L50 18 Z" fill="url(#vi-val-2)"/>
        </svg>
        <h3>Prompts that turn exports into decks</h3>
        <p>Paste a Frontier prompt into a coding agent and get a dashboard, exec summary, or ROI analysis.</p>
      </div>
      <div class="vi-value">
        <svg class="vi-value-icon" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <linearGradient id="vi-val-3" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#20BBC6"/>
              <stop offset="1" stop-color="#3F6CE9"/>
            </linearGradient>
          </defs>
          <path d="M10 12 H30 a4 4 0 0 1 4 4 V52 a4 4 0 0 0 -4 -4 H10 Z" fill="none" stroke="url(#vi-val-3)" stroke-width="2.5" stroke-linejoin="round"/>
          <path d="M54 12 H34 a4 4 0 0 0 -4 4 V52 a4 4 0 0 1 4 -4 H54 Z" fill="none" stroke="url(#vi-val-3)" stroke-width="2.5" stroke-linejoin="round"/>
          <path d="M14 24 L20 30 L26 22" fill="none" stroke="url(#vi-val-3)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M38 24 H50 M38 30 H50 M38 36 H46" stroke="url(#vi-val-3)" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <h3>Research-backed playbooks</h3>
        <p>Methodologies grounded in behavioural research — super users, causal inference, ONA.</p>
      </div>
    </div>
  </div>
</header>

<section id="overview" class="vi-section vi-section-lav">
  <div class="vi-section-inner">
    <span class="vi-eyebrow">Library overview</span>
    <h2 class="vi-section-heading">Browse the library by topic</h2>
    <p class="vi-section-desc">Pick a category to find tutorials, utility scripts, and analytical playbooks for that area of Viva Insights analysis. Most scripts are written in R or Python, with dedicated <a href="https://microsoft.github.io/vivainsights/">vivainsights</a> packages that handle the heavy data-processing work.</p>

    <div class="vi-card-grid">
      <a class="vi-card" href="{{ site.baseurl }}/getting-started/">
        <span class="vi-card-icon">🧭</span>
        <span class="vi-card-title">Getting Started</span>
        <span class="vi-card-desc">New here? Set up your environment and run your first Viva Insights analysis in R or Python.</span>
        <span class="vi-card-more">Begin →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/essentials/">
        <span class="vi-card-icon">🔧</span>
        <span class="vi-card-title">Essentials</span>
        <span class="vi-card-desc">Core utility functions and visualizations for everyday Viva Insights analysis.</span>
        <span class="vi-card-more">Explore →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/advanced/">
        <span class="vi-card-icon">📊</span>
        <span class="vi-card-title">Advanced Analytics</span>
        <span class="vi-card-desc">Regression models, machine learning, and statistical analysis techniques.</span>
        <span class="vi-card-more">Explore →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/network/">
        <span class="vi-card-icon">🔗</span>
        <span class="vi-card-title">Network Analysis</span>
        <span class="vi-card-desc">Organizational network analysis (ONA) — visualize and quantify collaboration patterns.</span>
        <span class="vi-card-more">Explore →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/copilot/">
        <span class="vi-card-icon">🤖</span>
        <span class="vi-card-title">Copilot Analytics</span>
        <span class="vi-card-desc">Measure Microsoft Copilot adoption, identify power users, and quantify productivity impact.</span>
        <span class="vi-card-more">Explore →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/frontier-analytics/">
        <span class="vi-card-icon">🚀</span>
        <span class="vi-card-title">Frontier</span>
        <span class="vi-card-desc">Turn a Viva Insights export into a dashboard, exec deck, or ROI analysis with a coding agent — plus prompt libraries and schema docs.</span>
        <span class="vi-card-more">Explore →</span>
      </a>
      <a class="vi-card" href="{{ site.baseurl }}/articles/">
        <span class="vi-card-icon">📰</span>
        <span class="vi-card-title">Articles</span>
        <span class="vi-card-desc">Long-form editorials and research briefs on meeting culture, Copilot adoption, and AI-enabled collaboration.</span>
        <span class="vi-card-more">Read →</span>
      </a>
    </div>

    <h3>Is this library for me?</h3>
    <p><strong>New to Viva Insights?</strong> Start with the <a href="https://learn.microsoft.com/en-us/viva/insights/tutorials/power-bi-intro">Viva Insights Power BI templates</a> and official documentation — they provide pre-built dashboards that deliver value without any coding.</p>
    <p><strong>Ready for advanced analysis?</strong> This library is for analysts, data scientists, and researchers who want to unlock deeper insights through custom analysis — predictive models, custom dashboards, hypothesis testing, ONA, Copilot impact measurement, or scaled automation.</p>
  </div>
</section>

<section id="sample-code" class="vi-section">
  <div class="vi-section-inner">
    <span class="vi-eyebrow">Sample code</span>
    <h2 class="vi-section-heading">Special focus areas</h2>
    <p class="vi-section-desc">Two areas where this library goes deeper than the standard packages: measuring Copilot adoption and impact, and using AI coding agents to turn raw Viva Insights exports into finished outputs.</p>

    <h3>Copilot analytics</h3>
    <p>With the rapid adoption of Microsoft Copilot, understanding usage patterns and measuring impact has become critical. Our dedicated <a href="{{ site.baseurl }}/copilot/">Copilot Analytics</a> section provides specialized scripts and methodologies for:</p>
    <ul>
      <li>Measuring Copilot adoption rates and user segmentation</li>
      <li>Identifying power users and building habit-based usage models</li>
      <li>Analyzing productivity impact and ROI of Copilot investments</li>
      <li>Creating executive dashboards for tracking deployment success</li>
    </ul>

    <h3>Frontier — AI-agent analytics</h3>
    <p>As organizations move from measuring Copilot adoption to measuring the impact of AI <strong>agents</strong>, a new class of analysis is required. Our <a href="{{ site.baseurl }}/frontier-analytics/">Frontier</a> section turns a Viva Insights export into a finished dashboard, executive deck, or ROI analysis by pasting a ready-made prompt into a coding agent. It provides analyst prompts, schema guides, and worked examples for:</p>
    <ul>
      <li>Profiling agent and Copilot usage across the organization</li>
      <li>Estimating the ROI and time savings of AI investments</li>
      <li>Building executive summaries, dashboards, and PowerPoint-ready outputs</li>
      <li>Validating findings against the underlying Copilot/agent data taxonomy</li>
    </ul>

    <h3>Quick links</h3>
    <ul>
      <li><a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/vivainsights-context.md">vivainsights-context.md</a> — reusable context file for AI coding agents</li>
      <li><a href="https://learn.microsoft.com/en-us/viva/insights/introduction">Viva Insights official documentation</a></li>
      <li><a href="https://microsoft.github.io/vivainsights-py/">vivainsights Python package</a></li>
      <li><a href="https://microsoft.github.io/vivainsights/">vivainsights R package</a></li>
      <li><a href="https://aka.ms/DecodingSuperUsage">Super Users Report for Copilot Analytics</a></li>
      <li><a href="https://github.com/microsoft/AI-in-One-Dashboard">Copilot Analytics All-in-one Dashboard</a></li>
      <li><a href="https://github.com/microsoft/superuserimpact">Super Users Impact Report</a></li>
    </ul>
  </div>
</section>

<section id="research" class="vi-section vi-section-warm">
  <div class="vi-section-inner">
    <span class="vi-eyebrow vi-eyebrow-warm">Research &amp; articles</span>
    <h2 class="vi-section-heading">Long-form reading and methodology</h2>
    <p class="vi-section-desc">Editorials, research briefs, and how-to guides on meeting culture, Copilot adoption, super-user research, and AI-enabled collaboration practices.</p>

    <p>Browse the full <a href="{{ site.baseurl }}/articles/">Articles index</a> for editorials and research briefs, or read on for how to use the rest of the library.</p>

    <h3>Built for data scientists</h3>
    <p>Most scripts here are written in <strong>R</strong> and <strong>Python</strong> — the most popular languages for automation, experimentation, and advanced statistical analysis. To accelerate your work, we've developed dedicated packages that handle the complexities of Viva Insights data processing:</p>
    <ul>
      <li><a href="https://microsoft.github.io/vivainsights/">vivainsights R package</a> — comprehensive toolkit for R users with 100+ functions for data manipulation and visualization</li>
      <li><a href="https://microsoft.github.io/vivainsights-py/">vivainsights Python package</a> — full-featured Python library optimized for data science workflows</li>
    </ul>
    <p>These packages let analysts hit the ground running without writing data-processing code from scratch.</p>

    <h3>How to use this library</h3>
    <p>Each script includes:</p>
    <ul>
      <li><strong>Purpose</strong> — what the script accomplishes</li>
      <li><strong>Prerequisites</strong> — required packages and data formats</li>
      <li><strong>Usage</strong> — how to run and customize the code</li>
      <li><strong>Download</strong> — direct link to the raw script file</li>
    </ul>

    <h3>Contributing &amp; license</h3>
    <p>This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA). For details, visit <a href="https://cla.opensource.microsoft.com">Microsoft CLA</a>. The project is licensed under the MIT License — see the <a href="https://github.com/microsoft/viva-insights-sample-code/blob/main/LICENSE">LICENSE</a> file for details.</p>
  </div>
</section>
