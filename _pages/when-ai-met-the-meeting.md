---
layout: article
title: "When AI Met the Meeting"
description: "A Copilot Analytics Lab long-read on how AI is changing meeting culture, which meeting benefits to protect, and which practices reduce the bad while preserving the good."
permalink: /articles/when-ai-met-the-meeting/
eyebrow: "Copilot Analytics Lab · PANDAS Team · May 2026"
dek: "Meetings are not broken, but meeting culture is uneven. Two years into enterprise AI rollout, data shows AI amplifies both the best and the worst of how teams meet."
byline: "By the PANDAS team · A Copilot Analytics Lab brief"
read_time: "22 min read"
css: "/assets/css/article.css"
---

<nav class="article-contents reveal" aria-label="Article contents">
<p>In this article</p>
<ol>
  <li><a href="#part-1">Part 1 · Meetings are valuable — unevenly</a></li>
  <li><a href="#part-2">Part 2 · The multitasking taxonomy</a></li>
  <li><a href="#part-3">Part 3 · What AI is doing to meetings</a></li>
  <li><a href="#part-4">Part 4 · Meeting benefits worth protecting</a></li>
  <li><a href="#part-5">Part 5 · Four practices to embrace</a></li>
  <li><a href="#part-6">Part 6 · Leadership take-aways</a></li>
  <li><a href="#measurement">Measurement · six signals worth tracking</a></li>
  <li><a href="#references">References</a></li>
</ol>
</nav>

<p class="lead">A typical knowledge worker now spends a large share of the week in meetings, increasingly accompanied by AI that summarizes, recaps, and follows meetings on their behalf. The original promise was straightforward: fewer meetings, less drudgery, and more deep work. The evidence now suggests a more complex reality: meeting load can still rise, and attention can fragment faster than culture evolves.</p>

<p>This brief synthesizes meeting science, human-AI collaboration research, and enterprise diagnostics. The core finding is not “fewer meetings.” It is <strong>better meeting design plus better AI-enabled team habits</strong>. Teams that pair AI with strong norms convert individual productivity gains into collective outcomes; teams that do not simply accelerate existing calendar debt.</p>

### The TL;DR

<ul class="tldr">
  <li><strong>Meetings are essential infrastructure</strong> for decisions, alignment, mentoring, and network formation. The issue is uneven quality, not meetings as a category <a href="#ref-1">[1]</a> <a href="#ref-16">[16]</a>.</li>
  <li><strong>AI is increasing both meeting intensity and in-meeting multitasking</strong> in many contexts; this is not inherently negative, but it is design-sensitive <a href="#ref-2">[2]</a> <a href="#ref-3">[3]</a> <a href="#ref-10">[10]</a>.</li>
  <li><strong>Productive multitasking and distracted multitasking are different phenomena</strong>; dashboards should diagnose structure, not moralize behavior.</li>
  <li><strong>Individual AI adoption is ahead of team AI adoption</strong>; recap/follow/meeting hygiene are where team-level gains are unlocked <a href="#ref-4">[4]</a> <a href="#ref-20">[20]</a>.</li>
  <li><strong>The practical agenda is to shrink bad meetings while protecting good ones</strong> — especially 1:1s, small decision forums, and cross-boundary collaboration contexts.</li>
</ul>

<aside class="callout is-bottomline reveal" markdown="1">
<span class="callout-label">Bottom line</span>
AI does not fix meeting culture. It amplifies the culture already present. Organizations that combine AI with explicit meeting norms preserve the social and leadership value of meetings while reducing avoidable load.
</aside>

<section id="part-1" class="part-head">
<span class="part-kicker">Part 1</span>
<h2>Meetings are valuable — unevenly</h2>
</section>

<p>Meeting research is unambiguous: well-designed meetings are fundamental to organizational performance, but poorly designed recurring meetings generate disproportionate cost <a href="#ref-1">[1]</a> <a href="#ref-16">[16]</a>. The “Lake Wobegon effect” in meetings persists — organizers tend to rate their meetings above average while attendees do not.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 360" role="img" aria-labelledby="fig1-title fig1-desc">
  <title id="fig1-title">Meeting quality spectrum</title>
  <desc id="fig1-desc">A spectrum from high-value small meetings to high-drift recurring meetings.</desc>
  <rect width="980" height="360" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">The Meeting Quality Spectrum</text>
  <g font-family="Segoe UI, Arial" font-size="16" fill="#1f2a33">
    <rect x="48" y="88" width="400" height="34" rx="8" fill="#2E7D4F"/><text x="60" y="111" fill="#fff">1:1s and small decision meetings</text>
    <rect x="48" y="134" width="505" height="34" rx="8" fill="#4C8C65"/><text x="60" y="157" fill="#fff">Working sessions (3-8)</text>
    <rect x="48" y="180" width="610" height="34" rx="8" fill="#8CA65F"/><text x="60" y="203" fill="#fff">Cross-team syncs (5-12)</text>
    <rect x="48" y="226" width="720" height="34" rx="8" fill="#C79A3A"/><text x="60" y="249" fill="#fff">All-hands and broadcasts (20+)</text>
    <rect x="48" y="272" width="830" height="34" rx="8" fill="#C0392B"/><text x="60" y="295" fill="#fff">Standing recurring meetings</text>
  </g>
  <text x="48" y="334" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Lower drift risk → Higher drift risk when purpose, cadence, and attendee list are not actively managed.</text>
</svg>
<figcaption><strong>Figure 1 —</strong> Meeting quality is mostly about design fit: right size, right duration, right cadence.</figcaption>
</figure>

<table class="is-wide reveal">
  <thead>
    <tr><th>Meeting type</th><th>Primary value</th><th>Primary risk when unmanaged</th></tr>
  </thead>
  <tbody>
    <tr><td>1:1s and small decision meetings</td><td>Mentorship, decisions, trust</td><td>Under-scheduling can reduce alignment</td></tr>
    <tr><td>Working sessions (3-8)</td><td>Co-creation and problem-solving</td><td>Drift into status updates</td></tr>
    <tr><td>Cross-team syncs</td><td>Boundary coordination</td><td>Persist after objective is gone</td></tr>
    <tr><td>Large broadcasts</td><td>Strategic context and visibility</td><td>Low participation density, passive attendance</td></tr>
    <tr><td>Recurring meetings</td><td>Cadence and accountability</td><td>Highest structural drift risk</td></tr>
  </tbody>
</table>

<blockquote class="pull-quote reveal">The science does not say “fewer meetings.” It says right meeting, right size, right cadence.</blockquote>

<section id="part-2" class="part-head">
<span class="part-kicker">Part 2</span>
<h2>The multitasking taxonomy</h2>
</section>

<p>Multitasking in meetings is not a single behavior with a single interpretation. Classic work identifies <strong>productive interleaving</strong> (notes, lookups, action capture) and <strong>distracted disengagement</strong>, both of which rise under different structural conditions <a href="#ref-2">[2]</a> <a href="#ref-3">[3]</a> <a href="#ref-21">[21]</a>.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 420" role="img" aria-labelledby="fig2-title fig2-desc">
  <title id="fig2-title">Productive versus distracted multitasking</title>
  <desc id="fig2-desc">Two-column taxonomy of productive and distracted multitasking patterns.</desc>
  <rect width="980" height="420" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">The Multitasking Taxonomy</text>
  <rect x="48" y="84" width="420" height="292" rx="12" fill="#E9F4EC" stroke="#BCDCC6"/>
  <rect x="512" y="84" width="420" height="292" rx="12" fill="#FCEEEE" stroke="#E7C1BB"/>
  <text x="72" y="122" font-size="20" fill="#2E7D4F" font-family="Segoe UI, Arial" font-weight="700">Productive multitasking</text>
  <text x="536" y="122" font-size="20" fill="#C0392B" font-family="Segoe UI, Arial" font-weight="700">Distracted multitasking</text>
  <g font-size="16" fill="#1f2a33" font-family="Segoe UI, Arial">
    <text x="72" y="164">• Capturing action items in real time</text>
    <text x="72" y="196">• Looking up data needed for the decision</text>
    <text x="72" y="228">• Drafting follow-ups while context is fresh</text>
    <text x="72" y="260">• Recap-assisted memory reinforcement</text>
    <text x="72" y="292">• Follow-not-attend catch-up in focus blocks</text>
    <text x="536" y="164">• Split-screen disengagement in low-value meetings</text>
    <text x="536" y="196">• Chronic dual-attendance in conflicts</text>
    <text x="536" y="228">• Large, long, recurring sessions with no agenda</text>
    <text x="536" y="260">• Reactive meetings collapsing focus windows</text>
    <text x="536" y="292">• No action-owner accountability after meetings</text>
  </g>
  <text x="48" y="398" font-size="13" fill="#5B6573" font-family="Segoe UI, Arial">Interpretation should be structural: high multitasking rates across a series often indicate meeting design debt, not individual failure.</text>
</svg>
<figcaption><strong>Figure 2 —</strong> Productive and distracted multitasking are analytically distinct and should be diagnosed differently.</figcaption>
</figure>

<aside class="callout reveal" markdown="1">
<span class="callout-label">Supporting evidence</span>
In Org B, meeting duration was the strongest predictor of distraction signal, and meetings 60+ minutes accounted for a majority of meeting hours despite minority share by count.
</aside>

<blockquote class="pull-quote reveal">AI does not fix meeting culture. It amplifies the culture already present.</blockquote>

<section id="part-3" class="part-head">
<span class="part-kicker">Part 3</span>
<h2>What AI is actually doing to meetings</h2>
</section>

<p>Recent research converges on a common pattern: AI speeds task execution but often increases total work throughput and parallelism <a href="#ref-13">[13]</a> <a href="#ref-14">[14]</a> <a href="#ref-17">[17]</a>. In calendars, this frequently appears as more meetings, more overlap, and more recap-mediated catch-up behavior.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 430" role="img" aria-labelledby="fig3-title fig3-desc">
  <title id="fig3-title">Signals of AI impact on meetings</title>
  <desc id="fig3-desc">Five headline statistics presented as cards.</desc>
  <rect width="980" height="430" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Signals in the field</text>
  <g font-family="Segoe UI, Arial">
    <rect x="48" y="88" width="166" height="128" rx="12" fill="#11365A"/>
    <text x="64" y="130" fill="#fff" font-size="30" font-weight="700">+252%</text>
    <text x="64" y="160" fill="#D7E4F1" font-size="13">
      <tspan x="64" dy="0">Teams meeting time</tspan>
      <tspan x="64" dy="16">vs. Feb 2020</tspan>
    </text>

    <rect x="228" y="88" width="166" height="128" rx="12" fill="#2E5C8A"/>
    <text x="244" y="130" fill="#fff" font-size="30" font-weight="700">62/mo</text>
    <text x="244" y="160" fill="#D7E4F1" font-size="13">
      <tspan x="244" dy="0">Average meetings</tspan>
      <tspan x="244" dy="16">per worker</tspan>
    </text>

    <rect x="408" y="88" width="166" height="128" rx="12" fill="#4C8C65"/>
    <text x="424" y="130" fill="#fff" font-size="30" font-weight="700">40-60</text>
    <text x="424" y="160" fill="#D7E4F1" font-size="13">
      <tspan x="424" dy="0">minutes/day</tspan>
      <tspan x="424" dy="16">individual AI savings</tspan>
    </text>

    <rect x="588" y="88" width="166" height="128" rx="12" fill="#C79A3A"/>
    <text x="604" y="130" fill="#fff" font-size="30" font-weight="700">8h/mo</text>
    <text x="604" y="160" fill="#FFF3DA" font-size="13">
      <tspan x="604" dy="0">Meeting content</tspan>
      <tspan x="604" dy="16">summarized async</tspan>
    </text>

    <rect x="768" y="88" width="166" height="128" rx="12" fill="#2E7D4F"/>
    <text x="784" y="130" fill="#fff" font-size="30" font-weight="700">37%</text>
    <text x="784" y="160" fill="#DDF2E4" font-size="13">
      <tspan x="784" dy="0">Sustained users report</tspan>
      <tspan x="784" dy="16">attending fewer meetings</tspan>
    </text>
  </g>
  <text x="48" y="266" font-family="Segoe UI, Arial" font-size="17" fill="#1f2a33">
    <tspan x="48" dy="0">Interpretation: growth and reduction signals can both be true at once.</tspan>
    <tspan x="48" dy="24">They usually describe different sub-populations and different norm maturity levels.</tspan>
  </text>
  <text x="48" y="332" font-family="Segoe UI, Arial" font-size="17" fill="#1f2a33">
    <tspan x="48" dy="0">AI deployment alone does not reduce meetings.</tspan>
    <tspan x="48" dy="24">Team norms determine whether saved time becomes better collaboration or more calendar load.</tspan>
  </text>
  <text x="48" y="398" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Sources: Microsoft WorkLab/WTI, Atlassian State of Teams, New Future of Work synthesis.</text>
</svg>
<figcaption><strong>Figure 3 —</strong> Individual gains are real; collective gains depend on team operating norms.</figcaption>
</figure>

<section id="part-4" class="part-head">
<span class="part-kicker">Part 4</span>
<h2>The meeting benefits worth protecting</h2>
</section>

<p>Calls to “reduce meetings” often cut the wrong layer. Three research streams show why: weak ties support mobility <a href="#ref-6">[6]</a> <a href="#ref-15">[15]</a>, remote work can harden network silos <a href="#ref-7">[7]</a> <a href="#ref-22">[22]</a>, and informal communication supports leadership and satisfaction <a href="#ref-8">[8]</a> <a href="#ref-18">[18]</a>. Good meeting policy protects those mechanisms while reducing structural waste.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 460" role="img" aria-labelledby="fig4-title fig4-desc">
  <title id="fig4-title">Three benefits worth protecting</title>
  <desc id="fig4-desc">Triad diagram showing weak ties, network bridging, and informal communication.</desc>
  <rect width="980" height="460" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">What meetings protect when designed well</text>
  <rect x="48" y="92" width="280" height="280" rx="14" fill="#EAF1F8" stroke="#B9C8D8"/>
  <rect x="350" y="92" width="280" height="280" rx="14" fill="#E9F4EC" stroke="#BCDCC6"/>
  <rect x="652" y="92" width="280" height="280" rx="14" fill="#FFF5E5" stroke="#EFD9AD"/>

  <text x="72" y="138" font-size="23" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Weak ties</text>
  <text x="72" y="168" font-size="15" fill="#1f2a33" font-family="Segoe UI, Arial">
    <tspan x="72" dy="0">Career mobility and opportunity</tspan>
    <tspan x="72" dy="22">depend on cross-network contact.</tspan>
  </text>

  <text x="374" y="138" font-size="23" fill="#2E7D4F" font-family="Segoe UI, Arial" font-weight="700">Network bridging</text>
  <text x="374" y="168" font-size="15" fill="#1f2a33" font-family="Segoe UI, Arial">
    <tspan x="374" dy="0">Cross-boundary collaboration</tspan>
    <tspan x="374" dy="22">is easier in well-run sync contexts.</tspan>
  </text>

  <text x="676" y="138" font-size="23" fill="#C79A3A" font-family="Segoe UI, Arial" font-weight="700">Informal exchange</text>
  <text x="676" y="168" font-size="15" fill="#1f2a33" font-family="Segoe UI, Arial">
    <tspan x="676" dy="0">Leadership trust and social glue</tspan>
    <tspan x="676" dy="22">often emerge around meetings.</tspan>
  </text>

  <rect x="48" y="392" width="884" height="42" rx="9" fill="#F2F7FC" stroke="#B9C8D8"/>
  <text x="68" y="418" font-size="16" fill="#11365A" font-family="Segoe UI, Arial">
    <tspan font-weight="700">Protect these mechanisms</tspan>
    <tspan fill="#1f2a33"> while shrinking long, large, low-contribution recurring meetings.</tspan>
  </text>
</svg>
<figcaption><strong>Figure 4 —</strong> Meeting reduction without design precision can unintentionally weaken networks, leadership visibility, and career mobility pathways.</figcaption>
</figure>

<aside class="callout reveal" markdown="1">
<span class="callout-label">Practical implication</span>
Aim reduction at long, large, recurring sessions with low contribution density — not at 1:1s, mentoring, and cross-team contexts that create social and decision value.
</aside>

<blockquote class="pull-quote reveal">The meetings worth keeping build people. The meetings worth shrinking drain attention.</blockquote>

<section id="part-5" class="part-head">
<span class="part-kicker">Part 5</span>
<h2>Four practices to embrace</h2>
</section>

<p>Individual AI adoption improves personal throughput. Team AI adoption improves collaborative outcomes. The following four practices are evidence-aligned and operationally simple.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 430" role="img" aria-labelledby="fig5-title fig5-desc">
  <title id="fig5-title">Four practices stack</title>
  <desc id="fig5-desc">Four stacked cards representing recap, follow-not-attend, focus catch-up, and meeting hygiene.</desc>
  <rect width="980" height="430" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Four practices for AI-era meeting culture</text>
  <g font-family="Segoe UI, Arial">
    <rect x="48" y="88" width="884" height="68" rx="12" fill="#11365A"/><text x="72" y="130" fill="#fff" font-size="20" font-weight="700">1. Use recap routinely across all meeting types</text>
    <rect x="48" y="170" width="884" height="68" rx="12" fill="#2E5C8A"/><text x="72" y="212" fill="#fff" font-size="20" font-weight="700">2. Accept one meeting, follow the conflicting one</text>
    <rect x="48" y="252" width="884" height="68" rx="12" fill="#4C8C65"/><text x="72" y="294" fill="#fff" font-size="20" font-weight="700">3. Use focus blocks for recap catch-up and follow-through</text>
    <rect x="48" y="334" width="884" height="68" rx="12" fill="#C79A3A"/>
    <text x="72" y="368" fill="#fff" font-size="20" font-weight="700">
      <tspan x="72" dy="0">4. Meeting hygiene: right-size, right-length, recurring audit,</tspan>
      <tspan x="72" dy="24">notice discipline</tspan>
    </text>
  </g>
</svg>
<figcaption><strong>Figure 5 —</strong> These practices are designed to reduce bad meetings while preserving high-value collaboration contexts.</figcaption>
</figure>

### Practice 1 · Use recap routinely

<p>Recap is the lowest-friction gain. It supports note quality, decision memory, and asynchronous catch-up. It is valuable not only for large meetings but also for 1:1s where presence and eye contact matter.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 520" role="img" aria-labelledby="fig6-title fig6-desc">
  <title id="fig6-title">Copilot recap adoption by anonymized function</title>
  <desc id="fig6-desc">Bar chart with anonymized functions A to E and adoption percentages.</desc>
  <rect width="980" height="520" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Copilot Recap Adoption by Function — Org B (anonymized)</text>
  <text x="48" y="82" font-size="14" fill="#5B6573" font-family="Segoe UI, Arial">Regular recap usage (% of users)</text>
  <line x1="150" y1="430" x2="900" y2="430" stroke="#B9C8D8"/>
  <line x1="150" y1="150" x2="150" y2="430" stroke="#B9C8D8"/>
  <g font-family="Segoe UI, Arial" font-size="14" fill="#5B6573">
    <text x="120" y="434">0</text><text x="112" y="370">2</text><text x="112" y="306">4</text><text x="112" y="242">6</text><text x="112" y="178">8</text>
  </g>
  <g font-family="Segoe UI, Arial" font-size="15" fill="#1f2a33">
    <rect x="190" y="190" width="110" height="240" rx="8" fill="#11365A"/><text x="220" y="452">Function A</text><text x="230" y="180" fill="#11365A" font-weight="700">7.0%</text>
    <rect x="330" y="270" width="110" height="160" rx="8" fill="#2E5C8A"/><text x="360" y="452">Function B</text><text x="370" y="260" fill="#2E5C8A" font-weight="700">4.0%</text>
    <rect x="470" y="310" width="110" height="120" rx="8" fill="#4C8C65"/><text x="500" y="452">Function C</text><text x="510" y="300" fill="#4C8C65" font-weight="700">3.0%</text>
    <rect x="610" y="350" width="110" height="80" rx="8" fill="#C79A3A"/><text x="640" y="452">Function D</text><text x="650" y="340" fill="#C79A3A" font-weight="700">2.0%</text>
    <rect x="750" y="350" width="110" height="80" rx="8" fill="#C0392B"/><text x="780" y="452">Function E</text><text x="790" y="340" fill="#C0392B" font-weight="700">2.0%</text>
  </g>
  <rect x="48" y="474" width="884" height="30" rx="8" fill="#EAF1F8" stroke="#B9C8D8"/>
  <text x="62" y="494" font-family="Segoe UI, Arial" font-size="14" fill="#11365A"><tspan font-weight="700">Overall Org B baseline: 5.3%</tspan><tspan fill="#1f2a33"> regular usage. Technology is deployed; behavior is uneven.</tspan></text>
</svg>
<figcaption><strong>Figure 6 —</strong> Function labels are intentionally anonymized. Core signal retained: low baseline adoption and material variance across functions.</figcaption>
</figure>

<aside class="callout reveal" markdown="1">
<span class="callout-label">Supporting evidence (anonymized)</span>
In Org B, only **5.3% of users** regularly used recap. Adoption variance by function indicates norm maturity differences, not tool-availability differences.
</aside>

### Practice 2 · Follow the conflict

<p>When meetings conflict, “accept one and follow one” outperforms dual-attendance. This is especially effective in large meetings with low individual contribution density.</p>

### Practice 3 · Use focus time for catch-up

<p>Batch recap processing in protected focus blocks. Fragmented recap checks throughout the day increase switch costs and residual cognitive load <a href="#ref-21">[21]</a>.</p>

### Practice 4 · Meeting hygiene

<p>Four hygiene levers consistently outperform generic “reduce meetings” mandates: right-size attendee lists, default to shorter durations, audit recurring series quarterly, and enforce notice discipline.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 500" role="img" aria-labelledby="fig7-title fig7-desc">
  <title id="fig7-title">Before during after scaffold</title>
  <desc id="fig7-desc">Three-column checklist for before, during, and after meeting behaviors.</desc>
  <rect width="980" height="500" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Operational scaffold: Before · During · After</text>
  <g font-family="Segoe UI, Arial">
    <rect x="48" y="88" width="280" height="372" rx="12" fill="#EAF1F8" stroke="#B9C8D8"/>
    <rect x="350" y="88" width="280" height="372" rx="12" fill="#E9F4EC" stroke="#BCDCC6"/>
    <rect x="652" y="88" width="280" height="372" rx="12" fill="#FFF5E5" stroke="#EFD9AD"/>
    <text x="72" y="124" font-size="22" fill="#11365A" font-weight="700">Before</text>
    <text x="374" y="124" font-size="22" fill="#2E7D4F" font-weight="700">During</text>
    <text x="676" y="124" font-size="22" fill="#C79A3A" font-weight="700">After</text>
    <g font-size="15" fill="#1f2a33">
      <text x="72" y="162">• Right-size attendee list</text><text x="72" y="188">• 25/50 minute defaults</text><text x="72" y="214">• Clear agenda shared early</text><text x="72" y="240">• 24+ hour notice norm</text><text x="72" y="266">• Ask: should this be async?</text>
      <text x="374" y="162">• Assign facilitator</text><text x="374" y="188">• Use AI for note/action capture</text><text x="374" y="214">• Keep contribution focus clear</text><text x="374" y="240">• Start/end on time</text><text x="374" y="266">• Resolve ownership live</text>
      <text x="676" y="162">• Share recap within 24h</text><text x="676" y="188">• One owner + one deadline</text><text x="676" y="214">• Send recap to followers</text><text x="676" y="240">• Quarterly recurring audit</text><text x="676" y="266">• Track completion outcomes</text>
    </g>
  </g>
</svg>
<figcaption><strong>Figure 7 —</strong> A simple before/during/after scaffold helps teams operationalize recap, follow, and hygiene practices consistently.</figcaption>
</figure>

<section id="part-6" class="part-head">
<span class="part-kicker">Part 6</span>
<h2>Leadership take-aways</h2>
</section>

<h3>1) Aim at fewer bad meetings, not fewer meetings.</h3>
<h3>2) Treat individual AI adoption as necessary but insufficient.</h3>
<h3>3) Make recap adoption a named, measurable objective.</h3>
<h3>4) Audit variation by function and region, then intervene locally.</h3>
<h3>5) Empower managers first; they are leverage multipliers.</h3>
<h3>6) Pair license investment with norms investment.</h3>
<h3>7) Use structural levers: defaults, meeting-free windows, and organizer targeting.</h3>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 500" role="img" aria-labelledby="fig8-title fig8-desc">
  <title id="fig8-title">Norms versus no norms trajectories</title>
  <desc id="fig8-desc">Two lines showing diverging outcomes after AI deployment with and without norms.</desc>
  <rect width="980" height="500" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="28" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Two trajectories after AI deployment</text>
  <line x1="120" y1="390" x2="900" y2="390" stroke="#B9C8D8"/>
  <line x1="120" y1="110" x2="120" y2="390" stroke="#B9C8D8"/>
  <text x="84" y="394" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Low</text>
  <text x="82" y="118" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">High</text>
  <text x="120" y="424" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Q1</text>
  <text x="320" y="424" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Q2</text>
  <text x="520" y="424" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Q3</text>
  <text x="720" y="424" font-family="Segoe UI, Arial" font-size="13" fill="#5B6573">Q4</text>
  <polyline points="120,200 320,230 520,260 720,300 900,335" fill="none" stroke="#C0392B" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="120,200 320,185 520,165 720,145 900,126" fill="none" stroke="#2E7D4F" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="700" y="342" font-family="Segoe UI, Arial" font-size="15" fill="#C0392B">AI without norms → meeting debt increases</text>
  <text x="668" y="122" font-family="Segoe UI, Arial" font-size="15" fill="#2E7D4F">AI + norms → better meetings, lower load</text>
  <rect x="48" y="456" width="14" height="14" fill="#2E7D4F"/><text x="68" y="468" font-family="Segoe UI, Arial" font-size="13" fill="#1f2a33">Collective productivity</text>
  <rect x="252" y="456" width="14" height="14" fill="#C0392B"/><text x="272" y="468" font-family="Segoe UI, Arial" font-size="13" fill="#1f2a33">Calendar burden</text>
</svg>
<figcaption><strong>Figure 8 —</strong> The divergence appears after deployment: norms determine whether AI time savings become collective gains or additional meeting load.</figcaption>
</figure>

<section id="measurement" class="part-head">
<span class="part-kicker">Measurement</span>
<h2>What to measure: six signals worth tracking</h2>
</section>

<ol>
  <li><strong>Meeting hours per week</strong> — baseline load signal.</li>
  <li><strong>Large and long meeting share</strong> (9+ attendees and 60+ minutes) — strongest structural risk indicator.</li>
  <li><strong>Multitasking rate</strong> — population-level quality signal, not an individual judgement.</li>
  <li><strong>Available focus hours</strong> — whether deep work remains possible.</li>
  <li><strong>After-hours collaboration</strong> — sustainability and burnout risk proxy.</li>
  <li><strong>Late join / late end frequency</strong> — operational discipline indicator.</li>
</ol>

<aside class="callout is-action reveal" markdown="1">
<span class="callout-label">Start this week</span>
1. Audit one recurring meeting you own.  
2. Change your defaults to 50/25 minutes.  
3. Turn recap on for your next recurring meeting and deliberately invite at least one “follower.”
</aside>

<p><strong>A coda.</strong> The meetings worth keeping build people: the 1:1s where mentorship happens, the small sessions where decisions are made, and the cross-team conversations where weak ties form. The meetings worth shrinking drain attention: long recurring broadcasts, stale syncs, and reactive fire-drills. AI will amplify whichever system teams choose to build.</p>

<section id="references" class="references">
<h2>References</h2>
<ol>
  <li id="ref-1">Rogelberg, S. G. (2019). <em>The Surprising Science of Meetings</em>. Oxford University Press.</li>
  <li id="ref-2">Iqbal, S. T., Grudin, J., & Horvitz, E. (2011). Peripheral computing during presentations. CHI.</li>
  <li id="ref-3">Cao, H. et al. (2021). Large-scale analysis of multitasking during remote meetings. CHI.</li>
  <li id="ref-4">Butler, J. et al. (Eds.). (2025). <em>Microsoft New Future of Work Report 2025</em>.</li>
  <li id="ref-5">Microsoft & LinkedIn. (2024). <em>Work Trend Index Annual Report</em>.</li>
  <li id="ref-6">Rajkumar, K. et al. (2022). A causal test of the strength of weak ties. <em>Science</em>.</li>
  <li id="ref-7">Yang, L. et al. (2022). Effects of remote work on collaboration. <em>Nature Human Behaviour</em>.</li>
  <li id="ref-8">Lutjens, M., & Felfe, J. (2025). Informal communication and job satisfaction in hybrid work.</li>
  <li id="ref-9">Microsoft. (2025). <em>Work Trend Index Annual Report</em>.</li>
  <li id="ref-10">Microsoft WorkLab. (2022). Too many meetings? Here's how AI could change that.</li>
  <li id="ref-11">Microsoft WorkLab. (2024). AI Data Drop: The 11-by-11 tipping point.</li>
  <li id="ref-12">Atlassian. (2025). <em>State of Teams 2025</em>.</li>
  <li id="ref-13">Ranganathan, A., & Ye, A. (2026). AI doesn't reduce work — it intensifies it. HBR.</li>
  <li id="ref-14">Brynjolfsson, E., Li, D., & Raymond, L. (2025). Generative AI at work. <em>QJE</em>.</li>
  <li id="ref-15">Granovetter, M. (1973). The strength of weak ties. <em>AJS</em>.</li>
  <li id="ref-16">Allen, J. A., Lehmann-Willenbrock, N., & Rogelberg, S. G. (Eds.). (2015). <em>The Cambridge Handbook of Meeting Science</em>.</li>
  <li id="ref-17">Dell'Acqua, F. et al. (2025). Navigating the jagged technological frontier. <em>Organization Science</em>.</li>
  <li id="ref-18">Saatci, B. et al. (2020). Reconfiguring hybrid meetings. <em>CSCW</em>.</li>
  <li id="ref-19">Amershi, S. et al. (2019). Guidelines for Human-AI Interaction. CHI.</li>
  <li id="ref-20">Asthana, S. et al. (2024). LLM-powered meeting recap system. PACM HCI / CSCW.</li>
  <li id="ref-21">Mark, G., Gudith, D., & Klocke, U. (2008). The cost of interrupted work. CHI.</li>
  <li id="ref-22">Olson, G. M., & Olson, J. S. (2000). Distance matters. <em>HCI</em>.</li>
</ol>
</section>

<p class="article-end-note">Org A = anonymized European mobility/IT enterprise diagnostic. Org B = anonymized global luxury brand diagnostic. Metrics shown in aggregate only.</p>
