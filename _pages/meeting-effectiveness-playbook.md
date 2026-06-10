---
layout: article
title: "The Meeting Effectiveness Playbook"
description: "An operational playbook for fixing meeting culture — what to change before, during, and after meetings, how Copilot helps, and how to measure progress with Viva Insights."
permalink: /articles/meeting-effectiveness-playbook/
eyebrow: "Copilot Analytics Lab · PANDAS Team · May 2026"
dek: "Meeting culture is a design problem with a design solution. This playbook turns the evidence into a short list of changes you can make this week — at the calendar, the meeting, and the organisation."
byline: "By the PANDAS team · A Copilot Analytics Lab brief"
read_time: "14 min read"
css: "/assets/css/article.css"
---

<nav class="article-contents reveal" aria-label="Article contents">
<p>In this article</p>
<ol>
  <li><a href="#part-1">Part 1 · The case for change</a></li>
  <li><a href="#part-2">Part 2 · Best practices, before / during / after</a></li>
  <li><a href="#part-3">Part 3 · Copilot as your meeting partner</a></li>
  <li><a href="#part-4">Part 4 · Scaling culture change</a></li>
  <li><a href="#part-5">Part 5 · Measure and improve</a></li>
  <li><a href="#start-this-week">Start this week</a></li>
  <li><a href="#references">References</a></li>
</ol>
</nav>

<p class="lead">Knowledge workers now spend most of the working week communicating and meeting. The original promise of AI was simple: fewer meetings, less drudgery, more deep work. Two years in, the evidence is more nuanced — meeting load can still rise, attention can fragment, and the calendar gets noisier before it gets quieter. This playbook is the operational answer. It is not a manifesto; it is a list of changes that work.</p>

### The TL;DR

<ul class="tldr">
  <li><strong>Meetings are not the problem — meeting <em>design</em> is.</strong> Duration, attendee count, and agenda clarity predict multitasking better than personality or intent <a href="#ref-3">[3]</a>.</li>
  <li><strong>A single in-meeting distraction costs ~23 minutes</strong> of recovered focus. That cost is structural, not behavioural <a href="#ref-7">[7]</a>.</li>
  <li><strong>Multitasking has three drivers</strong> — overload, passive attendance, and anxiety — and each needs a different response <a href="#ref-3">[3]</a>.</li>
  <li><strong>Copilot reduces the anxiety loop</strong> that drives most in-meeting task-switching. The safety net of recap and follow makes informed absence a real option.</li>
  <li><strong>The biggest gains come from defaults, not training.</strong> 25/50-minute meeting defaults, no-meeting blocks, and recurring-series audits change behaviour for people who never read the playbook.</li>
</ul>

<aside class="callout is-bottomline reveal" markdown="1">
<span class="callout-label">Bottom line</span>
The best-performing meetings are focused, right-sized (5–8 people), and short (≤30 minutes). These characteristics are not a function of facilitator talent; they are the result of design choices made before the meeting begins.
</aside>

<section id="part-1" class="part-head">
<span class="part-kicker">Part 1</span>
<h2>The case for change</h2>
</section>

<p>Since February 2020, the number of Teams meetings and calls per week has nearly tripled (<strong>+192%</strong>), and knowledge workers now spend roughly <strong>57% of their week</strong> in communication and meetings <a href="#ref-1">[1]</a>. That leaves less than half the working day for the focused, uninterrupted work that produces the highest-quality output. Over <strong>68% of employees</strong> report that they lack sufficient focus time during the day <a href="#ref-1">[1]</a>.</p>

<p>Meeting overload is one of the top contributors to digital exhaustion and burnout. Microsoft WorkLab research shows that employees whose meeting load has reduced report meaningfully higher wellbeing scores, stronger focus, and better quality of output <a href="#ref-3">[3]</a>. This is a wellbeing issue as much as a productivity issue — the two are inseparable.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 320" role="img" aria-labelledby="fig1-title fig1-desc">
  <title id="fig1-title">The hidden cost of a single distraction</title>
  <desc id="fig1-desc">A 3-minute attention break leads to a 23-minute recovery window.</desc>
  <rect width="980" height="320" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="26" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">The hidden time cost of losing the room</text>

  <g font-family="Segoe UI, Arial">
    <rect x="48" y="100" width="420" height="170" rx="14" fill="#2E5C8A"/>
    <text x="72" y="160" fill="#fff" font-size="64" font-weight="700">03</text>
    <text x="200" y="160" fill="#D7E4F1" font-size="18">minutes</text>
    <text x="72" y="200" fill="#fff" font-size="15">
      <tspan x="72" dy="0">The time before a distracted participant</tspan>
      <tspan x="72" dy="20">switches fully to another application.</tspan>
    </text>
    <text x="72" y="252" fill="#D7E4F1" font-size="13">Once attention breaks, the pull to task-switch is nearly immediate.</text>

    <rect x="512" y="100" width="420" height="170" rx="14" fill="#C0392B"/>
    <text x="536" y="160" fill="#fff" font-size="64" font-weight="700">23</text>
    <text x="664" y="160" fill="#FCEEEE" font-size="18">minutes</text>
    <text x="536" y="200" fill="#fff" font-size="15">
      <tspan x="536" dy="0">The time required to return to full</tspan>
      <tspan x="536" dy="20">cognitive productivity after switching.</tspan>
    </text>
    <text x="536" y="252" fill="#FCEEEE" font-size="13">A single distraction in a 30-minute meeting can erase the hour that follows.</text>
  </g>
</svg>
<figcaption><strong>Figure 1 —</strong> The cost of an interrupted meeting is not recovered at the end of the meeting; it is simply lost <a href="#ref-7">[7]</a>.</figcaption>
</figure>

<h3>Three drivers of in-meeting multitasking</h3>

<p>Not all multitasking has the same cause — or the same solution. Roughly <strong>39% of in-meeting multitasking is driven by task overload</strong>: people using meeting time to catch up on work they have not had space to do elsewhere <a href="#ref-3">[3]</a>. This is a rational response to being overloaded, not a sign of disengagement.</p>

<div class="three-up reveal" markdown="1">

**01 · Overload.** People use meeting time to clear a backlog they have nowhere else to do.
<br><em>Response:</em> Reduce overall meeting load; use AI to compress the catch-up burden.

**02 · Passive attendance.** Attendees are information recipients, not active contributors.
<br><em>Response:</em> Right-size invite lists; offer recordings or recaps as a genuine alternative.

**03 · Anxiety.** Self-interruption reflects sensitivity to what might be missed elsewhere.
<br><em>Response:</em> Deploy Copilot to reduce the anxiety loop that triggers task-switching.

</div>

<h3>Structure predicts multitasking — before anyone joins</h3>

<p>Analysis of large-scale remote meeting data shows that three structural conditions raise the odds of multitasking dramatically, independent of facilitation quality or topic <a href="#ref-3">[3]</a>.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 340" role="img" aria-labelledby="fig2-title fig2-desc">
  <title id="fig2-title">Multitasking odds by meeting structure</title>
  <desc id="fig2-desc">More than 10 attendees: 2x; more than 80 minutes: 6x; recurring format: 1.5x.</desc>
  <rect width="980" height="340" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="26" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Meeting structure predicts multitasking</text>

  <g font-family="Segoe UI, Arial">
    <rect x="48" y="100" width="280" height="200" rx="14" fill="#EAF1F8" stroke="#B9C8D8"/>
    <text x="68" y="156" fill="#11365A" font-size="48" font-weight="700">2×</text>
    <text x="68" y="186" fill="#1f2a33" font-size="15" font-weight="700">More than 10 attendees</text>
    <text x="68" y="216" fill="#1f2a33" font-size="13">
      <tspan x="68" dy="0">Social accountability disappears</tspan>
      <tspan x="68" dy="18">in groups larger than ten.</tspan>
    </text>
    <text x="68" y="278" fill="#11365A" font-size="12" font-weight="700">Aim for ≤ 8 for active discussion.</text>

    <rect x="350" y="100" width="280" height="200" rx="14" fill="#FCEEEE" stroke="#E7C1BB"/>
    <text x="370" y="156" fill="#C0392B" font-size="48" font-weight="700">6×</text>
    <text x="370" y="186" fill="#1f2a33" font-size="15" font-weight="700">More than 80 minutes</text>
    <text x="370" y="216" fill="#1f2a33" font-size="13">
      <tspan x="370" dy="0">Engagement collapses past a time</tspan>
      <tspan x="370" dy="18">threshold, regardless of topic.</tspan>
    </text>
    <text x="370" y="278" fill="#C0392B" font-size="12" font-weight="700">Break into focused 25–50 min blocks.</text>

    <rect x="652" y="100" width="280" height="200" rx="14" fill="#FFF5E5" stroke="#EFD9AD"/>
    <text x="672" y="156" fill="#C79A3A" font-size="48" font-weight="700">1.5×</text>
    <text x="672" y="186" fill="#1f2a33" font-size="15" font-weight="700">Recurring format</text>
    <text x="672" y="216" fill="#1f2a33" font-size="13">
      <tspan x="672" dy="0">Habit-driven meetings generate more</tspan>
      <tspan x="672" dy="18">multitasking than goal-driven ones.</tspan>
    </text>
    <text x="672" y="278" fill="#C79A3A" font-size="12" font-weight="700">Audit recurring series each quarter.</text>
  </g>
</svg>
<figcaption><strong>Figure 2 —</strong> Higher multitasking odds vs. baseline. Source: Cao et al. (2021), CHI <a href="#ref-3">[3]</a>.</figcaption>
</figure>

<blockquote class="pull-quote reveal">If poor meeting quality is a design problem, it is also a design solution.</blockquote>

<section id="part-2" class="part-head">
<span class="part-kicker">Part 2</span>
<h2>Best practices — before, during, and after</h2>
</section>

<h3>Not all meetings deserve a place in the calendar</h3>

<p>The first design decision is whether the meeting should exist at all. Use this as a quick keep-or-remove filter <a href="#ref-4">[4]</a>.</p>

<table class="is-wide reveal">
  <thead>
    <tr><th>Keep as a meeting</th><th>Replace or restructure</th></tr>
  </thead>
  <tbody>
    <tr><td>Decision-making (≤7 people, real-time back-and-forth)</td><td>Information-sharing → recorded video or Teams post</td></tr>
    <tr><td>Trust-building and conflict resolution</td><td>Large recurring syncs → trim list, reduce frequency, go async</td></tr>
    <tr><td>Brainstorming (≤18 people, ideas building in real time)</td><td>No-agenda standing meetings → audit quarterly</td></tr>
    <tr><td>Manager 1:1s — highest ROI, lowest multitasking</td><td>Just-in-case attendance → catch up via recap</td></tr>
    <tr><td>Genuine social connection</td><td>Passive broadcast / all-hands → async with live Q&amp;A</td></tr>
  </tbody>
</table>

<h3>Four questions to ask before accepting any meeting</h3>

<div class="four-up reveal" markdown="1">

**01 · Do I need to contribute?**
<br>If not, you are an FYI recipient — not a required attendee. Ask the organiser to share the recap or mark yourself as a follower in Teams.

**02 · Is this a real-time conversation?**
<br>If the purpose is information-sharing, async is almost always better. Suggest a recorded update or a shared document instead.

**03 · Can I catch up afterwards?**
<br>Recordings, recaps, and action summaries may cover everything you would take from the meeting. Request the Copilot Intelligent Recap.

**04 · Will being there strengthen the relationship?**
<br>Team cohesion and trust-building are legitimate reasons to attend. If so, attend intentionally and contribute to the relational dimension.

</div>

<h3>A checklist that works in every meeting you run</h3>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 380" role="img" aria-labelledby="fig3-title fig3-desc">
  <title id="fig3-title">Before, during, after checklist</title>
  <desc id="fig3-desc">Three-column structured approach for every meeting.</desc>
  <rect width="980" height="380" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="26" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">A structured approach to every meeting</text>

  <g font-family="Segoe UI, Arial">
    <rect x="48" y="92" width="288" height="260" rx="14" fill="#EAF1F8" stroke="#B9C8D8"/>
    <text x="68" y="124" fill="#11365A" font-size="13" font-weight="700">BEFORE</text>
    <text x="68" y="148" fill="#11365A" font-size="18" font-weight="700">Set the conditions</text>
    <g font-size="14" fill="#1f2a33">
      <text x="68" y="180">• Right-size the attendee list</text>
      <text x="68" y="204">• Choose 25 or 50 minutes</text>
      <text x="68" y="228">• Share a clear agenda</text>
      <text x="68" y="252">• Give 24+ hours' notice</text>
      <text x="68" y="276">• Consider async alternatives</text>
      <text x="68" y="300">• Name the meeting clearly</text>
    </g>

    <rect x="346" y="92" width="288" height="260" rx="14" fill="#E9F4EC" stroke="#BCDCC6"/>
    <text x="366" y="124" fill="#2E7D4F" font-size="13" font-weight="700">DURING</text>
    <text x="366" y="148" fill="#2E7D4F" font-size="18" font-weight="700">Facilitate with purpose</text>
    <g font-size="14" fill="#1f2a33">
      <text x="366" y="180">• Start and end on time</text>
      <text x="366" y="204">• Follow the agenda</text>
      <text x="366" y="228">• Designate a facilitator</text>
      <text x="366" y="252">• Use Copilot as note-taker</text>
      <text x="366" y="276">• Create conditions for focus</text>
      <text x="366" y="300">• Capture action items live</text>
    </g>

    <rect x="644" y="92" width="288" height="260" rx="14" fill="#FFF5E5" stroke="#EFD9AD"/>
    <text x="664" y="124" fill="#C79A3A" font-size="13" font-weight="700">AFTER</text>
    <text x="664" y="148" fill="#C79A3A" font-size="18" font-weight="700">Turn talk into action</text>
    <g font-size="14" fill="#1f2a33">
      <text x="664" y="180">• Share notes within 24 hours</text>
      <text x="664" y="204">• One owner per action item</text>
      <text x="664" y="228">• Use Copilot Intelligent Recap</text>
      <text x="664" y="252">• Share with non-attendees</text>
      <text x="664" y="276">• Ask: could this be shorter?</text>
      <text x="664" y="300">• Review recurring series</text>
    </g>
  </g>
</svg>
<figcaption><strong>Figure 3 —</strong> A simple operating standard that holds across team types, sizes, and contexts.</figcaption>
</figure>

<h3>Three fundamentals of good design</h3>

<div class="three-up reveal" markdown="1">

**01 · Right-size the room.**
Aim for 5 to 8 attendees. Every person added increases cognitive cost and multitasking likelihood. Ask: do they need to contribute, or only be informed? If the latter, share the notes instead.

**02 · Default to 25 or 50 minutes.**
Build in natural transition time and structurally prevent back-to-back scheduling. Many recurring check-ins lose nothing when 60 minutes becomes 45.

**03 · Give 24+ hours' notice.**
Last-minute meetings disrupt focus time and signal poor planning. Short-notice rates are a quality signal worth tracking — a high proportion across a team indicates a reactive scheduling culture.

</div>

<aside class="callout reveal" markdown="1">
<span class="callout-label">During the meeting</span>
The facilitator owns the agenda, the time, and the quality of the discussion. The note-taker owns the action items. Where Copilot is available, the note-taking role can be substantially automated — freeing that person to contribute fully rather than transcribing.
</aside>

<aside class="callout reveal" markdown="1">
<span class="callout-label">After the meeting</span>
Every recurring series deserves a periodic review: <em>Is this still needed? At this frequency? With this group? In this format?</em> A brief retrospective at the start of each quarter is enough. The answer is more often "no" or "shorter" than most people expect.
</aside>

<section id="part-3" class="part-head">
<span class="part-kicker">Part 3</span>
<h2>Copilot as your meeting partner</h2>
</section>

<p>Copilot is not just a productivity feature — it addresses a structural driver of meeting dysfunction. Most in-meeting multitasking is driven by anxiety about what is building up elsewhere. Copilot monitors and summarises in the background, allowing full presence without that fear. The safety net is always there; the anxiety loop is broken.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 380" role="img" aria-labelledby="fig4-title fig4-desc">
  <title id="fig4-title">Six Copilot capabilities for better meetings</title>
  <desc id="fig4-desc">Six capabilities organised by before, during, and after the meeting.</desc>
  <rect width="980" height="380" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="26" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Six Copilot capabilities for better meetings</text>

  <g font-family="Segoe UI, Arial">
    <rect x="48" y="92" width="288" height="124" rx="12" fill="#11365A"/>
    <text x="68" y="124" fill="#D7E4F1" font-size="13" font-weight="700">01 · BEFORE</text>
    <text x="68" y="156" fill="#fff" font-size="17" font-weight="700">Context briefing</text>
    <text x="68" y="184" fill="#D7E4F1" font-size="13">
      <tspan x="68" dy="0">Summarises emails, past meetings,</tspan>
      <tspan x="68" dy="16">and documents in Outlook.</tspan>
    </text>

    <rect x="346" y="92" width="288" height="124" rx="12" fill="#2E5C8A"/>
    <text x="366" y="124" fill="#D7E4F1" font-size="13" font-weight="700">02 · BEFORE</text>
    <text x="366" y="156" fill="#fff" font-size="17" font-weight="700">Agenda drafting</text>
    <text x="366" y="184" fill="#D7E4F1" font-size="13">
      <tspan x="366" dy="0">Drafts a suggested agenda from</tspan>
      <tspan x="366" dy="16">title and calendar context.</tspan>
    </text>

    <rect x="644" y="92" width="288" height="124" rx="12" fill="#4C8C65"/>
    <text x="664" y="124" fill="#DDF2E4" font-size="13" font-weight="700">03 · DURING</text>
    <text x="664" y="156" fill="#fff" font-size="17" font-weight="700">Live Q&amp;A</text>
    <text x="664" y="184" fill="#DDF2E4" font-size="13">
      <tspan x="664" dy="0">"What was just decided?"</tspan>
      <tspan x="664" dy="16">Catches late joiners up too.</tspan>
    </text>

    <rect x="48" y="232" width="288" height="124" rx="12" fill="#2E7D4F"/>
    <text x="68" y="264" fill="#DDF2E4" font-size="13" font-weight="700">04 · DURING</text>
    <text x="68" y="296" fill="#fff" font-size="17" font-weight="700">Action capture</text>
    <text x="68" y="324" fill="#DDF2E4" font-size="13">
      <tspan x="68" dy="0">Flags and attributes action items</tspan>
      <tspan x="68" dy="16">as they emerge live.</tspan>
    </text>

    <rect x="346" y="232" width="288" height="124" rx="12" fill="#C79A3A"/>
    <text x="366" y="264" fill="#FFF3DA" font-size="13" font-weight="700">05 · AFTER</text>
    <text x="366" y="296" fill="#fff" font-size="17" font-weight="700">Intelligent Recap</text>
    <text x="366" y="324" fill="#FFF3DA" font-size="13">
      <tspan x="366" dy="0">AI notes, owners, chapter markers</tspan>
      <tspan x="366" dy="16">— ready within minutes.</tspan>
    </text>

    <rect x="644" y="232" width="288" height="124" rx="12" fill="#B6722A"/>
    <text x="664" y="264" fill="#FFF3DA" font-size="13" font-weight="700">06 · AFTER</text>
    <text x="664" y="296" fill="#fff" font-size="17" font-weight="700">Follow, don't attend</text>
    <text x="664" y="324" fill="#FFF3DA" font-size="13">
      <tspan x="664" dy="0">Receive the full recap without</tspan>
      <tspan x="664" dy="16">being in the room.</tspan>
    </text>
  </g>
</svg>
<figcaption><strong>Figure 4 —</strong> Six Copilot capabilities that change the meeting equation. Learn more in the <a href="https://support.microsoft.com/en-us/office/get-started-with-copilot-in-microsoft-teams-meetings-0bf9dd3c-96f7-44e2-8bb8-790bedf066b1">Copilot in Teams meetings guide</a> <a href="#ref-9">[9]</a>.</figcaption>
</figure>

<h3>Two features that change the meeting equation</h3>

<p><strong>Intelligent Recap.</strong> After any recorded Teams meeting, Copilot generates a structured, AI-powered summary: key discussion points, attributed action items, and chapter markers with timestamps that let anyone jump to the relevant part. Notes are available within minutes. You no longer need to write up the meeting, sit through a recording, or rely on memory to reconstruct what was agreed.</p>

<p><strong>Follow, don't attend.</strong> Instead of attending, you can mark yourself as "following" a meeting and receive the full recap automatically. This changes the calculus of attendance: if your role is to be informed of outcomes rather than to contribute, there is now a better option than attending. Meeting organisers can explicitly invite non-contributors as followers for large meetings — removing the social pressure to attend and giving people a legitimate, supported way to opt out.</p>

<aside class="callout reveal" markdown="1">
<span class="callout-label">Why it works</span>
Most in-meeting multitasking is driven by three anxieties: <em>am I missing email?</em>, <em>am I missing a parallel meeting?</em>, and <em>am I missing context I'll need later?</em> Copilot addresses all three directly — drafting email faster, providing recaps for parallel meetings, and surfacing what was decided. It treats the cause, not just the symptom.
</aside>

<blockquote class="pull-quote reveal">AI does not fix meeting culture. It amplifies the culture already present.</blockquote>

<section id="part-4" class="part-head">
<span class="part-kicker">Part 4</span>
<h2>Scaling culture change</h2>
</section>

<p>Individual habits matter — but lasting change requires systemic levers. The most effective programmes change the <em>environment</em>, not the people.</p>

<h3>Three levers for organisation-wide change</h3>

<div class="three-up reveal" markdown="1">

**01 · No-meeting blocks.**
Programmes such as Focus Fridays — or any designated no-meeting block — create structural space for deep work. They only succeed if organisers respect them, which makes manager buy-in essential and measurement a prerequisite. Viva Insights can track Friday meeting hours as the primary KPI.

**02 · Default meeting durations.**
Configuring Outlook and Teams to default to 25- or 50-minute meetings costs nothing, requires no training, and has an immediate structural effect on everyone's calendar. Defaults influence behaviour even for people who never read the playbook.

**03 · Top-organiser engagement.**
In most organisations, the majority of meeting hours are generated by a relatively small number of people. Identifying and engaging the heaviest meeting organisers — particularly senior leaders — produces faster, more lasting impact than broad awareness campaigns. Use Viva Insights organiser data to find them.

</div>

<h3>Managers are the highest-leverage point</h3>

<p>Teams take their cues from how their manager behaves in meetings. If you start on time, end on time, share an agenda, and send a follow-up note, you set the standard for everyone around you. The reverse is equally true. Meeting culture flows downward — the most senior person in the room sets the norm, whether intentionally or not.</p>

<aside class="callout reveal" markdown="1">
<span class="callout-label">For managers</span>
Use Viva Insights to share meeting patterns with your team — <em>not</em> as a surveillance tool, but as a shared reference point. <em>Which of our recurring meetings are running longest? Where is multitasking highest? What can we change together?</em> Data makes these conversations objective, depersonalised, and actionable in a way that is difficult to achieve without it.
</aside>

<h3>The most valuable — and most vulnerable — meeting</h3>

<p>The 1:1 format consistently produces the lowest multitasking of any meeting type <a href="#ref-3">[3]</a>. In a 1:1, both parties are immediately aware if attention drops; there is nowhere to hide. This built-in accountability makes it uniquely effective. <strong>1:1s with direct reports should be the last meetings dropped when calendar pressure builds, not the first.</strong></p>

<table class="is-wide reveal">
  <thead>
    <tr><th>Team signal in Viva Insights</th><th>What it usually means</th></tr>
  </thead>
  <tbody>
    <tr><td>High multitasking hours</td><td>Likely too many large or long meetings — investigate structural drivers</td></tr>
    <tr><td>Low focus time</td><td>Calendar over-fragmented; help protect uninterrupted blocks</td></tr>
    <tr><td>High conflicting meeting hours</td><td>Frequently double-booked; resolve before it shows as disengagement</td></tr>
    <tr><td>Declining 1:1 frequency</td><td>Connection risk; restore cadence before it shows in engagement scores</td></tr>
    <tr><td>After-hours activity</td><td>Workload may be unsustainable; a workload conversation is needed</td></tr>
  </tbody>
</table>

<section id="part-5" class="part-head">
<span class="part-kicker">Part 5</span>
<h2>Measure and improve</h2>
</section>

<p>Six metrics, tracked over time in Viva Insights, are enough to know whether meeting culture is improving — and whether the work to improve it is landing <a href="#ref-4">[4]</a>.</p>

<figure class="article-figure reveal">
<svg viewBox="0 0 980 420" role="img" aria-labelledby="fig5-title fig5-desc">
  <title id="fig5-title">Six metrics for meeting culture</title>
  <desc id="fig5-desc">Six Viva Insights metrics organised in a 2x3 grid.</desc>
  <rect width="980" height="420" rx="16" fill="#F7FAFD"/>
  <text x="48" y="54" font-size="26" fill="#11365A" font-family="Segoe UI, Arial" font-weight="700">Six metrics to track over time</text>

  <g font-family="Segoe UI, Arial">
    <rect x="48" y="92" width="288" height="140" rx="12" fill="#EAF1F8" stroke="#B9C8D8"/>
    <text x="68" y="120" fill="#11365A" font-size="12" font-weight="700">01 · BASELINE</text>
    <text x="68" y="148" fill="#11365A" font-size="17" font-weight="700">Meeting hours</text>
    <text x="68" y="178" fill="#1f2a33" font-size="13">
      <tspan x="68" dy="0">Hours per person per week.</tspan>
      <tspan x="68" dy="16">The foundational signal — the</tspan>
      <tspan x="68" dy="16">baseline for everything else.</tspan>
    </text>

    <rect x="346" y="92" width="288" height="140" rx="12" fill="#FCEEEE" stroke="#E7C1BB"/>
    <text x="366" y="120" fill="#C0392B" font-size="12" font-weight="700">02 · STRUCTURAL</text>
    <text x="366" y="148" fill="#C0392B" font-size="17" font-weight="700">Large &amp; long meetings</text>
    <text x="366" y="178" fill="#1f2a33" font-size="13">
      <tspan x="366" dy="0">Meetings with 9+ attendees</tspan>
      <tspan x="366" dy="16">AND &gt; 1 hour. The strongest</tspan>
      <tspan x="366" dy="16">predictor of multitasking.</tspan>
    </text>

    <rect x="644" y="92" width="288" height="140" rx="12" fill="#FFF5E5" stroke="#EFD9AD"/>
    <text x="664" y="120" fill="#C79A3A" font-size="12" font-weight="700">03 · BEHAVIOURAL</text>
    <text x="664" y="148" fill="#C79A3A" font-size="17" font-weight="700">Multitasking hours</text>
    <text x="664" y="178" fill="#1f2a33" font-size="13">
      <tspan x="664" dy="0">Meeting hours with parallel</tspan>
      <tspan x="664" dy="16">email or chat. A design</tspan>
      <tspan x="664" dy="16">problem, not a behaviour.</tspan>
    </text>

    <rect x="48" y="248" width="288" height="140" rx="12" fill="#E9F4EC" stroke="#BCDCC6"/>
    <text x="68" y="276" fill="#2E7D4F" font-size="12" font-weight="700">04 · OUTCOME</text>
    <text x="68" y="304" fill="#2E7D4F" font-size="17" font-weight="700">Focus time</text>
    <text x="68" y="334" fill="#1f2a33" font-size="13">
      <tspan x="68" dy="0">Uninterrupted, meeting-free</tspan>
      <tspan x="68" dy="16">hours per working day. The</tspan>
      <tspan x="68" dy="16">primary goal of most programmes.</tspan>
    </text>

    <rect x="346" y="248" width="288" height="140" rx="12" fill="#FCEEEE" stroke="#E7C1BB"/>
    <text x="366" y="276" fill="#C0392B" font-size="12" font-weight="700">05 · WELLBEING</text>
    <text x="366" y="304" fill="#C0392B" font-size="17" font-weight="700">After-hours activity</text>
    <text x="366" y="334" fill="#1f2a33" font-size="13">
      <tspan x="366" dy="0">Work outside standard hours.</tspan>
      <tspan x="366" dy="16">A leading indicator of</tspan>
      <tspan x="366" dy="16">unsustainable workload.</tspan>
    </text>

    <rect x="644" y="248" width="288" height="140" rx="12" fill="#EAF1F8" stroke="#B9C8D8"/>
    <text x="664" y="276" fill="#11365A" font-size="12" font-weight="700">06 · NORMS</text>
    <text x="664" y="304" fill="#11365A" font-size="17" font-weight="700">On-time discipline</text>
    <text x="664" y="334" fill="#1f2a33" font-size="13">
      <tspan x="664" dy="0">Late starts and overruns.</tspan>
      <tspan x="664" dy="16">Reveals whether norms hold</tspan>
      <tspan x="664" dy="16">in practice, not just policy.</tspan>
    </text>
  </g>
</svg>
<figcaption><strong>Figure 5 —</strong> Six signals to track, available in the Viva Insights <a href="https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/templates/meeting-effectiveness">Meeting Effectiveness template</a>.</figcaption>
</figure>

<section id="start-this-week" class="part-head">
<span class="part-kicker">Start this week</span>
<h2>Three actions that need no approval, no budget, no new tools</h2>
</section>

<aside class="callout is-action reveal" markdown="1">
<span class="callout-label">Start this week</span>
1. **Audit one recurring meeting you own.** Pick the longest-running. Ask the group: *Is this still necessary? At this frequency? With everyone currently invited?* One meeting changed per month is twelve improved per year.
2. **Change your Outlook defaults to 25 / 50 minutes.** Under a minute to implement (File &gt; Options &gt; Calendar). Every new meeting you create will default to a shorter duration — and create natural gaps between calls.
3. **Turn on Copilot Intelligent Recap for your next recurring meeting.** Share the output with attendees and at least one "follower" who didn't attend. This sets the norm that non-attendees can stay fully informed without being in the room.
</aside>

<p><strong>A coda.</strong> Meeting culture does not change because a policy is published. It changes because the calendar changes — defaults shift, recurring series get audited, recaps land in people's inboxes, and managers behave differently in the meetings they run. AI is not the fix. It is the amplifier. The choice is which culture you ask it to scale.</p>

<section id="references" class="references">
<h2>References</h2>
<ol>
  <li id="ref-1">Microsoft. (2023). <em>Work Trend Index 2023: Will AI Fix Work?</em> <a href="https://www.microsoft.com/en-us/worklab/work-trend-index/will-ai-fix-work">microsoft.com/worklab</a></li>
  <li id="ref-2">Microsoft. (2024). <em>Work Trend Index 2024: AI at Work Is Here</em>. <a href="https://www.microsoft.com/en-us/worklab/work-trend-index/ai-at-work-is-here-now-comes-the-hard-part">microsoft.com/worklab</a></li>
  <li id="ref-3">Cao, H. et al. (2021). Large-scale analysis of multitasking behaviour during remote meetings. <em>CHI 2021, ACM.</em></li>
  <li id="ref-4">Iqbal, S. &amp; Leach, A. <em>Towards More Effective Meetings.</em> Microsoft Viva Insights.</li>
  <li id="ref-5">Iqbal, S. <em>The Future of Hybrid Meetings.</em> Microsoft Viva Insights.</li>
  <li id="ref-6">Microsoft WorkLab. <em>Research hub.</em> <a href="https://www.microsoft.com/en-us/worklab">microsoft.com/worklab</a></li>
  <li id="ref-7">Mark, G., Gudith, D., &amp; Klocke, U. (2008). The cost of interrupted work. <em>CHI 2008, ACM.</em></li>
  <li id="ref-8">Microsoft Viva Insights. <em>Meeting Effectiveness Power BI template.</em> <a href="https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/templates/meeting-effectiveness">learn.microsoft.com</a></li>
  <li id="ref-9">Microsoft. <em>Get started with Copilot in Teams meetings.</em> <a href="https://support.microsoft.com/en-us/office/get-started-with-copilot-in-microsoft-teams-meetings-0bf9dd3c-96f7-44e2-8bb8-790bedf066b1">support.microsoft.com</a></li>
  <li id="ref-10">Microsoft. <em>Admin guide for Copilot and transcription in Teams.</em> <a href="https://learn.microsoft.com/en-us/microsoftteams/copilot-teams-transcription">learn.microsoft.com</a></li>
</ol>
<p class="article-end-note">Adapted from the internal <em>Meeting Effectiveness Playbook</em> (May 2026). Figures and prescriptions are presented operationally; metric thresholds should be calibrated to local context before being used as targets.</p>
</section>
