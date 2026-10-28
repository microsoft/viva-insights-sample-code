---
layout: article
editorial: true
title: "When AI Met the Meeting"
description: "What meeting research and Copilot usage tell us about attention, collaboration and the meeting habits teams can change."
permalink: /articles/when-ai-met-the-meeting/
eyebrow: "Copilot Analytics Lab · May 2026"
dek: "AI can help people catch up on meetings. Teams still need to decide which conversations deserve their time, who needs to attend and what happens afterwards."
byline: "By the PANDAS team"
read_time: "7 min read"
css: "/assets/css/article.css"
image: /assets/images/articles/when-ai-met-the-meeting-card.png
---

<p class="lead">A meeting recap can save someone from watching a recording or joining a second call. It cannot decide whether the meeting needed to happen. That decision still belongs to the people who organise it.</p>

<p>This brief brings together meeting research, published Copilot findings and a case study of meeting behaviour. The practical recommendation is to review long, large and recurring meetings while protecting time for decisions, mentoring and cross-team work. Use recap to make attendance more deliberate, then check whether the changes leave people with more time to focus.</p>

<details class="article-contents">
<summary>In this article</summary>
<nav aria-label="Article contents">
<ol>
  <li><a href="#part-1">What the evidence says</a></li>
  <li><a href="#part-4">Which meetings to protect</a></li>
  <li><a href="#part-5">Four changes to try</a></li>
  <li><a href="#measurement">What to measure</a></li>
  <li><a href="#references">References</a></li>
</ol>
</nav>
</details>

<section id="part-1" class="part-head">
<h2>What the evidence says</h2>
</section>

<h3 id="part-2">Multitasking needs context</h3>

<p>Looking up a figure for the discussion and answering unrelated messages both count as multitasking, but they serve different purposes. Cao and colleagues studied remote meetings using Microsoft employee telemetry and a diary study. They found that meeting size, length, time and type were associated with multitasking, which had both positive and negative outcomes <a href="#ref-3">[1]</a>.</p>

<div class="article-comparison">
  <section aria-labelledby="supports-meeting">
    <h4 id="supports-meeting">Supports the conversation</h4>
    <ul>
      <li>Capturing decisions and action items.</li>
      <li>Looking up information needed for a decision.</li>
      <li>Drafting a follow-up while the context is fresh.</li>
    </ul>
  </section>
  <section aria-labelledby="competes-meeting">
    <h4 id="competes-meeting">Competes for attention</h4>
    <ul>
      <li>Switching to unrelated work during the discussion.</li>
      <li>Trying to attend two conflicting meetings.</li>
      <li>Repeatedly checking messages and missing decisions.</li>
    </ul>
  </section>
</div>

<p>In Case Study B, meeting duration was the strongest predictor of the distraction signal. Meetings lasting at least 60 minutes accounted for a majority of meeting hours, despite being a minority of meetings. That points to a useful place to investigate. The signal alone cannot tell us why an individual was multitasking.</p>

<h3 id="part-3">Meeting growth and AI benefits are different findings</h3>

<p>Meeting load was already rising before Microsoft 365 Copilot. Later surveys describe how Copilot users experience the tool. These sources cover different periods and populations, so they should be read separately.</p>

<div class="article-evidence">
  <section aria-labelledby="meeting-growth">
    <h4 id="meeting-growth">Background: meeting load</h4>
    <p class="evidence-value">+252%</p>
    <p>Increase in weekly meeting time for the average Teams user between February 2020 and February 2022.</p>
    <p class="evidence-source"><a href="https://www.microsoft.com/en-us/worklab/work-trend-index/great-expectations-making-hybrid-work-work">Microsoft Work Trend Index, 16 March 2022</a>. Aggregated Microsoft 365 activity <a href="#ref-wti-2022">[2]</a>.</p>
  </section>
  <section aria-labelledby="copilot-meeting-relief">
    <h4 id="copilot-meeting-relief">Copilot users' experience</h4>
    <p class="evidence-value">37%</p>
    <p>Of respondents who had used Copilot for more than ten weeks said it helped them attend fewer meetings.</p>
    <p class="evidence-source"><a href="https://www.microsoft.com/en-us/worklab/ai-data-drop-the-11-by-11-tipping-point">Microsoft WorkLab, the 11-by-11 study</a>. Self-reported results from a survey launched in September 2023, with 1,300 users across all usage-duration groups <a href="#ref-11">[3]</a>.</p>
  </section>
</div>

<p>The first statistic describes a historical change in meeting time. The second describes a reported benefit among a subset of Copilot users. Neither establishes that deploying AI will reduce a team's meeting hours. Teams need to decide which meetings people can skip and how they will receive the decisions and actions afterwards.</p>

<section id="part-4" class="part-head">
<h2>Which meetings to protect</h2>
</section>

<p>Reducing meeting hours can also remove opportunities to ask for help, make a decision or meet someone outside the immediate team. Rajkumar and colleagues' LinkedIn experiment found that weaker ties can help people find new jobs <a href="#ref-6">[4]</a>. Yang and colleagues found that firm-wide remote work made collaboration networks at Microsoft more siloed <a href="#ref-7">[5]</a>. Those studies concern networks and work arrangements, rather than proving the value of a particular meeting format. They give teams a reason to consider relationships when reviewing the calendar.</p>

<dl class="article-benefits">
  <div><dt>Mentoring and trust</dt><dd>Keep space for 1:1s and informal conversations where people can raise concerns, ask for advice and discuss their work.</dd></div>
  <div><dt>Decisions and shared work</dt><dd>Bring together the people who can resolve a question or work through a problem. Make the decision and its owner clear.</dd></div>
  <div><dt>Connections across teams</dt><dd>Review who would lose contact if a recurring meeting disappeared, and how those people would stay connected.</dd></div>
</dl>

<p class="article-table-hint">Scroll across the table to read all three columns.</p>
<div class="article-table-scroll" role="region" aria-labelledby="meeting-review-caption" tabindex="0">
<table>
  <caption id="meeting-review-caption">Review each meeting against its purpose</caption>
  <thead>
    <tr><th scope="col">Meeting type</th><th scope="col">Value to preserve</th><th scope="col">What to review</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">1:1s and small decision meetings</th><td>Mentoring, decisions and trust</td><td>Whether people have enough time to discuss what matters</td></tr>
    <tr><th scope="row">Working sessions</th><td>Solving a problem together</td><td>Whether the session has drifted into status reporting</td></tr>
    <tr><th scope="row">Cross-team meetings</th><td>Coordination and relationships</td><td>Whether the original objective still applies</td></tr>
    <tr><th scope="row">Large broadcasts</th><td>Shared context and questions for leaders</td><td>Which parts need live attendance</td></tr>
    <tr><th scope="row">Recurring series</th><td>Regular decisions and accountability</td><td>Purpose, cadence, duration and attendee list</td></tr>
  </tbody>
</table>
</div>

<section id="part-5" class="part-head">
<h2>Four changes to try</h2>
</section>

<h3>1. Make recap part of the follow-up</h3>

<p>Use recap to revisit decisions, check actions and catch up on a missed meeting. It can also help people stay present during a conversation instead of taking extensive notes. For 1:1s, agree what to capture and who can access it before recording or transcribing.</p>

<figure class="article-figure recap-band" aria-labelledby="recap-band-caption">
  <div class="recap-band-heading">
    <h4>Regular recap usage</h4>
    <p class="recap-band-value">5% to under 10%</p>
    <p>Overall in Case Study B</p>
  </div>
  <div class="recap-band-scale" aria-hidden="true">
    <div class="recap-band-track"><span></span></div>
    <div class="recap-band-ticks"><span>0%</span><span>5%</span><span>10%</span><span>15%</span><span>20%</span></div>
    <p>Percentage of users regularly using recap</p>
  </div>
  <figcaption id="recap-band-caption">The shaded interval shows the reported range. Usage varied across functions.</figcaption>
</figure>

<p>Start by finding out what prevents routine use in each team. Agree who will share the recap and how recipients will confirm the actions assigned to them. Review AI-generated notes before treating them as the record of a decision.</p>

<h3>2. Attend one meeting and follow up on the other</h3>

<p>When meetings conflict, attend the one where you need to contribute. If you only need the outcome of the other, arrange to receive its recap. Confirm that the organiser can share the material you need and that someone will contact you if a decision requires your input.</p>

<h3>3. Set aside time for catch-up</h3>

<p>Read recaps and complete follow-ups together in a planned block, rather than checking them repeatedly during other work. Keep that catch-up time visible when reviewing the calendar. Replacing attendance with an unread queue of recaps does not resolve the workload.</p>

<h3>4. Review the meetings you organise</h3>

<p>Try 25- or 50-minute defaults, give attendees enough notice to prepare, and review recurring series quarterly. Ask managers to apply these changes to their own meetings. Agree who will review the results, then adjust the approach where teams have different needs.</p>

<div class="article-checklist">
  <section aria-labelledby="before-meeting">
    <h4 id="before-meeting">Before</h4>
    <ul>
      <li>Share the purpose and agenda.</li>
      <li>Invite the people needed for the discussion.</li>
      <li>Choose a duration that fits the work.</li>
      <li>Consider whether a written update would suffice.</li>
    </ul>
  </section>
  <section aria-labelledby="during-meeting">
    <h4 id="during-meeting">During</h4>
    <ul>
      <li>Name a facilitator.</li>
      <li>Start and end on time.</li>
      <li>Capture decisions and actions.</li>
      <li>Agree an owner and deadline for each action.</li>
    </ul>
  </section>
  <section aria-labelledby="after-meeting">
    <h4 id="after-meeting">After</h4>
    <ul>
      <li>Check and share the recap.</li>
      <li>Include people who followed up asynchronously.</li>
      <li>Track the agreed actions.</li>
      <li>Revisit the need for the next meeting.</li>
    </ul>
  </section>
</div>

<section id="measurement" class="part-head">
<span id="part-6" class="article-anchor" aria-hidden="true"></span>
<h2>What to measure</h2>
</section>

<p>Review these signals at team or population level alongside feedback from attendees. Compare the same groups over time and record when meeting practices change. A lower meeting total alone does not show whether decisions, relationships or follow-through improved.</p>

<dl class="article-measures">
  <div><dt>Meeting hours per week</dt><dd>Track the time committed to meetings and where that time goes after a change.</dd></div>
  <div><dt>Large and long meetings</dt><dd>Review the share of meeting hours in sessions with at least nine attendees and lasting at least 60 minutes.</dd></div>
  <div><dt>Multitasking</dt><dd>Look for patterns across meeting series. Use them to investigate meeting design without judging individual attention.</dd></div>
  <div><dt>Available focus hours</dt><dd>Check whether people have uninterrupted time for their work.</dd></div>
  <div><dt>After-hours collaboration</dt><dd>Check whether work is moving outside normal working hours as meeting habits change.</dd></div>
  <div><dt>Late joins and late finishes</dt><dd>Look for scheduling conflicts and meetings that repeatedly overrun.</dd></div>
</dl>

<p>For a more detailed implementation guide, see <a href="{{ site.baseurl }}/articles/meeting-effectiveness-playbook/">The Meeting Effectiveness Playbook</a>.</p>

<aside class="article-next-steps" aria-labelledby="start-this-week">
<h3 id="start-this-week">Start this week</h3>
<ol>
  <li>Review one recurring meeting you organise, including its purpose and attendee list.</li>
  <li>Try a 25- or 50-minute slot and tell attendees what needs to be decided.</li>
  <li>Arrange for a checked recap to reach anyone who only needs the outcome, in line with your organisation's recording and transcription policy.</li>
</ol>
</aside>

<section id="references" class="references">
<h2>References</h2>
<ol>
  <li id="ref-3">Cao, H. et al. (2021). <a href="https://www.microsoft.com/en-us/research/publication/large-scale-analysis-of-multitasking-behavior-during-remote-meetings/">Large Scale Analysis of Multitasking Behavior During Remote Meetings</a>. CHI. Microsoft employee telemetry from February to May 2020 and a 715-person diary study.</li>
  <li id="ref-wti-2022">Microsoft (2022). <a href="https://www.microsoft.com/en-us/worklab/work-trend-index/great-expectations-making-hybrid-work-work">Great Expectations: Making Hybrid Work Work</a>. Work Trend Index, 16 March. The meeting-time statistic comes from aggregated Microsoft 365 activity.</li>
  <li id="ref-11">Microsoft WorkLab. <a href="https://www.microsoft.com/en-us/worklab/ai-data-drop-the-11-by-11-tipping-point">AI Data Drop: The 11-by-11 Tipping Point</a>. Copilot Usage in the Workplace survey, launched September 2023. Respondents included Microsoft employees and early customer users, mostly in North America and Europe.</li>
  <li id="ref-6">Rajkumar, K. et al. (2022). <a href="https://doi.org/10.1126/science.abl4476">A Causal Test of the Strength of Weak Ties</a>. <em>Science</em>. See also the <a href="https://news.mit.edu/2022/weak-ties-linkedin-employment-0915">MIT account of the study</a>.</li>
  <li id="ref-7">Yang, L. et al. (2022). <a href="https://doi.org/10.1038/s41562-021-01196-4">The Effects of Remote Work on Collaboration Among Information Workers</a>. <em>Nature Human Behaviour</em>. See also the <a href="https://www.microsoft.com/en-us/research/publication/the-effects-of-remote-work-on-collaboration-among-information-workers/">authors' research summary</a>.</li>
</ol>
<p class="article-end-note">Case-study labels omit sector and location details. The recap percentage is reported as a broad band. Statistics attributed to public research retain their published values.</p>
</section>
