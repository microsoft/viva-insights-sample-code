# Output Spec: Executive Summary Memo

This specification defines the structure, tone, and content expectations for an **executive summary memo** generated from Viva Insights data. Use this spec as context for coding agents when you want them to produce a concise, actionable document for VP/C-suite audiences.

---

## Target format

| Property | Requirement |
|----------|-------------|
| **File type** | Markdown (`.md`) or self-contained HTML (`.html`) |
| **Length** | 1–3 pages when rendered (approximately 500–1,500 words) |
| **Audience** | VP, C-suite, senior leadership — assume limited time and no technical background |
| **Tone** | Professional, data-driven, actionable. Lead with findings, not methodology. |

---

## Recommended structure

### 1. Title and metadata

```
# Copilot Adoption: Executive Summary
**Period:** September 2 – November 24, 2024
**Prepared by:** People Analytics Team
**Date:** December 1, 2024
```

### 2. Executive summary paragraph

A single paragraph (3–5 sentences) that answers: **"What is the headline?"**

This paragraph should:
- State the overall adoption rate and trend direction
- Highlight the single most important finding
- Mention one recommendation
- Be self-sufficient — a reader who only reads this paragraph should get the key message

**Example:**

> Copilot adoption reached 68% of licensed users in the latest reporting week, up 5 percentage points from the prior month. Engineering and Product teams lead adoption at 78%, while Sales and Marketing trail at 45%. The strongest predictor of sustained usage is manager-level adoption — teams with active managers show 2.3× higher individual adoption rates. We recommend prioritizing manager enablement programs in underperforming segments.

### 3. Key metrics table

A compact table with 5–8 headline numbers:

| Metric | Value | Trend |
|--------|-------|-------|
| Adoption rate (latest week) | 68% | ↑ 5pp vs. prior 4 wks |
| Total licensed users | 3,450 | — |
| Total active users (latest week) | 2,340 | ↑ 12% vs. prior 4 wks |
| Avg Copilot Actions per active user/week | 34.2 | ↑ 8% vs. prior 4 wks |
| Avg Copilot Assisted Hours per active user/week | 2.8 hrs | ↑ 0.4 hrs vs. prior 4 wks |
| Top-performing segment | Engineering (78%) | — |
| Lowest-performing segment | Sales (42%) | ↓ 3pp vs. prior 4 wks |

### 4. Key findings (3–5 bullets)

Each finding should be one bullet with:
- A bolded headline statement
- 1–2 supporting sentences with specific numbers
- An implication or "so what"

**Example:**

- **Manager adoption drives team adoption.** Teams where the manager is an active Copilot user have a 78% individual adoption rate, compared to 34% in teams with inactive managers. Manager enablement is the highest-leverage intervention.
- **Usage depth is increasing, not just breadth.** Average Copilot Actions per active user rose from 28 to 34 over the past 8 weeks, indicating that users who adopt Copilot are finding more ways to use it.
- **Sales and Marketing segments are underperforming.** Adoption rates of 42% and 48% respectively lag the company average by 20+ percentage points. Interviews suggest these teams lack role-specific Copilot use cases.
- **Churn is low but concentrated.** 8% of previously active users became inactive in the last 4 weeks, with 60% of churned users in junior IC roles. Targeted re-engagement for these users may be warranted.

### 5. Trends section

A brief (2–3 sentence) description of how metrics have changed over the reporting period. Optionally include a single embedded chart (adoption rate over time).

**Example:**

> Adoption has grown steadily from 52% to 68% over the 12-week reporting period, with the steepest gains in the first 6 weeks as initial license rollout completed. Growth has moderated in recent weeks, suggesting the organization is approaching natural adoption saturation among early adopters. The next phase of growth will likely require targeted enablement for lagging segments.

### 6. Recommendations (2–4 bullets)

Each recommendation should be:
- Specific and actionable (not vague advice)
- Connected to a finding above
- Assigned to an owner or function if possible

**Example:**

- **Launch a manager enablement program** (Owner: L&D). Provide managers with Copilot quick-start guides and use-case examples relevant to their function. Target: cover 80% of managers in underperforming segments by end of Q1.
- **Develop role-specific Copilot playbooks for Sales and Marketing** (Owner: People Analytics + Sales Ops). Create curated prompt libraries and success stories demonstrating Copilot value for customer-facing workflows.
- **Implement quarterly Copilot usage reviews** (Owner: IT + People Analytics). Share this dashboard quarterly with business unit leaders to maintain visibility and accountability.

### 7. Methodology note

A brief (2–4 sentence) note for transparency:

> This analysis is based on Viva Insights person query data at person-week granularity, covering 3,450 licensed Copilot users over 12 weeks. "Active" is defined as having at least one Copilot action in a given week. Segments with fewer than 10 users are suppressed for privacy. Copilot metric definitions follow the standard Viva Insights measurement methodology.

### 8. Data coverage note

State any limitations or caveats:

> Data covers employees in the following regions: Americas, EMEA. APAC employees are excluded due to delayed license rollout. Contractors and vendor staff are not included. Copilot metrics for unlicensed users appear as blank (null) and are excluded from adoption calculations.

---

## What to include vs. exclude

| Include | Exclude |
|---------|---------|
| Headline adoption numbers | Detailed statistical methodology |
| Trend direction and magnitude | Raw data tables with hundreds of rows |
| Top 3–5 findings with supporting data | Exploratory analysis or "interesting but inconclusive" patterns |
| Specific, actionable recommendations | Vague suggestions ("continue monitoring") |
| Segment comparisons (high vs. low performers) | Individual user-level data |
| One well-chosen chart (optional) | Multiple charts that require interpretation |
| Methodology transparency (brief) | Code, SQL queries, or technical implementation details |
| Data limitations and caveats | Speculative conclusions without data support |

---

## Tone guidelines

- **Lead with "so what," not "how."** Executives want conclusions and actions, not process descriptions.
- **Use plain language.** Avoid jargon like "person-weeks," "panel data," or "p-values." Translate to business terms.
- **Be precise with numbers.** "68% adoption" is better than "most users adopted Copilot." Always include comparison baselines.
- **Be honest about uncertainty.** If a trend is unclear, say so. Don't overstate conclusions.
- **Use active voice.** "Engineering adopted Copilot at the highest rate" not "The highest adoption rate was observed in Engineering."

---

## Example outline with placeholders

```markdown
# Copilot Adoption: Executive Summary

**Period:** [START_DATE] – [END_DATE]
**Prepared by:** [TEAM_NAME]
**Date:** [GENERATION_DATE]

## Summary

[2-4 sentences: headline adoption rate, trend, top finding, key recommendation]

## Key Metrics

| Metric | Value | Trend |
|--------|-------|-------|
| Adoption rate | [X]% | [↑/↓] [N]pp vs. prior [period] |
| Licensed users | [N] | — |
| Active users | [N] | [↑/↓] [N]% vs. prior [period] |
| Avg actions/user/week | [N] | [↑/↓] [N]% vs. prior [period] |

## Findings

- **[Finding 1 headline].** [1-2 supporting sentences with numbers.]
- **[Finding 2 headline].** [1-2 supporting sentences with numbers.]
- **[Finding 3 headline].** [1-2 supporting sentences with numbers.]

## Trends

[2-3 sentences on how metrics evolved. Optional chart.]

## Recommendations

- **[Recommendation 1].** [Specific action, owner, timeline.]
- **[Recommendation 2].** [Specific action, owner, timeline.]

## Methodology

[2-3 sentences on data source, definitions, and privacy thresholds.]

## Data Coverage

[1-2 sentences on scope, exclusions, and limitations.]
```

---

## How to instruct a coding agent

When asking a coding agent to produce this output, include context from this spec. Here is a sample instruction snippet:

```
OUTPUT FORMAT INSTRUCTIONS:
Produce an executive summary memo in Markdown format. Target length: 1-3 pages (500-1500 words).

Structure:
1. Title with period dates and preparation date
2. Executive summary paragraph (3-5 sentences with the headline finding)
3. Key metrics table (5-8 numbers with trend indicators)
4. Key findings (3-5 bullets, each with a bold headline and supporting data)
5. Brief trends section (2-3 sentences)
6. Recommendations (2-4 specific, actionable bullets with owners)
7. Methodology note (2-3 sentences)
8. Data coverage note (1-2 sentences on scope and limitations)

Tone: professional, data-driven, actionable. Write for a VP/C-suite audience with no
technical background. Lead with findings, not methodology. Use plain language.
Save as "copilot_executive_summary_YYYYMMDD.md".
```
