# Common deliverables

Condensed guidance for the standard outputs analysts most often ask for. This
distills the same conventions used in the Frontier Analytics prompt cards, so a
Skill-driven session produces comparable quality without a prompt being pasted.
Adapt every number, column name, and threshold to the data in front of you.

Read [analysis policy](analysis-policy.md) and [workflow support](workflow-support.md)
before choosing a path. Prompt cards guide code generation; their validation
metadata does not establish that a new report or every agent has been tested.
Use [report verification](report-verification.md) for a claim-level evidence
ledger, compact comparison layouts, numerical/privacy/visual checks and safe
revisions of an existing user-edited report.

For a prescribed, reproducible structure, prefer the matching prompt card
under `frontier-analytics/prompts/` instead of reinventing the deliverable from
scratch. Use this file when there is no prompt card for the task, or when the
user wants a faster first pass.

## Licensed, active, and unlicensed (used by every Copilot deliverable below)

- **Licensed:** prefer Copilot Enabled User, positive Total Copilot enabled days
  or dated licence records. A verified native licensed-only M365 Copilot metric
  with documented null/zero semantics is a valid fallback.
- **Active:** positive observed activity in the selected, verified metric.
  Report adoption rates only when the eligible/licensed denominator is known.
- **Unlicensed:** native licensed-only M365 Copilot nulls, native null or zero
  enabled days, or explicit negative licence evidence.
- **Unknown:** retain unresolved eligibility and incomplete coverage separately.

Do not extend licensed-only M365 Copilot semantics to unlicensed Copilot Chat,
Consumption credits or arbitrary columns containing "Copilot".
For current cross-query integration, follow
[Joining Viva Insights queries](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/).
Recommend weekly exports and assume supplied Viva Insights queries are complete
within their configured population, date window and product filters. Omitted
inactive activity rows contribute zero without separate completeness evidence.
Keep failed joins, unavailable queries, out-of-scope periods and known export
problems separate from native unlicensed nulls. Use product-specific licence semantics for adoption
denominators. [copilot-query-contracts.md](copilot-query-contracts.md) retains
historical layouts and analytical context, rather than overriding the current guide.

## Copilot adoption dashboard

- Build a self-contained output (static HTML, or a rendered notebook) with a
  weekly adoption-rate trend, a usage-intensity trend, and segment breakdowns by
  HR attribute.
- Compute every weekly figure using distinct `PersonId` counts rather than row
  counts.
- Avoid anything that needs a running server. Use static images or inline
  charts.
- Suppress any segment under the organization's privacy threshold.

## Executive summary memo

- The audience is senior leadership. Keep it to roughly one or two pages, lead
  with the headline finding, and avoid jargon.
- Compute every number directly from the data rather than estimating or
  fabricating a plausible-looking figure.
- State the adoption trend using a recent window compared against the prior
  window of equal length, and say plainly when the available history is too
  short for the preferred window.
- Include a short methodology note covering the data source, the licensed and
  active definitions used, and an association-over-causation caveat.

## ROI estimation

- Define the hourly rate and license cost as adjustable variables at the top of
  the analysis instead of hard-coding them, since both vary by organization.
- Annualize from a recent multi-week average instead of a single week.
- Report a break-even threshold: the minimum usage needed for the license cost
  to be recovered.
- Any licensed-vs-unlicensed comparison is observational. Present it with an
  explicit caveat about selection effects, since more engaged employees may
  simply be more likely to adopt Copilot first.
- Assisted hours are not measured time saved. Require an explicit, adjustable
  realisation factor and any proxy conversion before estimating value; keep
  credits, activity, potential time value and realised business benefit separate.
- Include a small sensitivity analysis across a few hourly-rate and adoption
  scenarios.
- Report negative or uncertain ROI honestly, with the assumptions that produced
  it, instead of smoothing it over.

## Usage segmentation

- Prefer `identify_usage_segments()` over hand-rolled thresholds, since it
  accounts for both the volume and the consistency of usage over time.
- Define the licensed population by enabled days for the period rather than by
  whether a person happened to take an action, to avoid conflating "not
  licensed" with "licensed but idle."
- Use the package's documented segment definitions and required windows.
  Do not substitute arbitrary percentiles or credits/tokens for action counts.

## Shared requirements across all of these

- Suppress groups below `max(organisation requirement, workflow floor)`: generic
  5; joined-query, dashboard, causal and process 10 distinct people. Protect complementary
  cells and never publish individual rankings.
- State assumptions and caveats prominently in the output itself rather than
  only in a footnote.
- Keep each view's source, exact dates, aggregation, valid bases and denominator
  traceable to shared analytical evidence. Retain supporting detail when the
  narrative is condensed, and keep unavailable values distinct from observed zero.
- For an existing publication, snapshot the live version, preserve unrelated
  edits and resolve only the comments actually addressed. Verify the persisted
  result after an authorised replacement.
- When a Frontier Analytics prompt card exists for the deliverable, mention it
  to the user as the prescribed-structure alternative requiring verification.
