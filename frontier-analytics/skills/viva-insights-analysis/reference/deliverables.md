# Common deliverables

Condensed guidance for the standard outputs analysts most often ask for. This
distills the same conventions used in the Frontier Analytics prompt cards, so a
Skill-driven session produces comparable quality without a prompt being pasted.
Adapt every number, column name, and threshold to the data in front of you.

For an exact, reproducible, tested structure, prefer the matching prompt card
under `frontier-analytics/prompts/` instead of reinventing the deliverable from
scratch. Use this file when there is no prompt card for the task, or when the
user wants a faster first pass.

## Licensed, active, and unlicensed (used by every Copilot deliverable below)

- **Licensed:** use `Total_Copilot_enabled_days > 0` when that column exists. Otherwise, treat a person-week as licensed if at least one Copilot metric is non-null and greater than zero.
- **Active:** licensed and the primary activity metric (commonly `Total_Copilot_actions_taken`) is greater than zero.
- **Unlicensed:** `Total_Copilot_enabled_days == 0` when available. Otherwise, all Copilot metric values are null or zero.

Treat missing Copilot values as unlicensed rather than as zero usage, unless the
data clearly indicates otherwise.

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
- Use percentile-based thresholds so segment boundaries adapt to the data
  instead of relying on fixed absolute cutoffs.

## Shared requirements across all of these

- Suppress any reported group under the organization's minimum aggregation
  threshold (default 5).
- State assumptions and caveats prominently in the output itself rather than
  only in a footnote.
- When a Frontier Analytics prompt card exists for the deliverable, mention it
  to the user as the tested, exact-structure alternative.
