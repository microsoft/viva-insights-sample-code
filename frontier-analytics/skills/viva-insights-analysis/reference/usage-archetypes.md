# Usage archetypes (experimental)

> **Experimental.** This guidance describes reference rule v1 of the AI usage
> archetype framework. The rule has been applied to real exports in internal
> studies. Results are descriptive only, and the thresholds are experimental.
> They are not validated for prediction, causal inference or intervention use.
> Expect the rule, labels and requirements to change.

Use this reference when a task asks how AI use is distributed over time and
across tools, beyond adoption and volume. The worked examples
`examples/python/06_usage_archetypes.py` and `examples/r/06_usage_archetypes.R`
implement the rule on a synthetic panel.

## Purpose and boundary

Adoption indicators show whether use was observed. Activity totals show how
much was recorded. The archetypes add three descriptive questions:

1. How regularly is AI used?
2. Is use concentrated in one tool group?
3. How much M365 credit is consumed, relative to a stated reference?

An archetype describes recorded behaviour during one observation window. It
carries no information about motive, task fit, work quality, expertise or
individual performance. Use archetypes for aggregate research and workflow
discovery only.

## Relation to the usage ladder

The Power, Habitual, Novice, Low and Non-user ladder is unchanged. See the
[Copilot usage segments](https://microsoft.github.io/viva-insights-sample-code/copilot-usage-segments/)
page and `analysis-conventions.md`. Report the ladder and continuous measures
alongside the archetypes, and keep both. The ladder describes how established
use is. The archetypes describe the pattern of use among people who meet the
analysis eligibility contract. Do not merge the two into a single scheme, and
do not use an archetype to override a ladder segment.

## Reference rule v1

Each dimension assigns one of two labels. Every threshold is tied to its
measurement unit and window, so any change needs a new version label and a
full recomputation of dependent results. Never replace a threshold silently.

### Window and measurement unit

- **Window:** twelve consecutive Sunday-start weeks, matching the weekly
  Person Query.
- **Eligibility:** `Total Copilot enabled days` of 7 in every week of the
  window. Record any other eligibility rule in the context record.
- **Active week:** positive `Total Copilot actions taken`, positive activity
  in any measured M365 tool group, or positive `Total GitHub AI Credits used`.
- **Tool groups:** four groups that the exports observe separately.
  1. Chat and the Copilot app (Copilot Chat and the Microsoft 365 Copilot app).
  2. Copilot in work apps (Excel, Loop, OneNote, Outlook, PowerPoint, Teams, Word).
  3. Cowork (recorded Cowork tasks or active days).
  4. GitHub (positive `Total GitHub AI Credits used`).
- **Group-week:** a tool group used at least once in a week. Each group counts
  at most once per week, however many apps or days it covers.

### The three dimensions

| Dimension | Labels | Rule v1 |
|---|---|---|
| Consistency (C/E) | Consistent / Episodic | Consistent if active in at least 9 of the 12 weeks. Episodic otherwise. |
| Surface mix (S/B) | Specialised / Broad | Specialised if a unique leading group supplies at least 40% of the person's group-weeks and leads the second group by at least five percentage points. Broad otherwise. |
| Metered use (L/I) | Lean / Intensive | Among people with positive M365 credits summed over the window, Lean at or below the stated reference median (ties are Lean). Intensive above it. |

Surface mix uses integers to avoid floating-point edge cases. Let `T` be the
total group-weeks across the four groups, and `c1` and `c2` the largest and
second-largest group counts. Assign Specialised only when:

```text
no other group equals c1
5 * c1 >= 2 * T        # share of at least 40%
20 * (c1 - c2) >= T    # lead of at least five percentage points
```

Metered use rules:

- Use M365 `Total Copilot Credits used`, aggregated from daily facts to
  Sunday-start weeks and summed over the window.
- Calculate the median within the stated full-profile population. The
  cut-point is relative, so Lean and Intensive describe position within one
  population rather than a common level of use.
- Never add GitHub credit amounts to M365 credits. GitHub credits only flag
  the GitHub tool group.
- Credits are neither tokens nor currency. Do not read a high value as cost,
  value or efficiency.
- For comparisons across periods, fix the cut-point in advance so that a
  moving median does not drive the change. Comparisons across populations
  need a common absolute reference with aligned coverage, policy handling
  and windows.

### Profiles

Three binary labels give eight profiles, written as codes such as `C-B-I`
(Consistent, Broad, Intensive). Spell the labels out in reports. A profile is
assigned only when evidence for all three dimensions is complete.

## Surface-day alternative

The earlier candidate rule counted surface-days and required one group to
supply at least 60% of them. It needs daily activity that can be combined
within each tool group, so that a day with Word and Excel use counts once for
work apps. Weekly exports provide per-app active-day totals that cannot be
added safely, because different apps can be used on the same day. Use the
surface-day rule only where daily activity can be combined in this way.
Report it under its own version label, validate it separately, and never mix
its labels with rule v1 results. Do not apply the 60% threshold to weekly
group presence. Weekly presence compresses differences between groups, so a
person who uses three groups every week has a leading share of only one third,
however unevenly their effort is divided.

## Evidence states

Classify everyone into a named state before assigning labels. Never drop a
person silently or force a label.

| State | Meaning | Treatment |
|---|---|---|
| Full profile | Eligible in every week, complete required indicators, positive M365 credits that can be totalled, and at least one group-week | One label on each dimension |
| Confirmed zero M365 credits | Coverage and eligibility confirmed, with no M365 credits recorded | Report as "No M365 metered use", outside the Lean/Intensive split |
| No positive observed credits, coverage unconfirmed | Absent or zero credit facts without confirmation that capture is complete | Report separately. Never relabel as confirmed zero |
| Unresolved credit additivity | Positive credits on overlapping spending-policy rows that may not be safely added | Exclude from primary results. Examine only in labelled sensitivity scenarios |
| Incomplete evidence or eligibility | Missing required indicators, fewer than 7 enabled days in any week, conflicting licence evidence, or no group-weeks | Count and exclude from full profiles |
| Outside the analysis cohort | Not selected by the cohort rule, for example absent from an optional query | Describe in the population accounting |

A coverage confirmation applies to one export only. Unique keys on
`PersonId`, `ServiceId`, `SpendingPolicyId` and `MetricDate` do not show that
amounts across spending policies are disjoint. Confirm additivity separately.
See `copilot-query-contracts.md` for keys, joins and coverage.

## Required context record

Rule v1 is interpretable only with its context. Record, for every application:

- rule version label
- observation window and week boundaries
- cohort rule, and how it differs from any adoption reporting population
- tool groups with positive observations, and the capture assumption for each
- credit reference population and the median used as the cut-point
- coverage confirmation and its source, per export
- policy-row additivity decision and its evidence
- treatment of ties, missing values and each evidence state

Report a tool group with no positive observations as such. Its absence may
reflect capture, access or genuine non-use. Do not drop the group or
renormalise shares, because that changes what Surface mix measures.

## Implementation steps

1. Select the cohort and confirm eligibility in every week of the window.
2. Verify native keys and confirm that credit rows can be added. Aggregate
   daily credit facts to Sunday-start weeks, keeping M365 and GitHub separate.
   Never expand weekly Person rows into days.
3. Flag activity in each of the four tool groups for each person-week.
4. Count active weeks. Assign Consistent at nine or more, Episodic otherwise.
5. Count group-weeks per group and apply the integer Surface mix rule.
6. Sum positive M365 credits, calculate the median in the full-profile
   population, and assign Lean at or below it.
7. Record a named evidence state for everyone without a full profile.
8. Run the sensitivity diagnostics below.
9. Apply the publication rules and joint disclosure check before release.

## Sensitivity diagnostics

Report these routinely. They are diagnostics of label stability, rather than
observed rates of change.

- **Leave-one-week:** drop each week in turn, recompute Surface mix and record
  whether the label changes in at least one recalculation. Report the share
  changed overall and the share changed among Specialised people. The overall
  figure can look reassuring because most people hold the majority label.
- **First versus last six weeks:** compute Surface mix in each half and
  report disagreement among people with group-weeks in both halves.
- **Rule variants:** compare candidate thresholds without outcome data, and
  report each under its own label.

Treat a Specialised label as a description of a period. Consistency can show a
ceiling in established, positive-credit populations. Do not move its threshold
simply to balance the categories.

## Classification versus publication

Classification and publication are separate steps. The eight profiles are
mutually exclusive and exhaustive within the classified population, but that
does not make every cell reportable.

- Use a minimum group size of 10 distinct people for any published cell, or a
  stricter organisational requirement. This is stricter than the package
  default of 5.
- Check related tables, filters, periods and rule variants together. Exact
  margins can reveal a withheld cell by subtraction.
- Withholding conveys information, for example that a count lies below 10.
  State this in the disclosure audit.
- Do not change thresholds, categories or the window solely to avoid
  withholding a cell.
- Never publish person-level labels, identifiers or rankings in tables,
  hidden data or logs.

Small populations may not support the full eight-profile breakdown by an
organisational attribute. Report a coarser split, such as one dimension at a
time, when the full breakdown would require heavy withholding.

## Interpretation and governance

- Labels describe a period. They are not performance ratings.
- Consistent does not establish diligence. Episodic may reflect leave,
  project cycles or incomplete telemetry.
- Broad describes spread across recorded tool groups. Specialised describes a
  leading group rather than expertise. Intensive describes relative M365
  credit consumption rather than value or effort.
- Do not use archetypes for individual ranking, performance ratings,
  disciplinary decisions or punitive monitoring.
- A profile cannot show that someone needs support. Identify a need
  independently, choose a testable response and evaluate it with work
  quality, time and consumption.
- Agree purpose, access, retention, notice and consent with the relevant
  governance owners before each use.
- Describe differences between organisational groups before any outcome
  comparison. Such differences raise questions about tasks, access and
  workflow, without showing tool preference or a deficit.
- Describe same-period relationships with outcomes as associations. See
  `analysis-conventions.md`.

## Validation status

Applied to real exports in internal studies. Descriptive only.

Not yet established:

- predictive performance on a later period with fixed definitions and
  cut-points
- information beyond the existing ladder, continuous measures and
  organisational context
- causal effects, productivity, motives or task fit
- construct validity of the labels with intended users
- a general adapter for every tenant and query version
- coding-tool use beyond metered GitHub credits, which would need the
  dedicated GitHub activity query

If a dimension cannot be measured reliably for a population, omit it rather
than substituting a proxy that changes its meaning. If the archetypes add no
useful information for a defined decision, the ladder and continuous measures
remain sufficient.
