# Analysis conventions

How to turn Viva Insights outputs into honest, defensible findings. These
conventions matter more than any single chart: workplace-analytics results are
easy to over-claim.

Apply the [shared analysis policy](analysis-policy.md) first.

## 1. Prefer association over causation

Viva Insights data is observational. Unless a designed experiment or a valid
quasi-experimental design supports a causal claim, describe relationships as
associations.

| Avoid | Prefer |
|---|---|
| "X causes Y" | "X is associated with Y" / "X is the strongest observable predictor of Y" |
| "X drove the change" | "X coincides with the change. Worth investigating." |
| "Caused by a methodology update" | "Cause not confirmed. Possible explanations include ..." |

For predictive-model outputs (Information Value, odds ratios, feature
importance), report predictive strength and direction, and state explicitly that
predictive association is not proof of mechanism.

## 2. Label anything simulated or injected

If an example seeds data, injects an effect, or uses a simulator
(`p2p_data_sim()` and similar), say so prominently and give the reader the switch
to turn it off ("set effect to 0 / replace with real data"). Never let a
demonstrated effect read as an empirical finding.

## 3. Copilot usage segmentation

`identify_usage_segments()` classifies a licensed population by rolling usage.
Two rules prevent the most common misclassification:

1. **Resolve missingness before calling it.** Fill only absent activity rows
   within a supplied complete query's configured scope with zero. Native
   licensed-only M365 Copilot nulls mean unlicensed. Failed joins, out-of-scope
   periods and known coverage failures remain unresolved.
   A rolling window can propagate unknown values; report affected segments as
   unavailable rather than replacing those values to force classification.
2. **Include licensed people with zero activity.** Prefer Copilot Enabled User
   or Total Copilot enabled days for the period. Native null enabled days mean
   unlicensed. Without those fields, a verified native licensed-only metric
   can establish licensing through its documented null/zero semantics.
   Filtering to positive actions would exclude licensed Non-users.

Segment definitions (12-week version, `power_thres` default 15):

| Segment | Condition |
|---|---|
| Power User | habitual AND 12w rolling mean >= power threshold (default 15) |
| Habitual User | habitual AND 12w rolling mean < power threshold |
| Novice User | 12w rolling mean >= 1 (not habitual) |
| Low User | 12w rolling mean > 0 and < 1 |
| Non-user | 12w rolling mean == 0 |

```python
# df is a deliberately selected continuously licensed cohort with a full weekly calendar.
enabled = df["Total_Copilot_enabled_days"].astype("Float64")
if not enabled.gt(0).fillna(False).all():
    raise ValueError("This minimal recipe requires eligibility throughout the window")
if "licence_source_matched" in df and not df["licence_source_matched"].fillna(False).all():
    raise ValueError("Unresolved licence join: affected rolling segments are unavailable")
df_lic = df.copy()
omitted = df_lic["activity_state"].eq("omitted_in_scope")
df_lic.loc[omitted, "Total_Copilot_actions_taken"] = 0
# Do not execute segmentation until its full required history is known.
if df_lic["Total_Copilot_actions_taken"].isna().any():
    raise ValueError("Unresolved activity: affected rolling segments are unavailable")
seg = vi.identify_usage_segments(df_lic, metric="Total_Copilot_actions_taken",
                                 version="12w", return_type="data")
```

`activity_state` is a column you create, because no export provides it. Build it
while aligning each person to the full configured calendar. Use
`"omitted_in_scope"` for person-periods the supplied complete query left out
within its configured scope, `"observed"` for rows with a recorded value, and a
specific reason such as `"native_unlicensed"`, `"unexpected_null"`, `"source_unavailable"`,
`"out_of_scope"` or `"known_coverage_failure"` for everything else.
This minimal recipe requires an explicitly selected continuously licensed cohort.
For a mixed population, use the paired Python/R segmentation examples, which keep
unlicensed weeks in the calendar and withhold only affected rolling segments.
They do not abort because another person has native unlicensed nulls. Never drop
unlicensed or unknown weeks and treat the remaining rows as consecutive history.

If segments come back all Non-user or all None, check: metric renamed correctly?
missingness resolved? correct population filter and required calendar history applied?

## 4. Period framing as a pattern rather than fixed dates

Before/After comparisons are common (a programme launch, a policy change). Treat
the windows as a **parameter of the analysis**, chosen from the data and the
intervention date rather than as fixed constants.

- Use windows of equal length on each side of the intervention.
- Prefer **matched calendar periods** (same quarter year over year) to reduce
  seasonal differences. Holidays, weekday mix, scope and coverage can still differ.
- State the exact window boundaries in the output so a reader can reproduce them.
- Exclude or flag holiday / inactive weeks inside the windows (see
  `data-pitfalls.md`).

For a formal interrupted-time-series question, assess intervention timing,
pre-trend support, seasonality and serial dependence before choosing a model.
Inspect `create_itsa()` (R) and its explicit period arguments where appropriate.
Its availability does not establish that the design identifies an effect.

### Comparable reporting and refreshes

Keep the source endpoint, verified complete-month cutoff, complete-week cutoff
and process-cycle window separate. Advance every dependent chart and table
when the approved window advances, or label an intentionally different window.
Observed dates support diagnostics but do not prove complete capture.

When a refresh changes a headline, reconcile source revisions, scope revisions,
time-window changes and population/attribute changes on a common envelope.
Compare exact internal event sets before aggregating. An ordered bridge must
reconcile without counting an overlapping change twice. Preserve the previous
approved specification and evidence instead of rewriting them to make a new
version match. See [Meeting Query](meeting-query.md) and
[cross-query diagnostics](cross-query-diagnostics.md).

## 5. Respect privacy in every reported cut

- Use `max(organisation requirement, workflow floor)` for `mingroup`: generic
  5, joined-query/dashboard/causal 10 distinct people.
- Do not publish a group smaller than the threshold. Aggregate up instead.
- Keep individual identifiers and rankings out of aggregate publications.
  Anonymising or pseudonymising rows is not sufficient disclosure protection.
- When sharing row-level data with other analysts, still pseudonymise
  identifiers first (`anonymise()` in R).

## 6. Units and wording precision

- Hours columns are in **hours** (meeting/collaboration/email/attendee hours).
  A raw `Duration` column, where present, is also in hours.
- Prefer explicit, checkable units in text (for example "FTE-days assuming an
  8-hour day" rather than an ambiguous "FTE-months").
- When a shift is unexplained, say so plainly and list candidate explanations
  rather than asserting one.

## 7. House writing style

- Avoid em-dashes and semicolons. Prefer short sentences, commas, parentheses, or conjunctions.
- Avoid contrastive marketing phrasing (for example "simple, not simplistic"). State what something is directly.
- Lead with the finding, then the caveat, then the method.
- Every number in prose should be traceable to a cut in the code.

## 8. Separate volume, intensity and composition

For the same included meetings with observed attendee hours:
`total attendee hours = meeting count * mean attendee hours per meeting`.
Use this exact identity to separate meeting volume from time investment per
event. A change decomposition must state how it allocates the interaction.
Do not substitute mean duration multiplied by mean attendance: covariance,
attendance time and scheduled duration can make that product differ.

State whether a quality measure is a mean of per-meeting ratios or a ratio of
sums. A person-level mean answers another question. Keep total hours, hours per
meeting and hours per person separate, with their own valid bases.

For rates, distinguish percentage changes from percentage-point changes.
A share falling from 2.0% to 1.8% falls 0.2 percentage points and 10% relative
to its original level. Show the numerator and denominator when their changes
could make the share misleading.

Report effect size and uncertainty before a significance label. Agree practical
reporting yardsticks rather than inventing a success target. If inference is
requested, choose its independent units, temporal dependence assumptions and
test family before comparing p-values. Repeated meetings do not automatically
provide independent observations. Distinguish pointwise intervals from
multiple-testing-adjusted decisions. Non-significance does not establish
equivalence, and a significant change does not establish intervention impact.
