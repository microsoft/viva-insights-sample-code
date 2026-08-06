# Analysis conventions

How to turn Viva Insights outputs into honest, defensible findings. These
conventions matter more than any single chart: workplace-analytics results are
easy to over-claim.

## 1. Association, not causation

Viva Insights data is observational. Unless a designed experiment or a valid
quasi-experimental design supports a causal claim, describe relationships as
associations.

| Avoid | Prefer |
|---|---|
| "X causes Y" | "X is associated with Y" / "X is the strongest observable predictor of Y" |
| "X drove the change" | "X coincides with the change; worth investigating" |
| "Caused by a methodology update" | "Cause not confirmed; possible explanations include ..." |

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

1. **Fill missing metric values with 0 before calling it.** A rolling window
   propagates NaN and leaves people unclassified otherwise.
2. **Define the licensed population by enabled days, not by action presence.**
   Use an enabled-days-positive filter (for example `Copilot_enabled_days > 0`)
   as the denominator for a target-week snapshot. Counting only rows where an
   action metric is non-null understates the Non-user group.

Segment definitions (12-week version, `power_thres` default 15):

| Segment | Condition |
|---|---|
| Power User | habitual AND 12w rolling mean >= power threshold (default 15) |
| Habitual User | habitual AND 12w rolling mean < power threshold |
| Novice User | 12w rolling mean >= 1 (not habitual) |
| Low User | 12w rolling mean > 0 and < 1 |
| Non-user | 12w rolling mean == 0 |

```python
df_lic = df[df["Copilot_enabled_days"].fillna(0) > 0].copy()
df_lic["Total_Copilot_actions_taken"] = df_lic["Total_Copilot_actions_taken"].fillna(0)
seg = vi.identify_usage_segments(df_lic, metric="Total_Copilot_actions_taken",
                                 version="12w", return_type="data")
```

If segments come back all Non-user or all None, check: metric renamed correctly?
NaNs filled? correct population filter applied?

## 4. Period framing (pattern, not fixed dates)

Before/After comparisons are common (a programme launch, a policy change). Treat
the windows as a **parameter of the analysis**, chosen from the data and the
intervention date, not as fixed constants.

- Use windows of equal length on each side of the intervention.
- Prefer **matched calendar periods** (same quarter year over year) so
  seasonality and holidays cancel.
- State the exact window boundaries in the output so a reader can reproduce them.
- Exclude or flag holiday / inactive weeks inside the windows (see
  `data-pitfalls.md`).

For a formal interrupted-time-series test, use `create_itsa()` (R) with explicit
`before_start / before_end / after_start / after_end` arguments rather than
eyeballing a line chart.

## 5. Respect privacy in every reported cut

- Keep `mingroup` at or above the organisation's agreed threshold (default 5).
- Do not publish a group smaller than the threshold; aggregate up instead.
- Anonymise identifiers in any shared artefact (`anonymise()` in R).

## 6. Units and wording precision

- Hours columns are in **hours** (meeting/collaboration/email/attendee hours).
  A raw `Duration` column, where present, is also in hours.
- Prefer explicit, checkable units in text (for example "FTE-days assuming an
  8-hour day" rather than an ambiguous "FTE-months").
- When a shift is unexplained, say so plainly and list candidate explanations
  rather than asserting one.

## 7. House writing style

- Minimise em-dashes; prefer commas, parentheses, or conjunctions.
- Lead with the finding, then the caveat, then the method.
- Every number in prose should be traceable to a cut in the code.
