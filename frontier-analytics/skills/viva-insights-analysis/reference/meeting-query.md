# Meeting Query specifics

Read [shared analysis policy](analysis-policy.md) and
[query schemas](query-schemas.md) first. This reference adds Meeting Query (MQ)
interpretation and an optional overlapping-export reconstruction. It does not
change the current Person, Consumption, GitHub or Agent join contract.

## 1. Grain and organiser attribution

| | Person Query | Meeting Query |
|---|---|---|
| One row | One person in a configured period | One calendar item |
| Typical key | `PersonId` + `MetricDate` | `MeetingId`, after uniqueness checks |
| Population | Configured people and periods | Meetings selected by query conditions |
| Organisational attributes | The person | The organiser |
| Useful unit | Hours per person-period | Attendee hours per meeting |

Confirm population conditions and metric rules from the query configuration.
Do not assume every organiser is measured or has usable HR attributes. A raw
calendar item can be a personal hold, all-day marker or cancelled item. A
meeting-row average is not an average per person, and meeting count alone
cannot establish meetings per employee.

An organisational cut describes the organiser's **recorded reporting line**,
not verified process ownership, attendee composition or attendee-side workload.
Write "meetings organised by people recorded in Function X total 4,000 attendee
hours". People in that function also attend meetings organised elsewhere.
Establish whether the attributes are historical or export-time labels. A
leadership succession is not automatically an alias or a behavioural change.
Keep exact source keys for joins and a separate display-name mapping.

Report attribution coverage for each grouping field, including the share of
known attendee hours with a usable attribute and any missing hours. Define blank
tokens explicitly. Do not silently drop unattributed rows or infer their cause.
Attendee composition requires a suitable attendee-side source. A Person Query
can describe person-level meeting load, but cannot alone recover each room's
membership.

## 2. Quality filters and attrition

The base filter in `query-schemas.md` excludes solo items, intended-participant
counts outside 2 to 150, cancelled items and all-day items. Where `Duration` is
available in hours, the worked convention also requires 1/60 through 8 hours,
inclusive. These thresholds are analytical choices, not product definitions.

Parse boolean text explicitly before filtering. The text `"False"` is truthy
under Python `astype(bool)`. Reject unknown tokens rather than treating them as
false. Missing quality inputs fail the example filter but stay in raw data.

The bundled [quality helpers](../examples/meeting_quality.py) and
[matched R helper](../examples/meeting_quality.R) implement this seven-rule
example. Their flag parser accepts true/false, yes/no, y/n and 1/0 variants.
Unknown nonmissing tokens stop execution. Missing flags remain unknown and fail
the mask. Reconcile R/Python results when regex or data backends differ.

Report both per-rule flagged counts and a sequential retained-row funnel.
Rules overlap, so their independent attrition counts must not be added.
Report missing inputs separately. Disclosure rules also apply to these counts.
Keep reviewed subject exclusions in one shared project configuration, and
record their additional effect after the structural filter.

## 3. Attendee hours and duration

`Attendee_meeting_hours` is the sum of hours attendees spent in a meeting.
Use it to describe time investment. It is not measured financial cost, time
saved or productivity. `Duration`, where supplied as in the package samples,
is in hours. Multiply by 60 for minutes.

Prefer the exported attendee-hours metric. Do not replace it with
`Number_of_attendees * Duration`. Scheduled duration and time actually spent
can differ. In the Python 0.4.3 synthetic sample, eight rows differ from that
product by more than `1e-8` hours. The maximum absolute difference is 2 hours in
both the Python 0.4.3 and R 0.7.2 samples.

Validate numeric units, missingness and nonnegative values before summing.
Any conversion to FTE-days needs an explicit assumed working day, for example
`attendee_hours / 8` under an 8-hour-day scenario. It does not measure capacity
that can actually be recovered. See [analysis conventions](analysis-conventions.md).

## 4. Behavioural counts and weighting

Use the installed sample columns and the export's metric definitions, not
guessed aliases. Fields depend on what the analyst selected.

| Imported column | Possible derived measure |
|---|---|
| `Number_of_attendees_multitasking` | Fraction of attendees flagged as multitasking |
| `Number_of_attendees_who_didn_t_join_the_meeting_on_time` | Fraction joining late |
| `Number_of_attendees_who_didn_t_end_the_meeting_on_time` | Fraction leaving late |
| `Number_of_redundant_attendees` | Fraction flagged redundant |
| `Number_of_emails_sent_during_the_meeting` | Emails per attendee, not a fraction |
| `Number_of_chats_sent_during_the_meeting` | Chats per attendee, not a fraction |

Divide by a known positive `Number_of_attendees`. Leave missing or invalid
denominators unresolved rather than clipping them to 1. Check whether each
numerator and denominator covers the same population. Investigate fractions
outside 0 to 1, but do not impose that bound on message counts per attendee.

Multitasking counts attendees who sent or read emails or Teams chats during a
meeting. Microsoft's definition excludes messages in the meeting/call chat.
It is a binary attendee flag, not a measure of how long someone multitasked or
proof that a meeting was ineffective. Late-join and late-end metrics follow
their documented timing rules and are not direct measures of meeting quality.

A mean of per-meeting fractions weights each meeting equally. A ratio of summed
counts weights by attendees. Neither is a distinct-person prevalence measure,
because a person can attend several meetings. State which estimand you use.
A majority threshold is another analytical choice, not a product standard.

`Accept_count`, `Decline_count`, `Tentatively_accepted_count` and
`No_response_count` describe responses, not actual attendance. `Recurring`
supports a standing-cadence versus ad hoc comparison.

## 5. Subject handling and privacy

Subjects can be suppressed by organisational settings. Do not require them for
structural analysis or attempt to recover suppressed text. Treat available
subjects as sensitive free text and keep raw examples out of published outputs.

For an approved topical analysis, retain the original locally and create a
separate normalised field. Review whitespace, case, accents, mail prefixes
(`RE:`, `FW:` and local equivalents) and client status decorations. Prefixes
can stack, so remove reviewed prefixes repeatedly until stable. Do not strip
everything after `@` or a date-like token without checking that it is decoration.
Those tokens can be meaningful subject content.

Anchor a series-matching expression when exact forum membership is intended.
A subject that merely mentions a forum can be preparation rather than the
forum itself. Keyword coverage is language-specific, so investigate regional
coverage before interpreting a thin topical cut. Travel-like words can be
business terms. Review matches locally and record deliberate keeps.

R 0.7.2 exports `tm_clean`, `tm_freq`, `tm_cooc`, `tm_wordcloud` and
`meeting_tm_report`. Inspect installed signatures and expected input before
use. Python 0.4.3 has no corresponding `tm_*` exports.
See [package inventory](packages.md) for discovery.

## 6. Partial months and coverage

Use saved query bounds, refresh completion and known ingestion problems to
establish temporal coverage. The maximum observed meeting timestamp is a
diagnostic, not proof of completeness. A complete month can have no meetings
on its final day, and an export with a meeting on that day can still have gaps.
Do not infer coverage from a filename or portal label alone.

Check the full raw export before topical or quality filtering. Persist the
reviewed complete window so sparse subsets cannot silently move the cutoff.
Check both leading and trailing partial periods. Exclude incomplete periods
from complete-month comparisons, or label them explicitly if partial periods
are the question. If configuration is unavailable, state that completeness is
unresolved rather than inventing a cutoff.

Keep coverage and low volume as separate flags. A fully covered holiday month
may have low volume. Annotate it rather than treating it as missing data.

## 7. Optional reconstruction of overlapping exports

Compare vintages separately first. Attribute availability, attendance metrics,
subjects and quality-filter membership can change for shared IDs. A newer
export is not necessarily a superset. Identical event membership does not
establish identical hours. Do not merge exports merely to validate a rerun.

For an explicitly agreed historical reconstruction, one possible rule is:

1. Confirm comparable tenant/key namespaces, settings, units, schema and
   timestamps. Require canonical, unique, nonblank `MeetingId` values per input.
2. Preserve the full raw ID union, including quality failures. Filter only
   afterward for analysis.
3. Prefer the newer row, except when only the older row passes the reviewed
   seven-rule quality filter. Record that exception. This reconstructs a
   study population and is not an authoritative current snapshot.
4. Refuse conflicting populated organiser/HR labels on every shared ID.
   Backfill only blank attributes from the other vintage. This includes
   newer attributes when the older row was preferred. Record backfill sources.
5. Retain row-level source labels and aggregate audit counts. Verify exact
   union cardinality and preserve input bytes and source hashes.

The quality-based preference can favour higher attendance and change totals.
Compare its results with a newest-only sensitivity before interpreting trends.
Do not silently treat an earlier HR label as a dated or current assignment.
Newer cancellation or attendance changes may be genuine corrections. The
older-passing preference must be explicitly approved for the reconstruction,
not silently applied to every refresh. A newer-populated HR policy needs its
own old/new audit and approval rather than bypassing conflict checks.

Person Query history requires a separate key and compatibility assessment.
Do not apply a MeetingId policy to `(PersonId, MetricDate)`. A complete-row
newer-wins policy, if approved, retains observed zeros and blank attributes
instead of blending older values into the winning row. Check metric definition
changes before stitching, and keep source-vintage provenance.

[merge_meeting_exports.py](../examples/merge_meeting_exports.py) implements this
optional in-memory rule. It leaves inputs unchanged and returns `(combined,
audit)`. Required columns are `MeetingId`, `Collaboration_start_time`,
`Number_of_attendees`, `Intended_participant_count`, `Cancelled`,
`All_Day_Meeting` and `Duration`. Inputs must have identical column sets/dtypes,
nonmissing parsed timestamps, finite numeric quality values when present and
boolean flags. Supply every organiser/HR text column through `attribute_columns`.
Missing quality values remain in the raw union but fail the filter.

Blank tokens are missing values, whitespace, `NA`, `N/A`, `#N/A` and `null`,
case-insensitively. Populated labels are whitespace-trimmed only for conflict
comparison. No alias mapping is inferred. Shared timestamps must agree.
Pre-existing `source_export` columns are refused. `source_export` identifies
the selected metric row, while the audit records attribute backfills separately.
It is not complete cell-level provenance.

From the skill directory, run the synthetic tests:

```powershell
python -B examples\test_merge_meeting_exports.py
```

The helper does not read or write files. Production handling additionally needs
output/lock guards, immutable input hashes, persisted cell provenance and the
workstream's approval. Prefer an established production merger when available.
See [cross-query diagnostics](cross-query-diagnostics.md).

## 8. Publication checklist

- State grain, units, organiser attribution, window and grouping configuration.
- Report the filter funnel, missingness and attribution coverage safely.
- Distinguish meeting-weighted and attendee-weighted summaries.
- Treat subjects and derived topical labels as sensitive.
- Establish completeness from configuration, supported by raw-data diagnostics.
- Label reconstruction assumptions, backfill vintage and sensitivity results.
- Apply the shared distinct-person privacy floor, never a meeting-row threshold.
  Many meetings from one organiser do not establish a safe group. If a valid
  person base cannot be established, withhold the breakdown.
- Keep any minimum-event stability rule separate from privacy. A person can
  organise many events, and attendee counts cannot be added to recover distinct
  people. Check both bases before publishing an organisational comparison.

## Sources

- [Microsoft Meeting query](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/meeting-query)
  documents organiser attributes, selection conditions and meeting metric definitions.
- [Metric definitions](https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics)
  and the saved query's metric rules govern product interpretation.
- Installed R 0.7.2 and Python 0.4.3 samples and APIs were inspected for the
  column names and sample-specific caveats above. Synthetic samples do not
  validate a customer's population or export settings.
