# Reviewable classification and event decisions

Treat subjects as sensitive text. Keep raw titles, occurrence identifiers and
manual decision registers local. Shared examples must be independently fabricated.
Retain a normalised matching field separately from the original.

## Rules need worked examples

Record each category's positive terms, strong contextual qualifiers, weak aliases,
negative terms and date gate. Define whether an exclusion can be overridden.
Distinguish a process name from a generic term such as "review" or "alignment".
Broad discovery patterns and approved rules are separate artifacts.

Check supported year codes symmetrically across comparison years. Distinguish
the target fiscal/audit year from the meeting's calendar year. Do not label all
titles mentioning another process as excluded if the actual regex permits them.
Generate explanatory tables from the executable rule or test the prose against it.

Review language coverage, accents, punctuation and Unicode word boundaries.
R ICU/PCRE and Python string/regex backends can differ. Keep an adversarial
positive/negative fixture and compare actual classifications across the runtimes
used. A normalised wording family does not establish a recurring meeting series.

## A manageable review pack

Stratify candidate examples across matched years/phases, categories, ambiguous
wording and available organisational context. Keep it bounded and explain the
sampling or pagination. Retain the complete internal register without implying
that a sample is exhaustive.

Show the rule result, decision status and available organiser context. Missing
attributes stay unavailable. Do not infer identity or ownership from a title.
Ask whether feedback applies to an occurrence, a verified series or an entire
rule. Do not propagate an occurrence decision across repeated titles.

Exact event-ID matching is preferred when the reviewer has an approved mapping.
Otherwise require an unambiguous date/title match with justified corroboration.
Unmatched or multiply matched decisions are pending, not automatically applied.
Conflicting or invalid decisions stop the affected classification.

## Explicit precedence

Agree a truth table for mandatory gates, category gates and exact decisions.
A defensible example, also used in the fabricated rehearsal, is:

1. Failed mandatory meeting-quality/shared-exclusion gate stays excluded.
2. Outside the approved overall annual window stays excluded.
3. Exact Remove vetoes every category.
4. Exact Keep overrides topical defaults only inside the mandatory gates.
5. A pending review retains its recorded prior disposition. If that disposition
   is required but absent, stop rather than inventing it.
6. Otherwise apply category-specific date/title gates and approved exclusions.

This ordering is an example, not product behaviour. A narrower category window
does not exclude an event that independently qualifies for another category.
Deduplicate the final union by the verified event key.

Retain baseline, candidate and final flags, decision reason, rule version and
source fingerprints. Test quality failures, outside-window Keeps, exact Removes,
pending states, overlapping categories and unrelated repeated titles.

After approval, report exact additions/removals and their contribution to each
comparison. Do not describe the scope effect as a behavioural change. Where
effects overlap, use an explicit ordered bridge or set intersections that reconcile.
