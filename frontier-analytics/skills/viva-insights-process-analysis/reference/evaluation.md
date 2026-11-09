# Evaluation scenarios and evidence boundaries

Test routing and behaviour separately. A deterministic fixture pass does not
prove automatic Skill discovery or a successful real-export trial.

| Scenario | Expected behaviour |
|---|---|
| Annual audit collaboration question | Use this workflow and request missing scope/coverage choices |
| Purview audit-log parsing | Keep the existing audit-log workflow; do not treat it as a business process |
| Business-process mining from transaction logs | State that this skill does not reconstruct process sequences |
| Generic Copilot dashboard | Route to the relevant dashboard owner |
| Person Query intervention-effect question | Assess design and hand off to the causal skill where compatible |
| Unapproved topical definition | Discovery/readiness only; no approved headline totals |
| Asymmetric year-code seed | Identify the unequal capture and require review before a matched-year claim |
| Accented text and different regex engines | Test actual classifications, including false-positive word boundaries |
| Overlapping categories | Count each event once in the union and reconcile intersections |
| Outside-window Keep or failed quality gate | Mandatory gates prevail |
| Pending review without required prior state | Stop rather than inventing a disposition |
| Same subject on another date | Do not transfer an exact occurrence decision |
| Partial source period | Retain reviewable observations but withhold a full-period comparison |
| Many meetings from a small organiser base | Keep event stability separate from person privacy and complementary cells |
| HR succession or revised source metrics | Separate attribution/source revisions from behavioural differences |
| Unexpected custom PQ metric | Validate definition and vintage; do not assert a product defect |
| Editorial condensation | Keep underlying evidence fixed and retain eligible detail |
| Concurrent live-deck change | Refuse a stale overwrite and verify a rebase before publication |

The [fabricated rehearsal](../examples/README.md) executes scoped classification,
union reconciliation, partial-period and privacy checks. Its R companion checks
the same title/decision fixtures, including Unicode boundaries. It does not
provide a generic export merger, inferential estimator, workbook builder or
PowerPoint renderer.

For an explicit-read agent trial, point an agent to this SKILL.md and its paired
core references. Ask it to run the rehearsal and assess a process-analysis ask,
then challenge it with a Purview log request and an unsupported causal claim.
Record the actual commands, observed outcomes, runtime and whether the skill
was explicitly supplied or discovered automatically. Do not mark an unexecuted
scenario as passed. Use fabricated inputs only.

## Recorded explicit-read trial, 9 November 2026

A coding agent explicitly read this skill and its paired core references, then
executed the following from the repository checkout:

- `python -B frontier-analytics/skills/viva-insights-process-analysis/examples/run_example.py`
- `python -B frontier-analytics/skills/viva-insights-process-analysis/examples/test_examples.py -v`
- `Rscript --vanilla frontier-analytics/skills/viva-insights-process-analysis/examples/run_classifier.R`

All commands exited successfully. Eight Python regression cases and 18 matched R
classifier fixtures passed. The rehearsal reported 13 meetings/25.0 hours for
2020 and 19 meetings/25.0 hours for 2021, each with 13 distinct synthetic
organisers. The partial 2022 window retained null reporting values.

The agent also returned readiness-only for an unapproved audit definition and
unknown intervention exposure, routed a Purview system-log request outside this
skill, and withheld a requested group with 30 meetings from one organiser.
No actual user files, source edits, external data or publications were involved.
This is explicit-read synthetic execution evidence, not automatic discovery,
real-export validation or evidence that a causal design is supported.
