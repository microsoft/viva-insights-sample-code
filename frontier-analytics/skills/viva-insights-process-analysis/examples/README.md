# Fabricated audit rehearsal

All subjects, identifiers, dates, rules and values here were independently
fabricated. They do not describe an organisation, customer or product default.
The fictional audit window is February-June, with three explicitly supported
year codes. The 2022 source ends partway through the window.

The Python example exercises approved-scope classification, overlapping-category
union, exact event decisions, coverage gates, person-based privacy and aggregate
reporting. Its `quality_pass` input is a deliberately supplied **prepared gate**,
not a computed raw Meeting Query quality filter. For real data, apply the core
Skill's reviewed structural filter and shared exclusions first.

Run from this directory:

```powershell
python -B run_example.py
python -B test_examples.py
Rscript --vanilla run_classifier.R
```

The Python rehearsal uses only the standard library. The R companion requires
`jsonlite` and the paired core Skill's flag helper. It checks the same CSV
title/decision fixtures with Unicode-aware PCRE boundaries.
It does not implement the Python coverage/privacy summary.
The shared core Skill provides separate matched R/Python seven-rule helpers.

The JSON output is explicitly labelled fabricated. It includes fully observed
annual summaries and an unavailable partial year, with rule/fixture provenance.
No person/event identifiers or raw subjects are exported by `run_example.py`.
Group breakdowns are not published, so the small synthetic review group cannot
be reconstructed from complementary cells. Thirty events from one organiser
still fail the distinct-person floor.

`fixtures/synthetic-audit-cases.csv` is the complete small worked-example register.
Its `decision` fields are already matched to exact occurrences. The example
does not match an external review workbook, infer series membership, inspect
actual query coverage or choose a source-merging policy.

These are deterministic synthetic checks, not automatic Skill-discovery
evaluations or a real-export trial. No statistical inference, forecasting,
PowerPoint editing or real-data report renderer is implemented here. Use
[evaluation scenarios](../reference/evaluation.md) for a separate agent trial.
