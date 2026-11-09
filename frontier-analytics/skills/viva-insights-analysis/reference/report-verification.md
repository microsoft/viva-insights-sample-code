# Evidence-led reporting and revision

Read [analysis policy](analysis-policy.md), [analysis conventions](analysis-conventions.md)
and the selected workflow's capability limits first. This is format-independent
guidance, not an implemented PowerPoint editor or report-validation engine.

## 1. Establish the evidence before the narrative

Keep a local evidence ledger with one stable identifier per published claim or
view. Record the question, source/version fingerprints, scope-rule version,
exact dates, grain, raw metric names, units, aggregation, numerator/denominator,
valid bases, privacy decision and caveat. Record the configured population and
whether organisational labels are historical or latest. Raw headers use spaces;
imported code may use underscores.

Retain full precision in the analytical evidence and round only for display.
Derive prose, tables and chart data from the same aggregates. Reconcile
partition identities, such as all = included + outside and attributed =
reportable + withheld. Overlapping categories need a deduplicated union rather
than a sum. Displayed rounded components may differ slightly from a rounded total.

Keep event/person identifiers and sensitive titles out of shareable ledgers,
notes, tooltips, embedded workbooks and downloadable data. Private working
membership records stay separate. Protect complementary cells and apply the
same disclosure decision to the chart and its embedded data.

## 2. Match the structure to the discussion

Confirm audience and purpose: exploratory findings, operational monitoring or
a supported decision. Do not invent a process owner, target or intervention
verdict to make an exploratory deck sound decisive.

Use commentary-led titles answering what the evidence says, with its material
qualification. Retain dates, units, bases, denominator and caveat visibly.
Speaker notes supplement those details. They do not replace them.

Organise each slide or report section around one question. It can contain two
tables or several related charts when their labels and evidence remain readable.
Use comparison panels for a shared question, with explicit differences in date
windows or measures. Move supporting detail to an indexed appendix or disclosure
panel, preserving complete eligible tables and explaining unavailable rows.
Paginate or simplify before shrinking text beyond readability.

Use stable colours for the same year or category. Rank each metric by its own
values unless a shared order is deliberate and labelled. Keep a common scale
for like-for-like panels, or explain different scales. Treemap area represents
the chosen denominator within each panel; equally sized maps do not imply equal
absolute totals. Show missing attribution explicitly or state that the view
conditions on attributed records. Identify any pooled categories.

## 3. Verify numerical, privacy and visual results separately

- Recompute every headline and badge from the registered aggregate and check
  signs, percentage points versus relative changes, units and rounding.
- Check chart caches, embedded workbooks, table cells, notes and accessible text
  against the same evidence. Updating a visible label alone is insufficient.
- Verify privacy across every embedded or downloadable surface. A minimum
  meeting count is a stability rule, not a distinct-person privacy base.
- Open the actual generated file in a compatible renderer when available.
  Inspect clipping, wrapping, legends, data labels, colour contrast and evidence
  placement. A valid ZIP/XML package does not prove a readable presentation.
- Fix and re-render affected views. For HTML, check offline operation and the
  relevant browser interactions. Record which checks actually ran.

If a renderer is unavailable, state that visual verification is unavailable.
Do not label the output visually approved. A scenario range is not a confidence
interval, and a significant difference is not evidence of intervention impact.

## 4. Revise an existing publication safely

1. Treat the current user-edited file as the source. Save a recoverable snapshot
   and fingerprint it before editing. Work on an unused candidate path.
2. Read the actual slide/section order and review comments. Preserve unrelated
   manual edits, native objects, notes and styles. Prefer bounded replacements
   to a full rebuild when the user has edited the document.
3. Keep the analytical evidence frozen during editorial changes. A scope, source
   or window change needs a separate reconciled analytical refresh.
4. Update affected titles, commentary, footers, contents, links and references.
   Record a before/after mapping when sections are merged or moved.
5. Verify the candidate and the unchanged content. Remove only resolved comments
   after their edits pass verification; retain unresolved or unrelated threads.
6. Recheck that the live source has not changed. A newer edit blocks a stale
   overwrite. Review and rebase onto it, then repeat affected checks.
7. Publish only to the user's authorised target. Do not close or overwrite an
   unsaved user presentation. Retain a rollback and verify the persisted result,
   including the review-comment state.

Whole-file hashes detect drift but do not explain it. Office or synchronisation
can change package metadata. Compare package parts to determine whether a
specific change is metadata-only. Do not ignore arbitrary XML or relax content
checks simply to make hashes match. Record any narrowly reviewed metadata rebase
and preserve the actual publication source.

Use script-file execution and short local paths where the installed Office/R
toolchain needs them. Keep those platform mechanics optional and documented,
without requiring a particular customer's folder structure or proprietary service.
