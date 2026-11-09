# Population and missing-organisational-data diagnostics

Read [shared analysis policy](analysis-policy.md) first. Use this reference for
missing HR attributes, differing query populations or original-versus-rerun
comparisons. It does not replace the
[current join guide](https://microsoft.github.io/viva-insights-sample-code/query-join-guide/)
or authorise stitching historical exports. Additional queries and historical
archives are optional evidence, not prerequisites.

## 1. Start with the supplied export

Confirm query purpose, grain, required fields and configured window before
diagnosing missing columns. Purpose-specific Person Queries can intentionally
expose different metrics and HR fields. Select a suitable source rather than
forcing one export into another workflow.

Validate identity keys, person-period uniqueness where applicable, actual
observed dates and metric types. Other query types have different keys.
Count distinct people separately from rows. Read only the columns required.
Do not silently apply an activity filter or discard observed zero-activity
rows when measuring query coverage.

Define missingness explicitly. Blank or whitespace `Organization`, literal
missing-value tokens and missingness across every checked HR field are different
measures. Record the fields checked. Distinguish missingness in any observed
period from missingness throughout a person's observed history. Preserve these
rows during diagnostics rather than applying a generic missing-HR exclusion.

Assess missingness by period alongside observed population, licence exposure
if supplied and activity. Activity without HR does not establish why attributes
are missing. With one export, report within-file findings and state that
cross-query coverage, historical availability and rerun recovery were not checked.
Do not block otherwise supported analysis.

## 2. Compare populations only when a suitable second export exists

Confirm comparable identity namespaces before exact `PersonId` matching.
Do not infer identities from names, titles or organisational labels.
Count each person once even when they have many rows.

| Group | Meaning |
|---|---|
| In both exports | At least one row in each file |
| Only in the first | Rows in the first file but none in the second |
| Only in the second | Rows in the second file but none in the first |

Check that shared plus first-only equals the first distinct-person count, and
shared plus second-only equals the second count. "Only" describes these files,
not the tenant, all queries or the HR roster.

This raw-file comparison diagnoses membership. It does not decide activity
imputation. Under the shared policy and current join contract, absent activity
inside a supplied complete query's configured scope contributes zero.
An unavailable source, unresolved join, out-of-scope period or known coverage
failure remains unknown. Preserve match indicators and reasons. Query absence
alone does not establish eligibility, no activity or missing entitlement.

Locate people with missing HR in each group. For shared people, inspect whether
the second export actually has usable attributes. Identity overlap alone does
not make enrichment possible. Do not backfill automatically.

Where dates overlap, repeat over a common observed window to assess coverage
differences. This does not establish identical licensing, inclusion settings or
HR availability. Filename dates, filesystem timestamps and report creation dates
are not verified query execution dates. Treat reported matching filters as
unverified until saved configurations are available. Inspect `IsActive` if
supplied rather than assuming an active-only selection.

## 3. Historical evidence and reruns

If historical exports exist, inspect available raw files independently.
A backfilled combined cache cannot establish which export originally supplied
an HR value. Distinguish identity presence from usable attributes, including
whether an attribute occurs in a strictly earlier observed period.
An earlier value does not establish when HR data was uploaded, accepted or
available to a query. Per-export matching counts overlap and must not be summed.
Report the distinct identity union separately.

Keep original and rerun exports separate. Compare exact keys, shared rows,
original-only rows, rerun-only rows and distinct-person membership changes.
Compare HR fields and metrics separately on shared keys. Harmonise only reviewed
header aliases and reject ambiguous aliases or unexplained schema differences.
Do not modify or merge raw exports to assess recovery.

Trace all originally affected people through every observed rerun period,
including people absent from the rerun. An unchanged affected-person count can
conceal different membership. Separate persistent missingness, recovery, absence
and newly affected people. Do not equate absence with recovery.

## 4. Weighting, observation bases and label vintage

For a within-file usage comparison, use people with usage observations and retain
observed zeros. If a contracted complete panel is appropriate, construct its
in-scope absent activity using the shared policy first. Do not impute unavailable
or out-of-scope weeks or treat people in a reference-only export as observed
usage without establishing scope and completeness.

For an equal-person summary, compute each person's mean over the agreed window,
then describe the distribution of person means, including mean, median and
relevant percentiles. Do not confuse it with pooled person-period statistics.
Report distinct-person and person-period bases and history lengths. Where bases
allow, use a justified minimum-history sensitivity and a same-calendar-period
comparison. These do not establish identical licence histories or remove role
and population differences.

For a refresh, read the workstream's window, activity normalisation, minimum
history, privacy floor, weighting and matched-population rule. A rolling-window
mean and full-history mean target different estimands. Thresholds belong to the
analysis contract, not a universal diagnostic default.

State whether organisational labels are valid at each historical date or are
each person's latest usable label applied to a whole-window average.
The latter is a current-profile lens, not historical assignment. Membership and
label changes can move group ranks without within-population behaviour change.
Audit transitions and use matched-window/population sensitivity before making
behavioural interpretations.

Read total metrics directly rather than summing overlapping application or
feature metrics. Label percentage and proportion units consistently. Generate
comparative prose from the current results, handling a zero reference explicitly.

## 5. Bound explanations by the evidence

Missing uploads, failed identity linkage, effective-date coverage, source
configuration and query inclusion settings are competing explanations.
Blank HR or absence from another query does not prove a record was never
uploaded or a licence was absent. Enabled days describe exposure in observed
periods rather than current service-plan state.

An authorised administrator can verify effective-dated HR records,
accepted/rejected upload evidence, linkage checks, actual execution dates,
licence status at execution and saved inclusion configurations.
Request that evidence rather than inferring the mechanism. Assess whether
missing HR could explain the entire population gap or only a subset.
Recheck current documentation before interpreting licensing:

- [Licensing](https://learn.microsoft.com/en-us/viva/insights/advanced/setup-maint/assign-licenses)
- [Organisational-data preparation](https://learn.microsoft.com/en-us/viva/insights/advanced/admin/prepare-org-data)
- [Person Query inclusion and IsActive](https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/person-query)

## 6. Reproducibility and disclosure controls

Keep approved local provenance with source paths, byte sizes, SHA-256 hashes,
observed windows, configurations, methods and limitations. Verify input hashes
before combining derived evidence. Keep current deliverables separate from
labelled historical vintages. If archiving, preserve historical bytes and record
old/new paths and hashes in a relocation manifest.

Keep customer identifiers and sensitive paths out of publications and skill
examples. Apply the shared distinct-person privacy floor to counts, percentages,
embedded data and complementary cells. A one-off disclosure request does not
override governing policy. Historical backfill does not confer permission to
publish a person's attributes.

Use readable labels and explicit denominators. Common-scale bars often show
population overlap more clearly than schematic Venn diagrams. Keep source
vintage and inference limits visible.
