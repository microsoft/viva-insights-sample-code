# Synthetic causal-analysis examples

Python standard library and base R consume **the same CSV files**. These are
bounded teaching examples, not a general-purpose real-data estimator or a route
to causal claims about employees. No external data, packages, network calls,
PPML, bootstrap, confidence intervals, or p-values are used.

## Run from the repository root (PowerShell)

Requires Python 3.10+ and R 4.1+. No installation step is needed.

```powershell
Set-Location .\frontier-analytics\skills\viva-insights-causal-analysis\examples
$Rscript = (Get-Command Rscript -ErrorAction Stop).Source
# If R is not on PATH, use the installed path, for example:
# $Rscript = 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe'

python -B run_example.py fixtures\synthetic-effect.csv fixtures\synthetic-effect.metadata.csv output-python
& $Rscript --vanilla run_example.R fixtures\synthetic-effect.csv fixtures\synthetic-effect.metadata.csv output-r
python -B test_examples.py --rscript $Rscript --output test-results
```

All output directories must be **new**, with an existing parent. Existing files
or directories are rejected rather than merged, deleted, or overwritten. Pick
another directory name on subsequent runs. Inputs are read-only. The checked-in
fixtures are reproducible:

```powershell
python -B generate_fixtures.py regenerated-fixtures
```

Do not replace these with business files. The synthetic filename, fictional-key
format, generator marker and metadata contract are guardrails, **not proof of
provenance**: metadata is a declaration, not cryptographic authentication.

## Input contract and meaning

Each `fixtures\synthetic-CASE.csv` pairs with `synthetic-CASE.metadata.csv`.
Data CSV columns, in order:

| Field | Meaning and validation |
|---|---|
| `PersonId` | Fictional `synthetic-P0000`-style key; no actual person identities |
| `MetricDate` | Calendar-valid ISO month-start date, years 2000-2100 |
| `CollaborationHours` | Fictional monthly collaboration hours, continuous exposure in [0,744]; empty means unknown, never zero |
| `CopilotActions` | Fictional nonnegative integer action count; descriptive, not used in the model |
| `CustomOutcome` | Finite continuous custom organisational score attached to a person, **not** a standard Viva metric |
| `OutcomePeriod` | Actual outcome measurement window: `YYYY-MM`, `YYYY`, `static`, or `YYYY-Qn` |

UTF-8 comma-separated inputs must have unique person-month keys, the exact
header, 1-10,000 data records and at most 2 MB per file. Internal and trailing
blank CSV records are rejected in both data and metadata files; a normal final
line terminator is allowed. Nonmissing numerics must
be decimal/scientific notation, finite and at most 1,000,000 in absolute value.
Blank outcomes/counts and invalid numerics/dates are errors. Only exposure
blanks have missing-value semantics. No imputation or complete-case deletion
occurs. Each person-window must have one consistent outcome value even if it
appears on multiple monthly rows. Monthly completeness/balance is required for
the estimator; irregular common month sets are not interpolated.

Metadata is a `key,value` CSV containing generator/version, synthetic-only flag,
case, row and outcome grains, declared mechanical dependency, known synthetic
effect, selected fields, units, meanings and design. The runner checks these
semantic declarations separately from numeric parsing: a numeric annual rating
is not thereby an admissible monthly outcome. Copying the annual value onto
12 rows cannot create 12 measurements. No general grain-discovery or hidden
dependency detection is claimed.

## Cases and expected decisions

All multi-period cases have 40 fictional people x 12 months (480 rows).

| Case | Ground truth / gate | Expected result |
|---|---|---|
| `effect` | Independent seeded exposure shocks plus person/month effects; structural slope 2 | Synthetic TWFE slope within 0.05 of 2 |
| `confounding` | Stable person factor drives both exposure and outcome; structural slope 0 | Pooled slope >1.5; TWFE within 0.05 of 0 |
| `annual` | One annual outcome per person repeated monthly | Block within-person inference; 40 actual measurements |
| `static` | One static attribute per person repeated monthly | Block within-person inference; 40 actual measurements |
| `quarterly` | One outcome per person-quarter repeated in its three months | Readiness only; 160 actual measurements, **no regression on 480 pseudo-observations** |
| `missing` | 40 unknown exposures | Block estimation, do not replace with zero |
| `mechanical` | Outcome is defined as twice the exposure | Block causal interpretation despite perfect numerical association |
| `one-period` | Only January retained (40 rows) | Insufficient for panel estimation |

Quarterly readiness reports only the distinct measurement count; it does not
write individual records, aggregate the exposure, or choose a lag/window.
A subsequent analysis would need one row per person-quarter and an explicitly
justified exposure aggregation and temporal ordering. This example stops there.

## Estimator and limitations

For the two estimable synthetic cases only, the illustration computes
`x_it - mean_i(x) - mean_t(x) + mean(x)` and the corresponding outcome residual,
then `sum(x_residual * y_residual) / sum(x_residual^2)`. This is the balanced
two-way fixed-effect OLS coefficient with person and observed-month intercepts.
It rejects an unbalanced panel or effectively zero residual exposure variation.
The pooled slope is descriptive and is displayed only alongside an estimable
synthetic panel; it is deliberately misleading in the confounding case.

The data-generating process fixes a constant linear effect and independent
exposure shocks/noise. Person and calendar effects are removed by construction.
Recovery of the programmed coefficient tests arithmetic, **not causal
identification in observational Viva data**. Actual analysis would still need
an estimand, defensible timing, confounding/selection assumptions, interference
assessment, outcome provenance and measurement support, and appropriate
uncertainty. This demonstration has no clustered standard errors, causal
identification test, staggered-adoption estimator or automatic metric selection.
Changing a dtype, adding fixed effects, or obtaining a small p-value would not
provide those missing justifications.

## Output and tests

Both runners take exactly three positional arguments: data CSV, metadata CSV,
new output directory. Success (including a scientific refusal/readiness result)
returns exit code 0 and one aggregate `summary.csv`:

`status,reason,rows,persons,periods,outcome_measurements,missing_exposures,pooled_slope,twfe_slope`

`status` is `synthetic_demo`, `blocked`, or `readiness_only`. Slopes are blank
unless a model is allowed. Invalid input, unsupported metadata, CLI misuse or
output I/O failure returns exit code 2 with a fixed error code. Invalid inputs
are rejected before output creation; a failure during output writing can leave
the newly created directory, which is never silently reused. No row identities,
row-level residuals or input rows appear in summaries or runner diagnostics.
Gate precedence is mechanical dependency, annual/static grain, quarterly grain,
panel size, unknown exposures, balance, residual exposure variation.

`test_examples.py` executes both actual CLIs, compares status/count/reason
exactly and coefficients within 1e-8, and checks base-R dummy-variable `lm`
against double demeaning. It tests all eight cases, numerical truth (not
p-values), invalid schema/dates/numerics/grains/metadata, repeated outcomes,
missing-versus-zero, singular/unbalanced panels, CLI/I/O failures, unchanged
input bytes, deterministic generation/output, and output-directory rejection.
Tests require R: there is no skip that can masquerade as a cross-language pass.
The new test directory retains `report.json` (versions, counts, aggregate
estimates) and `tests.txt`; temporary test inputs/outputs are cleaned up.

Files: `generate_fixtures.py`, `run_example.py`, `run_example.R`,
`test_examples.py`, this README, and eight data/metadata pairs in `fixtures`.
