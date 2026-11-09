# Viva Insights export data pitfalls

The recurring traps when loading and aggregating Viva Insights exports. Each has
a symptom, a cause, and a fix. Most produce silent wrong answers rather than
errors, so check for them proactively.

The [shared analysis policy](analysis-policy.md) owns eligibility, missingness,
cadence and publication rules across workflows.

## 1. `IsManager` arrives as "Yes"/"No" text rather than a boolean

**Symptom:** manager filters match nothing, or every row goes NA.
**Cause:** the export delivers `"Yes"`/`"No"` strings instead of `TRUE`/`FALSE` or 1/0.
**Fix:** normalise robustly.

```r
dt$is_mgr <- dplyr::case_when(
  toupper(as.character(dt$IsManager)) %in% c("TRUE","YES","Y","1")  ~ TRUE,
  toupper(as.character(dt$IsManager)) %in% c("FALSE","NO","N","0")  ~ FALSE,
  TRUE ~ NA
)
```
```python
m = df["IsManager"].astype(str).str.upper()
df["is_mgr"] = m.map({"TRUE":True,"YES":True,"Y":True,"1":True,
                      "FALSE":False,"NO":False,"N":False,"0":False})
```
The same pattern applies to any other Yes/No flag column.

## 2. `"#N/A"` arrives as a literal string

**Symptom:** a numeric column is typed as text. `is.na()` / `.isna()` misses
blanks. Group counts include a bogus "#N/A" category.
**Cause:** some exports write `"#N/A"` (and variants) as text rather than a true
null. Common in HR attribute columns.
**Fix:** filter the string forms explicitly before typing/aggregating.

```r
bad <- c("", "NA", "NULL", "#N/A", "#n/a", "N/A")
data <- data[!data$Organization %in% bad, ]
```
```python
bad = {"", "NA", "NULL", "#N/A", "#n/a", "N/A"}
df = df[~df["Organization"].isin(bad)]
```

## 3. Non-English / accented-locale exports

**Symptom:** column matching breaks after import. Dictionary lookups miss.
`import_query()` quietly turns accented headers into underscores.
**Cause:** exports from non-English tenants use localised, accented column names
and often **smart-quote apostrophes** (U+2019) mixed with ASCII ones.
**Fix:** read with an explicit encoding, normalise the quote/space variants, then
map to canonical English names.

```python
import pandas as pd

path = "path/to/export.csv"  # TODO: replace with your file path

def _norm_col(s: str) -> str:
    for ch in ("\u2019","\u2018","\u02bc","\u02be","\u02b9","\u2032"):
        s = s.replace(ch, "'")
    return s.replace("\u00a0", " ")   # non-breaking space -> space

for enc in ("utf-8-sig", "utf-8", "latin-1"):
    try:
        df = pd.read_csv(path, encoding=enc, low_memory=False); break
    except Exception:
        pass
# then rename via a {normalised_local_name: english_name} map you maintain
```

Diagnose apostrophe type with `repr(col_name)` on a sample of headers before
building the map. Keep the local->English mapping in one place. This approach
works across locales rather than for a single language.

## 4. Privacy threshold (minimum group size)

**Symptom:** small groups look precise but are not privacy-safe. Results are not
reproducible against the portal, which suppresses small groups.
**Cause:** Viva Insights enforces a minimum aggregation group size (commonly 5).
Below it, a group should not be reported.
**Fix:** use the stricter of the organisation requirement and workflow floor:
5 distinct people for generic analysis, 10 for joined-query, dashboard and
causal outputs. Pass the applicable value as `mingroup` in supported package
calls, and audit all published groups and complementary cells explicitly.

```r
org_minimum <- 5L  # replace with the organisation's actual requirement
workflow_floor <- 5L  # 10L for joined-query, dashboard or causal outputs
min_group <- max(org_minimum, workflow_floor)
identify_privacythreshold(pq, hrvar = "Organization", mingroup = min_group, return = "table")
create_rank(pq, metric = "Collaboration_hours", hrvar = "Organization", mingroup = min_group)
```

Never lower `mingroup` below the organisation's agreed threshold to "get a
number" for a small team. Aggregate up to a coarser `hrvar` instead.

## 5. Trailing-window metrics plateau then step

**Symptom:** a weekly series holds the same value for several weeks, then jumps,
which looks like a data glitch or a "methodology change".
**Cause:** several metrics (network sizes, tie counts, and similar) are computed
over a **trailing multi-week window**. The same windowed value repeats until the
window slides. This is how the metric is defined. It is not an error.
**Fix:** do not interpret the plateaus-and-steps as behavioural change. For
cross-period comparison, pool over multi-month windows or normalise per person,
and avoid week-on-week deltas on these metrics.

## 6. Low-activity weeks need interpretation

**Symptom:** population means shift around late December, regional new-year
periods, and summer, with no real behaviour change.
**Cause:** holidays, missing coverage or changing composition can change observed
activity. In a holiday week a sizeable share of people may go inactive, and those
who remain active often skew toward particular roles or regions, so a population
mean can move on composition alone. A statistical flag alone cannot distinguish
these causes.
**Fix:** retain flags for review before any documented exclusion. Confirm
holidays with approved calendar evidence rather than a universal inactivity
percentage. Inspect the installed package API first.

```r
pq <- identify_holidayweeks(pq, return = "data")
pq <- identify_inactiveweeks(pq, return = "data")
```
```python
pq = vi.identify_holidayweeks(pq.copy(), return_type="labelled_data")
assert isinstance(pq, pd.DataFrame) and "holidayweek" in pq
pq = vi.identify_inactiveweeks(pq, return_type="data")
assert isinstance(pq, pd.DataFrame) and "inactiveweek" in pq
```

R `vivainsights` 0.7.2 uses `return = "data"` for both labelled datasets.
The Python recipe assumes `import pandas as pd` and `import vivainsights as vi`.
Verified with Python `vivainsights` 0.4.3: both defaults return **text**, not a
DataFrame. Holiday `"labelled_data"` and inactive `"data"` retain rows and add
flags. The inactive helper in this release falls back to its lowest z-score
quantile when the SD rule flags nothing, so its label is not always an SD breach.
Do not silently remove those candidates. See `scripts/test_analysis_recipes.py`
in a full checkout for the executable synthetic smoke test.

Prefer matched calendar periods and state when verified holidays fall inside
a comparison window.

### `identify_nkw` is a heuristic with an upstream cardinality caveat

Low average collaboration does not establish a job classification. In Python
`vivainsights` 0.4.3, `identify_nkw` groups by `PersonId, Organization` but joins
flags back on `PersonId` alone. People who transfer organisations can multiply
rows in `"data_with_flag"` and cleaned modes; summaries count person/organisation
records, not necessarily distinct people. Missing organisations also need review.
Inspect the installed version/source and assert original person-period keys,
row counts and distinct-person totals before using any affected output. Stop
on expansion rather than silently deduplicating. This repository does not fix
or fork the package helper, and no fixed upstream release is claimed here.

## 7. Load only the columns you need

**Symptom:** slow loads and high memory on wide exports (hundreds of columns).
**Cause:** Person Query exports can be very wide. Reading every column is
wasteful when an analysis needs a handful.
**Fix:** select columns at read time.

```r
# fst / arrow support column selection at read
cols <- c("PersonId","MetricDate","Collaboration_hours","Internal_network_size")
pq <- as.data.frame(arrow::read_parquet(path, col_select = dplyr::all_of(cols)))
```
```python
import pandas as pd

path = "path/to/export.parquet"  # TODO: replace with your file path

pq = pd.read_parquet(path, columns=["PersonId","MetricDate",
                                    "Collaboration_hours","Internal_network_size"])
```

## 8. Confirm the grain before aggregating

Re-stating the most costly mistake: a Meeting Query is meeting-level, a Person
Query is person-week level, a P2P query is edge-level. Averaging meeting rows as
if they were people, or summing person-week metrics without accounting for the
week grain, gives confidently wrong numbers. Check with `extract_date_range()`
and a quick `nrow` / `groupby` sanity count first.

## 9. Unique keys do not prove amounts can be added

**Symptom:** a key-uniqueness check passes, but summed totals are too high or
change depending on which rows are kept.
**Cause:** some Consumption exports carry several rows for the same person,
service and date, one per `SpendingPolicyId`. The full key is unique, yet more
than one of those rows can hold a positive amount. Uniqueness describes the
table structure. It says nothing about whether the amounts overlap.
**Fix:** check additivity as a separate step. Count person-service-date keys
with more than one positive-credit row. A positive row alongside zero rows is
safe to sum. Multiple positive rows need the data owner's confirmation that the
policies record disjoint consumption before you add them. See the
[shared query contract](copilot-query-contracts.md#key-uniqueness-and-additivity-are-separate-checks).

```python
key = ["PersonId", "ServiceId", "MetricDate"]
pos = m365[m365["Total Copilot Credits used"] > 0]
multi_positive = (pos.groupby(key).size() > 1).sum()   # report the count only
```

## 10. Network stocks repeat within months

**Symptom:** a person's monthly network size is several times larger than any
weekly value, or a "later" network outcome already reflects the exposure period.
**Cause:** `Internal_network_size` and `External_network_size` are stocks
measured over a window. In some exports they repeat across the weekly rows of a
person-month, so the weekly row grain is not their measurement frequency (see
pitfall 5). A value dated in a boundary month can include contacts outside the
selected weeks.
**Fix:** never sum network sizes across weeks. Average them, or keep one value
per measurement window, and record which. Check how many distinct values each
person-month holds. When comparing periods, or ordering exposure before outcome,
confirm that the measurement windows behind the dates do not overlap.

```r
pq |>
  dplyr::mutate(month = format(as.Date(MetricDate), "%Y-%m")) |>
  dplyr::group_by(PersonId, month) |>
  dplyr::summarise(distinct_values = dplyr::n_distinct(Internal_network_size),
                   Internal_network_size = mean(Internal_network_size), .groups = "drop")
```

## Quick diagnostic checklist

- [ ] Loaded with `import_query()` so columns are underscore-named?
- [ ] `IsManager` and other Yes/No flags normalised?
- [ ] Literal HR missing-value tokens identified, with affected rows retained for coverage diagnostics?
- [ ] Locale/encoding handled if headers are accented?
- [ ] `mingroup` >= agreed privacy threshold everywhere?
- [ ] Trailing-window plateaus not misread as change?
- [ ] Low-activity candidates reviewed, with calendar evidence before holiday claims?
- [ ] Grain confirmed before aggregation?
- [ ] Additivity checked separately from key uniqueness (multiple positive policy rows)?
- [ ] Network stocks averaged rather than summed, with boundary-month windows checked?
- [ ] Meeting organiser attribution, attendee hours and period coverage reviewed? See [Meeting Query specifics](meeting-query.md).
- [ ] Missing HR and optional population/rerun comparisons investigated without automatic backfill? See [cross-query diagnostics](cross-query-diagnostics.md).
