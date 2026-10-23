# Viva Insights export data pitfalls

The recurring traps when loading and aggregating Viva Insights exports. Each has
a symptom, a cause, and a fix. Most produce silent wrong answers rather than
errors, so check for them proactively.

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
**Fix:** respect `mingroup` (default 5) in every `create_*` / scan call, and
audit groups explicitly.

```r
identify_privacythreshold(pq, hrvar = "Organization", mingroup = 5, return = "table")
create_rank(pq, metric = "Collaboration_hours", hrvar = "Organization", mingroup = 5)
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

## 6. Holiday weeks shrink the active population

**Symptom:** population means shift around late December, regional new-year
periods, and summer, with no real behaviour change.
**Cause:** during holidays 20-25% of people go inactive, and the remaining
"active" set skews toward different roles, so the mean moves on composition
alone.
**Fix:** flag and, where appropriate, exclude holiday/low-activity weeks.

```r
pq <- identify_holidayweeks(pq)     # flags low-activity weeks
pq <- identify_inactiveweeks(pq)    # per-person inactive weeks
```
```python
pq = vi.identify_holidayweeks(pq)
pq = vi.identify_inactiveweeks(pq)
```

Prefer comparing matched calendar periods, and state when holidays fall inside a
compared window.

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
- [ ] `"#N/A"` strings filtered from HR attributes?
- [ ] Locale/encoding handled if headers are accented?
- [ ] `mingroup` >= agreed privacy threshold everywhere?
- [ ] Trailing-window plateaus not misread as change?
- [ ] Holiday / inactive weeks flagged before period comparison?
- [ ] Grain confirmed before aggregation?
- [ ] Additivity checked separately from key uniqueness (multiple positive policy rows)?
- [ ] Network stocks averaged rather than summed, with boundary-month windows checked?
