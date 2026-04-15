# Common Pitfalls — Viva Insights Data Analysis

This document catalogs the most common data quality issues, analytical mistakes, and edge cases that arise when working with Viva Insights exports. Each pitfall includes an explanation of why it happens, how to detect it, and how to fix or avoid it.

> **For coding agents:** If you are processing Viva Insights data, review this list before starting your analysis. Many of these pitfalls are subtle and can invalidate results silently.

---

## 1. Panel structure misunderstanding

**The mistake:** Treating rows as independent observations when they are a person-time panel.

**Why it matters:** Person query data has a `(PersonId, MetricDate)` panel structure — each person appears once per time period. Ignoring this leads to:
- Overstating sample sizes (counting person-weeks as "people")
- Invalid statistical tests (observations are not independent — the same person appears multiple times)
- Incorrect aggregation (e.g., computing an average across all rows instead of per-person first, then across persons)

**How to detect:**

```python
# Check: are there repeated PersonIds?
persons = df['PersonId'].nunique()
rows = len(df)
print(f"{rows} rows, {persons} unique persons → {rows/persons:.1f} periods per person on average")
```

**How to fix:** Always aggregate to the appropriate level before analysis. For cross-sectional summaries, compute per-person averages first, then summarize across persons:

```python
# Wrong: mean across all rows (mixes within-person and between-person variation)
wrong_avg = df['Collaboration_Hours'].mean()

# Right: person-level mean first, then grand mean
person_avg = df.groupby('PersonId')['Collaboration_Hours'].mean()
correct_avg = person_avg.mean()
```

---

## 2. Missing value confusion (NA ≠ zero)

**The mistake:** Treating `NA`/`null` values in Copilot metric columns as zero.

**Why it matters:** In Viva Insights person query data:
- `NA` / `null` in a Copilot column → the user was **not licensed** for Copilot that period
- `0` in a Copilot column → the user **was licensed but did not use** Copilot that period

Replacing `NA` with `0` conflates unlicensed users with inactive licensed users, which:
- Inflates the denominator for adoption rate calculations
- Makes "average usage per user" appear artificially low
- Produces misleading trends (e.g., "usage is declining" when actually licenses were just revoked)

**How to detect:**

```python
copilot_cols = [c for c in df.columns if c.startswith('Copilot_')]
for col in copilot_cols:
    null_count = df[col].isna().sum()
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {null_count} nulls, {zero_count} zeros")
```

**How to fix:** Never use `fillna(0)` on Copilot columns. Instead, create explicit flags:

```python
df['is_licensed'] = df[copilot_cols].notna().any(axis=1)
df['is_active'] = df['is_licensed'] & (df['Copilot_Actions'] > 0)

# Compute adoption rate correctly
adoption = df[df['is_licensed']].groupby('MetricDate')['is_active'].mean()
```

---

## 3. Survivorship bias

**The mistake:** Assuming the same set of people appears in every time period.

**Why it matters:** People leave the organization, go on extended leave, or join during the analysis window. If you only look at people present at the end of the period, you miss:
- People who churned (left the org or lost their Copilot license)
- People who were hired or onboarded mid-period
- Seasonal patterns driven by workforce changes, not behavior changes

**How to detect:**

```python
# Count persons per period
period_counts = df.groupby('MetricDate')['PersonId'].nunique()
print(period_counts)

# Check who appears in all periods
periods_per_person = df.groupby('PersonId')['MetricDate'].nunique()
total_periods = df['MetricDate'].nunique()
balanced = (periods_per_person == total_periods).mean()
print(f"Balanced panel: {balanced:.1%} of persons appear in all {total_periods} periods")
```

**How to fix:** Decide on your analytical approach:
- **Balanced panel:** Restrict to persons who appear in all periods (simplifies longitudinal analysis but may introduce selection bias).
- **Unbalanced panel:** Keep all person-periods but account for entry/exit in your calculations (e.g., use person-period counts rather than person counts).
- **Always report** the panel completeness rate and any restrictions applied.

---

## 4. Small group suppression

**The mistake:** Reporting metrics for groups with very few people, violating privacy thresholds.

**Why it matters:** Viva Insights applies minimum group size thresholds (typically 5–10 users, configurable by tenant admin) to protect individual privacy. When doing custom analytics, you must apply the same thresholds — otherwise:
- Individual behavior may be identifiable from small-group aggregates
- Results may be noisy and unreliable for very small groups
- You may violate your organization's data governance policies

**How to detect:**

```python
# Check group sizes for each segmentation variable
for attr in ['Organization', 'FunctionType', 'LevelDesignation']:
    sizes = df.groupby([attr, 'MetricDate'])['PersonId'].nunique()
    small = sizes[sizes < 10]
    if len(small) > 0:
        print(f"{attr}: {len(small)} segment-periods have fewer than 10 people")
```

**How to fix:** Suppress or aggregate small groups:

```python
MIN_GROUP_SIZE = 10  # or your org's threshold

def suppress_small_groups(summary_df, group_col, count_col, threshold=MIN_GROUP_SIZE):
    """Replace values with NaN for groups below the privacy threshold."""
    mask = summary_df[count_col] < threshold
    suppressed = summary_df.copy()
    metric_cols = [c for c in suppressed.columns if c not in [group_col, count_col]]
    suppressed.loc[mask, metric_cols] = float('nan')
    return suppressed
```

---

## 5. Time zone effects

**The mistake:** Ignoring time zone differences when interpreting `MetricDate` and collaboration metrics.

**Why it matters:**
- `MetricDate` in Viva Insights typically represents the start of a measurement period based on the **tenant's configured time zone** or the **individual's time zone** (depending on configuration).
- Collaboration hours are calculated relative to the individual's configured working hours and time zone.
- Comparing "after-hours collaboration" across time zones is misleading if you don't account for different working-hours definitions.
- Purview audit log timestamps are always UTC, while person query periods may not be.

**How to detect:**

```python
# Check for time zone information in the data
if 'TimeZone' in df.columns:
    print(df['TimeZone'].value_counts())
    
# Compare metric distributions across time zones
if 'TimeZone' in df.columns:
    tz_stats = df.groupby('TimeZone')['After_Hours_Collaboration_Hours'].describe()
    print(tz_stats)
```

**How to fix:**
- When comparing across geographies, normalize after-hours metrics or exclude them.
- When joining person query data (weekly, tenant time zone) with Purview audit logs (event-level, UTC), convert both to a consistent reference before aggregation.
- Document the time zone assumptions in your analysis output.

---

## 6. HR attribute changes over time

**The mistake:** Assuming a person's HR attributes (Organization, LevelDesignation, etc.) are static.

**Why it matters:** In a person-time panel, a person's attributes may change across periods due to:
- Organizational restructuring
- Promotions or role changes
- Transfers between teams or geographies

If you group by `Organization` without accounting for this, a person who transferred mid-period will appear in two different groups, which may cause:
- Double counting in aggregate summaries
- Misleading segment trends (a segment appears to lose a member that another segment gains)

**How to detect:**

```python
# Check for attribute changes within persons
changes = df.groupby('PersonId').agg(
    n_orgs=('Organization', 'nunique'),
    n_levels=('LevelDesignation', 'nunique')
)
changers = changes[(changes['n_orgs'] > 1) | (changes['n_levels'] > 1)]
print(f"{len(changers)} persons ({len(changers)/len(changes):.1%}) changed attributes during the period")
```

**How to fix:**
- **Point-in-time attribution:** Use each row's attributes as-is (the attribute reflects status during that `MetricDate`). This is the default and usually correct.
- **Fixed-period attribution:** Assign each person the attributes from a specific date (e.g., the last period) for consistent grouping across the entire analysis. Document which approach you chose.

---

## 7. Metric definition changes

**The mistake:** Assuming metric definitions are consistent across all time periods in the export.

**Why it matters:** Microsoft periodically updates how Viva Insights metrics are calculated. For example:
- The definition of "meeting" (minimum duration, minimum attendees) may change
- New Copilot metrics may be added (appearing as new columns in later periods)
- Existing metric calculations may be refined

This can cause apparent "trend changes" that are actually methodology changes, not behavior changes.

**How to detect:**
- Check for columns that appear only in a subset of `MetricDate` values.
- Look for sudden, uniform shifts in metric distributions that align with known Viva Insights update dates.

```python
# Check for columns that are all-null in early periods but populated later
for col in df.select_dtypes(include='number').columns:
    null_by_period = df.groupby('MetricDate')[col].apply(lambda x: x.isna().all())
    if null_by_period.any() and not null_by_period.all():
        print(f"{col}: appears only in some periods")
```

**How to fix:**
- Document the date range of your analysis and note any known metric definition changes.
- If a metric was added partway through, only analyze it from its first appearance onward.
- Avoid making trend claims that span known methodology changes without a caveat.

---

## 8. Double counting in aggregation

**The mistake:** Aggregating metrics without respecting the panel structure, leading to inflated totals.

**Why it matters:** Common errors include:
- Summing `Collaboration_Hours` across all rows to get "total collaboration hours" (this double-counts people who appear in multiple weeks)
- Counting rows instead of unique `PersonId` values to report "number of users"
- Computing averages across all rows without first aggregating to the person level

**How to detect:**

```python
# Sanity check: does the "total user count" make sense?
row_count = len(df)
person_count = df['PersonId'].nunique()
period_count = df['MetricDate'].nunique()
print(f"Rows: {row_count}, Persons: {person_count}, Periods: {period_count}")
print(f"Expected rows (persons × periods): {person_count * period_count}")
```

**How to fix:** Be explicit about the aggregation level:

```python
# Per-week summary (cross-sectional)
weekly = df.groupby('MetricDate').agg(
    n_users=('PersonId', 'nunique'),
    avg_collab=('Collaboration_Hours', 'mean'),
    total_copilot_actions=('Copilot_Actions', 'sum')
)

# Per-person summary (longitudinal)
person_level = df.groupby('PersonId').agg(
    avg_collab=('Collaboration_Hours', 'mean'),
    total_copilot_actions=('Copilot_Actions', 'sum'),
    active_weeks=('Copilot_Actions', lambda x: (x > 0).sum())
)
```

---

## 9. Column name variations across tenants

**The mistake:** Hard-coding column names from one tenant's export and assuming they work universally.

**Why it matters:** The same metric may appear under different names in different tenants. Common variations are documented in the [Person Query Data Dictionary](viva-person-query-data-dictionary.md#common-column-name-variations). Hard-coding column names causes code to fail silently (if the column is missing) or loudly (KeyError) when applied to a different tenant's data.

**How to detect:**

```python
# Check for expected columns
expected = ['PersonId', 'MetricDate', 'Collaboration_Hours', 'Copilot_Actions']
missing = [c for c in expected if c not in df.columns]
if missing:
    print(f"Missing expected columns: {missing}")
    print(f"Available columns: {list(df.columns)}")
```

**How to fix:**
- **Auto-detect columns** using pattern matching (e.g., columns containing `Copilot`, `Collab`, `Date`).
- **Print column names at the start** of every analysis and ask for confirmation or mapping.
- **Use flexible column mapping** in code:

```python
# Flexible column mapping
COLUMN_MAP = {
    'person_id': ['PersonId', 'Person_Id', 'HashId', 'EmployeeId'],
    'metric_date': ['MetricDate', 'Date', 'Week_Start_Date', 'Period_Start'],
    'organization': ['Organization', 'Org', 'Department', 'Business_Unit'],
}

def find_column(df, standard_name):
    """Find the actual column name from known variants."""
    for variant in COLUMN_MAP.get(standard_name, []):
        if variant in df.columns:
            return variant
    raise KeyError(f"Could not find column for '{standard_name}'. "
                   f"Available: {list(df.columns)}")
```

---

## Data validation checklist

Run through this checklist at the start of any Viva Insights analysis:

### Structure validation
- [ ] **Panel structure confirmed:** Each `(PersonId, MetricDate)` pair is unique
- [ ] **Granularity identified:** Weekly (7-day gaps) or daily (1-day gaps)
- [ ] **Date range reasonable:** Start and end dates match expected export period
- [ ] **Row count makes sense:** Approximately `n_persons × n_periods`

### Column validation
- [ ] **Core columns present:** `PersonId` and `MetricDate` (or variants) exist
- [ ] **HR attributes identified:** At least `Organization` (or variant) is present
- [ ] **Copilot columns detected:** Columns matching `Copilot_*` pattern exist (if Copilot analysis)
- [ ] **Column names documented:** Printed and verified against expected schema

### Data quality
- [ ] **Missing values understood:** Nulls in Copilot columns represent unlicensed users, not zero usage
- [ ] **Panel completeness assessed:** What percentage of persons appear in all periods?
- [ ] **Small groups identified:** Segments with < minimum group size flagged for suppression
- [ ] **No unexpected duplicates:** No duplicate `(PersonId, MetricDate)` rows

### Analysis setup
- [ ] **Licensed vs. unlicensed flagged:** `is_licensed` column created based on Copilot metric presence
- [ ] **Active vs. inactive flagged:** `is_active` column created based on `Copilot_Actions > 0`
- [ ] **Time zone assumptions documented:** Noted whether analysis assumes tenant TZ or individual TZ
- [ ] **Privacy threshold applied:** Minimum group size set and enforced in all outputs
