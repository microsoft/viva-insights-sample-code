"""
Example 4 - Meeting Query quality filter.

Raw Meeting Query rows can include personal holds, all-day markers, cancelled
items and broadcasts. This shows a generalised quality filter and a per-rule
attrition table. Exclusion rates depend on the export and analytical question.

Note: the built-in mt_data sample is small and synthetic, so very few rows
survive the strict filter. The point here is the filter mechanics and the
attrition breakdown rather than the surviving counts. Adapt thresholds to your data.

Reminder: Duration and *_hours columns are measured in hours rather than minutes.
"""
import pandas as pd
import vivainsights as vi

mt = vi.load_mt_data()
n0 = len(mt)

# Sample flags are boolean. Parse real-export text explicitly before filtering.
# Per-rule attrition: how many rows each rule would remove on its own.
rules = {
    "attendees < 2":            mt["Number_of_attendees"] < 2,
    "intended <= 1":            mt["Intended_participant_count"] <= 1,
    "intended > 150 (broadcast)": mt["Intended_participant_count"] > 150,
    "cancelled":                mt["Cancelled"].astype(bool),
    "all-day":                  mt["All_Day_Meeting"].astype(bool),
}
attrition = pd.DataFrame(
    {"rule": list(rules), "rows_flagged": [int(m.sum()) for m in rules.values()]}
)
print(f"Raw rows: {n0}")
print(attrition.to_string(index=False))

# Combined quality filter.
keep = (
    (mt["Number_of_attendees"] >= 2)
    & (mt["Intended_participant_count"] > 1)
    & (mt["Intended_participant_count"] <= 150)
    & (~mt["Cancelled"].astype(bool))
    & (~mt["All_Day_Meeting"].astype(bool))
)
mq = mt[keep]
print(f"\nAfter quality filter: {n0} -> {len(mq)} rows "
      f"({len(mq) / n0:.1%} retained)")
