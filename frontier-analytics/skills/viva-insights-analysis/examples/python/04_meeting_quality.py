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
import sys
from pathlib import Path

import pandas as pd
import vivainsights as vi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meeting_quality import normalise_flag, quality_mask

mt = vi.load_mt_data()
n0 = len(mt)

# Per-rule attrition: how many rows each rule would remove on its own.
rules = {
    "attendees < 2":            mt["Number_of_attendees"] < 2,
    "intended <= 1":            mt["Intended_participant_count"] <= 1,
    "intended > 150 (broadcast)": mt["Intended_participant_count"] > 150,
    "cancelled":                normalise_flag(mt["Cancelled"]),
    "all-day":                  normalise_flag(mt["All_Day_Meeting"]),
    "duration < 1 minute":      mt["Duration"] < 1 / 60,
    "duration > 8 hours":       mt["Duration"] > 8,
}
attrition = pd.DataFrame(
    {"rule": list(rules), "rows_flagged": [int(m.sum()) for m in rules.values()]}
)
print(f"Raw rows: {n0}")
print(attrition.to_string(index=False))

# Combined quality filter.
keep = quality_mask(mt)
mq = mt[keep]
missing = mt[["Number_of_attendees", "Intended_participant_count", "Duration"]].isna().any(axis=1)
missing |= normalise_flag(mt["Cancelled"]).isna() | normalise_flag(mt["All_Day_Meeting"]).isna()
print("Missing quality inputs:", int(missing.sum()))
print(f"\nAfter quality filter: {n0} -> {len(mq)} rows "
      f"({len(mq) / n0:.1%} retained)")
