"""
Example 3 - Copilot usage segmentation.

Builds a total Copilot actions metric from the per-app action columns, then
classifies the population into usage segments (Power / Habitual / Novice /
Low / Non-user) with identify_usage_segments.

Convention reminders (see reference/analysis-conventions.md):
- Native null enabled days mean unlicensed. Preserve the full weekly calendar.
- Require known eligible activity history throughout the rolling window,
  rather than inferring eligibility from action presence.
"""
import pandas as pd
import vivainsights as vi

pq = vi.load_pq_data()

# 1. Build a total Copilot actions metric from the per-app action columns.
action_cols = [c for c in pq.columns if c.startswith("Copilot_actions_taken_in_")]
print("Per-app action columns:", action_cols)
if not action_cols:
    raise ValueError("This sample recipe requires verified action columns")
pq["Total_Copilot_actions"] = pq[action_cols].sum(axis=1, skipna=False)
enabled = pd.to_numeric(pq["Total_Copilot_enabled_days"], errors="raise").astype("Float64")
pq["is_licensed"] = enabled.fillna(0).gt(0)
if "licence_source_matched" in pq:
    pq.loc[~pq["licence_source_matched"].fillna(False), "is_licensed"] = pd.NA
pq["MetricDate"] = pd.to_datetime(pq["MetricDate"], errors="raise")
if pq.duplicated(["PersonId", "MetricDate"]).any() or pq["MetricDate"].dt.dayofweek.nunique() != 1:
    raise ValueError("The weekly sample requires unique person-week keys and one week start")
pq = pq.sort_values(["PersonId", "MetricDate"]).reset_index(drop=True)

# 2. Keep the calendar intact for rolling classification, then gate each window.
seg = vi.identify_usage_segments(pq, metric="Total_Copilot_actions",
                                 version="12w", return_type="data")
known = (
    seg["is_licensed"].fillna(False)
    & seg["Total_Copilot_actions"].notna()
)
for width in (4, 12):
    complete = known.groupby(seg["PersonId"]).transform(
        lambda values: values.rolling(width, min_periods=width).sum().eq(width)
    )
    contiguous = seg.groupby("PersonId")["MetricDate"].diff(width - 1).eq(
        pd.Timedelta(weeks=width - 1)
    )
    columns = [f"UsageSegments_{width}w", f"IsHabit{width}w", f"target_metric_l{width}w"]
    seg[f"IsHabit{width}w"] = seg[f"IsHabit{width}w"].astype("boolean")
    seg.loc[~(complete & contiguous), columns] = pd.NA

# 3. Segment distribution across the whole panel of person-weeks.
#    Expect many NA rows. The first 11 weeks of each person's history lack a
#    full 12-week window. Unlicensed weeks (native null or zero enabled days),
#    unresolved joins and unknown licensed activity withhold affected windows.
#    They do not become licensed Non-users or stop other people's histories. On the
#    built-in sample, scattered zero-enabled weeks leave most person-weeks NA.
#    For a point-in-time adoption report, filter to a single MetricDate first
#    (for example seg[seg["MetricDate"] == seg["MetricDate"].max()]).
print("\nUsage segment distribution (person-weeks; NA = unavailable history):")
print(seg["UsageSegments_12w"].value_counts(dropna=False).to_string())
