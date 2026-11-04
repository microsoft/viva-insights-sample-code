"""
Example 3 - Copilot usage segmentation.

Builds a total Copilot actions metric from the per-app action columns, then
classifies the population into usage segments (Power / Habitual / Novice /
Low / Non-user) with identify_usage_segments.

Convention reminders (see reference/analysis-conventions.md):
- Preserve observed nulls. This built-in fixture has complete action cells;
  real unknown histories must not be forced into segments with zero-fill.
- Define the licensed population by enabled days for a snapshot rather than by
  action presence.
"""
import vivainsights as vi

pq = vi.load_pq_data()

# 1. Build a total Copilot actions metric from the per-app action columns.
action_cols = [c for c in pq.columns if c.startswith("Copilot_actions_taken_in_")]
print("Per-app action columns:", action_cols)
if not action_cols or pq[action_cols].isna().any().any():
    raise ValueError("This sample recipe requires complete verified action columns")
pq["Total_Copilot_actions"] = pq[action_cols].sum(axis=1, skipna=False)

# This demonstrates observed sample activity, not a licensed adoption denominator.

# 3. Classify into 12-week usage segments.
seg = vi.identify_usage_segments(pq, metric="Total_Copilot_actions",
                                 version="12w", return_type="data")

# 4. Segment distribution across the whole panel of person-weeks.
#    For a point-in-time adoption report, filter to a single MetricDate first
#    (for example seg[seg["MetricDate"] == seg["MetricDate"].max()]).
print("\nUsage segment distribution (all person-weeks):")
print(seg["UsageSegments_12w"].value_counts(dropna=False).to_string())
