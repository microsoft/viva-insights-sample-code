"""
Viva Insights - Copilot Analytics - Python Example Script
Tracking Copilot usage segments over time.

When measuring Copilot adoption, a single weekly snapshot can be misleading,
because usage is a habit that builds up (or decays) over many weeks. The
'vivainsights' package ships `identify_usage_segments()`, which classifies every
person-week into an adoption segment (Power / Habitual / Novice / Low / Non-user)
using a rolling window of Copilot actions.

This script shows how to:
  1. build a total Copilot-actions metric from the individual Copilot action
     columns in a Person Query,
  2. classify each person-week with identify_usage_segments(version="12w"), and
  3. visualise how the mix of segments evolves over time with a stacked-area
     chart, plus the trend in average Copilot actions.

It runs end-to-end on the built-in `pq_data` sample. To run this on your own
data, replace `vi.load_pq_data()` with `vi.import_query("your-person-query.csv")`.
"""

import vivainsights as vi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load data and build a total Copilot-actions metric
# ---------------------------------------------------------------------------
# A Person Query splits Copilot activity across several Copilot_actions_taken_in_*
# columns (Teams, Outlook, Word, Excel, ...). identify_usage_segments() expects a
# single intensity metric, so we sum them into one column and fill missing values
# with 0 (a missing action count simply means the action was not taken).
pq = vi.load_pq_data()

app_cols = [c for c in pq.columns if c.startswith("Copilot_actions_taken_in")]
pq[app_cols] = pq[app_cols].fillna(0)
pq["Total_Copilot_actions_taken"] = pq[app_cols].sum(axis=1)

print("Panel: {:,} persons x {} weeks ({} to {})".format(
    pq["PersonId"].nunique(),
    pq["MetricDate"].nunique(),
    pd.to_datetime(pq["MetricDate"]).min().date(),
    pd.to_datetime(pq["MetricDate"]).max().date(),
))

# ---------------------------------------------------------------------------
# 2. Classify each person-week into a usage segment
# ---------------------------------------------------------------------------
# version="12w" applies the standard 12-week rolling definition. return_type="data"
# appends the classification columns (including UsageSegments_12w) to the frame.
seg = vi.identify_usage_segments(
    data=pq,
    metric="Total_Copilot_actions_taken",
    version="12w",
    return_type="data",
)

seg_order = ["Power User", "Habitual User", "Novice User", "Low User", "Non-user"]
seg["UsageSegments_12w"] = pd.Categorical(
    seg["UsageSegments_12w"], categories=seg_order, ordered=True
)

print("\nOverall distribution of person-weeks by segment:")
print(seg["UsageSegments_12w"].value_counts(dropna=False))

# NOTE ON THE ROLLING WINDOW: because the 12-week version looks back up to 12
# weeks, the earliest weeks in any export are based on a shorter window and are
# less stable. With a long history it is common to drop the first ~12 weeks
# before interpreting the trend; with this short sample we keep all weeks.

# ---------------------------------------------------------------------------
# 3a. Segment mix over time (stacked area)
# ---------------------------------------------------------------------------
seg_share = (
    seg.groupby(["MetricDate", "UsageSegments_12w"], observed=False)
    .size()
    .reset_index(name="n")
)
seg_share["share"] = seg_share.groupby("MetricDate")["n"].transform(
    lambda x: x / x.sum()
)

share_wide = seg_share.pivot(
    index="MetricDate", columns="UsageSegments_12w", values="share"
).fillna(0)
share_wide = share_wide[seg_order]
share_wide.index = pd.to_datetime(share_wide.index)

seg_palette = {
    "Power User": "#1b4965",
    "Habitual User": "#5fa8d3",
    "Novice User": "#cae9ff",
    "Low User": "#f4a259",
    "Non-user": "#bc4b51",
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.stackplot(
    share_wide.index,
    [share_wide[s].values for s in seg_order],
    labels=seg_order,
    colors=[seg_palette[s] for s in seg_order],
    alpha=0.9,
)
ax.set_title("Copilot usage-segment mix over time")
ax.set_ylabel("Share of population")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.margins(x=0)
ax.legend(loc="upper center", ncol=5, fontsize=8, frameon=False)
fig.tight_layout()

# ---------------------------------------------------------------------------
# 3b. Average Copilot actions over time
# ---------------------------------------------------------------------------
actions_trend = (
    seg.groupby("MetricDate")["Total_Copilot_actions_taken"].mean().reset_index()
)
actions_trend["MetricDate"] = pd.to_datetime(actions_trend["MetricDate"])

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(
    actions_trend["MetricDate"],
    actions_trend["Total_Copilot_actions_taken"],
    marker="o",
    color="#1b4965",
)
ax2.set_title("Average Copilot actions per person-week")
ax2.set_ylabel("Mean Copilot actions")
ax2.margins(x=0)
fig2.tight_layout()

# Run plt.show() to display the charts interactively.
# plt.show()

# ---------------------------------------------------------------------------
# 4. Export the segment-share table (e.g. to paste into Excel)
# ---------------------------------------------------------------------------
share_out = share_wide.reset_index().rename(columns={"index": "MetricDate"})
vi.export(share_out)
