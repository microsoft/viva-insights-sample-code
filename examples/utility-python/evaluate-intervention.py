"""
Viva Insights - Evaluating a Workplace Intervention (Python)

Organisations frequently run interventions to improve how people work: a protected
"focus day", a meeting-reduction push, or a Microsoft 365 Copilot enablement wave
for one team. The natural question is "did it actually work?", and just as
importantly, "did anything simply move somewhere else?"

Answering this credibly needs more than a before-and-after comparison for the
group that took part, because a company-wide trend, a seasonal effect, or a change
in how a metric is calculated can all masquerade as a programme effect. The
antidote is a simple quasi-experiment:

  - Compare a TREATED group (received the intervention) against a CONTROL group.
  - Split time into Before, During, and After windows.
  - Use a DIFFERENCE-IN-DIFFERENCES read: treated change minus control change.
  - DISCOUNT any signal that moves the same way in the control group, because a
    change that appears where there was no intervention cannot have been caused
    by it.

We demonstrate this on the built-in `pq_data` sample Person Query.
"""

import numpy as np
import pandas as pd
import vivainsights as vi

# ---------------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------------
pq = vi.load_pq_data()
pq["MetricDate"] = pd.to_datetime(pq["MetricDate"])

# Treated vs control. In practice you would flag the population that actually
# received the intervention; here we use one organisation for illustration.
TREATED_ORG = "IT"
pq["group"] = np.where(pq["Organization"] == TREATED_ORG, "Treated", "Control")

# Before / During / After as three equal windows across the date range.
d_min, d_max = pq["MetricDate"].min(), pq["MetricDate"].max()
span = d_max - d_min
cut1, cut2 = d_min + span / 3, d_min + 2 * span / 3
pq["period"] = pd.Categorical(
    np.select(
        [pq["MetricDate"] < cut1, pq["MetricDate"] < cut2],
        ["Before", "During"],
        default="After",
    ),
    categories=["Before", "During", "After"],
    ordered=True,
)

print(pq.groupby(["group", "period"], observed=True).size())

# The sample data contains no real intervention, so purely for demonstration we
# inject a modest, clearly labelled reduction in multitasking for the treated
# group during and after the programme. DELETE this block when using your own
# data; it exists only so the method has a real effect to detect.
treated = pq["group"] == "Treated"
pq.loc[treated & (pq["period"] == "During"), "Multitasking_hours"] *= 0.92
pq.loc[treated & (pq["period"] == "After"), "Multitasking_hours"] *= 0.80


# ---------------------------------------------------------------------------
# 2. Two-stage aggregation
# ---------------------------------------------------------------------------
# Aggregate in two stages: first to a typical value per person (so a few very
# heavy/light weeks do not dominate), then to a mean across people within each
# group and period.
def two_stage_summary(data, metric, id="PersonId", group="group", period="period"):
    stage1 = (
        data.groupby([id, group, period], observed=True)[metric]
        .mean()
        .rename("person_mean")
        .reset_index()
    )
    return (
        stage1.groupby([group, period], observed=True)["person_mean"]
        .agg(value="mean", n_persons="size")
        .reset_index()
    )


summ = two_stage_summary(pq, metric="Multitasking_hours")
print("\nTwo-stage summary:")
print(summ)

# ---------------------------------------------------------------------------
# 3. Difference-in-differences
# ---------------------------------------------------------------------------
# Each group's change from Before to After; the difference between them isolates
# the treated-specific effect (the control change captures what happened anyway).
wide = summ.pivot(index="group", columns="period", values="value")
wide["change_before_after"] = wide["After"] - wide["Before"]
print("\nBefore/After change by group:")
print(wide[["Before", "During", "After", "change_before_after"]])

did = (
    wide.loc["Treated", "change_before_after"]
    - wide.loc["Control", "change_before_after"]
)
print(f"\nDifference-in-differences (treated change - control change): {did:.3f}")

# Optional plot (uncomment to display).
# import matplotlib.pyplot as plt
# for g, sub in summ.groupby("group", observed=True):
#     plt.plot(sub["period"], sub["value"], marker="o", label=g)
# plt.ylabel("Multitasking hours / person / week"); plt.legend()
# plt.title("Weekly multitasking hours by period"); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# Reading the result honestly
# ---------------------------------------------------------------------------
# The difference-in-differences is the number to trust, not the treated group's
# before/after change on its own. If the control group had moved by a similar
# amount, we would conclude the shift was an organisation-wide or seasonal effect
# (or a metric-definition change) and DISCOUNT it, no matter how good the treated
# group's raw change looked.
#
# Two habits worth building in:
#   - Look for displacement: re-run the summary on adjacent behaviours (e.g.
#     After_hours_collaboration_hours, Weekend_collaboration_hours) to confirm the
#     load did not simply move elsewhere.
#   - Prefer two-stage aggregation so a few extreme individuals or unequal week
#     counts do not skew the comparison.
#
# This design is reusable for any workplace intervention, including an AI-adoption
# programme: frame a Copilot enablement wave as the treated group and read the
# difference-in-differences against a comparable control, rather than relying on a
# before-and-after that a company-wide trend could confound.

# Copy the summary to the clipboard to paste into Excel.
vi.export(summ)
