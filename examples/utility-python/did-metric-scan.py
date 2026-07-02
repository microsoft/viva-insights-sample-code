"""
Viva Insights - Regression & Machine Learning - Python Example Script
Scanning many metrics with a difference-in-differences model (Power vs Low).

A frequent request in Copilot Analytics is a single table that answers: across
all our collaboration metrics, on which ones do heavier Copilot users change
their behaviour, by how much, and which changes are statistically significant?

A naive comparison of group means (Power users send more emails than Low users)
is cross-sectional and confounded. This script instead runs a within-person
difference-in-differences (DiD) model PER METRIC and assembles the results into
one tidy, sortable table plus a forest plot. The design compares two
both-licensed groups (Power vs Low Copilot users), so the contrast is usage
intensity, not licence access.

For a single-metric event-study and the parallel-trends assumption, see the
companion `event-study-did` example; here the focus is the scan across metrics
and honest reporting of significance, including metrics that do NOT move.

As in that example, a Person Query does not ship with a clean adoption event, so
we build a small SEEDED SIMULATION whose column names match a real Person Query.
Swap the simulation block for vi.import_query() and the scan runs unchanged. The
per-metric effects are injected for demonstration only and are not real results.

Requires: linearmodels (`pip install linearmodels`) in addition to vivainsights.
"""

import vivainsights as vi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# ---------------------------------------------------------------------------
# 1. Simulate a Person-Query-shaped panel with two intensity groups
# ---------------------------------------------------------------------------
# Everyone is a licensed adopter with an adoption week; they differ in intensity,
# where Power users ramp to heavy Copilot use and Low users stay light. We simulate
# several standard Person Query collaboration columns and inject a different
# post-adoption change for Power vs Low on each. One differential is zero on
# purpose, so the scan shows a realistic mix of significant / non-significant.
#
# REPLACE THIS BLOCK FOR REAL DATA: load with vi.import_query(), classify
# Power/Low with vi.identify_usage_segments(version="12w"), and derive each
# person's anchor week from their first Copilot action.
rng = np.random.default_rng(202)

n_persons, n_weeks = 500, 40
weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

# DEMO ONLY: extra post-adoption change for POWER relative to LOW users, per
# metric. Delete / set to 0 for real data. One metric is deliberately null.
EFFECTS = {
    "Emails_sent": 1.5,
    "Chats_sent": 3.0,
    "Meeting_hours": 0.4,
    "Collaboration_hours": 0.8,
    "After_hours_collaboration_hours": 0.15,
    "Channel_message_posts": 0.0,  # deliberately null
}
BASE_MEAN = {"Emails_sent": 22, "Chats_sent": 30, "Meeting_hours": 10,
             "Collaboration_hours": 14, "After_hours_collaboration_hours": 2,
             "Channel_message_posts": 6}
NOISE_SD = {"Emails_sent": 5, "Chats_sent": 7, "Meeting_hours": 2.5,
            "Collaboration_hours": 3, "After_hours_collaboration_hours": 0.8,
            "Channel_message_posts": 2}

persons = pd.DataFrame({
    "PersonId": [f"P{i:04d}" for i in range(n_persons)],
    "group": rng.choice(["Power User", "Low User"], n_persons, p=[0.55, 0.45]),
    "adopt_wk": rng.integers(10, 27, n_persons),
    "base_lvl": rng.normal(0, 1, n_persons),
})

week_idx = np.arange(1, n_weeks + 1)
grid = (
    persons.assign(key=1)
    .merge(pd.DataFrame({"key": 1, "week_idx": week_idx,
                         "MetricDate": weeks,
                         "season": 1.2 * np.sin(2 * np.pi * week_idx / 26)}),
           on="key")
    .drop(columns="key")
)
grid["post"] = grid["week_idx"] >= grid["adopt_wk"]
grid["is_power"] = grid["group"] == "Power User"

for m, eff in EFFECTS.items():
    grid[m] = (
        BASE_MEAN[m] + 3 * grid["base_lvl"] + grid["season"]
        + np.where(grid["post"] & grid["is_power"], eff, 0.0)
        + rng.normal(0, NOISE_SD[m], len(grid))
    )

panel = grid[["PersonId", "MetricDate", "week_idx", "group", "adopt_wk", *EFFECTS]]

# ---------------------------------------------------------------------------
# 2. Build event time and DiD indicators (balanced +/- 8-week window)
# ---------------------------------------------------------------------------
WINDOW = 8
panel_es = panel.assign(event_week=panel["week_idx"] - panel["adopt_wk"])
panel_es = panel_es[(panel_es["event_week"] >= -WINDOW) &
                    (panel_es["event_week"] <= WINDOW)].copy()
cover = panel_es.groupby("PersonId")["event_week"].agg(
    pre=lambda x: (x < 0).sum(), post=lambda x: (x > 0).sum())
keep = cover[(cover["pre"] >= 4) & (cover["post"] >= 4)].index
panel_es = panel_es[panel_es["PersonId"].isin(keep)].copy()

panel_es["treated"] = (panel_es["group"] == "Power User").astype(int)
panel_es["post"] = (panel_es["event_week"] >= 0).astype(int)
panel_es["treat_post"] = panel_es["post"] * panel_es["treated"]
print(panel_es.drop_duplicates("PersonId")["group"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 3. Run the DiD once per metric and collect results
# ---------------------------------------------------------------------------
# y ~ treat_post + person_FE + week_FE, SE clustered by person.
pdata = panel_es.set_index(["PersonId", "MetricDate"])


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


rows = []
for m in EFFECTS:
    fit = PanelOLS(pdata[m], pdata[["treat_post"]],
                   entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    est = fit.params["treat_post"]
    lo, hi = fit.conf_int().loc["treat_post"]
    p = fit.pvalues["treat_post"]
    base_pre = panel_es.loc[(panel_es["treated"] == 1) & (panel_es["post"] == 0), m].mean()
    rows.append({
        "metric": m, "estimate": est, "conf_low": lo, "conf_high": hi,
        "p_value": p, "sig": stars(p),
        "baseline_pre": base_pre, "pct_of_baseline": est / base_pre,
    })

results = pd.DataFrame(rows).sort_values("estimate", ascending=False).reset_index(drop=True)

display = results.assign(
    estimate=results["estimate"].round(3),
    CI=results.apply(lambda r: f"[{r.conf_low:+.2f}, {r.conf_high:+.2f}]", axis=1),
    pct=(results["pct_of_baseline"] * 100).round(1).astype(str) + "%",
    p=results["p_value"].map(lambda x: f"{x:.2g}"),
)[["metric", "estimate", "CI", "pct", "p", "sig"]]
display.columns = ["Metric", "Delta (units)", "95% CI", "% of baseline", "p", "Sig"]
print("\nPower vs Low DiD (one row per metric, sorted by effect):")
print(display.to_string(index=False))
# Read as: within-person, Power users changed this metric by Delta more than Low
# users after adoption. The Sig column separates signal from noise, and the
# deliberately null metric should land n.s.

# ---------------------------------------------------------------------------
# 4. Forest plot
# ---------------------------------------------------------------------------
fp = results.sort_values("pct_of_baseline")
colours = np.where(fp["sig"] != "n.s.", "#1b4965", "#bc4b51")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axvline(0, color="grey")
ax.errorbar(
    fp["pct_of_baseline"], range(len(fp)),
    xerr=[fp["pct_of_baseline"] - fp["conf_low"] / fp["baseline_pre"],
          fp["conf_high"] / fp["baseline_pre"] - fp["pct_of_baseline"]],
    fmt="none", ecolor=colours, elinewidth=1.5, capsize=3,
)
ax.scatter(fp["pct_of_baseline"], range(len(fp)), color=colours, zorder=3)
ax.set_yticks(range(len(fp)))
ax.set_yticklabels(fp["metric"])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set_title("Power vs Low Copilot users: DiD effect by metric")
ax.set_xlabel("Effect as % of pre-adoption baseline (95% CI)")
fig.tight_layout()
# Intervals crossing the zero line are not distinguishable from no change.
# Run plt.show() to display.
# plt.show()

# ---------------------------------------------------------------------------
# 5. Export the results table
# ---------------------------------------------------------------------------
vi.export(results)
