"""
Viva Insights - Regression & Machine Learning - Python Example Script
Within-person event-study & difference-in-differences (DiD).

A common question in Copilot Analytics is causal: did a person's collaboration
behaviour actually change after they started using Copilot, or are heavy users
simply different people to begin with? A plain users-vs-non-users comparison
cannot separate the two, because adoption is not random.

A within-person event-study paired with a two-way fixed-effects (TWFE)
difference-in-differences model is a standard way to address this. It aligns
every adopter on their own event time (weeks relative to adoption), compares
adopters against non-adopting controls over the same calendar weeks, and absorbs
each person's baseline (person fixed effects) and every week's common shocks
(week fixed effects). The coefficient on post x treated is the within-person
change for adopters, net of the controls.

Person Query exports do not ship with a clean adoption event, so this script
builds a small SEEDED SIMULATION whose column names match a real Person Query.
The downstream modelling code therefore runs on a real export loaded
with vi.import_query(); you swap out the data-generation block and map your
outcome columns (see below). The simulation injects a CLEARLY-LABELLED
illustrative effect so the model has something to recover; this is for
demonstration only and is not a real result.

Requires: linearmodels (`pip install linearmodels`) in addition to vivainsights.

To run on a real export instead of the simulation, replace section 1 with:

    pq = vi.import_query("your-person-query.csv")
    app_cols = [c for c in pq.columns if c.startswith("Copilot_actions_taken_in")]
    pq[app_cols] = pq[app_cols].fillna(0)
    pq["Total_Copilot_actions_taken"] = pq[app_cols].sum(axis=1)
    # pick the outcome column(s) you want to model, e.g. "Collaboration_hours",
    # "Chat_hours", "Emails_sent"; these are standard Person Query columns.
    panel = pq  # then continue from section 2 unchanged
"""

import vivainsights as vi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS

# ---------------------------------------------------------------------------
# 1. Simulate a Person-Query-shaped panel
# ---------------------------------------------------------------------------
# REPLACE THIS BLOCK FOR REAL DATA: load your export with vi.import_query() and
# derive the adoption week from the first week of non-zero Copilot actions (see
# section 2). Everything downstream stays the same.
rng = np.random.default_rng(100)

n_persons, n_weeks = 400, 40
weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")

# DEMO ONLY: the size of the behaviour change we inject into treated people after
# adoption. Set to 0 (or delete) for real data, because the model should ESTIMATE the
# effect, not have it baked in.
TREATMENT_EFFECT = 0.8  # extra collaboration hours/week once adopted

treated = rng.binomial(1, 0.6, n_persons)
base_collab = rng.normal(12, 3, n_persons)
adopt_week = np.where(treated == 1, rng.integers(10, 29, n_persons), -1)

persons = pd.DataFrame({
    "PersonId": [f"P{i:04d}" for i in range(n_persons)],
    "treated": treated,
    "base_collab": base_collab,
    "adopt_week": adopt_week,
})

week_idx = np.arange(1, n_weeks + 1)
season_by_week = 1.5 * np.sin(2 * np.pi * week_idx / 26)

# cross join person x week
panel = (
    persons.assign(key=1)
    .merge(pd.DataFrame({"key": 1, "week_idx": week_idx,
                         "MetricDate": weeks, "season": season_by_week}), on="key")
    .drop(columns="key")
)

panel["adopted_now"] = (panel["treated"] == 1) & (panel["week_idx"] >= panel["adopt_week"])

# Copilot actions: 0 before adoption, positive afterwards (treated only)
panel["Total_Copilot_actions_taken"] = np.where(
    panel["adopted_now"], rng.poisson(25, len(panel)), 0
)

# primary outcome: baseline + season + injected effect (post-adoption) + noise
panel["Collaboration_hours"] = (
    panel["base_collab"] + panel["season"]
    + np.where(panel["adopted_now"], TREATMENT_EFFECT, 0.0)
    + rng.normal(0, 2, len(panel))
)
# two correlated collaboration outcomes for the composite index later
panel["Chat_hours"] = 0.35 * panel["Collaboration_hours"] + rng.normal(0, 0.8, len(panel))
panel["Emails_sent"] = 20 + 1.2 * panel["Collaboration_hours"] + rng.normal(0, 4, len(panel))

panel = panel[[
    "PersonId", "MetricDate", "week_idx", "treated",
    "Total_Copilot_actions_taken",
    "Collaboration_hours", "Chat_hours", "Emails_sent",
]]

# ---------------------------------------------------------------------------
# 2. Derive the adoption week and event time
# ---------------------------------------------------------------------------
# On a real export you infer adoption from the data: a person is an ADOPTER if
# they ever record a non-zero Copilot action; their adoption week is the first
# such week. Controls (never any actions) get a PLACEBO adoption week (the median
# adopter week) so they contribute a comparable event-time window.
adopt = (
    panel[panel["Total_Copilot_actions_taken"] > 0]
    .groupby("PersonId")["MetricDate"].min().rename("adopt_date").reset_index()
)
placebo_date = adopt["adopt_date"].median()

panel = panel.merge(adopt, on="PersonId", how="left")
panel["is_adopter"] = panel["adopt_date"].notna()
panel["anchor"] = panel["adopt_date"].fillna(placebo_date)
panel["event_week"] = ((panel["MetricDate"] - panel["anchor"]).dt.days // 7).astype(int)

# Keep a balanced +/- 8-week window with >= 4 weeks each side
WINDOW = 8
panel_es = panel[(panel["event_week"] >= -WINDOW) & (panel["event_week"] <= WINDOW)].copy()
cover = panel_es.groupby("PersonId")["event_week"].agg(
    pre=lambda x: (x < 0).sum(), post=lambda x: (x > 0).sum()
)
keep = cover[(cover["pre"] >= 4) & (cover["post"] >= 4)].index
panel_es = panel_es[panel_es["PersonId"].isin(keep)].copy()

panel_es["treated_grp"] = panel_es["is_adopter"].astype(int)
panel_es["post"] = (panel_es["event_week"] >= 0).astype(int)
panel_es["treat_post"] = panel_es["post"] * panel_es["treated_grp"]

grp = panel_es.drop_duplicates("PersonId")["treated_grp"].value_counts()
print("Adopters (treated): {}   Non-adopters (control): {}".format(
    grp.get(1, 0), grp.get(0, 0)))

# ---------------------------------------------------------------------------
# 3. Headline TWFE difference-in-differences
# ---------------------------------------------------------------------------
# y_it = beta * (post_it x treated_i) + person_FE + week_FE + e_it, SE clustered
# by person. linearmodels PanelOLS wants a (entity, time) MultiIndex.
pdata = panel_es.set_index(["PersonId", "MetricDate"])

did = PanelOLS(
    pdata["Collaboration_hours"],
    pdata[["treat_post"]],
    entity_effects=True,
    time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

print(did.summary.tables[1])

did_beta = did.params["treat_post"]
print("\nDiD estimate: {:+.2f} collaboration hours/week "
      "(injected demo effect was {:+.2f})".format(did_beta, TREATMENT_EFFECT))
# The recovered coefficient should land close to TREATMENT_EFFECT. On real data
# you would NOT know the true value, and that is the point of estimating it.

# ---------------------------------------------------------------------------
# 4. Event-study: check pre-trends and watch the effect emerge
# ---------------------------------------------------------------------------
# One DiD number hides the parallel-trends assumption. The event-study estimates
# a separate coefficient per event week (relative to week -1). Pre-adoption
# coefficients should sit near zero; post-adoption ones step up to the effect.
es_df = panel_es.copy()
event_weeks = sorted(w for w in es_df["event_week"].unique() if w != -1)
dummy_cols = []
for k in event_weeks:
    col = f"evt_{k}"
    es_df[col] = ((es_df["event_week"] == k) & (es_df["treated_grp"] == 1)).astype(int)
    dummy_cols.append(col)

es_idx = es_df.set_index(["PersonId", "MetricDate"])
es = PanelOLS(
    es_idx["Collaboration_hours"],
    es_idx[dummy_cols],
    entity_effects=True,
    time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)

es_coefs = pd.DataFrame({
    "event_week": event_weeks,
    "estimate": [es.params[c] for c in dummy_cols],
    "se": [es.std_errors[c] for c in dummy_cols],
})
# add the reference week (-1) at exactly 0
es_coefs = pd.concat([
    es_coefs,
    pd.DataFrame({"event_week": [-1], "estimate": [0.0], "se": [0.0]}),
]).sort_values("event_week").reset_index(drop=True)
es_coefs["lwr"] = es_coefs["estimate"] - 1.96 * es_coefs["se"]
es_coefs["upr"] = es_coefs["estimate"] + 1.96 * es_coefs["se"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="grey")
ax.axvline(-0.5, ls="--", color="grey")
ax.fill_between(es_coefs["event_week"], es_coefs["lwr"], es_coefs["upr"],
                color="#5fa8d3", alpha=0.25)
ax.plot(es_coefs["event_week"], es_coefs["estimate"], color="#1b4965", marker="o")
ax.set_title("Event-study: collaboration hours around Copilot adoption")
ax.set_xlabel("Weeks relative to adoption")
ax.set_ylabel("Effect vs week -1 (hours/week)")
fig.tight_layout()
# A flat pre-period and a clean post-period step is the signature of a credible
# DiD. Trending pre-period coefficients would undermine the parallel-trends
# assumption. Run plt.show() to display.
# plt.show()

# ---------------------------------------------------------------------------
# 5. Bonus: run the same design on a z-scored composite index
# ---------------------------------------------------------------------------
# Individual metrics are noisy; combining related ones into a z-scored composite
# is more robust. Each component is standardised across the panel and averaged.
components = ["Collaboration_hours", "Chat_hours", "Emails_sent"]
for c in components:
    panel_es[f"z_{c}"] = (panel_es[c] - panel_es[c].mean()) / panel_es[c].std()
panel_es["collab_index"] = panel_es[[f"z_{c}" for c in components]].mean(axis=1)

idx_data = panel_es.set_index(["PersonId", "MetricDate"])
did_index = PanelOLS(
    idx_data["collab_index"],
    idx_data[["treat_post"]],
    entity_effects=True,
    time_effects=True,
).fit(cov_type="clustered", cluster_entity=True)
print("\nComposite-index DiD: {:+.3f} z-units".format(did_index.params["treat_post"]))

traj = (
    panel_es.assign(grp=np.where(panel_es["treated_grp"] == 1, "Adopter", "Control"))
    .groupby(["grp", "event_week"])["collab_index"].mean().reset_index()
)
fig2, ax2 = plt.subplots(figsize=(9, 4))
for g, colour in [("Adopter", "#1b4965"), ("Control", "#bc4b51")]:
    sub = traj[traj["grp"] == g]
    ax2.plot(sub["event_week"], sub["collab_index"], marker="o", label=g, color=colour)
ax2.axvline(-0.5, ls="--", color="grey")
ax2.set_title("Composite collaboration index around adoption")
ax2.set_xlabel("Weeks relative to adoption")
ax2.set_ylabel("Composite index (z-units)")
ax2.legend(frameon=False)
fig2.tight_layout()

# ---------------------------------------------------------------------------
# 6. Export the event-study coefficient table
# ---------------------------------------------------------------------------
vi.export(es_coefs)
