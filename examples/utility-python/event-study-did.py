"""
Viva Insights - Regression & Machine Learning - Python Example Script
Within-person event-study & difference-in-differences (DiD).

A common question in Copilot Analytics is causal: did a person's collaboration
behaviour change after they started using Copilot, or are heavy users simply
different people to begin with? A plain users-vs-non-users comparison cannot
separate the two, because adoption is not random.

A within-person event-study paired with a difference-in-differences (DiD)
design is a standard way to approach this. It aligns every adopter on their own
event time (weeks relative to adoption), compares adopters against non-adopting
controls over the same calendar weeks, and absorbs each person's baseline
(person fixed effects) and every week's common shocks (week fixed effects).

Adoption here is STAGGERED: people adopt in different weeks. A single two-way
fixed-effects (TWFE) regression can give misleading estimates in that setting
when effects differ by adoption cohort or change over time, because already
adopted people can end up acting as comparisons for later adopters
(Goodman-Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020). The PRIMARY
estimate in this script is therefore a saturated event study in the spirit of
Sun and Abraham (2021), fitted with pyfixest: one coefficient per adoption
cohort and event week, never-adopters as the comparison group, then averaged
across cohorts. The naive TWFE event study is kept only as a labelled
comparison. All results are estimates under assumptions (chiefly parallel
trends and no anticipation) and are best read as associations consistent with
an effect rather than proof of one. People who already use Copilot in the first
weeks of the panel have no observed "before", so they cannot contribute to an
adoption estimate and the script drops them.

Person Query exports do not ship with a clean adoption event, so this script
builds a small SEEDED SIMULATION whose column names match a real Person Query.
The downstream modelling code therefore runs on a real export loaded
with vi.import_query(); you swap out the data-generation block and map your
outcome columns (see below). The simulation injects a CLEARLY-LABELLED
illustrative effect so the model has something to recover; this is for
demonstration only and is not a real result.

The primary estimate also conditions on organisational function. Copilot
rollouts often follow function, and functions can have their own trends, so the
model uses function-by-week fixed effects (Function^week_idx): adopters are
compared with never-adopters in the same function and the same calendar week.
See https://microsoft.github.io/viva-insights-sample-code/articles/causal-questions-ai-usage/#function-confounding
for why. The simulation adds a synthetic `Function` column with function-specific
trends and function-specific adoption timing.

Requires: pyfixest (`pip install pyfixest`; tested with 0.60.0 and 0.70.0) in addition to
vivainsights. Check the version you have installed. pyfixest's `pf.event_study` does not accept
extra fixed effects, so the estimate is fitted with `pf.feols` using saturated
cohort-by-relative-time interactions, `i(rel_time, C(cohort), ref=-1, ref2=0)`,
and `PersonId + Function^week_idx` fixed effects, then averaged across cohorts
with cell-size weights.

To run on a real export instead of the simulation, replace section 1 with:

    pq = vi.import_query("your-person-query.csv")
    app_cols = [c for c in pq.columns if c.startswith("Copilot_actions_taken_in")]
    pq[app_cols] = pq[app_cols].fillna(0)
    pq["Total_Copilot_actions_taken"] = pq[app_cols].sum(axis=1)
    # map your HR function attribute (for example FunctionType or Organization)
    # to Function. Use a value measured before adoption, so that it cannot be
    # changed by Copilot use. Freezing each person's first observed value keeps
    # a later move between functions out of the fixed effects.
    pq["Function"] = (pq.sort_values("MetricDate")
                        .groupby("PersonId")["FunctionType"].transform("first"))
    # pick the outcome column(s) you want to model, e.g. "Collaboration_hours",
    # "Chat_hours", "Emails_sent"; these are standard Person Query columns.
    panel = pq  # then continue from section 2 unchanged
"""

import re
import warnings
import vivainsights as vi
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt

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

# Synthetic organisational function. Rollout follows function: earlier-rolled-out
# functions also have rising collaboration trends, so function confounds adoption.
# Within each function, adoption weeks still vary and some people never adopt.
FUNCTIONS = ["Sales", "Engineering", "Operations", "Finance"]
P_ADOPT = {"Sales": 0.8, "Engineering": 0.6, "Operations": 0.5, "Finance": 0.4}
ROLLOUT_START = {"Sales": 10, "Engineering": 12, "Operations": 16, "Finance": 18}
FUNCTION_TREND = {"Sales": 0.04, "Engineering": 0.0, "Operations": -0.02, "Finance": -0.04}

function = rng.choice(FUNCTIONS, n_persons)
treated = rng.binomial(1, [P_ADOPT[f] for f in function])
base_collab = rng.normal(12, 3, n_persons)
start = np.array([ROLLOUT_START[f] for f in function])
adopt_week = np.where(treated == 1, start + rng.integers(0, 11, n_persons), -1)

persons = pd.DataFrame({
    "PersonId": [f"P{i:04d}" for i in range(n_persons)],
    "Function": function,
    "treated": treated,
    "base_collab": base_collab,
    "adopt_week": adopt_week,
})

week_idx = np.arange(1, n_weeks + 1)
season_by_week = 1.5 * np.sin(2 * np.pi * week_idx / 26)

# function-specific week shocks: a linear trend plus random weekly noise
function_shock = pd.DataFrame(
    [(f, w, FUNCTION_TREND[f] * w + rng.normal(0, 0.5))
     for f in FUNCTIONS for w in week_idx],
    columns=["Function", "week_idx", "function_shock"],
)

# cross join person x week
panel = (
    persons.assign(key=1)
    .merge(pd.DataFrame({"key": 1, "week_idx": week_idx,
                         "MetricDate": weeks, "season": season_by_week}), on="key")
    .drop(columns="key")
    .merge(function_shock, on=["Function", "week_idx"])
)

panel["adopted_now"] = (panel["treated"] == 1) & (panel["week_idx"] >= panel["adopt_week"])

# Copilot actions: 0 before adoption, positive afterwards (treated only)
panel["Total_Copilot_actions_taken"] = np.where(
    panel["adopted_now"], rng.poisson(25, len(panel)), 0
)

# primary outcome: baseline + season + function shock + injected effect + noise
panel["Collaboration_hours"] = (
    panel["base_collab"] + panel["season"] + panel["function_shock"]
    + np.where(panel["adopted_now"], TREATMENT_EFFECT, 0.0)
    + rng.normal(0, 2, len(panel))
)
# two correlated collaboration outcomes for the composite index later
panel["Chat_hours"] = 0.35 * panel["Collaboration_hours"] + rng.normal(0, 0.8, len(panel))
panel["Emails_sent"] = 20 + 1.2 * panel["Collaboration_hours"] + rng.normal(0, 4, len(panel))

panel = panel[[
    "PersonId", "Function", "MetricDate", "week_idx", "treated",
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

# Calendar-week index derived from MetricDate (1 = first week in the panel), so
# this section works unchanged on a real export.
panel["week_idx"] = (
    (panel["MetricDate"] - panel["MetricDate"].min()).dt.days // 7 + 1
).astype(int)

# Adoption cohort (calendar week index of first adoption) for the staggered-robust
# estimator. pyfixest expects 0 for never-treated units.
first_idx = (
    panel[panel["Total_Copilot_actions_taken"] > 0]
    .groupby("PersonId")["week_idx"].min().rename("cohort").reset_index()
)
panel = panel.merge(first_idx, on="PersonId", how="left")
panel["cohort"] = panel["cohort"].fillna(0).astype(int)

# Early adopters with no observed pre-period cannot contribute to an adoption
# estimate: they look like people treated for the whole window, so the estimator
# has no "before" for them (see "check pre-adoption" in
# https://microsoft.github.io/viva-insights-sample-code/articles/causal-questions-ai-usage/#check-pre-adoption ).
# Drop people with fewer than MIN_PRE observed pre-adoption weeks.
MIN_PRE = 4
n_pre = panel[(panel["cohort"] > 0) & (panel["week_idx"] < panel["cohort"])].groupby("PersonId").size()
adopter_ids = panel.loc[panel["cohort"] > 0, "PersonId"].unique()
early = [p for p in adopter_ids if n_pre.get(p, 0) < MIN_PRE]
print("Dropped {} people who adopted with fewer than {} observed pre-adoption weeks "
      "(no usable 'before').".format(len(early), MIN_PRE))
panel = panel[~panel["PersonId"].isin(early)].copy()
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
# (used for the naive TWFE comparison only)
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
# 3. Primary estimate: staggered-robust (saturated) event study
# ---------------------------------------------------------------------------
# pyfixest fits one coefficient per adoption cohort x event week, with the never-
# adopters (cohort 0) as the comparison group, then averages the cohort-specific
# estimates into one estimate per event week (Sun and Abraham style weighting).
# Person fixed effects and function-by-week fixed effects are absorbed, so
# adopters are compared with never-adopters in the same function and calendar
# week. Standard errors are clustered by person. Event week -1 is the reference.
# pf.event_study() cannot take extra fixed effects, so the saturated model is
# fitted directly with pf.feols() and averaged below.


def saturated_event_study(df, yname, fe="PersonId + Function^week_idx"):
    """Fit the saturated event study and return (fit, by-week table, ATT, lead test)."""
    d = df.assign(rel_time=np.where(df["cohort"] > 0, df["week_idx"] - df["cohort"], -1))
    with warnings.catch_warnings():
        # Silence only the expected reference-period collinearity notice
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=r"(?s).*variables dropped due to multicollinearity.*")
        fit = pf.feols(
            f"{yname} ~ i(rel_time, C(cohort), ref=-1, ref2=0) | {fe}",
            data=d, vcov={"CRV1": "PersonId"},
        )

    # Linear combinations of the cohort x week coefficients, tested with the
    # public fit.wald_test(). Weights are cohort x event-week cell sizes.
    b = fit.coef()
    cell_n = (
        d[d["cohort"] > 0]
        .groupby(["cohort", "rel_time"]).size()
    )
    parsed = []
    for i, nm in enumerate(b.index):
        m = re.search(r"::(-?\d+(?:\.\d+)?):.*::(\d+)$", str(nm))
        parsed.append((i, float(m.group(1)), int(m.group(2))) if m else None)

    def combo(keep):
        w = np.zeros(len(b))
        for p in parsed:
            if p and keep(p[1]):
                w[p[0]] = cell_n.get((p[2], int(p[1])), 0)
        return w / w.sum() if w.sum() > 0 else w

    def wald(R):
        # pyfixest 0.60.0 returns a Series, 0.70.0 returns a WaldTest object
        res = fit.wald_test(R=R, distribution="chi2")
        if hasattr(res, "stat"):
            return float(res.stat), float(res.pvalue)
        return float(res.iloc[0]), float(res.iloc[1])

    def estimate(r):
        # For one combination the Wald statistic is estimate^2 / variance, which
        # recovers the standard error without touching private attributes.
        # An estimate of exactly zero gives 0/0, so report NaN rather than dividing.
        est = float(r @ b.values)
        wald_stat, _ = wald(r.reshape(1, -1))
        se = abs(est) / np.sqrt(wald_stat) if est != 0 and wald_stat > 0 else np.nan
        return est, se

    # One estimate per event week, averaged over cohorts
    rows = []
    for k in range(-WINDOW, WINDOW + 1):
        r = combo(lambda t, k=k: t == k)
        if k != -1 and r.any():
            est, se = estimate(r)
            rows.append({"event_week": k, "estimate": est, "se": se,
                         "lwr": est - 1.96 * se, "upr": est + 1.96 * se})
    by_week = pd.DataFrame(rows)

    # ATT: average over all post-adoption event weeks, weighted by cell size.
    att = estimate(combo(lambda t: t >= 0))

    # Joint Wald test on the pre-adoption leads in the plotted window (-8 to -2)
    leads = [t for t in range(-WINDOW, -1)]
    R = np.vstack([combo(lambda t, k=k: t == k) for k in leads])
    lead_stat, lead_p = wald(R)
    return (fit, by_week, att, (lead_stat, len(leads), lead_p))


fit_sa, sa_week, (att, att_se), (pre_chi2, pre_df, pre_p) = saturated_event_study(
    panel, "Collaboration_hours")

print("\nSun-Abraham style ATT estimate (function-by-week FE): {:+.2f} collaboration "
      "hours/week (SE {:.2f}; injected demo effect was {:+.2f})".format(
          att, att_se, TREATMENT_EFFECT))
print("Joint test of {} pre-adoption leads: chi-squared = {:.2f}, p = {:.3f}".format(
    pre_df, pre_chi2, pre_p))

# Comparison: the same saturated model with week fixed effects only, which does
# not condition on function.
_, _, (att_nf, att_nf_se), (pre_chi2_nf, _, pre_p_nf) = saturated_event_study(
    panel, "Collaboration_hours", fe="PersonId + week_idx")
print("Same model, week FE only (no function conditioning): ATT {:+.2f} (SE {:.2f}); "
      "lead test chi-squared = {:.2f}, p = {:.3f}".format(
          att_nf, att_nf_se, pre_chi2_nf, pre_p_nf))
# A non-significant pre-test does not establish parallel trends. The test can
# have low power and cannot rule out a trend that starts at adoption. Read it
# together with the plot and with what you know about why people adopted when
# they did.
# The recovered estimate should land close to TREATMENT_EFFECT. On real data you
# would NOT know the true value, and the estimate depends on the assumptions above.

# ---------------------------------------------------------------------------
# 4. Event-study plot and coefficient table
# ---------------------------------------------------------------------------
# One average hides the parallel-trends assumption. The event-study shows an
# estimate for each event week (relative to week -1). Pre-adoption estimates
# should sit near zero; post-adoption ones should step up to the effect.
es_coefs = sa_week[["event_week", "estimate", "se", "lwr", "upr"]].astype(float)
es_coefs = es_coefs[(es_coefs["event_week"] >= -WINDOW) & (es_coefs["event_week"] <= WINDOW)]
# add the reference week (-1) at exactly 0
es_coefs = pd.concat([
    es_coefs,
    pd.DataFrame({"event_week": [-1.0], "estimate": [0.0], "se": [0.0],
                  "lwr": [0.0], "upr": [0.0]}),
]).sort_values("event_week").reset_index(drop=True)
es_coefs["event_week"] = es_coefs["event_week"].astype(int)

fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0, color="grey")
ax.axvline(-0.5, ls="--", color="grey")
ax.fill_between(es_coefs["event_week"], es_coefs["lwr"], es_coefs["upr"],
                color="#5fa8d3", alpha=0.25)
ax.plot(es_coefs["event_week"], es_coefs["estimate"], color="#1b4965", marker="o")
ax.set_title("Event-study (saturated, staggered-robust): collaboration hours around adoption")
ax.set_xlabel("Weeks relative to adoption")
ax.set_ylabel("Estimate vs week -1 (hours/week)")
fig.tight_layout()
# Trending pre-period estimates would undermine the parallel-trends assumption.
# Run plt.show() to display.
# plt.show()

# ---------------------------------------------------------------------------
# 4b. Comparison: naive TWFE (shown for comparison)
# ---------------------------------------------------------------------------
# For comparison only. With staggered adoption and possibly different effects by
# cohort, a single TWFE coefficient and the TWFE event study can be contaminated,
# so treat them as a sensitivity check against the estimate above, not as the
# headline. They use the +/- 8-week window around each person's anchor, with a
# placebo anchor for controls.
did = pf.feols("Collaboration_hours ~ treat_post | PersonId + week_idx",
               data=panel_es, vcov={"CRV1": "PersonId"})
print("\nNaive TWFE DiD estimate (shown for comparison): {:+.2f} collaboration "
      "hours/week (injected demo effect was {:+.2f})".format(
          did.coef()["treat_post"], TREATMENT_EFFECT))

es_twfe = pf.feols("Collaboration_hours ~ i(event_week, treated_grp, ref=-1) | PersonId + week_idx",
                   data=panel_es, vcov={"CRV1": "PersonId"})
tw = es_twfe.tidy().reset_index()
tw["event_week"] = tw["Coefficient"].str.extract(r"event_week::(-?\d+)")[0].astype(float)
tw = tw.dropna(subset=["event_week"]).sort_values("event_week")

fig_tw, ax_tw = plt.subplots(figsize=(9, 5))
ax_tw.axhline(0, color="grey")
ax_tw.axvline(-0.5, ls="--", color="grey")
ax_tw.fill_between(tw["event_week"], tw["2.5%"], tw["97.5%"], color="#bc4b51", alpha=0.2)
ax_tw.plot(tw["event_week"], tw["Estimate"], color="#bc4b51", marker="o")
ax_tw.set_title("Naive TWFE event-study (shown for comparison)")
ax_tw.set_xlabel("Weeks relative to adoption")
ax_tw.set_ylabel("Estimate vs week -1 (hours/week)")
fig_tw.tight_layout()

# ---------------------------------------------------------------------------
# 5. Bonus: run the same design on a z-scored composite index
# ---------------------------------------------------------------------------
# Individual metrics are noisy; combining related ones into a z-scored composite
# can help. Each component is standardised across the panel and averaged.
components = ["Collaboration_hours", "Chat_hours", "Emails_sent"]
for c in components:
    panel[f"z_{c}"] = (panel[c] - panel[c].mean()) / panel[c].std()
panel["collab_index"] = panel[[f"z_{c}" for c in components]].mean(axis=1)
panel_es = panel_es.merge(
    panel[["PersonId", "week_idx", "collab_index"]], on=["PersonId", "week_idx"])

_, _, (att_index, att_index_se), _ = saturated_event_study(panel, "collab_index")
print("\nComposite-index ATT (saturated event study): {:+.3f} z-units (SE {:.3f})".format(
    att_index, att_index_se))

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
