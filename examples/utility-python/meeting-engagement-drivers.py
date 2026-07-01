"""
Viva Insights - Meeting Engagement Drivers (Python)

Many meeting-culture conversations start with "we have too many meetings". A more
useful question is often "are we meeting WELL?", that is, are people engaged, or
quietly working on something else while the meeting runs?

This script treats in-meeting messaging (chats and emails sent during a meeting)
as a proxy for disengagement, and asks: which characteristics of a meeting best
predict how much backchannel messaging it generates? We:

  1. Prepare a meeting-level dataset (one row per meeting).
  2. Fit a random forest to rank the design features that drive per-attendee messaging.
  3. Take a closer look at the top driver (meeting duration) and separate a genuine
     effect from the trivial "a longer meeting simply has more minutes" explanation.

NOTE ON DATA
------------
The package ships a sample Meeting Query, `vi.load_mt_data()`, but it is small and
dominated by single-person focus blocks, so it does not contain enough multi-person
meetings or in-meeting messaging to illustrate this model. We therefore SIMULATE a
realistic meeting-level dataset below, using the same column names as a Meeting
Query export. To run on your own data, replace the `simulate_meeting_query()` call
with `vi.import_query()` of your own Meeting Query, and the downstream code is
unchanged. (`vi.load_mt_data()` is useful for inspecting the expected schema, but is
too small and focus-block-heavy to fit this model.)

Results are described as ASSOCIATION, not causation.
"""

import numpy as np
import pandas as pd
import vivainsights as vi
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# 0. Simulate a realistic Meeting Query (swap for real data on your own tenant)
# ---------------------------------------------------------------------------
def simulate_meeting_query(n=2000, seed=42):
    """Return a data frame shaped like a Viva Insights Meeting Query.

    Messaging intensity is generated so that DURATION is the dominant driver,
    with a sub-linear (exponent < 1) relationship, so total messaging rises
    with length while per-minute messaging gently declines. This creates a
    plausible pattern for demonstration and lets the example recover it.
    """
    rng = np.random.default_rng(seed)

    duration_min = rng.choice([15, 30, 45, 60, 90, 120, 180], size=n,
                              p=[.14, .28, .20, .18, .10, .06, .04])
    n_attendees = 2 + rng.poisson(4, n)                 # at least 2
    intended = n_attendees + rng.poisson(1, n)
    recurring = rng.binomial(1, 0.5, n)

    # RSVP responses split out of the intended invitees
    accept = rng.binomial(intended, 0.70)
    remaining = intended - accept
    no_response = rng.binomial(remaining, 0.60)
    decline = remaining - no_response
    redundant = rng.binomial(n_attendees, 0.10)

    no_response_rate = no_response / np.maximum(intended, 1)

    # Per-attendee messaging rate: duration dominates, sub-linearly (^0.85)
    lam = (0.03 * duration_min ** 0.85
           * (1 + 0.8 * no_response_rate)
           * (1 + 0.05 * np.log(n_attendees)))
    chats = rng.poisson(lam * 0.5 * n_attendees)
    emails = rng.poisson(lam * 0.5 * n_attendees)

    return pd.DataFrame({
        "Number_of_attendees": n_attendees,
        "Number_of_attendees_multitasking": rng.binomial(n_attendees, 0.3),
        "Number_of_chats_sent_during_the_meeting": chats,
        "Number_of_emails_sent_during_the_meeting": emails,
        "Number_of_redundant_attendees": redundant,
        "All_Day_Meeting": "FALSE",
        "Cancelled": "FALSE",
        "Recurring": np.where(recurring == 1, "TRUE", "FALSE"),
        "Accept_count": accept,
        "No_response_count": no_response,
        "Decline_count": decline,
        "Intended_participant_count": intended,
        "Duration": duration_min / 60.0,          # Duration is in HOURS
    })


# To use real data instead, comment the line below and use an exported Meeting Query:
#   mt = vi.import_query("my_meeting_query.csv")
# (vi.load_mt_data() is useful only for inspecting the expected schema here.)
mt = simulate_meeting_query()

# ---------------------------------------------------------------------------
# 1. Clean and prepare the modelling frame
# ---------------------------------------------------------------------------
def as_flag(series):
    """Coerce TRUE/FALSE-style flags to booleans regardless of storage type."""
    return series.astype(str).str.upper().map({"TRUE": True, "FALSE": False})


mt["Cancelled"] = as_flag(mt["Cancelled"])
mt["All_Day_Meeting"] = as_flag(mt["All_Day_Meeting"])
mt["Recurring"] = as_flag(mt["Recurring"])

# Light quality filter: genuine meetings only (drop solo holds, cancelled, all-day).
mt = mt[
    (mt["Number_of_attendees"] >= 2)
    & (~mt["Cancelled"])
    & (~mt["All_Day_Meeting"])
    & (mt["Duration"] > 0)
].copy()

print(f"Meetings after filtering: {len(mt)}")

# `Duration` is in HOURS; convert to minutes. Normalise messaging PER ATTENDEE so
# large meetings don't look busier by default, and derive RSVP commitment signals.
intended = mt["Intended_participant_count"].clip(lower=1)
attendees = mt["Number_of_attendees"].clip(lower=1)

model_df = pd.DataFrame({
    "chats_per_att":  mt["Number_of_chats_sent_during_the_meeting"] / attendees,
    "emails_per_att": mt["Number_of_emails_sent_during_the_meeting"] / attendees,
    "duration_min":          mt["Duration"] * 60,
    "n_attendees":           mt["Number_of_attendees"],
    "intended_participants": mt["Intended_participant_count"],
    "recurring":             mt["Recurring"].astype(int),
    "accept_rate":           mt["Accept_count"] / intended,
    "no_response_rate":      mt["No_response_count"] / intended,
    "decline_rate":          mt["Decline_count"] / intended,
    "redundant_share":       mt["Number_of_redundant_attendees"] / attendees,
}).dropna()

FEATURES = [
    "duration_min", "n_attendees", "intended_participants", "recurring",
    "accept_rate", "no_response_rate", "decline_rate", "redundant_share",
]

# ---------------------------------------------------------------------------
# 2. Rank the drivers with a random forest (permutation importance)
# ---------------------------------------------------------------------------
# We fit one model per outcome (chats and emails per attendee) to read off the
# out-of-bag R-squared for each, then compute permutation importance on the
# primary (chats) model. Permutation importance shuffles each feature in turn and
# measures the loss in accuracy, which is more faithful than impurity importance.
# n_jobs is left at 1 for portability (avoids multiprocessing issues on Windows).
X = model_df[FEATURES]


def fit_rf(outcome, n_estimators=300):
    rf = RandomForestRegressor(
        n_estimators=n_estimators, oob_score=True, random_state=123, n_jobs=1
    )
    rf.fit(X, model_df[outcome])
    return rf


rf_chats = fit_rf("chats_per_att")
rf_emails = fit_rf("emails_per_att")

# OOB R-squared: how much of the variation the features explain. Modest values are
# expected for human behaviour; we report ASSOCIATION, not cause.
print("\nOOB R-squared:")
print(f"  chats per attendee : {rf_chats.oob_score_:.3f}")
print(f"  emails per attendee: {rf_emails.oob_score_:.3f}")

perm = permutation_importance(
    rf_chats, X, model_df["chats_per_att"],
    n_repeats=10, random_state=123, n_jobs=1
)
imp_chats = (
    pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nPermutation importance (chats per attendee):")
print(imp_chats)

# Optional bar chart of the driver ranking (uncomment to display).
# import matplotlib.pyplot as plt
# imp_chats.sort_values("importance").plot.barh(
#     x="feature", y="importance", color="#1B3A6B", legend=False)
# plt.title("What drives in-meeting chat, per attendee")
# plt.tight_layout(); plt.show()

# Meeting duration typically ranks at or near the top, which makes it a practical
# lever, because duration is one of the few meeting properties you can change with
# a calendar setting.

# ---------------------------------------------------------------------------
# 3. A closer look at duration: rate vs exposure
# ---------------------------------------------------------------------------
# Objection: isn't "longer meetings have more messaging" trivial, since a longer
# meeting simply contains more minutes? We compare TOTAL messaging per attendee
# against the same figure PER MINUTE across duration bands.
dose = model_df.assign(
    msgs_per_att=lambda d: d["chats_per_att"] + d["emails_per_att"],
    band=lambda d: pd.cut(
        d["duration_min"],
        bins=[0, 15, 30, 60, 120, np.inf],
        labels=["<=15m", "16-30m", "31-60m", "61-120m", ">120m"],
    ),
)
dose_summary = (
    dose.groupby("band", observed=True)
    .apply(lambda g: pd.Series({
        "n_meetings": len(g),
        "msgs_per_att": g["msgs_per_att"].mean(),
        "msgs_per_att_permin": (g["msgs_per_att"] / g["duration_min"]).mean(),
    }), include_groups=False)
    .reset_index()
)

print("\nDuration dose-response (total vs per-minute):")
print(dose_summary)

# Usual pattern: total messaging per attendee RISES with length, while the
# per-minute rate is roughly flat or gently declining. Longer meetings are not
# disproportionately distracting minute-for-minute, but disengagement does not
# self-correct, and instead it accumulates across a long session.

# Simple linear model: marginal effect of length, holding meeting size constant.
lm = LinearRegression().fit(
    model_df[["duration_min", "n_attendees"]],
    model_df["chats_per_att"] + model_df["emails_per_att"],
)
print(f"\nExtra messages per attendee per +30 min: {lm.coef_[0] * 30:.2f}")

# ---------------------------------------------------------------------------
# What this means
# ---------------------------------------------------------------------------
# The result is NOT that multitasking is unavoidable or that the only fix is to
# cut meeting volume. The messaging rate responds to how a meeting is designed,
# and length is the part of the design that most reliably manufactures
# disengagement. Shorter meetings help twice: fewer people drift off, and the
# drift that happens has less time to accumulate. In-meeting messaging is a
# MOVABLE signal worth tracking.
#
# Transferable practices: model at the meeting level (not person level);
# normalise messaging per attendee; rank with permutation importance and
# corroborate the top driver with an interpretable model; split any "bigger
# total" into rate and exposure; and report association, not causation.
#
# Link to AI adoption: some in-meeting messaging is people trying to stay
# productive while stuck in a meeting that does not need all of them. Copilot
# meeting recaps let people skip or leave lower-value meetings and catch up
# afterwards, complementing (not replacing) the lever of shortening meetings.

# Copy the driver ranking to the clipboard to paste into Excel.
vi.export(imp_chats)
