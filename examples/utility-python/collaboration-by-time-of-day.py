"""
Viva Insights - Collaboration by Time of Day (Python)

A common question is: "what time does the typical person start and end their
working day, and how does that differ by team, region, or role?" Viva Insights
exposes hourly "collaboration by time of day" metrics, which count the emails and
chats sent and the fractions of the hour spent in meetings and unscheduled calls
for each hour of the day, and these let us answer it.

This script shows how to turn those hourly buckets into a typical START and END
of day, using the same definition of an "active hour" as the Microsoft product:
an hour is active if ANY of chats sent, emails sent, meetings, or unscheduled
calls is greater than zero.

NOTE ON DATA
------------
The package's sample datasets do not include the hourly "by time of day" columns,
so this script SIMULATES a person-by-day dataset with those columns (named as an
`import_query()` of a Person or Daily query would name them). To run on your own
data, select the "Emails sent", "Chats sent", "Meetings", and "Unscheduled calls"
time-of-day metrics in a Person or Daily query, import it, and replace the
`simulate_time_of_day()` call below, and the downstream code is unchanged.
"""

import numpy as np
import pandas as pd
import vivainsights as vi

HOURS = range(24)
METRICS = ["Chats_sent", "Emails_sent", "Meetings", "Unscheduled_calls"]


# ---------------------------------------------------------------------------
# 0. Simulate a person-by-day dataset with hourly buckets
# ---------------------------------------------------------------------------
def simulate_time_of_day(n_person=200, n_days=20, seed=7):
    """Return a person-by-day frame with 24 hourly columns per collaboration metric.

    Each person has a latent start and end of day; managers end later. Activity is
    generated per hour, denser inside working hours and sparse outside them.
    """
    rng = np.random.default_rng(seed)
    persons = [f"P{i:03d}" for i in range(n_person)]
    is_manager = rng.random(n_person) < 0.25
    start_true = np.clip(rng.normal(8.5, 0.7, n_person), 6, 10)
    end_true = np.clip(rng.normal(18.0, 1.0, n_person) + is_manager * 1.0, 15, 22)

    # One row per person-day (weekdays only).
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([persons, dates], names=["PersonId", "MetricDate"])
    m = len(idx)
    person_pos = np.repeat(np.arange(n_person), n_days)

    hours = np.tile(np.arange(24), (m, 1))                # (m, 24)
    in_work = (hours >= start_true[person_pos, None]) & (hours < end_true[person_pos, None])
    active = rng.random((m, 24)) < np.where(in_work, 0.85, 0.02)

    # Metric buckets consistent with the active mask.
    data = {}
    for h in HOURS:
        a = active[:, h]
        col = f"{h:02d}_{h + 1:02d}"
        data[f"Chats_sent_{col}"] = a * rng.poisson(1.0, m)
        data[f"Emails_sent_{col}"] = a * rng.poisson(0.6, m)
        data[f"Meetings_{col}"] = a * rng.random(m) * 0.5           # fraction of hour
        data[f"Unscheduled_calls_{col}"] = a * (rng.random(m) < 0.1) * rng.random(m)

    out = pd.DataFrame(data, index=idx).reset_index()
    out["IsManager"] = np.repeat(is_manager, n_days)
    return out


# To use real data instead, replace this with vi.import_query("your_query.csv").
df = simulate_time_of_day()

# ---------------------------------------------------------------------------
# 1. Build the 24-hour activity matrix and find first / last active hour
# ---------------------------------------------------------------------------
# An hour is active if ANY of the four metrics is > 0 in that hour bucket.
def hour_columns(prefix):
    return [f"{prefix}_{h:02d}_{h + 1:02d}" for h in HOURS]


active_matrix = np.zeros((len(df), 24), dtype=bool)
for metric in METRICS:
    active_matrix |= df[hour_columns(metric)].to_numpy() > 0

has_any = active_matrix.any(axis=1)

# First active hour = index of the first True; last active hour = index of the
# last True. Start is the bucket START hour; End is the bucket END hour (+1).
first_hr = active_matrix.argmax(axis=1).astype(float)
last_hr = 23 - active_matrix[:, ::-1].argmax(axis=1)
start = first_hr
end = last_hr + 1.0

day_level = df[["PersonId", "MetricDate", "IsManager"]].copy()
day_level["start"] = np.where(has_any, start, np.nan)
day_level["end"] = np.where(has_any, end, np.nan)
day_level = day_level.dropna(subset=["start", "end"])   # drop days with no activity

# ---------------------------------------------------------------------------
# 2. Aggregate to a typical start and end of day
# ---------------------------------------------------------------------------
# Two-stage: each person's median across their days, then the mean across people.
# We also report the pooled median as a coarse cross-check.
def hhmm(x):
    """Format decimal hours (e.g. 8.5) as HH:MM (08:30)."""
    if pd.isna(x):
        return None
    h = int(x)
    return f"{h:02d}:{int(round((x - h) * 60)):02d}"


person_med = day_level.groupby("PersonId")[["start", "end"]].median()
typical = person_med.mean()          # two-stage mean (primary metric)
pooled = day_level[["start", "end"]].median()

print("Typical working day (two-stage person-then-mean):")
print(f"  start: {hhmm(typical['start'])}   end: {hhmm(typical['end'])}")
print("Pooled median (coarse check):")
print(f"  start: {hhmm(pooled['start'])}   end: {hhmm(pooled['end'])}")

# ---------------------------------------------------------------------------
# 3. Cuts by day of week and by role
# ---------------------------------------------------------------------------
day_level["weekday"] = pd.Categorical(
    day_level["MetricDate"].dt.day_name().str[:3],
    categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ordered=True,
)


def two_stage_by(group_cols):
    p = (day_level.groupby(["PersonId"] + group_cols, observed=True)[["start", "end"]]
         .median().reset_index())
    g = p.groupby(group_cols, observed=True)[["start", "end"]].mean().reset_index()
    g["start_hhmm"] = g["start"].map(hhmm)
    g["end_hhmm"] = g["end"].map(hhmm)
    return g


print("\nBy day of week:")
print(two_stage_by(["weekday"])[["weekday", "start_hhmm", "end_hhmm"]])

print("\nBy role (managers typically end later):")
by_role = two_stage_by(["IsManager"])
by_role["role"] = np.where(by_role["IsManager"], "Manager", "Non-manager")
print(by_role[["role", "start_hhmm", "end_hhmm"]])

# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
# - The two-stage (person-median then mean-across-people) metric picks up sub-hour
#   shifts that a single pooled median would miss, while staying robust to a few
#   unusual days per person.
# - Days with no recorded activity are excluded rather than counted as a 00:00
#   start, which would badly bias the result.
# - On real data, be careful about the timezone basis of the hourly buckets for a
#   globally distributed population; compare within a region where possible.

vi.export(two_stage_by(["weekday"]))
