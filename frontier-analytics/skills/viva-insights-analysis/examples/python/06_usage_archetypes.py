"""
Example 6 - Experimental usage archetypes (reference rule v1).

EXPERIMENTAL, SYNTHETIC DATA ONLY. The script generates its own seeded,
synthetic person-week panel. It reads no files and contains no real export,
customer data or published result. The printed numbers illustrate the
mechanics only and say nothing about real Copilot use. See
reference/usage-archetypes.md for the rule, evidence states, context record
and publication rules.

Steps:
1. Generate a synthetic 12-week person-week panel (Sunday-start weeks).
2. Assign every person a named evidence state before any label.
3. For full profiles, assign Consistency (C/E), Surface mix (S/B, integer
   rule on group-weeks) and Metered use (L/I, median split of positive M365
   credits, ties Lean).
4. Print the context record, a privacy-safe summary that withholds groups
   under 10, and leave-one-week and split-half sensitivity diagnostics.

The Power / Habitual / Novice / Low / Non-user ladder (example 3) is unchanged
and should be reported alongside these descriptive labels. Labels describe a
period. Never use them to rank or monitor individuals.

The random generator is a minimal-standard (Park-Miller) generator written out
identically in r/06_usage_archetypes.R, so both scripts build the same
synthetic panel. Identical output was checked manually for the committed seed.
"""
import pandas as pd

SEED = 20250105
N_PEOPLE = 600
N_WEEKS = 12
WINDOW_START = pd.Timestamp("2025-01-05")  # a Sunday
RULE_VERSION = "usage archetypes, reference rule v1 (experimental)"
MIN_GROUP_SIZE = 10
CONSISTENT_MIN_WEEKS = 9

# Synthetic stand-ins for real export fields:
#   copilot_enabled_days      -> Person Query `Total Copilot enabled days`
#   copilot_actions           -> Person Query `Total Copilot actions taken`
#   chat/work_apps/cowork_*   -> tool-specific Person Query indicators
#   m365_credits              -> Consumption `Total Copilot Credits used`,
#                                aggregated from daily facts to Sunday weeks
#   github_credits            -> `Total GitHub AI Credits used`
#   multi_policy_positive_keys -> person-service-date keys with more than one
#                                positive spending-policy row
GROUPS = ["chat", "work_apps", "cowork", "github"]
GROUP_LABELS = {
    "chat": "Chat and the Copilot app",
    "work_apps": "Copilot in work apps",
    "cowork": "Cowork",
    "github": "GitHub (positive GitHub credits)",
}

# Synthetic analysis decisions recorded in the context record. In a real
# application these need evidence from the export owner.
COVERAGE_CONFIRMED = {"A": True, "B": False}  # per synthetic export partition
POLICY_ADDITIVITY_CONFIRMED = False

STATES = [
    "Full profile",
    "Confirmed zero M365 credits (No M365 metered use)",
    "No positive observed credits, coverage unconfirmed",
    "Unresolved credit additivity",
    "Incomplete evidence or eligibility",
    "Outside the analysis cohort",
]
PROFILES = [
    (c + "-" + s + "-" + m, f"{cl}, {sl}, {ml}")
    for c, cl in [("C", "Consistent"), ("E", "Episodic")]
    for s, sl in [("S", "Specialised"), ("B", "Broad")]
    for m, ml in [("L", "Lean"), ("I", "Intensive")]
]


class MinimalStandardGenerator:
    """Park-Miller generator. Exact in both Python integers and R doubles."""

    def __init__(self, seed):
        self.state = seed % 2147483647 or 1

    def uniform(self):
        self.state = (16807 * self.state) % 2147483647
        return self.state / 2147483647


def generate_panel(seed):
    """Return (people, person_weeks) for a SYNTHETIC population."""
    rng = MinimalStandardGenerator(seed)
    people, rows = [], []
    for i in range(N_PEOPLE):
        u = [rng.uniform() for _ in range(20)]
        partition = "A" if u[0] < 0.7 else "B"
        in_cohort = u[1] >= 0.03
        short_week = int(u[2] * N_WEEKS) if u[3] < 0.05 else -1
        missing_week = int(u[4] * N_WEEKS) if u[5] < 0.02 else -1
        p_active = 0.35 + 0.40 * u[6] if u[7] < 0.12 else 0.85 + 0.15 * u[6]
        p_group = [0.88 + 0.12 * u[8], 0.88 + 0.12 * u[9], 0.15 + 0.60 * u[10],
                   0.20 + 0.75 * u[11] if u[12] < 0.30 else 0.0]
        if u[13] < 0.15:
            focus = int(u[14] * 4)
            p_group = [0.97 if g == focus else 0.3 * p for g, p in enumerate(p_group)]
        zero_credits = u[15] < 0.08
        m365_scale = 40 + int(960 * u[16])
        overlap_week = int(u[18] * N_WEEKS) if u[17] < 0.03 else -1
        github_scale = 5 + int(195 * u[19])
        person_id = f"SYN{i + 1:04d}"
        people.append({"PersonId": person_id, "partition": partition,
                       "in_cohort": in_cohort})
        for w in range(N_WEEKS):
            v = [rng.uniform() for _ in range(8)]
            active = v[0] < p_active
            flags = [active and v[1 + g] < p_group[g] for g in range(4)]
            group_actions = [1 + int(20 * v[1 + g]) if flags[g] else 0 for g in range(3)]
            m365 = 0 if (zero_credits or not active) else 1 + int(v[5] * m365_scale)
            actions = sum(group_actions) + (1 + int(10 * v[7]) if active else 0)
            rows.append({
                "PersonId": person_id,
                "MetricDate": WINDOW_START + pd.Timedelta(weeks=w),
                "copilot_enabled_days": 4 if w == short_week else 7,
                "copilot_actions": None if w == missing_week else actions,
                "chat_actions": group_actions[0],
                "work_apps_actions": group_actions[1],
                "cowork_actions": group_actions[2],
                "github_credits": 1 + int(v[6] * github_scale) if flags[3] else 0,
                "m365_credits": m365,
                "multi_policy_positive_keys": 1 if (w == overlap_week and m365 > 0) else 0,
            })
    return pd.DataFrame(people), pd.DataFrame(rows)


def surface_mix(counts):
    """Reference rule v1 with integers: share >= 40% and lead >= 5 points."""
    total = sum(counts)
    if total == 0:
        return None
    c1, c2 = sorted(counts, reverse=True)[:2]
    if counts.count(c1) == 1 and 5 * c1 >= 2 * total and 20 * (c1 - c2) >= total:
        return "Specialised"
    return "Broad"


# Boundary self-checks of the integer rule (explicit so `python -O` keeps them).
SURFACE_MIX_CHECKS = [
    ([4, 3, 2, 1], "Specialised"),  # 40% share, 10-point lead
    ([4, 4, 1, 1], "Broad"),        # tied leading groups
    ([8, 7, 5, 0], "Specialised"),  # exactly 40% and 5 points
    ([8, 7, 6, 0], "Broad"),        # share 8/21 is under 40%
    ([0, 0, 0, 0], None),           # no group-weeks
]
for check_counts, expected in SURFACE_MIX_CHECKS:
    if surface_mix(check_counts) != expected:
        raise AssertionError(f"surface_mix({check_counts}) should be {expected}")


def protect(counts, margins=()):
    """Return a withheld flag per cell. Cells under MIN_GROUP_SIZE are withheld.
    Each margin is a list of cell indices whose total is printed or can be
    inferred (the table total is always included). Within every margin the
    withheld cells must number at least two and sum to at least
    MIN_GROUP_SIZE, so no single withheld cell can be recovered by
    subtraction. Otherwise the smallest visible cell in that margin is also
    withheld. Real releases must check all related outputs jointly."""
    n = len(counts)
    margins = list(margins) + [list(range(n))]
    hidden = [c < MIN_GROUP_SIZE for c in counts]
    changed = True
    while changed and not all(hidden):
        changed = False
        for margin in margins:
            withheld = [k for k in margin if hidden[k]]
            if not withheld or (len(withheld) > 1
                                and sum(counts[k] for k in withheld) >= MIN_GROUP_SIZE):
                continue
            visible = sorted((counts[k], k) for k in margin if not hidden[k])
            if visible:
                hidden[visible[0][1]] = True
            else:
                hidden = [True] * n
            changed = True
            break
    return hidden


def show(counts, hidden):
    return ["withheld" if h else str(c) for c, h in zip(counts, hidden)]


def publishable(numerator, denominator, parent=None):
    """A share needs at least MIN_GROUP_SIZE people in the numerator, in its
    complement, and in any group excluded from a printed parent total."""
    excluded = 0 if parent is None else parent - denominator
    return (numerator >= MIN_GROUP_SIZE
            and denominator - numerator >= MIN_GROUP_SIZE
            and not 0 < excluded < MIN_GROUP_SIZE)


def share(numerator, denominator, parent=None):
    if not publishable(numerator, denominator, parent):
        return "withheld (a count, its complement or an excluded group is under 10)"
    return f"{numerator} of {denominator} ({100 * numerator / denominator:.1f}%)"


# 1. Synthetic panel ---------------------------------------------------------
roster, pw = generate_panel(SEED)
pw = pw.sort_values(["PersonId", "MetricDate"]).reset_index(drop=True)
pw["week"] = (pw["MetricDate"] - WINDOW_START).dt.days // 7

# 2. Person-week flags -------------------------------------------------------
pw["chat"] = pw["chat_actions"] > 0
pw["work_apps"] = pw["work_apps_actions"] > 0
pw["cowork"] = pw["cowork_actions"] > 0
pw["github"] = pw["github_credits"] > 0
pw["active"] = (pw["copilot_actions"].fillna(0) > 0) | pw[GROUPS].any(axis=1)

people = pw.groupby("PersonId").agg(
    active_weeks=("active", "sum"),
    chat=("chat", "sum"), work_apps=("work_apps", "sum"),
    cowork=("cowork", "sum"), github=("github", "sum"),
    m365_total=("m365_credits", "sum"),
    min_enabled_days=("copilot_enabled_days", "min"),
    missing_indicators=("copilot_actions", lambda s: int(s.isna().sum())),
    multi_policy_keys=("multi_policy_positive_keys", "sum"),
).reset_index().merge(roster, on="PersonId")
people["group_weeks"] = people[GROUPS].sum(axis=1)


# 3. Evidence states before any label ----------------------------------------
def evidence_state(p):
    if not p["in_cohort"]:
        return STATES[5]
    if p["min_enabled_days"] < 7 or p["missing_indicators"] > 0:
        return STATES[4]
    if p["multi_policy_keys"] > 0 and not POLICY_ADDITIVITY_CONFIRMED:
        return STATES[3]
    if p["m365_total"] == 0:
        return STATES[1] if COVERAGE_CONFIRMED[p["partition"]] else STATES[2]
    if p["group_weeks"] == 0:
        return STATES[4]
    return STATES[0]


people["state"] = people.apply(evidence_state, axis=1)
full = people["state"] == STATES[0]

# 4. Labels for full profiles only -------------------------------------------
reference_median = people.loc[full, "m365_total"].median()
people["consistency"] = None
people["surface_mix"] = None
people["metered_use"] = None
people.loc[full, "consistency"] = people.loc[full, "active_weeks"].map(
    lambda n: "Consistent" if n >= CONSISTENT_MIN_WEEKS else "Episodic")
people.loc[full, "surface_mix"] = people.loc[full, GROUPS].apply(
    lambda r: surface_mix([int(x) for x in r]), axis=1)
people.loc[full, "metered_use"] = people.loc[full, "m365_total"].map(
    lambda x: "Lean" if x <= reference_median else "Intensive")
people["profile"] = None
people.loc[full, "profile"] = (people.loc[full, "consistency"].str[0] + "-"
                               + people.loc[full, "surface_mix"].str[0] + "-"
                               + people.loc[full, "metered_use"].str[0])

# 5. Context record -----------------------------------------------------------
print("SYNTHETIC DATA. Illustrates the mechanics of an EXPERIMENTAL framework.")
print("\n=== Context record ===")
window_end = WINDOW_START + pd.Timedelta(weeks=N_WEEKS) - pd.Timedelta(days=1)
print(f"Rule version: {RULE_VERSION}")
print(f"Window: {WINDOW_START.date()} to {window_end.date()} "
      f"({N_WEEKS} Sunday-start weeks)")
print("Cohort rule: synthetic roster members present in all required queries")
print("Eligibility: 7 enabled days in every week, no missing required indicators")
print("Tool groups (person-weeks with positive observations, capture assumption):")
cohort_pw = pw[pw["PersonId"].isin(people.loc[people["in_cohort"], "PersonId"])]
for g in GROUPS:
    n = int(cohort_pw[g].sum())
    status = "observed" if n > 0 else "NO POSITIVE OBSERVATIONS, report as such"
    print(f"  {GROUP_LABELS[g]:<36} {status}, synthetic capture assumed complete")
print(f"Credit reference: full-profile population, median M365 credits "
      f"{reference_median:.1f}, ties Lean")
print("Coverage confirmation: synthetic partition A confirmed, partition B unconfirmed")
print(f"Policy-row additivity confirmed: {POLICY_ADDITIVITY_CONFIRMED}")
print("GitHub credits flag the GitHub group only and are never added to M365 credits")

# 6. Privacy-safe summaries ---------------------------------------------------
print(f"\n=== Evidence states (groups under {MIN_GROUP_SIZE} withheld) ===")
state_counts = [int((people["state"] == s).sum()) for s in STATES]
state_hidden = protect(state_counts)
for s, shown in zip(STATES, show(state_counts, state_hidden)):
    print(f"  {s:<52} {shown:>9}")
print(f"  {'Supplied synthetic roster':<52} {len(people):>9}")

# The full-profile count reappears as the profile-table total and the
# sensitivity denominator. If the state table withholds it, for example
# because the other states together number fewer than 10, both sections are
# withheld instead.
breakdown_ok = not state_hidden[0]
profile_counts = [int((people["profile"] == code).sum()) for code, _ in PROFILES]
# A median split makes the Lean and Intensive totals close to half of the
# full profiles, so treat both as known margins in the disclosure check.
lean_cells = [k for k, (code, _) in enumerate(PROFILES) if code.endswith("L")]
intensive_cells = [k for k, (code, _) in enumerate(PROFILES) if code.endswith("I")]
profile_hidden = protect(profile_counts, [lean_cells, intensive_cells])
if breakdown_ok:
    print(f"\n=== Profiles among full profiles (groups under {MIN_GROUP_SIZE} withheld) ===")
    for (code, label), shown in zip(PROFILES, show(profile_counts, profile_hidden)):
        print(f"  {code}  {label:<36} {shown:>9}")
    print(f"  {'Full profiles':<42} {int(full.sum()):>9}")
    print("  'withheld' marks cells under 10 and further cells chosen so that, within")
    print("  the printed total and the Lean and Intensive totals implied by the median")
    print("  split, withheld cells number at least two and sum to at least 10. Only")
    print("  their combined total can be derived. Withholding itself conveys information.")
else:
    print("\n=== Profiles and sensitivity ===")
    print("  withheld: population too small for this breakdown")
    print("  (the full-profile count is withheld in the evidence-state table)")

# 7. Sensitivity diagnostics (Surface mix) -------------------------------------
flags_by_person = {pid: g[GROUPS].astype(int).values.tolist()
                   for pid, g in pw.groupby("PersonId")}
loo_changed, split_eligible, split_disagree = {}, 0, 0
for pid, base in people.loc[full, ["PersonId", "surface_mix"]].itertuples(index=False):
    weeks = flags_by_person[pid]
    changed = False
    for drop in range(N_WEEKS):
        counts = [sum(wk[g] for k, wk in enumerate(weeks) if k != drop) for g in range(4)]
        if surface_mix(counts) != base:
            changed = True
            break
    loo_changed[pid] = changed
    first = [sum(wk[g] for wk in weeks[:6]) for g in range(4)]
    last = [sum(wk[g] for wk in weeks[6:]) for g in range(4)]
    if sum(first) > 0 and sum(last) > 0:
        split_eligible += 1
        split_disagree += surface_mix(first) != surface_mix(last)
people["loo_surface_changed"] = people["PersonId"].map(loo_changed)

specialised = full & (people["surface_mix"] == "Specialised")
n_full = int(full.sum())
all_changed = int(people.loc[full, "loo_surface_changed"].sum())
s_total = int(specialised.sum())
s_changed = int(people.loc[specialised, "loo_surface_changed"].sum())
# The Specialised total is printable only if it adds nothing to the profile
# table: every Specialised cell, or every Broad cell, must be visible. The
# Broad row follows by subtraction, so it must also be publishable.
s_cells = [k for k, (code, _) in enumerate(PROFILES) if code[2] == "S"]
b_cells = [k for k, (code, _) in enumerate(PROFILES) if code[2] == "B"]
s_margin_safe = (not any(profile_hidden[k] for k in s_cells)
                 or not any(profile_hidden[k] for k in b_cells))
if (s_margin_safe and publishable(s_changed, s_total)
        and publishable(all_changed - s_changed, n_full - s_total)):
    specialised_row = share(s_changed, s_total)
else:
    specialised_row = "withheld (would reveal a withheld profile cell or a group under 10)"

if breakdown_ok:
    print("\n=== Sensitivity: Surface mix (stability diagnostics rather than observed change) ===")
    print("Label changed in at least one leave-one-week recalculation:")
    print(f"  all full profiles: {share(all_changed, n_full)}")
    print(f"  Specialised:       {specialised_row}")
    print("First versus last six weeks, label disagreement among people with")
    print(f"group-weeks in both halves: {share(split_disagree, split_eligible, parent=n_full)}")
