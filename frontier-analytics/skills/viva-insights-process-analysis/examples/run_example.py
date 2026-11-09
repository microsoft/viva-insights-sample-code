"""Fabricated scope rehearsal, not a real-export adapter or inference engine."""

import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load_policy():
    return json.loads((HERE / "fixtures" / "synthetic-audit-policy.json").read_text(encoding="utf-8"))


def validate_policy(policy):
    years = policy["years"]
    if not years or len(set(years)) != len(years) or set(map(str, years)) != set(policy["year_codes"]):
        raise ValueError("Every comparison year needs an explicit supported code")
    if (not all(isinstance(v, str) and v.strip() for v in policy["year_codes"].values())
            or len(set(policy["year_codes"].values())) != len(years)):
        raise ValueError("Year codes must be distinct nonblank text")
    first, last = policy["annual_start"], policy["annual_end"]
    date.fromisoformat("2020-" + first)
    date.fromisoformat("2020-" + last)
    if first > last:
        raise ValueError("This bounded example supports a single calendar-year window")
    if date.fromisoformat(policy["coverage_start"]) > date.fromisoformat(policy["coverage_through"]):
        raise ValueError("Reversed coverage interval")
    if not isinstance(policy["privacy_floor"], int) or policy["privacy_floor"] < 1:
        raise ValueError("Invalid privacy floor")
    for pattern in [*policy["categories"].values(), policy["exclude_pattern"]]:
        re.compile(pattern, re.IGNORECASE)


def classify(event, policy):
    """Return local category membership and final scope, retaining observed zeros."""
    validate_policy(policy)
    if type(event["quality_pass"]) is not bool:
        raise ValueError("quality_pass must be an explicitly prepared boolean")
    decision = event.get("decision", "")
    if decision not in ("", "Keep", "Remove", "Pending"):
        raise ValueError("Invalid manual decision")
    day = date.fromisoformat(event["Date"])
    if not date.fromisoformat(policy["coverage_start"]) <= day <= date.fromisoformat(policy["coverage_through"]):
        raise ValueError("Event outside configured source coverage")
    subject = event["Subject"]
    if not isinstance(subject, str):
        raise ValueError("This title-based rehearsal needs an observed subject")
    md = day.strftime("%m-%d")
    categories = {name for name, pattern in policy["categories"].items()
                  if re.search(pattern, subject, re.IGNORECASE)}
    code = r"\b(?:" + "|".join(re.escape(v) for v in policy["year_codes"].values()) + r")\b"
    if re.search(code, subject, re.IGNORECASE):
        categories.add("Audit")
    baseline = bool(categories) and not re.search(policy["exclude_pattern"], subject, re.IGNORECASE)
    prior = event.get("prior_in_scope")
    if decision == "Pending" and type(prior) is not bool:
        raise ValueError("Pending review requires its recorded prior disposition")
    if not event["quality_pass"] or not first_last(policy, md):
        final = False
    elif decision == "Remove":
        final = False
    elif decision == "Keep":
        final = True
    elif decision == "Pending":
        final = prior
    else:
        final = bool(baseline)
    return {"categories": sorted(categories), "final_in_scope": final}


def first_last(policy, month_day):
    return policy["annual_start"] <= month_day <= policy["annual_end"]


def boolean(value):
    if value not in ("true", "false", ""):
        raise ValueError("Invalid fabricated boolean")
    return {"true": True, "false": False, "": None}[value]


def load_cases():
    with (HERE / "fixtures" / "synthetic-audit-cases.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("quality_pass", "prior_in_scope", "expected"):
            row[field] = boolean(row[field])
    return rows


def fabricated_events(policy):
    events = []
    for year in policy["years"]:
        for person in range(12):
            events.append({
                "MeetingId": f"fabricated-{year}-{person}", "Date": f"{year}-03-10",
                "Subject": f"{policy['year_codes'][str(year)]} audit controls review",
                "quality_pass": True, "Organizer": f"synthetic-{person}",
                "Group": "Synthetic audit team", "attendee_hours": 2.0 if year == 2020 else 1.5,
            })
    for case in load_cases():
        events.append(case | {"Organizer": "synthetic-review", "Group": "Synthetic review group",
                              "attendee_hours": 1.0})
    return events


def analyse(events, policy, organisation_minimum=10):
    validate_policy(policy)
    if policy["status"] != "approved" or policy["coverage_confirmed"] is not True:
        raise ValueError("Approved scope and confirmed coverage are required for reporting")
    if not isinstance(organisation_minimum, int) or organisation_minimum < 1:
        raise ValueError("Invalid organisational privacy requirement")
    ids = [e["MeetingId"] for e in events]
    if (any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Duplicate or invalid event keys")
    for event in events:
        value = event["attendee_hours"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid attendee hours")
    selected = [e for e in events if classify(e, policy)["final_in_scope"]]
    minimum = max(10, policy["privacy_floor"], organisation_minimum)
    results = []
    for year in policy["years"]:
        start = date.fromisoformat(f"{year}-{policy['annual_start']}")
        end = date.fromisoformat(f"{year}-{policy['annual_end']}")
        complete = (date.fromisoformat(policy["coverage_start"]) <= start and
                    end <= date.fromisoformat(policy["coverage_through"]))
        rows = [e for e in selected if date.fromisoformat(e["Date"]).year == year]
        people = {e["Organizer"] for e in rows if e.get("Organizer")}
        status = "observed" if complete else "partial"
        if complete and rows and len(people) < minimum:
            status = "privacy_withheld"
        publish = status == "observed"
        results.append({
            "year": year, "status": status, "meetings": len(rows) if publish else None,
            "attendee_hours": sum(e["attendee_hours"] for e in rows) if publish else None,
            "distinct_organisers": len(people) if publish and rows else None,
        })
    # No subgroup cells are exported, avoiding complementary disclosure from the small review group.
    return {
        "mode": "fabricated demonstration", "scope_version": policy["version"],
        "years": results, "group_breakdowns": "not published by this rehearsal",
        "interpretation": "Descriptive captured collaboration, with no causal or compliance claim",
    }


def main():
    policy = load_policy()
    result = analyse(fabricated_events(policy), policy)
    result["fixture_sha256"] = {
        name: hashlib.sha256((HERE / "fixtures" / name).read_bytes()).hexdigest()
        for name in ("synthetic-audit-policy.json", "synthetic-audit-cases.csv")
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
