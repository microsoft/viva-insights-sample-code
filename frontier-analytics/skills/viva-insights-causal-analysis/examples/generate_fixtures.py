"""Generate bounded, fictional Person Query-shaped teaching data (stdlib only)."""

import csv
import random
import sys
from pathlib import Path

FIELDS = [
    "PersonId", "MetricDate", "CollaborationHours", "CopilotActions",
    "CustomOutcome", "OutcomePeriod",
]
CASES = {
    "effect": ("monthly", "false", "2"),
    "confounding": ("monthly", "false", "0"),
    "annual": ("annual", "false", "not_identified"),
    "static": ("static", "false", "not_identified"),
    "quarterly": ("quarterly", "false", "not_identified"),
    "missing": ("monthly", "false", "2"),
    "mechanical": ("monthly", "true", "not_causal"),
    "one-period": ("monthly", "false", "2"),
}


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(output):
    output.mkdir()  # Never merge into or overwrite an existing directory.
    for case, (grain, mechanical, truth) in CASES.items():
        rng = random.Random(1729)
        rows = []
        for person in range(40):
            for month in range(1, 13):
                shock = rng.uniform(-3, 3)
                noise = rng.uniform(-0.4, 0.4)
                hours = 12 + 0.08 * person + 0.15 * month + shock
                outcome = 30 + 0.7 * person + 0.3 * month + 2 * hours + noise
                if case == "confounding":
                    hours = 10 + 0.6 * person + 0.1 * month + shock
                    outcome = 40 + 1.4 * person + 0.3 * month + noise
                period = f"2024-{month:02d}"
                if grain == "annual":
                    period, outcome = "2024", 50 + person * 0.5
                elif grain == "static":
                    period, outcome = "static", 50 + person * 0.5
                elif grain == "quarterly":
                    quarter = (month - 1) // 3 + 1
                    period, outcome = f"2024-Q{quarter}", 50 + person * 0.5 + quarter
                elif mechanical == "true":
                    outcome = 2 * hours
                rows.append({
                    "PersonId": f"synthetic-P{person:04d}",
                    "MetricDate": f"2024-{month:02d}-01",
                    "CollaborationHours": "" if case == "missing" and month == 4 else f"{hours:.10f}",
                    "CopilotActions": str(5 + (person * 7 + month * 11) % 60),
                    "CustomOutcome": f"{outcome:.10f}",
                    "OutcomePeriod": period,
                })
        if case == "one-period":
            rows = [row for row in rows if row["MetricDate"] == "2024-01-01"]
        meta = {
            "generator": "viva-causal-synthetic-v1",
            "synthetic_only": "true",
            "case": case,
            "row_grain": "person-month",
            "outcome_grain": grain,
            "outcome_derived_from_exposure": mechanical,
            "known_effect": truth,
            "exposure_field": "CollaborationHours",
            "outcome_field": "CustomOutcome",
            "exposure_meaning": "Fictional monthly collaboration hours; blank means unknown, never zero.",
            "count_meaning": "Fictional monthly Copilot action count; descriptive only, not a covariate.",
            "outcome_meaning": "Fictional custom organisational operational score attached to a person; not a Viva built-in metric.",
            "outcome_units": "score points",
            "identity_meaning": "Generated fictional person key, not a real identifier.",
            "measurement_meaning": "OutcomePeriod is the actual measurement window; repeated values are not new measurements.",
            "design": (
                "Stable person factor causes both exposure and outcome; true exposure effect is zero."
                if case == "confounding" else
                "Independent seeded exposure shocks plus person and month effects; no customer observations."
            ),
        }
        write_csv(output / f"synthetic-{case}.csv", FIELDS, rows)
        write_csv(output / f"synthetic-{case}.metadata.csv", ["key", "value"],
                  [{"key": key, "value": value} for key, value in meta.items()])


def main():
    if len(sys.argv) != 2:
        print("ERROR usage: python generate_fixtures.py NEW_DIRECTORY", file=sys.stderr)
        return 2
    try:
        generate(Path(sys.argv[1]))
    except OSError:
        print("ERROR output_io_or_directory_exists", file=sys.stderr)
        return 2
    print("Generated 8 synthetic cases and metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
