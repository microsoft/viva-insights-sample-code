"""Bounded synthetic readiness gates and balanced two-way FE illustration."""

import csv
import math
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from generate_fixtures import CASES, FIELDS, write_csv

SUMMARY = [
    "status", "reason", "rows", "persons", "periods", "outcome_measurements",
    "missing_exposures", "pooled_slope", "twfe_slope",
]


def require(condition, code):
    if not condition:
        raise ValueError(code)


def read_csv(path, fields):
    require(path.stat().st_size <= 2_000_000, "file_too_large")
    with path.open(newline="", encoding="utf-8") as stream:
        # DictReader silently skips blank records; reader preserves them for validation.
        reader = csv.reader(stream, strict=True)
        require(next(reader, None) == fields, "invalid_columns")
        rows = list(reader)
    require(0 < len(rows) <= 10000, "invalid_row_count")
    require(all(len(row) == len(fields) for row in rows), "invalid_csv_shape")
    return [dict(zip(fields, row)) for row in rows]


def numeric(value, missing=False):
    if value == "" and missing:
        return None
    require(bool(re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value)),
            "invalid_numeric")
    result = float(value)
    require(math.isfinite(result) and abs(result) <= 1_000_000, "numeric_out_of_bounds")
    return result


def demean(values, people, periods):
    person_values, period_values = defaultdict(list), defaultdict(list)
    for value, person, period in zip(values, people, periods):
        person_values[person].append(value)
        period_values[period].append(value)
    person_mean = {k: sum(v) / len(v) for k, v in person_values.items()}
    period_mean = {k: sum(v) / len(v) for k, v in period_values.items()}
    grand = sum(values) / len(values)
    return [v - person_mean[p] - period_mean[t] + grand
            for v, p, t in zip(values, people, periods)]


def analyze(data_path, metadata_path):
    require(data_path.name.startswith("synthetic-") and data_path.suffix == ".csv",
            "synthetic_filename_required")
    metadata_rows = read_csv(metadata_path, ["key", "value"])
    metadata = {row["key"]: row["value"] for row in metadata_rows}
    require(len(metadata) == len(metadata_rows), "duplicate_metadata_key")
    fixed = {
        "generator": "viva-causal-synthetic-v1", "synthetic_only": "true",
        "row_grain": "person-month", "exposure_field": "CollaborationHours",
        "outcome_field": "CustomOutcome", "outcome_units": "score points",
    }
    require(all(metadata.get(k) == v for k, v in fixed.items()), "unsupported_metadata")
    require(all(metadata.get(k, "").strip() for k in [
        "exposure_meaning", "count_meaning", "outcome_meaning", "identity_meaning",
        "measurement_meaning", "design",
    ]), "missing_field_meaning")
    case = metadata.get("case")
    require(case in CASES, "unsupported_case")
    grain, mechanical, truth = CASES[case]
    require((metadata.get("outcome_grain"), metadata.get("outcome_derived_from_exposure"),
             metadata.get("known_effect")) == (grain, mechanical, truth),
            "inconsistent_metadata")
    rows = read_csv(data_path, FIELDS)
    people, periods, x, y = [], [], [], []
    keys, measurements = set(), {}
    for row in rows:
        person, raw_date = row["PersonId"], row["MetricDate"]
        require(bool(re.fullmatch(r"synthetic-P[0-9]{4}", person)), "invalid_synthetic_identity")
        require(bool(re.fullmatch(r"[0-9]{4}-[0-9]{2}-01", raw_date)), "invalid_date")
        try:
            observed = date.fromisoformat(raw_date)
        except ValueError:
            raise ValueError("invalid_date") from None
        require(2000 <= observed.year <= 2100, "invalid_date")
        key = (person, raw_date)
        require(key not in keys, "duplicate_person_period")
        keys.add(key)
        expected = {
            "monthly": raw_date[:7], "annual": raw_date[:4], "static": "static",
            "quarterly": f"{observed.year}-Q{(observed.month - 1) // 3 + 1}",
        }[grain]
        require(row["OutcomePeriod"] == expected, "measurement_period_mismatch")
        exposure = numeric(row["CollaborationHours"], missing=True)
        count = numeric(row["CopilotActions"])
        outcome = numeric(row["CustomOutcome"])
        require(exposure is None or 0 <= exposure <= 744, "invalid_hours")
        require(count >= 0 and count.is_integer(), "invalid_count")
        measurement_key = (person, expected)
        require(measurement_key not in measurements or measurements[measurement_key] == outcome,
                "inconsistent_repeated_outcome")
        measurements[measurement_key] = outcome
        people.append(person)
        periods.append(raw_date)
        x.append(exposure)
        y.append(outcome)
    result = dict(zip(SUMMARY, [
        "blocked", "", len(rows), len(set(people)), len(set(periods)),
        len(measurements), sum(v is None for v in x), "", "",
    ]))
    if mechanical == "true":
        result["reason"] = "mechanical_outcome"
    elif grain in ("annual", "static"):
        result["reason"] = "no_within_person_measurement"
    elif grain == "quarterly":
        result.update(status="readiness_only", reason="quarterly_reaggregation_required")
    elif result["periods"] < 2 or result["persons"] < 2:
        result["reason"] = "insufficient_panel"
    elif result["missing_exposures"]:
        result["reason"] = "unknown_exposure"
    elif len(rows) != result["persons"] * result["periods"]:
        result["reason"] = "unbalanced_panel"
    else:
        xd, yd = demean(x, people, periods), demean(y, people, periods)
        denominator = sum(v * v for v in xd)
        if denominator <= 1e-10:
            result["reason"] = "no_within_exposure_variation"
        else:
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            pooled = sum((a - xm) * (b - ym) for a, b in zip(x, y))
            pooled /= sum((a - xm) ** 2 for a in x)
            beta = sum(a * b for a, b in zip(xd, yd)) / denominator
            result.update(status="synthetic_demo", reason="not_real_data_inference",
                          pooled_slope=f"{pooled:.10f}", twfe_slope=f"{beta:.10f}")
    return result


def main():
    if len(sys.argv) != 4:
        print("ERROR usage: python run_example.py INPUT.csv METADATA.csv NEW_OUTPUT_DIRECTORY",
              file=sys.stderr)
        return 2
    try:
        output = Path(sys.argv[3])
        require(not output.exists(), "output_exists")
        result = analyze(Path(sys.argv[1]), Path(sys.argv[2]))
        output.mkdir()
        write_csv(output / "summary.csv", SUMMARY, [result])
    except UnicodeError:
        print("ERROR input_or_output_io", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    except (OSError, csv.Error):
        print("ERROR input_or_output_io", file=sys.stderr)
        return 2
    print(f"{result['status']}: {result['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
