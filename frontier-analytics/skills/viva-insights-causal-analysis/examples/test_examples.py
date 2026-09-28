"""Exercise both CLIs on identical inputs; no third-party test dependencies."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generate_fixtures import CASES, FIELDS, generate, write_csv

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
RSCRIPT = None
WORK = None
RUNS = 0
RESULTS = {}


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ExamplesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def invoke(self, language, args, expected=0):
        global RUNS
        prefix = ([sys.executable, "-B", str(HERE / "run_example.py")] if language == "python"
                  else [RSCRIPT, "--vanilla", str(HERE / "run_example.R")])
        completed = subprocess.run(prefix + [str(a) for a in args], capture_output=True,
                                   text=True, timeout=30)
        RUNS += 1
        # Check privacy before asserting errors, so a parser failure cannot dump rows in a log.
        self.assertNotIn("synthetic-P", completed.stdout + completed.stderr)
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        return completed

    def pair(self, case="effect", data=None, metadata=None, expected=0, reason=None):
        data = data or FIXTURES / f"synthetic-{case}.csv"
        metadata = metadata or FIXTURES / f"synthetic-{case}.metadata.csv"
        before = (data.read_bytes(), metadata.read_bytes())
        results = []
        for language in ("python", "r"):
            self.serial += 1
            output = self.root / f"{language}-{self.serial}"
            result = self.invoke(language, [data, metadata, output], expected)
            if expected:
                self.assertFalse(output.exists())
                if reason:
                    self.assertIn("ERROR " + reason, result.stderr)
            else:
                self.assertEqual([p.name for p in output.iterdir()], ["summary.csv"])
                rows = read_rows(output / "summary.csv")
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertNotIn("synthetic-P", (output / "summary.csv").read_text())
                if reason:
                    self.assertEqual(row["reason"], reason)
                results.append(row)
        self.assertEqual(before, (data.read_bytes(), metadata.read_bytes()))
        if results:
            for key in results[0]:
                if key.endswith("_slope") and results[0][key]:
                    self.assertAlmostEqual(float(results[0][key]), float(results[1][key]), places=8)
                else:
                    self.assertEqual(results[0][key], results[1][key], key)
            return results[0]

    def mutate(self, change, case="effect"):
        rows = read_rows(FIXTURES / f"synthetic-{case}.csv")
        change(rows)
        path = self.root / "synthetic-mutated.csv"
        write_csv(path, FIELDS, rows)
        return path

    def test_01_all_cases_and_numerical_truth(self):
        expected = {
            "effect": ("synthetic_demo", "not_real_data_inference", 480),
            "confounding": ("synthetic_demo", "not_real_data_inference", 480),
            "annual": ("blocked", "no_within_person_measurement", 40),
            "static": ("blocked", "no_within_person_measurement", 40),
            "quarterly": ("readiness_only", "quarterly_reaggregation_required", 160),
            "missing": ("blocked", "unknown_exposure", 480),
            "mechanical": ("blocked", "mechanical_outcome", 480),
            "one-period": ("blocked", "insufficient_panel", 40),
        }
        for case, (status, reason, measurements) in expected.items():
            with self.subTest(case=case):
                row = self.pair(case, reason=reason)
                RESULTS[case] = row
                self.assertEqual(row["status"], status)
                self.assertEqual(int(row["outcome_measurements"]), measurements)
                if status != "synthetic_demo":
                    self.assertEqual(row["twfe_slope"], "")
                    self.assertEqual(row["pooled_slope"], "")
        self.assertAlmostEqual(float(RESULTS["effect"]["twfe_slope"]), 2, delta=0.05)
        self.assertAlmostEqual(float(RESULTS["confounding"]["twfe_slope"]), 0, delta=0.05)
        self.assertGreater(float(RESULTS["confounding"]["pooled_slope"]), 1.5)
        self.assertEqual(RESULTS["missing"]["missing_exposures"], "40")

    def test_02_generator_determinism(self):
        output = self.root / "regenerated"
        generate(output)
        self.assertEqual(digest_tree(FIXTURES), digest_tree(output))
        before = digest_tree(output)
        with self.assertRaises(FileExistsError):
            generate(output)
        self.assertEqual(before, digest_tree(output))

    def test_03_output_determinism(self):
        for language in ("python", "r"):
            snapshots = []
            for number in (1, 2):
                output = self.root / f"{language}-{number}"
                self.invoke(language, [FIXTURES / "synthetic-effect.csv",
                                       FIXTURES / "synthetic-effect.metadata.csv", output])
                snapshots.append((output / "summary.csv").read_bytes())
            self.assertEqual(*snapshots)

    def test_04_duplicate_key(self):
        path = self.mutate(lambda rows: rows.append(rows[0].copy()))
        self.pair(data=path, expected=2, reason="duplicate_person_period")

    def test_05_dates(self):
        for value in ("2024-13-01", "2024-02-30", "2024-01-02", "01/01/2024", "", "1999-01-01"):
            with self.subTest(value=value):
                path = self.mutate(lambda rows: rows[0].update(MetricDate=value))
                self.pair(data=path, expected=2, reason="invalid_date")

    def test_06_numeric_validation(self):
        for field, value, reason in [
            ("CollaborationHours", "NaN", "invalid_numeric"),
            ("CollaborationHours", "Inf", "invalid_numeric"),
            ("CustomOutcome", "-Inf", "invalid_numeric"),
            ("CustomOutcome", "1e309", "numeric_out_of_bounds"),
            ("CustomOutcome", "", "invalid_numeric"),
            ("CollaborationHours", "oops", "invalid_numeric"),
            ("CollaborationHours", "-1", "invalid_hours"),
            ("CollaborationHours", "745", "invalid_hours"),
            ("CopilotActions", "1.5", "invalid_count"),
            ("CopilotActions", "-1", "invalid_count"),
            ("CopilotActions", "", "invalid_numeric"),
        ]:
            with self.subTest(field=field, value=value):
                path = self.mutate(lambda rows: rows[0].update({field: value}))
                self.pair(data=path, expected=2, reason=reason)

    def test_07_quarterly_measurement_consistency(self):
        path = self.mutate(lambda rows: rows[1].update(CustomOutcome="999"), "quarterly")
        self.pair(case="quarterly", data=path, expected=2, reason="inconsistent_repeated_outcome")
        path = self.mutate(lambda rows: rows[0].update(OutcomePeriod="2024-Q2"), "quarterly")
        self.pair(case="quarterly", data=path, expected=2, reason="measurement_period_mismatch")

    def test_08_annual_and_static_consistency(self):
        for case in ("annual", "static"):
            path = self.mutate(lambda rows: rows[1].update(CustomOutcome="999"), case)
            self.pair(case=case, data=path, expected=2, reason="inconsistent_repeated_outcome")

    def test_09_unbalanced_panel(self):
        path = self.mutate(lambda rows: rows.pop())
        self.pair(data=path, reason="unbalanced_panel")

    def test_10_no_within_exposure(self):
        for varying in (False, True):
            def replace(rows):
                for row in rows:
                    value = 10 + int(row["PersonId"][-4:]) if varying else 10
                    row["CollaborationHours"] = str(value)
            self.pair(data=self.mutate(replace), reason="no_within_exposure_variation")

    def test_11_missing_is_not_zero(self):
        path = self.mutate(lambda rows: rows[0].update(CollaborationHours=""))
        row = self.pair(data=path, reason="unknown_exposure")
        self.assertEqual(row["missing_exposures"], "1")
        path = self.mutate(lambda rows: rows[0].update(CollaborationHours="0"))
        row = self.pair(data=path, reason="not_real_data_inference")
        self.assertEqual(row["missing_exposures"], "0")

    def test_12_metadata_required_and_semantic(self):
        for key, value, reason in [
            ("generator", "other", "unsupported_metadata"),
            ("synthetic_only", "false", "unsupported_metadata"),
            ("outcome_grain", "annual", "inconsistent_metadata"),
            ("outcome_derived_from_exposure", "true", "inconsistent_metadata"),
            ("known_effect", "7", "inconsistent_metadata"),
            ("outcome_meaning", "", "missing_field_meaning"),
            ("case", "real-data", "unsupported_case"),
        ]:
            with self.subTest(key=key):
                meta = read_rows(FIXTURES / "synthetic-effect.metadata.csv")
                for row in meta:
                    if row["key"] == key:
                        row["value"] = value
                path = self.root / "metadata.csv"
                write_csv(path, ["key", "value"], meta)
                self.pair(metadata=path, expected=2, reason=reason)
        meta.append(meta[0])
        write_csv(path, ["key", "value"], meta)
        self.pair(metadata=path, expected=2, reason="duplicate_metadata_key")

    def test_13_cli_failures_and_existing_output(self):
        for language in ("python", "r"):
            self.invoke(language, [], 2)
            self.invoke(language, ["a", "b", "c", "d"], 2)
            self.invoke(language, [self.root / "synthetic-absent.csv",
                                   FIXTURES / "synthetic-effect.metadata.csv",
                                   self.root / "absent-output"], 2)
            self.invoke(language, [FIXTURES / "synthetic-effect.csv", self.root / "absent-meta",
                                   self.root / "absent-output"], 2)
            self.assertFalse((self.root / "absent-output").exists())
            sentinel = self.root / "sentinel.txt"
            sentinel.write_text("unchanged", encoding="utf-8")
            before = digest_tree(self.root)
            self.invoke(language, [FIXTURES / "synthetic-effect.csv",
                                   FIXTURES / "synthetic-effect.metadata.csv", self.root], 2)
            self.invoke(language, [FIXTURES / "synthetic-effect.csv",
                                   FIXTURES / "synthetic-effect.metadata.csv", sentinel], 2)
            self.assertEqual(before, digest_tree(self.root))
            self.invoke(language, [FIXTURES / "synthetic-effect.csv",
                                   FIXTURES / "synthetic-effect.metadata.csv",
                                   self.root / "absent-parent" / "output"], 2)

    def test_14_schema_and_malformed_csv(self):
        path = self.root / "synthetic-malformed.csv"
        for content in (
            "Wrong,Header\n1,2\n",
            ",".join(FIELDS) + "\n",
            ",".join(FIELDS) + "\nsynthetic-P0000,2024-01-01\n",
            ",".join(FIELDS) + '\n"synthetic-P0000,2024-01-01\n',
        ):
            path.write_text(content, encoding="utf-8")
            self.pair(data=path, expected=2)
        for kind in ("data", "metadata"):
            source = FIXTURES / ("synthetic-effect.csv" if kind == "data"
                                 else "synthetic-effect.metadata.csv")
            lines = source.read_text(encoding="utf-8").splitlines()
            for position in ("internal", "trailing"):
                for newline in ("\n", "\r\n"):
                    with self.subTest(kind=kind, position=position, newline=repr(newline)):
                        records = lines.copy()
                        records.insert(2 if position == "internal" else len(records), "")
                        path.write_bytes((newline.join(records) + newline).encode("utf-8"))
                        self.pair(**{kind: path}, expected=2, reason="invalid_csv_shape")

    def test_15_bounds_and_synthetic_contract(self):
        path = self.root / "synthetic-large.csv"
        path.write_bytes(b"x" * 2_000_001)
        self.pair(data=path, expected=2, reason="file_too_large")
        path = self.mutate(lambda rows: rows[0].update(PersonId="not-a-synthetic-key"))
        self.pair(data=path, expected=2, reason="invalid_synthetic_identity")
        other = self.root / "customer.csv"
        other.write_bytes((FIXTURES / "synthetic-effect.csv").read_bytes())
        self.pair(data=other, expected=2, reason="synthetic_filename_required")

    def test_16_dummy_ols_reference(self):
        # Independently check the double-demeaning formula against base-R dummy-variable OLS.
        script = (
            'd <- read.csv(commandArgs(TRUE)[1]); '
            'm <- lm(CustomOutcome ~ CollaborationHours + factor(PersonId) + factor(MetricDate), d); '
            'cat(sprintf("%.12f", coef(m)[["CollaborationHours"]]))'
        )
        for case, truth in (("effect", 2), ("confounding", 0)):
            completed = subprocess.run(
                [RSCRIPT, "--vanilla", "-e", script, str(FIXTURES / f"synthetic-{case}.csv")],
                capture_output=True, text=True, timeout=30, check=True)
            beta = float(completed.stdout)
            self.assertAlmostEqual(beta, truth, delta=0.05)
            row = self.pair(case)
            self.assertAlmostEqual(beta, float(row["twfe_slope"]), places=8)

    def test_17_single_person(self):
        def single(rows):
            rows[:] = [row for row in rows if row["PersonId"] == "synthetic-P0000"]
        self.pair(data=self.mutate(single), reason="insufficient_panel")

    def test_18_outcome_period_required(self):
        path = self.mutate(lambda rows: rows[0].update(OutcomePeriod=""))
        self.pair(data=path, expected=2, reason="measurement_period_mismatch")


def main():
    global RSCRIPT, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", required=True, help="Path to Rscript executable")
    parser.add_argument("--output", required=True, type=Path, help="New report directory")
    args = parser.parse_args()
    RSCRIPT = args.rscript
    try:
        version = subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True,
                                 check=True, timeout=30)
        args.output.mkdir()
    except (OSError, subprocess.SubprocessError):
        print("ERROR Rscript unavailable or output directory cannot be created", file=sys.stderr)
        return 2
    before = digest_tree(FIXTURES)
    log = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="test-work-", dir=args.output) as directory:
        WORK = directory
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExamplesTest)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    preserved = before == digest_tree(FIXTURES)
    report = {
        "python_version": sys.version.split()[0],
        "r_version": (version.stdout + version.stderr).strip(),
        "test_methods": result.testsRun,
        "runner_cli_invocations": RUNS,
        "failures": len(result.failures), "errors": len(result.errors),
        "fixtures_unchanged": preserved,
        "cases": RESULTS,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "tests.txt").write_text(log.getvalue(), encoding="utf-8")
    print(log.getvalue(), end="")
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() and preserved else 1


if __name__ == "__main__":
    sys.exit(main())
