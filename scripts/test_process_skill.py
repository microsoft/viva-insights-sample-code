"""Execute the fabricated process rehearsal without private services or data."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "frontier-analytics" / "skills" / "viva-insights-process-analysis" / "examples"


class ProcessSkillTests(unittest.TestCase):
    def test_bundled_scope_and_safeguard_regressions(self):
        run = subprocess.run([sys.executable, "-B", str(EXAMPLES / "test_examples.py")],
                             capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_rehearsal_exports_aggregate_evidence_and_unavailable_partial_year(self):
        run = subprocess.run([sys.executable, "-B", str(EXAMPLES / "run_example.py")],
                             capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["mode"], "fabricated demonstration")
        self.assertEqual([r["status"] for r in report["years"]], ["observed", "observed", "partial"])
        self.assertIsNone(report["years"][-1]["meetings"])
        self.assertNotIn("MeetingId", run.stdout)
        self.assertNotIn("synthetic-review", run.stdout)
        self.assertNotIn("Subject", run.stdout)
        self.assertIn("fixture_sha256", report)


if __name__ == "__main__":
    unittest.main()
