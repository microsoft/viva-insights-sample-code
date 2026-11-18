"""Regression tests for the synthetic DiD/ITSA notebook event-study baseline."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "utility-python" / "causal-inference" / "did-itsa" / "DiD_ITSA_Analysis.ipynb"


def event_study_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "6.3. DiD Event Study Plot" in source:
            return source
    raise AssertionError("DiD/ITSA notebook event-study cell not found")


def sample_panel() -> pd.DataFrame:
    rows = []
    for person, licensed in [("treated-a", 1), ("treated-b", 1), ("control-a", 0), ("control-b", 0)]:
        licensing_week = 5 if licensed else -1
        for week in range(16):
            outcome = 10 + week + (2 if licensed and week >= licensing_week else 0)
            rows.append({
                "PersonID": person,
                "Week": week,
                "Licensed": licensed,
                "LicensingWeek": licensing_week,
                "Outcome": outcome,
            })
    return pd.DataFrame(rows)


def run_event_study(data: pd.DataFrame) -> tuple[dict, str]:
    namespace = {
        "data": data,
        "TIME_VAR": "Week",
        "TREATMENT_INDICATOR": "Licensed",
        "TREATMENT_WEEK_VAR": "LicensingWeek",
        "OUTCOME_VARIABLE": "Outcome",
        "PERSON_ID_VAR": "PersonID",
        "TREATMENT_NAME": "Synthetic tool",
        "OUTCOME_DISPLAY_NAME": "Outcome",
    }
    output = io.StringIO()
    import matplotlib.pyplot as plt
    original_show = plt.show
    plt.show = lambda: None
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(event_study_source(), str(NOTEBOOK), "exec"), namespace)
    finally:
        plt.show = original_show
        plt.close("all")
    return namespace, output.getvalue()


class NotebookEventStudyTests(unittest.TestCase):
    def test_calendar_week_panel_uses_week_before_license_as_baseline(self):
        namespace, output = run_event_study(sample_panel())
        self.assertIn("Descriptive group-mean event-time plot created successfully", output)
        self.assertEqual(namespace["intervention_week"], 5)
        self.assertEqual(namespace["baseline_week"], 4)
        self.assertEqual(namespace["event_times"], list(range(-5, 11)))
        self.assertEqual(namespace["effects"][namespace["event_times"].index(-1)], 0)
        self.assertAlmostEqual(namespace["effects"][namespace["event_times"].index(0)], 2)
        self.assertTrue(all(pd.notna(namespace["effects"])))

    def test_missing_week_before_license_fails_with_clear_message(self):
        panel = sample_panel()
        panel = panel[panel["Week"] != 4]
        namespace, output = run_event_study(panel)
        self.assertIn("baseline week 4", output)
        self.assertNotIn("Event study plot created successfully", output)
        self.assertNotIn("effects", namespace)

    def test_multiple_licensing_cohorts_fail_with_clear_message(self):
        panel = sample_panel()
        panel.loc[panel["PersonID"] == "treated-b", "LicensingWeek"] = 6
        namespace, output = run_event_study(panel)
        self.assertIn("requires exactly one treatment cohort", output)
        self.assertNotIn("Descriptive group-mean event-time plot created successfully", output)
        self.assertNotIn("effects", namespace)


if __name__ == "__main__":
    unittest.main()
