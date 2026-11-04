"""Synthetic smoke checks for the documented installed-package recipes.

Run: python -m unittest discover -s scripts -p test_analysis_recipes.py
Install: python -m pip install -r scripts/requirements-doc-tests.txt
Requires the pinned vivainsights 0.4.3 environment. Missing dependencies fail
collection rather than presenting skipped recipes as a successful validation.
No customer data, files or upstream package patches are used.
"""
from importlib.metadata import version
import contextlib
import io
import inspect
from pathlib import Path
import re
import runpy
import unittest
from unittest import mock

import pandas as pd
import vivainsights as vi


PACKAGE_VERSION = version("vivainsights")
ROOT = Path(__file__).resolve().parents[1]
CORE_REFERENCE = ROOT / "frontier-analytics" / "skills" / "viva-insights-analysis" / "reference"
REFERENCE = CORE_REFERENCE / "data-pitfalls.md"
PITFALLS = ROOT / "frontier-analytics" / "schemas" / "common-pitfalls.md"
CONVENTIONS = CORE_REFERENCE / "analysis-conventions.md"
NOTEBOOK = ROOT / "frontier-analytics" / "examples" / "sample-output-specs" / "notebook-analysis-spec.md"


def documented_python(path, contains):
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
    matches = [block for block in blocks if contains in block]
    if len(matches) != 1:
        raise AssertionError(f"Expected one recipe containing {contains!r} in {path.name}")
    return compile(matches[0], str(path), "exec")


def execute_documented(path, contains, **namespace):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(documented_python(path, contains), namespace)
    return namespace


def synthetic_panel():
    dates = pd.date_range("2026-01-05", periods=8, freq="7D")
    return pd.DataFrame(
        [
            {
                "PersonId": f"P{person:03d}",
                "MetricDate": date,
                "Organization": "Synthetic",
                "Collaboration_hours": (
                    1 + person / 10 if week == 4 else 20 + week + person / 10
                ),
            }
            for person in range(12)
            for week, date in enumerate(dates)
        ]
    )


class AnalysisRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if PACKAGE_VERSION != "0.4.3":
            raise AssertionError("Install scripts/requirements-doc-tests.txt: only vivainsights 0.4.3 is verified")
        print(f"Recipe smoke tests: vivainsights {PACKAGE_VERSION}")

    def test_documented_python_recipe_retains_dataframe_and_keys(self):
        original = synthetic_panel()
        namespace = execute_documented(
            REFERENCE, "vi.identify_holidayweeks(", pq=original.copy(), vi=vi, pd=pd
        )
        result = namespace["pq"]
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(original))
        pd.testing.assert_frame_equal(
            result[["PersonId", "MetricDate"]].reset_index(drop=True),
            original[["PersonId", "MetricDate"]].reset_index(drop=True),
        )
        self.assertTrue({"holidayweek", "inactiveweek"}.issubset(result.columns))
        self.assertTrue(result["holidayweek"].any())
        self.assertTrue(result["inactiveweek"].any())

    def test_installed_return_signatures_and_default_types(self):
        for helper in (vi.identify_holidayweeks, vi.identify_inactiveweeks):
            with self.subTest(helper=helper.__name__):
                self.assertIn("return_type", inspect.signature(helper).parameters)
                self.assertIsInstance(helper(synthetic_panel()), str)

    def test_identify_nkw_transfer_caveat_is_reproducible(self):
        panel = synthetic_panel()
        transferred = (panel["PersonId"] == "P000") & (
            panel["MetricDate"] >= pd.Timestamp("2026-02-02")
        )
        panel.loc[transferred, "Organization"] = "Synthetic transfer"
        result = vi.identify_nkw(panel, return_type="data_with_flag")
        # Characterise the upstream bug, not a local replacement for the helper.
        self.assertGreater(len(result), len(panel))
        self.assertTrue(result.duplicated(["PersonId", "MetricDate"]).any())

    def test_explicit_eligibility_is_independent_of_activity(self):
        panel = pd.DataFrame({
            "PersonId": ["positive_unknown", "zero_eligible", "positive_ineligible", "null_eligible"],
            "MetricDate": pd.to_datetime(["2026-01-05"] * 4),
            "Total_Copilot_enabled_days": [None, 5, 0, 5],
            "Copilot_Actions": [4, 0, 7, None],
        })
        result = execute_documented(PITFALLS, "enabled = pd.to_numeric", df=panel, pd=pd)
        output = result["df"].set_index("PersonId")
        self.assertTrue(pd.isna(output.loc["positive_unknown", "is_licensed"]))
        self.assertTrue(output.loc["zero_eligible", "is_licensed"])
        self.assertFalse(output.loc["zero_eligible", "is_active"])
        self.assertFalse(output.loc["positive_ineligible", "is_licensed"])
        self.assertTrue(pd.isna(output.loc["null_eligible", "is_active"]))
        self.assertTrue(result["adoption"].isna().all())

    def test_absent_in_scope_activity_becomes_zero_not_recorded_null(self):
        panel = pd.DataFrame({
            "Total_Copilot_enabled_days": [5, 5],
            "Total_Copilot_actions_taken": [None, 3.0],
            "activity_state": ["omitted_in_scope", "observed"],
        })

        class CaptureSegments:
            def identify_usage_segments(self, data, **kwargs):
                return data.copy()

        output = execute_documented(
            CONVENTIONS, "activity_state", df=panel, vi=CaptureSegments()
        )["seg"]
        self.assertEqual(output["Total_Copilot_actions_taken"].tolist(), [0, 3])
        self.assertTrue(panel["Total_Copilot_actions_taken"].isna().iloc[0])

    def test_unknown_missingness_reasons_block_rolling_segmentation(self):
        class NoSegmentation:
            def identify_usage_segments(self, *args, **kwargs):
                raise AssertionError("Unknown history must fail before the package call")

        for reason in ("observed_null", "source_unavailable", "out_of_scope", "known_coverage_failure"):
            with self.subTest(reason=reason):
                panel = pd.DataFrame({
                    "Total_Copilot_enabled_days": [5],
                    "Total_Copilot_actions_taken": [None],
                    "activity_state": [reason],
                })
                with self.assertRaisesRegex(ValueError, "Unresolved activity"):
                    execute_documented(
                        CONVENTIONS, "activity_state", df=panel, vi=NoSegmentation()
                    )

    def test_unknown_or_ineligible_history_blocks_segmentation(self):
        for enabled in (None, 0):
            with self.subTest(enabled=enabled):
                panel = pd.DataFrame({
                    "Total_Copilot_enabled_days": [5, enabled, 5],
                    "Total_Copilot_actions_taken": [1, 0, 2],
                    "activity_state": ["observed"] * 3,
                })
                with self.assertRaisesRegex(ValueError, "eligibility throughout"):
                    execute_documented(CONVENTIONS, "activity_state", df=panel)

    def test_irregular_dates_do_not_supply_missing_cadence_configuration(self):
        for dates in (
            ["2026-01-05"],
            ["2026-01-05", "2026-01-12"],  # Could be sparse daily activity.
            ["2026-01-05", "2026-01-19", "2026-02-02"],
            ["2026-01-05", "2026-01-06", "2026-01-19"],
        ):
            with self.subTest(dates=dates):
                panel = pd.DataFrame({"PersonId": ["P000"] * len(dates), "MetricDate": dates})
                with self.assertRaisesRegex(ValueError, "Confirm configured query grouping"):
                    execute_documented(NOTEBOOK, "CONFIGURED_GROUPING", df=panel, pd=pd)

    def test_privacy_threshold_boundaries_use_distinct_people(self):
        namespace = execute_documented(PITFALLS, "def suppress_small_groups")
        self.assertEqual(namespace["MIN_GROUP_SIZE"], 5)
        suppress = namespace["suppress_small_groups"]
        for org_minimum, workflow_floor in ((3, 5), (3, 10), (12, 10)):
            threshold = max(org_minimum, workflow_floor)
            with self.subTest(org=org_minimum, workflow=workflow_floor):
                panel = pd.DataFrame(
                    [
                        {"group": group, "PersonId": f"{group}-{person}", "activity": 1}
                        for group, n in (("below", threshold - 1), ("at", threshold), ("above", threshold + 1))
                        for person in range(n)
                        for _ in range(3)
                    ]
                )
                summary = panel.groupby("group").agg(
                    people=("PersonId", "nunique"), activity=("activity", "sum")
                ).reset_index()
                output = suppress(summary, "group", "people", threshold).set_index("group")
                self.assertTrue(output.loc["below", ["people", "activity"]].isna().all())
                self.assertEqual(output.loc["at", "people"], threshold)
                self.assertEqual(output.loc["above", "people"], threshold + 1)
                self.assertFalse(summary["people"].isna().any())

    def test_identify_nkw_threshold_boundary_without_transfer(self):
        panel = pd.DataFrame({
            "PersonId": ["below", "at", "above"] * 2,
            "Organization": ["Synthetic"] * 6,
            "Collaboration_hours": [4.9, 5, 5.1] * 2,
        })
        output = vi.identify_nkw(panel, collab_threshold=5, return_type="data_with_flag")
        self.assertEqual(len(output), len(panel))
        flags = output.groupby("PersonId")["flag_nkw"].first().to_dict()
        self.assertEqual(flags, {"below": "nkw", "at": "kw", "above": "kw"})

    def run_segmentation_example(self, panel):
        script = ROOT / "frontier-analytics" / "skills" / "viva-insights-analysis" / "examples" / "python" / "03_copilot_segmentation.py"
        with mock.patch.object(vi, "load_pq_data", return_value=panel.copy()):
            with contextlib.redirect_stdout(io.StringIO()):
                return runpy.run_path(str(script))["seg"]

    def segmentation_panel(self):
        return pd.DataFrame({
            "PersonId": ["synthetic"] * 30,
            "MetricDate": pd.date_range("2026-01-05", periods=30, freq="7D"),
            "Total_Copilot_enabled_days": [5.0] * 30,
            "Copilot_actions_taken_in_Teams": [10.0] * 30,
        })

    def test_segmentation_example_withholds_affected_eligibility_and_null_windows(self):
        for column, invalid in [
            ("Total_Copilot_enabled_days", 0),
            ("Total_Copilot_enabled_days", None),
        ]:
            with self.subTest(column=column, invalid=invalid):
                panel = self.segmentation_panel()
                panel.loc[12, column] = invalid
                result = self.run_segmentation_example(panel).reset_index(drop=True)
                self.assertEqual(len(result), len(panel))
                for width in (4, 12):
                    columns = [f"UsageSegments_{width}w", f"IsHabit{width}w", f"target_metric_l{width}w"]
                    self.assertTrue(result.loc[:width - 2, columns].isna().all().all())
                    self.assertTrue(result.loc[12:12 + width - 1, columns].isna().all().all())
                    self.assertTrue(result.loc[12 + width:, columns].notna().all().all())
                    self.assertTrue(result.loc[11, columns].notna().all())

    def test_segmentation_example_stops_on_unknown_activity(self):
        panel = self.segmentation_panel()
        panel.loc[12, "Copilot_actions_taken_in_Teams"] = None
        with self.assertRaisesRegex(ValueError, "Unknown activity"):
            self.run_segmentation_example(panel)

    def test_segmentation_example_keeps_gaps_and_restores_only_full_windows(self):
        panel = self.segmentation_panel().drop(index=12).sample(frac=1, random_state=4)
        result = self.run_segmentation_example(panel).set_index("MetricDate")
        for width in (4, 12):
            blocked = pd.date_range("2026-01-05", periods=30, freq="7D")[13:12 + width]
            restored = pd.date_range("2026-01-05", periods=30, freq="7D")[12 + width]
            self.assertTrue(result.loc[blocked, f"UsageSegments_{width}w"].isna().all())
            self.assertTrue(pd.notna(result.loc[restored, f"UsageSegments_{width}w"]))
        self.assertEqual(len(result), 29)

    def test_segmentation_example_known_eligible_zero_is_non_user(self):
        panel = self.segmentation_panel()
        panel["Copilot_actions_taken_in_Teams"] = 0.0
        result = self.run_segmentation_example(panel).reset_index(drop=True)
        self.assertEqual(result.loc[11:, "UsageSegments_12w"].unique().tolist(), ["Non-user"])
        self.assertTrue(result.loc[:10, "UsageSegments_12w"].isna().all())


if __name__ == "__main__":
    unittest.main()
