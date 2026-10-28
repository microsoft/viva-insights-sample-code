"""Regression checks for the generated site page and its machine-readable contract."""
import copy
import html
from html.parser import HTMLParser
import json
import re
import unittest

import pandas as pd

import build_join_guide as guide


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.anchors = []
        self.contract = ""
        self.in_contract = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.anchors.append(attrs["href"][1:])
        if tag == "script" and attrs.get("id") == "join-contract":
            self.in_contract = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_contract = False

    def handle_data(self, data):
        if self.in_contract:
            self.contract += data


class JoinGuideTests(unittest.TestCase):
    def test_generated_files_are_current(self):
        self.assertEqual(guide.PAGE.read_text(encoding="utf-8"), guide.build_html())
        self.assertEqual(
            guide.CONTRACT_PATH.read_text(encoding="utf-8"),
            json.dumps(guide.CONTRACT, indent=2, ensure_ascii=False) + "\n",
        )

    def test_contract_and_anchor_integrity(self):
        parser = PageParser()
        parser.feed(guide.build_html())
        self.assertEqual(json.loads(parser.contract), guide.CONTRACT)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        self.assertTrue(set(parser.anchors) <= set(parser.ids))

    def test_native_site_layout(self):
        page = guide.build_html()
        self.assertTrue(page.startswith("---\nlayout: page\n"))
        self.assertNotIn("<html", page)
        self.assertNotIn("<body", page)
        self.assertNotIn("<h1", page)
        self.assertNotIn('<nav class="toc"', page)
        self.assertIn("permalink: /query-join-guide/", page)
        self.assertIn("'/assets/data/query-join-contract.json' | relative_url", page)
        for query in guide.CONTRACT["queries"]:
            self.assertIn(f'id="{query["id"]}-heading"', page)
        self.assertEqual(page.count("Week (Recommended)"), 4)

    def test_invalid_contract_references_fail(self):
        contract = copy.deepcopy(guide.CONTRACT)
        contract["derived_datasets"][0]["build_steps"] = [999]
        with self.assertRaises(AssertionError):
            guide.validate_contract(contract)
        contract = copy.deepcopy(guide.CONTRACT)
        contract["relationships"][0]["to_id"] = "unknown"
        with self.assertRaises(AssertionError):
            guide.validate_contract(contract)

    def test_query_entry_points_defer_to_current_guide(self):
        for path in [
            "frontier-analytics/schemas/README.md",
            "frontier-analytics/schemas/viva-person-query-data-dictionary.md",
            "frontier-analytics/skills/viva-insights-analysis/SKILL.md",
            "frontier-analytics/skills/viva-insights-analysis/reference/query-schemas.md",
            "frontier-analytics/skills/viva-insights-analysis/reference/copilot-query-contracts.md",
            "frontier-analytics/skills/viva-insights-analysis/reference/deliverables.md",
            "frontier-analytics/skills/viva-insights-copilot-dashboards/SKILL.md",
            "frontier-analytics/skills/viva-insights-copilot-dashboards/reference/consumption.md",
            "frontier-analytics/skills/viva-insights-copilot-dashboards/reference/github.md",
            "examples/utility-r/_data/README.md",
        ]:
            with self.subTest(path=path):
                text = guide.ROOT.joinpath(*path.split("/")).read_text(encoding="utf-8")
                self.assertIn("https://microsoft.github.io/viva-insights-sample-code/query-join-guide/", text)
                self.assertNotIn("this reference owns the analytical rules", text)

    def test_documented_cadence_uses_configuration_not_sparse_dates(self):
        text = (guide.ROOT / "frontier-analytics" / "schemas" / "viva-person-query-data-dictionary.md").read_text(encoding="utf-8")
        section = text.split("### Detecting time granularity\n", 1)[1]
        code = re.search(r"```python\n(.*?)\n```", section, re.S).group(1)
        for grouping, dates, expected in [
            ("day", ["2026-10-19", "2026-10-26"], "daily"),
            ("day", ["2026-10-19"], "daily"),
            ("week", ["2026-10-19", "2026-11-02"], "weekly"),
            ("month", ["2026-01-01", "2026-02-01"], "monthly"),
            (None, ["2026-10-19", "2026-10-26"], None),
        ]:
            with self.subTest(grouping=grouping, dates=dates):
                scope = {
                    "pd": pd,
                    "df": pd.DataFrame({"MetricDate": dates}),
                    "export_config": {"date_grouping": grouping},
                }
                if expected is None:
                    with self.assertRaisesRegex(ValueError, "configured date grouping"):
                        exec(code, scope)
                else:
                    exec(code, scope)
                    self.assertEqual(scope["granularity"], expected)

    def test_coverage_defaults_and_aggregation_exceptions_are_visible(self):
        page = guide.build_html()
        for rule in guide.CONTRACT["coverage_assumption"].values():
            if isinstance(rule, str):
                self.assertIn(html.escape(rule, quote=True), page)
        gates = {g["id"]: g["rule"] for g in guide.CONTRACT["gates"]}
        self.assertIn("panel-bound summaries", gates["G07"])
        self.assertIn("G05b", gates["G07"])
        self.assertIn("G11", gates["G07"])
        self.assertNotIn("in every case", gates["G07"])
        files = {f["id"]: f for q in guide.CONTRACT["queries"] for f in q["files"]}
        for measure in guide.CONTRACT["measures"]:
            if measure["file_id"].startswith("github.by_"):
                rule = measure["person_week_rule"]
                self.assertIn("linked detail", rule)
                for dimension in files[measure["file_id"]]["native_key"]:
                    if dimension not in {"PersonId", "MetricDate"}:
                        self.assertIn(dimension, rule)

    def test_documented_week_alignment_uses_detected_weekday(self):
        text = (guide.ROOT / "frontier-analytics" / "schemas" / "join-patterns.md").read_text(encoding="utf-8")
        alignment = text.split("### Time alignment\n", 1)[1].split("### Pitfalls", 1)[0]
        self.assertIn("person_query['MetricDate'].dt.dayofweek.unique()", alignment)
        self.assertIn("event_days.dt.dayofweek - int(week_starts[0])", alignment)
        self.assertIn("unique(wday(person_query$MetricDate, week_start = 1))", alignment)
        self.assertIn("week_start = week_starts[[1]]", alignment)
        self.assertNotIn("to_period('W-SUN')", alignment)

    def test_documented_python_join_preserves_calendar_weeks(self):
        text = (guide.ROOT / "frontier-analytics" / "schemas" / "join-patterns.md").read_text(encoding="utf-8")
        alignment = text.split("### Time alignment\n", 1)[1].split("### Pitfalls", 1)[0]
        code = re.search(r"```python\n(.*?)\n```", alignment, re.S).group(1)
        for start, zone in [
            ("2026-10-19", "UTC"),
            ("2026-10-25", "UTC"),
            ("2026-11-01", "America/Los_Angeles"),
            ("2026-10-26", "America/Los_Angeles"),
            ("2026-03-08", "America/Los_Angeles"),
            ("2026-03-02", "America/Los_Angeles"),
        ]:
            with self.subTest(start=start, zone=zone):
                scope = {
                    "pd": pd,
                    "person_query": pd.DataFrame({
                        "PersonId": ["synthetic"] * 2,
                        "MetricDate": pd.date_range(start, periods=2, freq="7D", tz=zone),
                    }),
                    "purview_mapped": pd.DataFrame({
                        "PersonId": ["synthetic"] * 14,
                        "creation_time": pd.date_range(start, periods=14, tz=zone),
                        "event_id": range(14),
                        "operation": ["test"] * 14,
                        "copilot_app_host": ["test"] * 14,
                    }),
                }
                exec(code, scope)
                self.assertEqual(scope["enriched"]["copilot_events"].tolist(), [7, 7])
                self.assertEqual(len(scope["purview_weekly"]), 2)


if __name__ == "__main__":
    unittest.main()
