"""Regression checks for the generated site page and its machine-readable contract."""
import copy
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
