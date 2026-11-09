import copy
import unittest

from run_example import analyse, classify, fabricated_events, load_cases, load_policy


class ProcessExampleTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_all_worked_cases_and_unicode_boundaries(self):
        for event in load_cases():
            with self.subTest(case=event["case"]):
                self.assertEqual(classify(event, self.policy)["final_in_scope"], event["expected"])

    def test_overlap_is_one_event_and_manual_decisions_do_not_transfer(self):
        events = {e["case"]: e for e in load_cases()}
        selected = classify(events["overlap"], self.policy)
        self.assertEqual(len(selected["categories"]), 2)
        self.assertTrue(selected["final_in_scope"])
        self.assertFalse(classify(events["same_title_other_date"], self.policy)["final_in_scope"])

    def test_asymmetric_year_codes_and_invalid_decisions_are_refused(self):
        bad = copy.deepcopy(self.policy)
        del bad["year_codes"]["2020"]
        with self.assertRaisesRegex(ValueError, "Every comparison year"):
            classify(load_cases()[0], bad)
        for change, message in (({"decision": "Maybe"}, "Invalid manual"),
                                ({"decision": "Pending", "prior_in_scope": None}, "prior disposition")):
            with self.assertRaisesRegex(ValueError, message):
                classify(load_cases()[0] | change, self.policy)

    def test_partial_periods_are_unavailable_but_complete_empty_periods_are_zero(self):
        report = analyse(fabricated_events(self.policy), self.policy)
        self.assertEqual([r["status"] for r in report["years"]], ["observed", "observed", "partial"])
        self.assertIsNone(report["years"][2]["attendee_hours"])
        empty = analyse([], self.policy)
        self.assertEqual(empty["years"][0]["meetings"], 0)
        self.assertIsNone(empty["years"][2]["meetings"])

    def test_event_count_does_not_replace_people_privacy(self):
        events = [load_cases()[0] | {"MeetingId": f"unique-{i}", "Organizer": "one-person",
                                    "attendee_hours": 1.0} for i in range(30)]
        row = analyse(events, self.policy)["years"][1]
        self.assertEqual(row["status"], "privacy_withheld")
        self.assertIsNone(row["meetings"])
        self.assertIsNone(row["attendee_hours"])

    def test_unapproved_or_unknown_coverage_cannot_publish(self):
        for change in ({"status": "proposed"}, {"coverage_confirmed": False}):
            with self.assertRaisesRegex(ValueError, "Approved scope"):
                analyse(fabricated_events(self.policy), self.policy | change)

    def test_duplicate_keys_and_invalid_values_stop_without_mutating_sources(self):
        events = fabricated_events(self.policy)
        before = copy.deepcopy(events)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            analyse(events + [events[0]], self.policy)
        with self.assertRaisesRegex(ValueError, "Invalid attendee"):
            analyse([events[0] | {"attendee_hours": float("nan")}], self.policy)
        self.assertEqual(events, before)

    def test_quality_and_source_coverage_are_not_inferred_from_strings_or_latest_event(self):
        event = load_cases()[0]
        with self.assertRaisesRegex(ValueError, "prepared boolean"):
            classify(event | {"quality_pass": "False"}, self.policy)
        with self.assertRaisesRegex(ValueError, "source coverage"):
            classify(event | {"Date": "2022-04-01"}, self.policy)


if __name__ == "__main__":
    unittest.main()
