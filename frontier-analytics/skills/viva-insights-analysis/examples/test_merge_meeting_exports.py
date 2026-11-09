"""Synthetic regression tests for the documented in-memory merge example."""

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from merge_meeting_exports import merge_mq, quality_mask


ATTRIBUTES = ["Organization", "Function"]


def frame(ids):
    n = len(ids)
    return pd.DataFrame({
        "MeetingId": pd.Series(ids, dtype="string"),
        "Collaboration_start_time": pd.to_datetime(["2020-06-15 09:00"] * n),
        "Number_of_attendees": pd.Series([3] * n, dtype="Int64"),
        "Intended_participant_count": pd.Series([3] * n, dtype="Int64"),
        "Cancelled": pd.Series([False] * n, dtype="boolean"),
        "All_Day_Meeting": pd.Series([False] * n, dtype="boolean"),
        "Duration": pd.Series([1.0] * n, dtype="Float64"),
        "Attendee_meeting_hours": pd.Series([3.0] * n, dtype="Float64"),
        "Organization": pd.Series(["Ops"] * n, dtype="string"),
        "Function": pd.Series(["Planning"] * n, dtype="string"),
    })


class MergeTests(unittest.TestCase):
    def merge(self, old, new):
        before = [old.copy(deep=True), new.copy(deep=True)]
        try:
            return merge_mq(old, new, attribute_columns=ATTRIBUTES)
        finally:
            assert_frame_equal(old, before[0])
            assert_frame_equal(new, before[1])

    def test_full_union_preferences_backfill_and_audit(self):
        old = frame(["old-invalid", "old-valid", "prefer-old", "both-valid", "both-invalid", "blank"])
        new = frame(["prefer-old", "both-valid", "both-invalid", "blank", "new-only"])
        old.loc[[0, 4], "Cancelled"] = True
        new.loc[[0, 2], "Cancelled"] = True
        new.loc[3, "Organization"] = " #n/a "
        new.loc[1, "Attendee_meeting_hours"] = 4.0
        new.loc[2, "Attendee_meeting_hours"] = 5.0
        combined, audit = self.merge(old, new)
        rows = combined.set_index("MeetingId")
        self.assertEqual(set(rows.index), set(old.MeetingId) | set(new.MeetingId))
        self.assertEqual(len(rows), 7)
        self.assertEqual(audit["combined_rows"], len(new) + audit["old_only_ids"])
        self.assertEqual(audit["older_valid_preferred_ids"], 1)
        self.assertEqual(audit["old_rows_written"], 3)
        self.assertEqual(audit["new_rows_written"], 4)
        self.assertEqual(audit["shared_ids"], 4)
        self.assertEqual(audit["new_only_ids"], 1)
        self.assertEqual(audit["true_conflicts"], 0)
        self.assertEqual(rows.loc["old-invalid", "source_export"], "old")
        self.assertTrue(rows.loc["old-invalid", "Cancelled"])
        self.assertEqual(rows.loc["prefer-old", "source_export"], "old")
        self.assertFalse(rows.loc["prefer-old", "Cancelled"])
        self.assertEqual(rows.loc["new-only", "source_export"], "new")
        self.assertEqual(rows.loc["both-valid", "Attendee_meeting_hours"], 4.0)
        self.assertEqual(rows.loc["both-invalid", "Attendee_meeting_hours"], 5.0)
        self.assertEqual(rows.loc["blank", "Organization"], "Ops")
        self.assertEqual(audit["backfilled_cells_by_column"], {"Organization": 1, "Function": 0})
        self.assertEqual(audit["backfilled_cells_by_source"]["Organization"], {"old": 1, "new": 0})

    def test_newer_attributes_fill_older_preferred_row(self):
        old, new = frame(["shared"]), frame(["shared"])
        old.loc[0, "Organization"] = pd.NA
        new.loc[0, "Cancelled"] = True
        merged, audit = self.merge(old, new)
        self.assertEqual(merged.source_export.iloc[0], "old")
        self.assertEqual(merged.Organization.iloc[0], "Ops")
        self.assertEqual(audit["backfilled_cells_by_source"]["Organization"], {"old": 0, "new": 1})

    def test_all_seven_rules_and_boundaries(self):
        cases = [
            ("Number_of_attendees", 1, False), ("Number_of_attendees", 2, True),
            ("Intended_participant_count", 1, False), ("Intended_participant_count", 151, False),
            ("Intended_participant_count", 2, True), ("Intended_participant_count", 150, True),
            ("Cancelled", True, False), ("All_Day_Meeting", True, False),
            ("Duration", 0.01, False), ("Duration", 8.1, False),
            ("Duration", 1 / 60, True), ("Duration", 8.0, True),
        ]
        for col, value, expected in cases:
            with self.subTest(column=col, value=value):
                data = frame(["event"])
                data.loc[0, col] = value
                self.assertEqual(bool(quality_mask(data).iloc[0]), expected)

    def test_missing_quality_retained_in_raw_union(self):
        for col in ("Cancelled", "All_Day_Meeting", "Number_of_attendees",
                    "Intended_participant_count", "Duration"):
            with self.subTest(column=col):
                old, new = frame(["old-only"]), frame(["new-only"])
                old.loc[0, col] = pd.NA
                self.assertFalse(quality_mask(old).iloc[0])
                merged, _ = self.merge(old, new)
                self.assertEqual(len(merged), 2)

    def test_blank_tokens_and_nonblank_new_value(self):
        for blank in (None, pd.NA, "", " ", "NA", " n/a ", "#N/A", "null"):
            with self.subTest(blank=repr(blank)):
                old, new = frame(["shared"]), frame(["shared"])
                new.loc[0, "Organization"] = blank
                merged, _ = self.merge(old, new)
                self.assertEqual(merged.Organization.iloc[0], "Ops")
                old.loc[0, "Organization"], new.loc[0, "Organization"] = blank, "New org"
                merged, audit = self.merge(old, new)
                self.assertEqual(merged.Organization.iloc[0], "New org")
                self.assertEqual(audit["backfilled_cells_by_column"]["Organization"], 0)

    def test_conflicts_refused_on_every_shared_row(self):
        for old_invalid, new_invalid in ((False, False), (False, True), (True, True)):
            with self.subTest(old_invalid=old_invalid, new_invalid=new_invalid):
                old, new = frame(["shared"]), frame(["shared"])
                old.loc[0, "Cancelled"], new.loc[0, "Cancelled"] = old_invalid, new_invalid
                new.loc[0, "Organization"] = "Finance"
                with self.assertRaisesRegex(ValueError, "Populated organiser/HR conflict"):
                    self.merge(old, new)

    def test_whitespace_not_alias_merging(self):
        old, new = frame(["shared"]), frame(["shared"])
        new.loc[0, "Organization"] = " Ops "
        merged, _ = self.merge(old, new)
        self.assertEqual(merged.Organization.iloc[0], " Ops ")
        new.loc[0, "Organization"] = "OPS"
        with self.assertRaisesRegex(ValueError, "conflict"):
            self.merge(old, new)

    def test_duplicate_blank_and_noncanonical_keys(self):
        for ids in (["same", "same"], [""], [" "], [None], [" padded"], ["#N/A"]):
            for invalid_side in ("old", "new"):
                with self.subTest(ids=ids, side=invalid_side):
                    old, new = frame(["good"]), frame(["good"])
                    if invalid_side == "old":
                        old = frame(ids)
                    else:
                        new = frame(ids)
                    with self.assertRaisesRegex(ValueError, "MeetingId"):
                        self.merge(old, new)

    def test_schema_and_types_refused(self):
        mutations = [
            lambda d: d.drop(columns="Duration"),
            lambda d: d.assign(Unexpected="x"),
            lambda d: pd.concat([d, d[["Duration"]]], axis=1),
            lambda d: d.assign(source_export="prior"),
            lambda d: d.astype({"Number_of_attendees": "string"}),
            lambda d: d.astype({"Cancelled": "string"}),
            lambda d: d.astype({"Duration": "float64"}),
            lambda d: d.assign(Organization=7),
            lambda d: d.assign(Duration=float("inf")),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                with self.assertRaises(ValueError):
                    self.merge(frame(["shared"]), mutation(frame(["shared"])))

    def test_timestamp_drift_and_missing_refused(self):
        for value in (pd.NaT, pd.Timestamp("2020-06-16 09:00")):
            with self.subTest(value=value):
                old, new = frame(["shared"]), frame(["shared"])
                new.loc[0, "Collaboration_start_time"] = value
                with self.assertRaises(ValueError):
                    self.merge(old, new)

    def test_column_order_and_empty_exports(self):
        old, new = frame(["old-only"]), frame(["new-only"])
        merged, _ = self.merge(old, new[new.columns[::-1]])
        self.assertEqual(set(merged.MeetingId), {"old-only", "new-only"})
        for prior, current in ((old.iloc[:0], new), (old, new.iloc[:0]), (old.iloc[:0], new.iloc[:0])):
            with self.subTest(old=len(prior), new=len(current)):
                merged, _ = self.merge(prior, current)
                self.assertEqual(len(merged), len(prior) + len(current))

    def test_invalid_attribute_contract(self):
        for attributes in ("Organization", ["MeetingId"], ["Missing"], ["Function", "Function"], [1]):
            with self.subTest(attributes=attributes):
                with self.assertRaises(ValueError):
                    merge_mq(frame(["a"]), frame(["b"]), attribute_columns=attributes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
