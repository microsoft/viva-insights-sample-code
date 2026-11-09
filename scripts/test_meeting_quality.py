"""Synthetic regressions for the bundled seven-rule Meeting Query recipe."""

from pathlib import Path
import sys
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

EXAMPLES = Path(__file__).resolve().parents[1] / "frontier-analytics" / "skills" / "viva-insights-analysis" / "examples"
sys.path.insert(0, str(EXAMPLES))
from meeting_quality import normalise_flag, quality_mask


def frame(n=1):
    return pd.DataFrame({
        "Number_of_attendees": [3] * n, "Intended_participant_count": [3] * n,
        "Cancelled": ["False"] * n, "All_Day_Meeting": ["No"] * n,
        "Duration": [1.0] * n,
    })


class MeetingQualityTests(unittest.TestCase):
    def test_text_flags_preserve_valid_meetings_and_accept_common_variants(self):
        data = frame(6)
        data["Cancelled"] = ["False", " No ", "N", "0", False, 0.0]
        data["All_Day_Meeting"] = ["false", "NO", "n", "0.0", False, 0]
        before = data.copy(deep=True)
        self.assertTrue(quality_mask(data).all())
        assert_frame_equal(data, before)
        truth = normalise_flag(pd.Series(["True", "YES", "Y", "1", True, 1.0]))
        self.assertTrue(truth.all())

    def test_unknown_tokens_stop_and_missing_flags_remain_unknown(self):
        for token in ("maybe", "off", 2):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "Unrecognised"):
                quality_mask(frame().assign(Cancelled=token))
        flags = normalise_flag(pd.Series([None, "", " #N/A ", "null"]))
        self.assertTrue(flags.isna().all())
        self.assertFalse(quality_mask(frame().assign(Cancelled="")).iloc[0])

    def test_duration_boundaries_and_missing_inputs(self):
        data = frame(6)
        data["Duration"] = [1 / 60, 8, 0.01, 8.01, None, 1]
        data.loc[5, "Intended_participant_count"] = 151
        self.assertEqual(quality_mask(data).tolist(), [True, True, False, False, False, False])
        self.assertTrue(quality_mask(data, max_intended=200).iloc[5])
        with self.assertRaisesRegex(ValueError, "including Duration"):
            quality_mask(data.drop(columns="Duration"))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            quality_mask(frame().assign(Duration=float("inf")))

    def test_documented_python_recipe_uses_the_same_full_filter(self):
        import re
        import vivainsights as vi
        path = EXAMPLES.parent / "reference" / "query-schemas.md"
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
        code = next(b for b in blocks if "from meeting_quality import quality_mask" in b)
        namespace = {"vi": vi}
        exec(compile(code, str(path), "exec"), namespace)
        expected = namespace["mt"].loc[quality_mask(namespace["mt"])]
        assert_frame_equal(namespace["mq"], expected)


if __name__ == "__main__":
    unittest.main()
