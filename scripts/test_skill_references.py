"""Verify mandatory installed Skill references without the source checkout."""

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prompt_metadata import ROOT

SKILLS = ROOT / "frontier-analytics" / "skills"
NAMES = ("viva-insights-analysis", "viva-insights-copilot-dashboards", "viva-insights-causal-analysis")


def mandatory_references(skill):
    text = (skill / "SKILL.md").read_text(encoding="utf-8")
    paths = re.findall(r"`((?:reference/|\.\./viva-insights-[^/]+/reference/)[^`\s]+\.md)`", text)
    paths += re.findall(r"\[[^\]]+\]\(((?:reference/|\.\./viva-insights-[^/]+/reference/)[^)#]+\.md)(?:#[^)]*)?\)", text)
    return [skill.joinpath(*path.split("/")) for path in paths]


class InstalledReferencesTests(unittest.TestCase):
    def test_bundled_meeting_merge_regressions(self):
        test = SKILLS / NAMES[0] / "examples" / "test_merge_meeting_exports.py"
        result = subprocess.run(
            [sys.executable, "-B", str(test)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_matched_R_segmentation_fixture_is_a_PR_gate(self):
        workflow = (ROOT / ".github" / "workflows" / "prompt-drift-check.yml").read_text(encoding="utf-8")
        self.assertIn("  pull_request:", workflow)
        self.assertIn("  r-segmentation:", workflow)
        self.assertIn("cran::vivainsights@0.7.2", workflow)
        self.assertIn("Rscript --vanilla scripts/test_segmentation.R", workflow)

    def test_core_standalone_and_companion_installations(self):
        for names in [(NAMES[0],), NAMES]:
            with self.subTest(skills=names), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for name in names:
                    shutil.copytree(SKILLS / name, root / name)
                for name in names:
                    references = mandatory_references(root / name)
                    self.assertTrue(references, f"{name}: no mandatory references checked")
                    for target in references:
                        self.assertTrue(target.resolve().is_relative_to(root.resolve()), target)
                        self.assertTrue(target.is_file(), f"{name}: missing installed reference {target.name}")

    def test_missing_shared_reference_is_detectable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in NAMES:
                shutil.copytree(SKILLS / name, root / name)
            target = root / NAMES[0] / "reference" / "analysis-policy.md"
            target.unlink()
            self.assertTrue(any(not path.is_file() for name in NAMES for path in mandatory_references(root / name)))


if __name__ == "__main__":
    unittest.main()
