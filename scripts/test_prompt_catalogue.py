#!/usr/bin/env python3
"""Focused regression tests for the generated Prompt Library catalogue."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_prompt_catalogue as generator
import check_prompt_drift as drift


class BadgeTests(unittest.TestCase):
    def test_string_scalar_styles_are_not_silently_dropped(self):
        page = """---
badges:
  data_source: 'Person''s query'
  language: "R or Python"
  output: HTML report # a comment
  min_data: 8–12+ weeks
---
"""
        self.assertEqual(generator.badge_values(page), {
            "data_source": "Person's query",
            "language": "R or Python",
            "output": "HTML report",
            "min_data": "8–12+ weeks",
        })

    def test_unsupported_badge_value_fails_explicitly(self):
        for value in ("|", "[R, Python]", "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generator.badge_values(f"---\nbadges:\n  language: {value}\n---\n")

    def test_badges_are_optional(self):
        self.assertEqual(generator.badge_values("---\ntitle: Example\n---\n"), {})

    def test_inline_badges_mapping_is_rejected(self):
        with self.assertRaises(ValueError):
            generator.badge_values("---\nbadges: {language: R}\n---\n")


class CatalogueChecksTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.prompts = self.root / "frontier-analytics" / "prompts"
        self.pages = self.root / "_pages"
        self.source = self.prompts / "copilot-adoption" / "example.md"
        self.page = self.pages / "frontier-analytics-prompt-example.md"
        self.data = self.root / "_data" / "prompts.json"
        self.source.parent.mkdir(parents=True)
        self.pages.mkdir(parents=True)
        self.mapping = {"copilot-adoption/example.md": self.page.name}
        self.body = """# Example

## Purpose

Summary with a [related prompt](../related.md).

## Quick prompt (short version)

```
quick prompt
```

## Prompt 1: First part

```
first part
```

## Prompt 2: Second part

```
second part
```
"""
        self.front_matter = """---
permalink: /frontier-analytics-prompt-example/
badges:
  data_source: Person Query
  output: "HTML report"
---
"""
        self.source.write_text(self.body, encoding="utf-8")
        self.page.write_text(self.front_matter + self.body, encoding="utf-8")
        patches = [
            mock.patch.object(generator, "PROMPTS_DIR", self.prompts),
            mock.patch.object(generator, "PAGES_DIR", self.pages),
            mock.patch.object(generator, "SITE_DATA", self.data),
            mock.patch.object(generator, "MAPPING", self.mapping),
            mock.patch.object(drift, "PROMPTS_DIR", self.prompts),
            mock.patch.object(drift, "PAGES_DIR", self.pages),
            mock.patch.object(drift, "REPO_ROOT", self.root),
            mock.patch.object(drift, "MAPPING", self.mapping),
            mock.patch.object(drift, "KNOWN_SITE_LESS_CARDS", set()),
            mock.patch.object(drift, "PARITY_KEYWORDS", []),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.assertEqual(self.generate(), 0)

    def generate(self, *args):
        with mock.patch.object(sys, "argv", ["generate_prompt_catalogue.py", *args]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return generator.main()

    def check_drift(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = drift.main()
        return result, output.getvalue()

    def test_check_detects_stale_metadata_without_writing(self):
        original = self.data.read_text(encoding="utf-8")
        self.assertEqual(self.generate("--check"), 0)
        self.page.write_text(self.front_matter.replace("HTML report", "Notebook") + self.body, encoding="utf-8")
        self.assertEqual(self.generate("--check"), 1)
        self.assertEqual(self.data.read_text(encoding="utf-8"), original)
        self.assertIn("Summary with a related prompt.", original)
        self.assertNotIn("../related.md", original)

    def test_missing_quick_fence_fails_even_when_both_copies_match(self):
        body = self.body.replace("```\nquick prompt\n```", "quick prompt")
        self.source.write_text(body, encoding="utf-8")
        self.page.write_text(self.front_matter + body, encoding="utf-8")
        result, output = self.check_drift()
        self.assertEqual(result, 1)
        self.assertIn("no complete fenced block", output)

    def test_second_full_prompt_block_must_match(self):
        self.assertEqual(self.check_drift()[0], 0)
        self.page.write_text(
            self.front_matter + self.body.replace("second part\n```", "changed part\n```"),
            encoding="utf-8",
        )
        result, output = self.check_drift()
        self.assertEqual(result, 1)
        self.assertIn("full prompt code blocks differ", output)

    def test_missing_full_prompt_fence_fails(self):
        body = self.body.replace("```\nsecond part\n```", "second part")
        self.source.write_text(body, encoding="utf-8")
        self.page.write_text(self.front_matter + body, encoding="utf-8")
        result, output = self.check_drift()
        self.assertEqual(result, 1)
        self.assertIn("no fenced prompt", output)


if __name__ == "__main__":
    unittest.main()
