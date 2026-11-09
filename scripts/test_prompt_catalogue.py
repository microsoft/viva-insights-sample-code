"""Regression gates for canonical prompt pages, metadata and copyable content."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import generate_prompt_catalogue as catalogue
import generate_prompt_pages as pages
import prompt_metadata as metadata


class PromptGenerationTests(unittest.TestCase):
    def test_all_generated_outputs_are_exact(self):
        for path, expected in pages.generated_pages().items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), expected)
        self.assertEqual(catalogue.SITE_DATA.read_text(encoding="utf-8"), catalogue.generated_catalogue())

    def test_catalogue_does_not_read_generated_pages(self):
        with mock.patch.object(pages, "PAGES_DIR", Path("nonexistent")):
            self.assertEqual(catalogue.generated_catalogue(), catalogue.SITE_DATA.read_text(encoding="utf-8"))

    def test_fenced_prompt_and_code_bytes_preserved(self):
        for item in metadata.PAGES:
            source = (metadata.PROMPTS_DIR / item["source"]).read_text(encoding="utf-8")
            rendered = pages.render_page(item, source)
            self.assertEqual(
                [part for fenced, part in metadata.fenced_parts(source) if fenced],
                [part for fenced, part in metadata.fenced_parts(rendered) if fenced],
            )
            self.assertEqual(metadata.full_prompt_blocks(source), metadata.full_prompt_blocks(rendered))

    def test_purview_audit_samples_require_identifier_masking(self):
        source = (metadata.PROMPTS_DIR / "purview-augmentation/audit-log-parsing.md").read_text(encoding="utf-8")
        for instruction in (
            "sample records with direct identifiers such as UserId, UPNs and email addresses masked",
            "Print masked examples, recursively redacting direct identifiers",
            "Sample of 5 Copilot event rows with direct user identifiers masked",
            "Mask or omit direct user identifiers in examples.",
        ):
            with self.subTest(instruction=instruction):
                self.assertIn(instruction, source)

    def test_roi_prompt_resolves_null_assisted_hours_before_proxying(self):
        source = (metadata.PROMPTS_DIR / "copilot-adoption/roi-estimation.md").read_text(encoding="utf-8")
        self.assertIn("If Copilot_Assisted_Hours is null despite positive licence or activity evidence", source)
        self.assertIn("withhold affected estimates until resolved", source)
        self.assertIn("propose a documented proxy and conversion for approval", source)

    def test_multiple_prompts_liquid_and_code_links_are_literal(self):
        body = """# Example

## Purpose

Description.

## Quick prompt (short version)

```
quick {{ value }}
```

## Prompt 1: First

```text
{{ site.baseurl }} [literal](../missing.md)
```

## Prompt 2: Second

~~~
second {% include missing %}
~~~
"""
        result = pages.render_page(metadata.PAGES[0], body)
        self.assertEqual(metadata.full_prompt_blocks(body), metadata.full_prompt_blocks(result))
        self.assertEqual(result.count("{% raw %}"), 3)
        self.assertIn("[literal](../missing.md)", result)

    def test_relative_link_routes_and_fragments(self):
        source = "copilot-adoption/copilot-causal-toolkit.md"
        result = pages.rewrite_links(
            "[adoption](dashboard-overview.md#prompt) [repo-only](../person-query/causal-diagnostics.md#purpose)",
            source,
        )
        self.assertIn("{{ site.baseurl }}/frontier-analytics-prompt-dashboard/#prompt", result)
        self.assertIn("blob/main/frontier-analytics/prompts/person-query/causal-diagnostics.md#purpose", result)
        with self.assertRaisesRegex(ValueError, "unmapped relative link"):
            pages.rewrite_links("[bad](missing.md)", source)

    def test_missing_and_orphan_sources_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(pages, "PROMPTS_DIR", Path(temp)):
                with self.assertRaisesRegex(ValueError, "Unmapped/missing"):
                    pages.generated_pages()
        with mock.patch.object(pages, "MAPPING", {**metadata.MAPPING, "extra.md": "orphan.md"}):
            with self.assertRaisesRegex(ValueError, "Unmapped/missing"):
                pages.generated_pages()

    def test_check_is_read_only_and_detects_page_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            page = Path(temp) / "page.md"
            page.write_text("stale", encoding="utf-8")
            with mock.patch.object(pages, "generated_pages", return_value={page: "expected"}):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertFalse(pages.sync_pages(True))
                self.assertEqual(page.read_text(encoding="utf-8"), "stale")
                self.assertTrue(pages.sync_pages(False))
                self.assertTrue(pages.sync_pages(True))

    def test_missing_full_prompt_fence_fails(self):
        with self.assertRaisesRegex(ValueError, "no fenced prompt"):
            metadata.full_prompt_blocks("## Prompt 1\n\n```\none\n```\n\n## Prompt 2\nmissing\n")
        with self.assertRaisesRegex(ValueError, "Unclosed"):
            list(metadata.fenced_parts("```\nunclosed"))

    def test_metadata_routes_unique_and_existing(self):
        slugs = [item["slug"] for item in metadata.PAGES]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertEqual(len(metadata.MAPPING), len(metadata.PAGES))
        self.assertTrue(set(metadata.METADATA["navigation"]) <= set(slugs))
        for target in metadata.METADATA["links"]:
            self.assertTrue(metadata.ROOT.joinpath(*target.split("/")).is_file(), target)
        for group in json.loads(catalogue.generated_catalogue()):
            for item in group["prompts"]:
                self.assertTrue(item["quick_prompt"])
                self.assertTrue(item["page_url"].endswith("/"))


if __name__ == "__main__":
    unittest.main()
