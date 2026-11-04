"""Negative fixtures for the post-Jekyll prompt/link regression gate."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_rendered_prompts as rendered


class RenderedPromptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.site = root / "site"
        prompts = root / "prompts"
        prompts.mkdir()
        (prompts / "example.md").write_text("# Example\n\n```\nliteral {{ code }}\n```\n", encoding="utf-8")
        route = self.site / "frontier-analytics-prompt-example"
        route.mkdir(parents=True)
        self.page = route / "index.html"
        ids = ["purpose", "audience", "when-to-use", "required-inputs", "assumptions",
               "quick-prompt-short-version", "common-failure-modes", "recommended-output", "prompt", "adaptation-notes"]
        self.html = "".join(f'<h2 id="{key}"></h2>' for key in ids)
        self.html += '<pre><code>literal {{ code }}\n</code></pre><a href="/prefix/target/#dest">target</a>'
        self.page.write_text(self.html, encoding="utf-8")
        target = self.site / "target"
        target.mkdir()
        (target / "index.html").write_text('<h2 id="dest">Target</h2>', encoding="utf-8")
        for patch in [
            mock.patch.object(rendered, "PAGES", [{"source": "example.md", "slug": "example"}]),
            mock.patch.object(rendered, "PROMPTS_DIR", prompts),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def test_valid_literal_prompt_and_non_root_route(self):
        rendered.check_site(self.site, "/prefix")

    def test_broken_anchor_is_rejected(self):
        self.page.write_text(self.html.replace("#dest", "#missing"), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "broken anchor"):
            rendered.check_site(self.site, "/prefix")

    def test_baseurl_escape_is_rejected(self):
        self.page.write_text(self.html.replace("/prefix/target/", "/target/"), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "escapes non-root"):
            rendered.check_site(self.site, "/prefix")

    def test_changed_rendered_prompt_is_rejected(self):
        self.page.write_text(self.html.replace("{{ code }}", "evaluated"), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "rendered code differs"):
            rendered.check_site(self.site, "/prefix")


if __name__ == "__main__":
    unittest.main()
