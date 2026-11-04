"""Check generated shared policy/support facts and catalogue usage."""

import unittest

import generate_analysis_references as references
from prompt_metadata import ROOT


class AnalysisReferencesTests(unittest.TestCase):
    def test_bundled_policy_is_exact_and_idempotent(self):
        summary = references.policy_summary()
        for path in references.PROMPTS_DIR.rglob("*.md"):
            if path.name == "README.md" or path.parent.name == "purview-augmentation":
                continue
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertEqual(text, references.bundle_policy(text, summary))
                self.assertEqual(text, references.bundle_policy(references.bundle_policy(text, summary), summary))

    def test_policy_update_replaces_only_full_prompt_bundle(self):
        original = "# Card\n\n## Quick prompt (short version)\n\n```\nquick\n```\n\n## Prompt\n\n```\nfull\n```\n"
        summary = references.POLICY_START + "\nrule\n" + references.POLICY_END
        result = references.bundle_policy(original, summary)
        self.assertIn("```\nquick\n```", result)
        self.assertEqual(result.count(summary), 1)
        changed = references.bundle_policy(result, summary.replace("rule", "updated"))
        self.assertNotIn("\nrule\n", changed)
        self.assertIn("\nupdated\n", changed)
        self.assertEqual(changed.count(references.POLICY_START), 1)

    def test_generated_references_and_support_facts_are_current(self):
        for path, content in references.generated_references().items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_catalogue_uses_generated_capability_limits(self):
        template = (ROOT / "_includes" / "skill-catalogue.html").read_text(encoding="utf-8")
        self.assertIn("site.data.workflow_support[skill.slug]", template)
        metadata = (ROOT / "_data" / "agent_skills.yml").read_text(encoding="utf-8")
        self.assertNotIn("  limitations:", metadata)

    def test_missing_reference_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "missing/out-of-repository"):
            references.site_links("[missing](missing.md)", references.REFERENCE / "analysis-policy.md")

    def test_shared_refs_use_site_routes_and_other_links_remain_explicit(self):
        source = references.REFERENCE / "analysis-policy.md"
        result = references.site_links("[support](workflow-support.md#capabilities)", source)
        self.assertIn("{{ site.baseurl }}/frontier-analytics-support/#capabilities", result)


if __name__ == "__main__":
    unittest.main()
