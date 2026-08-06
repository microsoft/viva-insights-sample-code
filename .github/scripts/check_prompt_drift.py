#!/usr/bin/env python3
"""Check for content drift between Frontier Analytics prompt cards and their
site-facing mirror pages.

Background: frontier-analytics/ is excluded from the Jekyll build (see the
exclude list in _config.yml), so the pages under _pages/frontier-analytics-
prompt-*.md are separate, manually maintained copies of the prompt cards
under frontier-analytics/prompts/**/*.md rather than generated pages. Nothing
keeps the two in sync today, which has repeatedly caused real bugs:

  - A prompt card existed only on the website with no corresponding source
    file in the repo.
  - A "Quick prompt (short version)" section was added to a prompt card but
    never added to its site-page mirror.
  - A fix to the licensing classification rule was applied to a prompt card
    but not to its site-page mirror.

This script cannot make the two copies identical by construction (that is
a bigger structural fix, tracked separately). It exists to make future drift
loud instead of silent. It hard-fails the specific checks we know how to make
reliable, and warns (without failing) on everything else, since some cards
still have known, pre-existing wording differences between the two copies
that have not been fully reconciled.

Exit code 0: no hard-fail conditions found (warnings may still be printed).
Exit code 1: at least one hard-fail condition found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "frontier-analytics" / "prompts"
PAGES_DIR = REPO_ROOT / "_pages"

# Map each prompt card (relative to frontier-analytics/prompts/) to its
# site-page mirror (relative to _pages/). Add new cards here when they are
# added to both locations. If a new card is intentionally repo-only (no site
# page), add its path to KNOWN_SITE_LESS_CARDS instead.
MAPPING: dict[str, str] = {
    "copilot-adoption/dashboard-overview.md": "frontier-analytics-prompt-dashboard.md",
    "copilot-adoption/executive-summary.md": "frontier-analytics-prompt-executive-summary.md",
    "copilot-adoption/roi-estimation.md": "frontier-analytics-prompt-roi.md",
    "copilot-adoption/segmentation-and-churn.md": "frontier-analytics-prompt-segmentation.md",
    "copilot-adoption/executive-powerpoint-deck.md": "frontier-analytics-prompt-powerpoint.md",
    "copilot-adoption/copilot-causal-toolkit.md": "frontier-analytics-prompt-causal-toolkit.md",
    "purview-augmentation/audit-log-parsing.md": "frontier-analytics-prompt-audit-parsing.md",
    "purview-augmentation/agent-usage-analysis.md": "frontier-analytics-prompt-agent-usage.md",
}

# Prompt cards that are intentionally repo-only, with no site-page mirror.
KNOWN_SITE_LESS_CARDS: set[str] = set()

# Keywords that must appear in both copies of a pair, or neither, since they
# encode a specific classification rule that has drifted between copies
# before. Extend this list if another cross-cutting rule like this is found.
PARITY_KEYWORDS = [
    "Total_Copilot_enabled_days",
    "enabled-days column",
]

HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def section_text(text: str, heading: str) -> str | None:
    """Return the body of a level-2 markdown section, or None if absent."""
    pattern = rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)"
    m = re.search(pattern, text, re.S | re.M)
    if not m:
        return None
    body = m.group(1)
    # Strip a trailing standalone "---" divider and surrounding blank lines.
    # These are presentational only and are not always present in both copies.
    body = re.sub(r"\n+---\s*\Z", "", body)
    return body.strip()


def headings(text: str) -> list[str]:
    return HEADING_RE.findall(text)


def main() -> int:
    hard_failures: list[str] = []
    warnings: list[str] = []

    card_files = sorted(
        p.relative_to(PROMPTS_DIR).as_posix()
        for p in PROMPTS_DIR.rglob("*.md")
        if p.name != "README.md"
    )
    for card in card_files:
        if card not in MAPPING and card not in KNOWN_SITE_LESS_CARDS:
            hard_failures.append(
                f"'{card}' has no entry in MAPPING (in this script) and is not "
                "listed in KNOWN_SITE_LESS_CARDS. Add a site-page mapping, or "
                "add it to KNOWN_SITE_LESS_CARDS if it is intentionally not on "
                "the website."
            )

    known_site_pages = set(MAPPING.values())
    for page in PAGES_DIR.glob("frontier-analytics-prompt-*.md"):
        if page.name not in known_site_pages:
            hard_failures.append(
                f"'_pages/{page.name}' is not listed as a target in MAPPING. "
                "Either this page is orphaned (its source prompt card was "
                "deleted or renamed), or MAPPING needs a new entry."
            )

    for card_rel, page_name in MAPPING.items():
        card_path = PROMPTS_DIR / card_rel
        page_path = PAGES_DIR / page_name

        if not card_path.exists():
            hard_failures.append(f"Mapped prompt card is missing: frontier-analytics/prompts/{card_rel}")
            continue
        if not page_path.exists():
            hard_failures.append(f"Mapped site page is missing: _pages/{page_name}")
            continue

        card_text = card_path.read_text(encoding="utf-8")
        page_text = page_path.read_text(encoding="utf-8")

        pair_label = f"frontier-analytics/prompts/{card_rel} <-> _pages/{page_name}"

        quick_card = section_text(card_text, "Quick prompt (short version)")
        quick_page = section_text(page_text, "Quick prompt (short version)")
        if quick_card is None or quick_page is None:
            hard_failures.append(
                f"{pair_label}: 'Quick prompt (short version)' section is missing "
                f"from {'the prompt card' if quick_card is None else 'the site page'}."
            )
        elif quick_card != quick_page:
            hard_failures.append(
                f"{pair_label}: 'Quick prompt (short version)' content differs "
                "between the prompt card and the site page. They must match exactly."
            )

        for keyword in PARITY_KEYWORDS:
            in_card = keyword in card_text
            in_page = keyword in page_text
            if in_card != in_page:
                hard_failures.append(
                    f"{pair_label}: '{keyword}' appears in "
                    f"{'the prompt card' if in_card else 'the site page'} but not "
                    "the other. This keyword encodes a classification rule that "
                    "must be stated consistently in both copies."
                )

        card_headings = set(headings(card_text))
        page_headings = set(headings(page_text))
        only_in_card = card_headings - page_headings
        only_in_page = page_headings - card_headings
        if only_in_card:
            warnings.append(f"{pair_label}: section(s) only in the prompt card: {sorted(only_in_card)}")
        if only_in_page:
            warnings.append(f"{pair_label}: section(s) only in the site page: {sorted(only_in_page)}")

        for heading in sorted(card_headings & page_headings):
            body_card = section_text(card_text, heading)
            body_page = section_text(page_text, heading)
            if body_card is not None and body_page is not None and body_card != body_page:
                warnings.append(f"{pair_label}: section '{heading}' differs between the two copies.")

    if warnings:
        print("Warnings (not blocking, but worth a look):")
        for w in warnings:
            print(f"  - {w}")
        print()

    if hard_failures:
        print("Prompt drift check FAILED:")
        for f in hard_failures:
            print(f"  - {f}")
        print()
        print(
            "See frontier-analytics/prompts/**/*.md and the matching "
            "_pages/frontier-analytics-prompt-*.md file. Both copies must agree "
            "on the items above. If this check is becoming a recurring source of "
            "friction, see the tracking issue about generating these pages from "
            "a single source instead of maintaining two copies by hand."
        )
        return 1

    print("Prompt drift check passed (no hard-fail conditions found).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
