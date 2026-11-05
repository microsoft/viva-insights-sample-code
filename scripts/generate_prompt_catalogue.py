#!/usr/bin/env python3
"""Generate prompt site pages and catalogue from canonical prompt cards."""

from __future__ import annotations

import argparse
import json
import re
import sys

from prompt_metadata import ROOT, PAGES, PROMPTS_DIR, SOURCE_URL, section_text
from generate_prompt_pages import page_url, sync_pages
from generate_analysis_references import sync_policy_bundles, sync_references

SITE_DATA = ROOT / "_data" / "prompts.json"
CATEGORY_NAMES = {
    "copilot-adoption": "Copilot adoption",
    "copilot-dashboards": "Query dashboards",
    "purview-augmentation": "Purview augmentation",
}
CATEGORY_ORDER = list(CATEGORY_NAMES.values())


def prompt_text(text: str, label: str) -> str:
    section = section_text(text, label)
    if section is None:
        raise ValueError(f"Missing section: {label}")
    match = re.search(r"```[^\n]*\n(.*?)\n```", section, re.S)
    if not match:
        raise ValueError(f"Missing fenced prompt in section: {label}")
    return match.group(1)


def prompt_summary(text: str) -> str:
    section = section_text(text, "Purpose")
    if not section:
        raise ValueError("Missing Purpose section")
    first_paragraph = re.split(r"\n\s*\n", section, maxsplit=1)[0].strip()
    plain_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", first_paragraph)
    return " ".join(plain_text.splitlines())


def generated_catalogue() -> str:
    groups = {name: {"title": name, "prompts": []} for name in CATEGORY_ORDER}

    for metadata in PAGES:
        source_path = metadata["source"]
        source = PROMPTS_DIR / source_path
        source_text = source.read_text(encoding="utf-8")
        title_match = re.search(r"^#\s+(.+?)\s*$", source_text, re.M)
        if not title_match:
            raise ValueError(f"Missing prompt title in {source}")

        category_key = source_path.split("/", 1)[0]
        category = CATEGORY_NAMES[category_key]
        groups[category]["prompts"].append({
            "slug": metadata["slug"],
            "title": title_match.group(1),
            "summary": prompt_summary(source_text),
            "page_url": page_url(metadata),
            "source_card": source_path,
            "source_url": SOURCE_URL + source_path,
            "badges": metadata.get("badges", {}),
            "quick_prompt": prompt_text(source_text, "Quick prompt (short version)"),
        })

    result = [groups[name] for name in CATEGORY_ORDER if groups[name]["prompts"]]
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated data without modifying it")
    args = parser.parse_args()
    policy_current = sync_policy_bundles(args.check)
    pages_current = sync_pages(args.check)
    references_current = sync_references(args.check)
    expected = generated_catalogue()
    if args.check:
        if not SITE_DATA.exists() or SITE_DATA.read_text(encoding="utf-8") != expected:
            print("Prompt catalogue data is stale. Run scripts/generate_prompt_catalogue.py.", file=sys.stderr)
            return 1
        if policy_current and pages_current and references_current:
            print("Prompt pages, catalogue and policy/support references match their sources.")
        return 0 if policy_current and pages_current and references_current else 1
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(expected, encoding="utf-8")
    print(f"Wrote prompt catalogue data to {SITE_DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
