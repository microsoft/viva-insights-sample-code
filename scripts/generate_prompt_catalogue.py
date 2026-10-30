#!/usr/bin/env python3
"""Generate the Prompt Library catalogue data from its canonical prompt cards."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from check_prompt_drift import MAPPING, PAGES_DIR, PROMPTS_DIR, section_text

SITE_DATA = ROOT / "_data" / "prompts.json"
SOURCE_URL = "https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/prompts/"
CATEGORY_NAMES = {
    "copilot-adoption": "Copilot adoption",
    "copilot-dashboards": "Query dashboards",
    "purview-augmentation": "Purview augmentation",
}
CATEGORY_ORDER = list(CATEGORY_NAMES.values())


def front_matter(text: str) -> str:
    parts = text.split("---", 2)
    if len(parts) != 3:
        raise ValueError("Expected YAML front matter")
    return parts[1]


def badge_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    in_badges = False
    for line in front_matter(text).splitlines():
        if line == "badges:":
            in_badges = True
            continue
        if in_badges and line and not line.startswith(" "):
            break
        if in_badges:
            match = re.match(r'\s{2}([a-z_]+):\s*"([^"]*)"\s*$', line)
            if match:
                result[match.group(1)] = match.group(2)
    return result


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
    return first_paragraph


def main() -> None:
    groups = {name: {"title": name, "prompts": []} for name in CATEGORY_ORDER}

    for source_path, page_name in MAPPING.items():
        source = PROMPTS_DIR / source_path
        page = PAGES_DIR / page_name
        source_text = source.read_text(encoding="utf-8")
        page_text = page.read_text(encoding="utf-8")
        slug_match = re.search(
            r"^permalink:\s*/frontier-analytics-prompt-([^/]+)/\s*$",
            front_matter(page_text),
            re.M,
        )
        if not slug_match:
            raise ValueError(f"Missing prompt permalink in {page}")
        title_match = re.search(r"^#\s+(.+?)\s*$", source_text, re.M)
        if not title_match:
            raise ValueError(f"Missing prompt title in {source}")

        category_key = source_path.split("/", 1)[0]
        category = CATEGORY_NAMES[category_key]
        groups[category]["prompts"].append({
            "slug": slug_match.group(1),
            "title": title_match.group(1),
            "summary": prompt_summary(source_text),
            "page_url": f"/frontier-analytics-prompt-{slug_match.group(1)}/",
            "source_card": source_path,
            "source_url": SOURCE_URL + source_path,
            "badges": badge_values(page_text),
            "quick_prompt": prompt_text(source_text, "Quick prompt (short version)"),
        })

    result = [groups[name] for name in CATEGORY_ORDER if groups[name]["prompts"]]
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(group['prompts']) for group in result)} prompts to {SITE_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
