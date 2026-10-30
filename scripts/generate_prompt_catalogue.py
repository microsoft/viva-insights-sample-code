#!/usr/bin/env python3
"""Generate the Prompt Library catalogue data from its canonical prompt cards."""

from __future__ import annotations

import argparse
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
    match = re.match(r"\A---[ \t]*\n(.*?)^---[ \t]*$", text, re.S | re.M)
    if not match:
        raise ValueError("Expected YAML front matter")
    return match.group(1)


def badge_scalar(raw: str) -> str:
    """Parse simple YAML string scalars; reject other forms explicitly."""
    raw = raw.strip()
    if raw.startswith('"'):
        try:
            value, end = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"Unsupported double-quoted badge value: {raw}") from error
        trailing = raw[end:]
        if not isinstance(value, str) or not re.fullmatch(r"[ \t]*(?:[ \t]+#.*)?", trailing):
            raise ValueError(f"Unsupported badge value: {raw}")
        return value
    if raw.startswith("'"):
        match = re.fullmatch(r"'((?:[^']|'')*)'(?:[ \t]+#.*)?", raw)
        if not match:
            raise ValueError(f"Unsupported single-quoted badge value: {raw}")
        return match.group(1).replace("''", "'")

    value = re.split(r"[ \t]+#", raw, maxsplit=1)[0].strip()
    non_string = re.fullmatch(
        r"(?i:true|false|yes|no|on|off|null|~|[-+]?(?:[0-9][0-9_.]*)(?:[eE][-+]?[0-9]+)?|[0-9]{4}-[0-9]{2}-[0-9]{2})",
        value,
    )
    if not value or value[0] in "[]{}>|&*!%@`'\"" or re.search(r":(?:[ \t]|$)", value) or non_string:
        raise ValueError(f"Unsupported plain badge value: {raw}")
    return value


def badge_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    in_badges = False
    for line in front_matter(text).splitlines():
        if re.match(r"^badges[ \t]*:", line):
            if in_badges:
                raise ValueError("Duplicate badges section")
            if not re.fullmatch(r"badges[ \t]*:[ \t]*(?:#.*)?", line):
                raise ValueError(f"Unsupported badges mapping: {line}")
            in_badges = True
            continue
        if in_badges and line and not line.startswith((" ", "#")):
            break
        if in_badges:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"  ([a-z_]+):[ \t]*(.*?)\s*", line)
            if not match or match.group(1) in result:
                raise ValueError(f"Unsupported or duplicate badge entry: {line}")
            result[match.group(1)] = badge_scalar(match.group(2))
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
    plain_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", first_paragraph)
    return " ".join(plain_text.splitlines())


def generated_catalogue() -> str:
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
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check generated data without modifying it")
    args = parser.parse_args()
    expected = generated_catalogue()
    if args.check:
        if not SITE_DATA.exists() or SITE_DATA.read_text(encoding="utf-8") != expected:
            print("Prompt catalogue data is stale. Run scripts/generate_prompt_catalogue.py.", file=sys.stderr)
            return 1
        print("Prompt catalogue data matches its sources.")
        return 0
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(expected, encoding="utf-8")
    print(f"Wrote prompt catalogue data to {SITE_DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
