"""Shared prompt routing and Markdown helpers, independent of generated pages."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "frontier-analytics" / "prompts"
PAGES_DIR = ROOT / "_pages"
METADATA = json.loads((ROOT / "scripts" / "prompt_pages.json").read_text(encoding="utf-8"))
PAGES = METADATA["pages"]
MAPPING = {
    page["source"]: f'frontier-analytics-prompt-{page["slug"]}.md'
    for page in PAGES
}
KNOWN_SITE_LESS_CARDS = set(METADATA["repo_only"])
SOURCE_URL = "https://github.com/microsoft/viva-insights-sample-code/blob/main/frontier-analytics/prompts/"


def section_text(text: str, heading: str) -> str | None:
    match = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=\n## |\Z)", text, re.S | re.M)
    if not match:
        return None
    return re.sub(r"\n+---\s*\Z", "", match.group(1)).strip()


def fenced_parts(text: str):
    """Yield (is_fenced, text), retaining fence bytes and all line endings."""
    buffer = []
    fence = None
    for line in text.splitlines(keepends=True):
        opening = re.match(r"^(`{3,}|~{3,})", line)
        if fence is None and opening:
            if buffer:
                yield False, "".join(buffer)
            buffer = [line]
            fence = opening.group(1)
        elif fence is not None:
            buffer.append(line)
            if re.fullmatch(rf"{re.escape(fence[0])}{{{len(fence)},}}[ \t]*\r?\n?", line):
                yield True, "".join(buffer)
                buffer = []
                fence = None
        else:
            buffer.append(line)
    if fence is not None:
        raise ValueError("Unclosed Markdown fence")
    if buffer:
        yield False, "".join(buffer)


def full_prompt_blocks(text: str) -> list[str]:
    blocks = []
    in_prompt = False
    count = 0
    found = False
    for fenced, part in fenced_parts(text):
        if fenced and in_prompt:
            blocks.append("\n".join(part.splitlines()[1:-1]))
            count += 1
        elif not fenced:
            for heading in re.findall(r"^## (.+)$", part, re.M):
                if in_prompt and not count:
                    raise ValueError("A full Prompt section has no fenced prompt")
                in_prompt = bool(re.match(r"Prompt(?:\s|$)", heading))
                if in_prompt:
                    found = True
                    count = 0
    if in_prompt and not count:
        raise ValueError("A full Prompt section has no fenced prompt")
    if not found:
        raise ValueError("No full Prompt section found")
    return blocks
