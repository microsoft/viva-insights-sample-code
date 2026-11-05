#!/usr/bin/env python3
"""Publish the core Skill's policy/support references without maintaining site copies."""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

from prompt_metadata import ROOT, PAGES_DIR, PROMPTS_DIR, MAPPING, KNOWN_SITE_LESS_CARDS, fenced_parts

REFERENCE = ROOT / "frontier-analytics" / "skills" / "viva-insights-analysis" / "reference"
ROUTES = {"analysis-policy.md": "policy", "workflow-support.md": "support"}
POLICY_START = "<!-- analysis-policy:start -->"
POLICY_END = "<!-- analysis-policy:end -->"


def policy_summary():
    text = (REFERENCE / "analysis-policy.md").read_text(encoding="utf-8")
    match = re.search(re.escape(POLICY_START) + r"\s*\n(.*?)\n" + re.escape(POLICY_END), text, re.S)
    if not match:
        raise ValueError("Missing canonical copyable analysis-policy block")
    return POLICY_START + "\n" + match.group(1).strip() + "\n" + POLICY_END


def bundle_policy(text, summary):
    """Keep full copyable prompts self-contained without hand-maintained policy copies."""
    parts = []
    in_prompt = False
    pattern = re.escape(POLICY_START) + r".*?" + re.escape(POLICY_END) + r"\n*"
    for fenced, part in fenced_parts(text):
        if not fenced:
            for heading in re.findall(r"^## (.+)$", part, re.M):
                in_prompt = bool(re.match(r"Prompt(?:\s|$)", heading))
        elif in_prompt:
            lines = re.sub(pattern, "", part, flags=re.S).splitlines(keepends=True)
            part = "".join(lines[:-1]).rstrip() + "\n\n" + summary + "\n" + lines[-1]
        parts.append(part)
    return "".join(parts)


def sync_policy_bundles(check):
    summary = policy_summary()
    current = True
    for name in sorted(set(MAPPING) | KNOWN_SITE_LESS_CARDS):
        if name.split("/")[0] not in {"copilot-adoption", "copilot-dashboards", "person-query"}:
            continue
        path = PROMPTS_DIR / name
        text = path.read_text(encoding="utf-8")
        expected = bundle_policy(text, summary)
        if check:
            if text != expected:
                print(f"Stale bundled policy: {name}", file=sys.stderr)
                current = False
        elif text != expected:
            path.write_text(expected, encoding="utf-8")
    return current


def site_links(text, source):
    def replace(match):
        label, target = match.groups()
        url = urlsplit(target)
        if url.scheme or url.netloc or target.startswith("#"):
            return match.group(0)
        path = source.parent.joinpath(*url.path.split("/")).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError(f"{source.name}: missing/out-of-repository reference {target}")
        if path.parent == REFERENCE and path.name in ROUTES:
            mapped = "{{ site.baseurl }}/frontier-analytics-" + ROUTES[path.name] + "/"
        else:
            mapped = "https://github.com/microsoft/viva-insights-sample-code/blob/main/" + path.relative_to(ROOT).as_posix()
        return f"[{label}]({mapped}{'#' + url.fragment if url.fragment else ''})"
    return re.sub(r"\[([^\]\n]+)\]\(([^)\n]+)\)", replace, text)


def generated_references():
    outputs = {}
    support = {}
    for filename, route in ROUTES.items():
        source = REFERENCE / filename
        text = source.read_text(encoding="utf-8")
        title = re.search(r"^# (.+)$", text, re.M).group(1)
        body = "".join(
            "{% raw %}\n" + part + "{% endraw %}\n" if fenced else site_links(part, source)
            for fenced, part in fenced_parts(text)
        )
        front = '---\nlayout: page\ntitle: ' + json.dumps(title) + f'\npermalink: /frontier-analytics-{route}/\n---\n'
        outputs[PAGES_DIR / f"frontier-analytics-{route}.md"] = (
            front + "<!-- Generated from the core Skill reference. Do not edit. -->\n\n" + body.rstrip()
            + "\n\n[Back to Skills]({{ site.baseurl }}/frontier-analytics-skills/)\n"
        )
        if filename == "workflow-support.md":
            for slug, block in re.findall(
                r"<!-- catalogue:([\w-]+):start -->\s*\n(.*?)<!-- catalogue:\1:end -->", text, re.S
            ):
                lines = [line[2:].strip() for line in block.splitlines() if line.strip()]
                if not lines or any(not line.startswith("- ") for line in block.splitlines() if line.strip()):
                    raise ValueError(f"{slug}: catalogue facts must be single-line bullets")
                if slug in support:
                    raise ValueError(f"Duplicate catalogue support facts: {slug}")
                support[slug] = lines
    expected = {"viva-insights-analysis", "viva-insights-copilot-dashboards", "viva-insights-causal-analysis"}
    if set(support) != expected:
        raise ValueError("Missing or unexpected canonical Skill catalogue support blocks")
    outputs[ROOT / "_data" / "workflow_support.json"] = json.dumps(support, indent=2) + "\n"
    return outputs


def sync_references(check):
    current = True
    for path, content in generated_references().items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                print(f"Stale generated reference: {path.name}", file=sys.stderr)
                current = False
        else:
            path.write_text(content, encoding="utf-8")
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    sys.exit(0 if sync_references(parser.parse_args().check) else 1)
