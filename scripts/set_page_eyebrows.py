"""Author per-page `eyebrow:` front-matter for inner pages.

Inserts an `eyebrow: "<LABEL>"` line into the YAML front-matter of each
mapped page, immediately after the `title:` line. Idempotent: skips files
that already have an `eyebrow:` key.

Run from repo root:  python scripts/set_page_eyebrows.py
"""
import os
import re
import sys

# section-label mapping. Filenames are relative to repo root.
EYEBROW_MAP = {
    # Essentials hub
    "_pages/essentials.md": "Essentials",
    "_pages/getting-started.md": "Essentials",
    "_pages/generate-custom-kpi.md": "Essentials",
    # Advanced
    "_pages/advanced.md": "Advanced analytics",
    "_pages/network.md": "Advanced analytics \u00b7 Network",
    # Copilot Analytics
    "_pages/copilot.md": "Copilot analytics",
    "_pages/copilot-usage-segments.md": "Copilot analytics",
    "_pages/dax-calculated-columns.md": "Copilot analytics \u00b7 Power BI",
    # Causal Inference (under Copilot in the nav, but its own section editorially)
    "_pages/causal-inference.md": "Causal inference",
    "_pages/causal-inference-data-prep.md": "Causal inference",
    "_pages/causal-inference-did.md": "Causal inference",
    "_pages/causal-inference-doubly-robust.md": "Causal inference",
    "_pages/causal-inference-experimentation-guide.md": "Causal inference",
    "_pages/causal-inference-iv.md": "Causal inference",
    "_pages/causal-inference-propensity.md": "Causal inference",
    "_pages/causal-inference-regression.md": "Causal inference",
    "_pages/causal-inference-technical.md": "Causal inference",
    "_pages/causal-inference-validation.md": "Causal inference",
    "_pages/copilot-causal-toolkit.md": "Causal inference \u00b7 Toolkit",
    "_pages/copilot-causal-toolkit-interpretation-guide.md": "Causal inference \u00b7 Toolkit",
    # Frontier
    "_pages/frontier-analytics.md": "Frontier",
    "_pages/frontier-analytics-prompts.md": "Frontier",
    "_pages/frontier-analytics-schemas.md": "Frontier",
    "_pages/frontier-analytics-prompt-agent-usage.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-audit-parsing.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-causal-toolkit.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-dashboard.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-executive-summary.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-powerpoint.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-roi.md": "Frontier \u00b7 Prompt library",
    "_pages/frontier-analytics-prompt-segmentation.md": "Frontier \u00b7 Prompt library",
    # People Skills
    "_pages/skills-example-join.md": "People Skills",
    "_pages/skills-example-join-requirements.md": "People Skills",
    # Articles
    "_pages/articles.md": "Articles",
    "_pages/when-ai-met-the-meeting.md": "Articles",
    "_pages/meeting-effectiveness-playbook.md": "Articles",
    # Search
    "_pages/search.md": "Search",
}


def insert_eyebrow(path: str, eyebrow: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---"):
        return "no front-matter"
    # Find the closing --- of the front matter
    m = re.match(r"^---\r?\n(.*?\r?\n)---\r?\n", text, flags=re.DOTALL)
    if not m:
        return "front-matter not closed"
    fm = m.group(1)
    if re.search(r"(?m)^eyebrow:\s*", fm):
        return "already has eyebrow"
    # Insert after the `title:` line; fall back to top of front-matter
    title_match = re.search(r"(?m)^title:.*\r?\n", fm)
    insert_line = f'eyebrow: "{eyebrow}"\n'
    if title_match:
        idx = title_match.end()
        new_fm = fm[:idx] + insert_line + fm[idx:]
    else:
        new_fm = insert_line + fm
    new_text = "---\n" + new_fm + "---\n" + text[m.end():]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_text)
    return "updated"


def main() -> int:
    changed = 0
    skipped = 0
    missing = 0
    for relpath, eyebrow in EYEBROW_MAP.items():
        path = relpath.replace("/", os.sep)
        if not os.path.exists(path):
            print(f"  MISSING {relpath}")
            missing += 1
            continue
        status = insert_eyebrow(path, eyebrow)
        if status == "updated":
            print(f"  + {relpath:60s} eyebrow=\"{eyebrow}\"")
            changed += 1
        else:
            print(f"  - {relpath:60s} ({status})")
            skipped += 1
    print(f"\nUpdated: {changed}  Skipped: {skipped}  Missing: {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
