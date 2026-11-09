#!/usr/bin/env python3
"""Verify prompt routes, local links/anchors and literal code in a built Jekyll site."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from prompt_metadata import PAGES, PROMPTS_DIR, fenced_parts


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.blocks = []
        self.in_pre = False
        self.current = ""
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "pre":
            self.in_pre = True
            self.current = ""

    def handle_endtag(self, tag):
        if tag == "pre":
            self.blocks.append(self.current)
            self.in_pre = False

    def handle_data(self, data):
        if self.in_pre:
            self.current += data


def check_site(site, baseurl):
    cache = {}

    def load(path):
        if path not in cache:
            if not path.is_file():
                raise AssertionError(f"Missing built link target: {path}")
            cache[path] = Page(path)
        return cache[path]

    count = 0
    for metadata in PAGES:
        route = f'/frontier-analytics-prompt-{metadata["slug"]}/'
        page = load(site / route.strip("/") / "index.html")
        if len(page.ids) != len(set(page.ids)):
            raise AssertionError(f"{route}: duplicate HTML IDs")
        legacy = {"purpose", "audience", "when-to-use", "required-inputs", "assumptions",
                  "quick-prompt-short-version", "common-failure-modes"}
        if metadata["slug"] == "causal-toolkit":
            legacy.update(metadata["heading_aliases"].values())
        else:
            legacy.update({"recommended-output", "prompt", "adaptation-notes"})
        if metadata.get("title_anchor"):
            legacy.add(metadata["title_anchor"])
        if not legacy <= set(page.ids):
            raise AssertionError(f"{route}: missing legacy anchors: {sorted(legacy - set(page.ids))}")
        source = (PROMPTS_DIR / metadata["source"]).read_text(encoding="utf-8")
        expected = ["".join(part.splitlines(keepends=True)[1:-1])
                    for fenced, part in fenced_parts(source) if fenced]
        if page.blocks != expected:
            raise AssertionError(f"{route}: rendered code differs from source fences")
        for href in page.links:
            url = urlsplit(urljoin(baseurl + route, href))
            if url.scheme or url.netloc:
                continue
            if baseurl and not url.path.startswith(baseurl + "/"):
                raise AssertionError(f"{route}: link escapes non-root base URL: {href}")
            relative = unquote(url.path[len(baseurl):]).lstrip("/")
            target = site / relative
            if url.path.endswith("/"):
                target /= "index.html"
            if url.fragment:
                target_page = load(target)
                if unquote(url.fragment) not in target_page.ids:
                    raise AssertionError(f"{route}: broken anchor: {href}")
            elif not target.is_file():
                raise AssertionError(f"{route}: broken link: {href}")
            count += 1
    print(f"Rendered prompt checks passed: {len(PAGES)} pages, {count} local links, exact code blocks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("_site"))
    parser.add_argument("--baseurl", default="/viva-insights-sample-code")
    args = parser.parse_args()
    check_site(args.site, args.baseurl.rstrip("/"))
