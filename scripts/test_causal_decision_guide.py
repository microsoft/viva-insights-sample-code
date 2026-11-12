"""Structural checks for _data/causal_decision_guide.yml."""
import re
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "_data" / "causal_decision_guide.yml"
ARTICLE = ROOT / "_pages" / "causal-questions-ai-usage.md"
INCLUDE = ROOT / "_includes" / "causal-decision-guide.html"
PAGES = ROOT / "_pages"
ARTICLE_URL = "/articles/causal-questions-ai-usage/"
MODES = ("access", "intensity")

ID_ATTR = re.compile(r'\bid\s*=\s*"([^"]+)"')


def article_ids():
    """Ids written in the article source plus ids rendered by the include."""
    ids = set(ID_ATTR.findall(ARTICLE.read_text(encoding="utf-8")))
    ids |= set(ID_ATTR.findall(INCLUDE.read_text(encoding="utf-8")))
    return ids


def targets(answer):
    if "to" in answer:
        return [answer["to"]]
    return list(answer.get("next", {}).values())


def stateful_walk(data):
    """Walk (node, mode) states from start, mirroring the widget.

    `next` is resolved against the mode held before the answer; `set` updates
    the mode afterwards. Returns (visited ids, error messages).
    """
    nodes = {n["id"]: n for n in data["nodes"]}
    leaves = {l["id"] for l in data["leaves"]}
    seen, visited, errors = set(), set(), []
    stack = [(data["start"], None)]
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        node_id, mode = state
        visited.add(node_id)
        if node_id in leaves or node_id not in nodes:
            continue
        for answer in nodes[node_id]["answers"]:
            if "to" in answer:
                dest = answer["to"]
            else:
                if mode is None:
                    errors.append(f"{node_id}: next fork reachable with no mode")
                    continue
                dest = answer.get("next", {}).get(mode)
                if dest is None:
                    errors.append(f"{node_id}: no next entry for mode {mode}")
                    continue
            stack.append((dest, answer.get("set", mode)))
    return visited, errors


def parse_stub(path):
    text = path.read_text(encoding="utf-8")
    front = text.split("---")[1]
    return yaml.safe_load(front)


class DecisionGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
        cls.nodes = {n["id"]: n for n in cls.data["nodes"]}
        cls.leaves = {l["id"]: l for l in cls.data["leaves"]}
        cls.ids = article_ids()

    def edges(self, node_id):
        return [t for a in self.nodes[node_id]["answers"] for t in targets(a)]

    def test_ids_unique(self):
        ids = [n["id"] for n in self.data["nodes"]] + [l["id"] for l in self.data["leaves"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_start_exists(self):
        self.assertIn(self.data["start"], self.nodes)

    def test_answers_resolve(self):
        for node in self.data["nodes"]:
            for answer in node["answers"]:
                self.assertTrue(targets(answer), f"{node['id']}: answer without target")
                for t in targets(answer):
                    self.assertTrue(t in self.nodes or t in self.leaves, f"{node['id']} -> {t}")

    def test_conditional_routes_cover_both_modes(self):
        for node in self.data["nodes"]:
            for answer in node["answers"]:
                if "next" in answer:
                    self.assertEqual(set(answer["next"]), {"access", "intensity"})

    def test_anchors_exist_in_article(self):
        self.assertIn("decision-guide-widget", self.ids)
        for item in self.data["nodes"] + self.data["leaves"]:
            anchor = item.get("anchor")
            if item in self.data["leaves"]:
                self.assertTrue(anchor, f"{item['id']}: leaf without anchor")
            if anchor:
                self.assertTrue(anchor.startswith("#"), item["id"])
                self.assertIn(anchor[1:], self.ids, f"{item['id']}: {anchor} not in article")

    def test_redirect_stubs(self):
        stubs = sorted(PAGES.glob("causal-inference*.md"))
        self.assertEqual(len(stubs), 10, [s.name for s in stubs])
        for stub in stubs:
            meta = parse_stub(stub)
            self.assertEqual(meta.get("layout"), "redirect", stub.name)
            self.assertEqual(meta.get("permalink"), f"/{stub.stem}/", stub.name)
            target, _, fragment = meta["redirect_to"].partition("#")
            self.assertEqual(target, ARTICLE_URL, stub.name)
            if fragment:
                self.assertIn(fragment, self.ids, f"{stub.name}: #{fragment}")

    def test_stateful_walk(self):
        visited, errors = stateful_walk(self.data)
        self.assertEqual(errors, [])
        everything = set(self.nodes) | set(self.leaves)
        self.assertEqual(everything - visited, set(), "unreachable under stateful walk")

    def test_reachability(self):
        seen, stack = set(), [self.data["start"]]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.nodes:
                seen.add(current)
                continue
            seen.add(current)
            stack.extend(self.edges(current))
        self.assertEqual(set(self.nodes) - seen, set(), "unreachable nodes")
        self.assertEqual(set(self.leaves) - seen, set(), "unreachable leaves")

    def test_no_cycles(self):
        state = {}

        def visit(node_id):
            if node_id not in self.nodes:
                return
            self.assertNotEqual(state.get(node_id), "open", f"cycle at {node_id}")
            if state.get(node_id) == "done":
                return
            state[node_id] = "open"
            for t in self.edges(node_id):
                visit(t)
            state[node_id] = "done"

        visit(self.data["start"])


if __name__ == "__main__":
    unittest.main()
