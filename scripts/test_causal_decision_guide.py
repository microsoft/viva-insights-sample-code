"""Structural checks for _data/causal_decision_guide.yml."""
from pathlib import Path
import unittest

import yaml

DATA = Path(__file__).resolve().parent.parent / "_data" / "causal_decision_guide.yml"

ALLOWED_ANCHORS = {
    "#route-descriptive", "#route-randomised", "#route-rdd",
    "#route-staggered-did", "#route-between-function", "#route-synthetic",
    "#route-itsa", "#route-dose", "#route-selection", "#route-no-estimate",
    "#joined-panel",
}


def targets(answer):
    if "to" in answer:
        return [answer["to"]]
    return list(answer.get("next", {}).values())


class DecisionGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
        cls.nodes = {n["id"]: n for n in cls.data["nodes"]}
        cls.leaves = {l["id"]: l for l in cls.data["leaves"]}

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

    def test_anchors_allowed(self):
        for leaf in self.data["leaves"]:
            self.assertIn(leaf["anchor"], ALLOWED_ANCHORS, leaf["id"])

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
