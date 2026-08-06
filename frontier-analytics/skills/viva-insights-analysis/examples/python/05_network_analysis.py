"""
Example 5 - Person-to-person network analysis.

A P2P export is an edge list: one row per collaboration tie between two people,
with tie-strength scores. network_p2p builds the graph and detects communities.
network_summary computes node centrality.

Data-shape note: a real P2P export carries HR attributes for each collaborator
(columns like PrimaryCollaborator_Organization). The built-in Python sample is a
bare edge list, so below we attach an ILLUSTRATIVE group derived deterministically
from the PersonId. This is only to make the example run. Replace it with the real
HR attributes from your query.
"""
import hashlib
import matplotlib
matplotlib.use("Agg")
import vivainsights as vi

p2p = vi.load_p2p_data().copy()
print("Edge-list columns:", list(p2p.columns))
print("Ties:", len(p2p))

# network_p2p needs a single-date snapshot.
one_date = sorted(p2p["MetricDate"].unique())[0]
p2p = p2p[p2p["MetricDate"] == one_date].copy()

# Illustrative HR group (deterministic, NOT a real attribute). In real data these
# columns come from the query export, so delete this block and use those instead.
def _demo_group(pid: str) -> str:
    h = int(hashlib.md5(str(pid).encode()).hexdigest(), 16)
    return f"Group_{h % 4}"

p2p["PrimaryCollaborator_Organization"] = p2p["PrimaryCollaborator_PersonId"].map(_demo_group)
p2p["SecondaryCollaborator_Organization"] = p2p["SecondaryCollaborator_PersonId"].map(_demo_group)

# Build the graph and summarise node centrality.
graph = vi.network_p2p(p2p, hrvar="Organization", return_type="network")
print("Graph object:", type(graph).__name__)

summary = vi.network_summary(graph, return_type="table")
print(f"\nNode centrality table: {summary.shape[0]} nodes x {summary.shape[1]} metrics")
print("Top 5 nodes by pagerank:")
print(summary.sort_values("pagerank", ascending=False).head().to_string(index=False))
