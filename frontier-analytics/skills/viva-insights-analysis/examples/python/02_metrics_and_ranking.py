"""
Example 2 - Compare a metric across groups and scan several metrics at once.

Demonstrates: create_rank (ranked group table + plot) and keymetrics_scan
(multi-metric heatmap across groups). Plots are written headlessly to the
system temp directory so nothing is left in the skill folder.
"""
import os, tempfile
import matplotlib
matplotlib.use("Agg")  # headless backend, never blocks
import vivainsights as vi

pq = vi.load_pq_data()
outdir = tempfile.mkdtemp(prefix="vi_ex2_")

# 1. Rank Organizations by mean weekly collaboration hours (table).
rank_tbl = vi.create_rank(pq, metric="Collaboration_hours", hrvar="Organization",
                          mingroup=5, return_type="table")
print("Ranked groups by Collaboration_hours:")
print(rank_tbl.head(10).to_string(index=False))

# 2. Same, as a plot saved to disk.
fig = vi.create_rank(pq, metric="Collaboration_hours", hrvar="Organization",
                     mingroup=5, return_type="plot")
rank_png = os.path.join(outdir, "rank_collaboration.png")
fig.savefig(rank_png, dpi=150, bbox_inches="tight")
print(f"Saved rank plot: {rank_png}")

# 3. Scan several metrics across groups at once (explicit metric list for
#    reproducibility, since defaults differ between R and Python).
scan = vi.keymetrics_scan(
    pq, hrvar="Organization", mingroup=5,
    metrics=["Collaboration_hours", "Email_hours", "Meetings",
             "After_hours_collaboration_hours", "Internal_network_size"],
    return_type="table",
)
print("\nKey-metrics scan (head):")
print(scan.head().to_string(index=False))
