"""
This code provides an example on how to plot a network graph with the igraph object generated by the `network_g2g` function.
Use `return_type = 'network'` to generate the igraph object, and use functions from matplotlib and igraph to generate a network visualization.
"""

import vivainsights as vi
import igraph as ig
import matplotlib.pyplot as plt

g = vi.network_g2g(
    data = vi.load_g2g_data(), # change dataset here
    primary = "PrimaryCollaborator_Organization",
    secondary = "SecondaryCollaborator_Organization",
    metric = "Meeting_Count", # update metric
    return_type = "network"
    )

g = g.simplify()
fig, ax = plt.subplots(figsize=(8, 8))

g.vs["org_size"] = [x*50 for x in g.vs["org_size"]] # scale the size of the nodes

ig.plot(
    g,
    layout=g.layout("mds"),
    target=ax,
    vertex_color="blue", # set vertex color here
    vertex_label=g.vs["name"],
    vertex_frame_width=1,
    vertex_size=g.vs["org_size"],
    edge_alpha=0.5,
    edge_color="grey"
)

plt.show()