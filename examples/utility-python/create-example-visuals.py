"""
Viva Insights - Python Utility Script
================================
This script provides a demo on how to generate example visuals using the 'vivainsights' Python library
All outputs are saved as SVG in a local output folder that you can customize the path to. 
"""

# Setup ---------------------------------------------------------------------------------------------
# load packages
import vivainsights as vi
import os
import matplotlib.pyplot as plt

# load in-built datasets
pq_data = vi.load_pq_data() # load and assign in-built person query
g2g_data = vi.load_g2g_data() # load and assign in-built group-to-group query
p2p_data = vi.p2p_data_sim(size = 400) # Simulate a person-to-person query, using the `size` parameter to control for size

# set output path
out_path = os.getcwd() + '/examples/utility-python/example-visuals/'

# SVG output: `create_bar()` ------------------------------------------------------------------------

plot_bar = vi.create_bar(data=pq_data, metric='Emails_sent', hrvar='Organization', mingroup=5)

vi.export(
    plot_bar,
    file_format = 'svg',
    path = out_path + 'create_bar',
    timestamp = False
)

# SVG output: `create_line()` -----------------------------------------------------------------------

plot_line = vi.create_line(data=pq_data, metric='Emails_sent', hrvar='Organization', mingroup=5, return_type='plot')

vi.export(
    plot_line,
    file_format = 'svg',
    path = out_path + 'create_line',
    timestamp = False
)

# SVG output: `create_boxplot()` --------------------------------------------------------------------

plot_box = vi.create_boxplot(data=pq_data, metric='Emails_sent', hrvar='Organization', mingroup=5, return_type='plot')

vi.export(
    plot_box,
    file_format = 'svg',
    path = out_path + 'create_boxplot',
    timestamp = False
)

# SVG output: `create_rank()` -----------------------------------------------------------------------

plot_rank = vi.create_rank(
    data=pq_data,
    metric='Collaboration_hours',
    hrvar = ['Organization', 'FunctionType', 'LevelDesignation', 'SupervisorIndicator'],
    mingroup=5,
    return_type = 'plot'
)

vi.export(
    plot_rank,
    file_format = 'svg',
    path = out_path + 'create_rank',
    timestamp = False
)

# SVG output: `network_g2g()` -----------------------------------------------------------------------

plot_g2g = vi.network_g2g(
    data = g2g_data,
    primary = "PrimaryCollaborator_Organization",
    secondary = "SecondaryCollaborator_Organization",
    return_type = "plot"
    )

# Workaround whilst # 15 is being fixed
# vi.export(
#     plot_g2g,
#     file_format = 'svg',
#     path = out_path + 'network_g2g',
#     timestamp = False
# )

# SVG output: `network_p2p()` -----------------------------------------------------------------------

plot_p2p = vi.network_p2p(data = p2p_data, return_type = "plot") 

# Workaround whilst # 15 is being fixed
# vi.export(
#     plot_p2p,
#     file_format = 'svg',
#     path = out_path + 'network_p2p',
#     timestamp = False
# )