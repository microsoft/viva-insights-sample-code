# ==================================================================================================
# Viva Insights - Copilot Analytics - Example Python Script
# This script provides a demo on how to generate example visuals using the 'vivainsights' Python library,
# using Copilot metrics from Viva Insights.
# ==================================================================================================

# Load libraries
import vivainsights as vi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
demo_pq = vi.import_query("data/pq_data.csv")

# In the code below, you can replace: 
# - the dataset name with your own dataset name
# - the metric names with your own metric names (specified in string)
# - the organizational attribute (`hrvar`) with your own organizational attribute

# Metrics: Copilot actions taken in
metrics_cop_actions_taken_in = [
    col for col in demo_pq.columns if "Copilot_actions_taken_in" in col
]

# Metrics: Copilot actions involving summarisation
metrics_summarise_cop = [
    col for col in demo_pq.columns if "Summarise" in col
]

# Key metrics scan: Copilot actions taken in by Organization -----------------------

vi.keymetrics_scan(
    data=demo_pq,
    hrvar="Organization",
    metrics=metrics_cop_actions_taken_in,
    return_type="plot"
)

# Run `plt.show()` to display the plot, or `vi.export()` to save the plot as an image file.

# Key metrics scan: High vs Medium vs Low Copilot users ------------------

demo_pq["Total_Copilot_actions"] = demo_pq[metrics_cop_actions_taken_in].sum(axis=1)

tb_cop_usage_segments = (
    demo_pq.groupby("PersonId")["Total_Copilot_actions"]
    .mean()
    .reset_index()
    .assign(CopilotUsageSegment=lambda df: pd.cut(
        df["Total_Copilot_actions"],
        bins=[-1, 0, 3, 9, float('inf')],
        labels=["Non-user", "Low\n(1-3 actions)", "Medium\n(4-9 actions)", "Heavy\n(10+ actions)"]
    ))
)

vi.keymetrics_scan(
    data=demo_pq.merge(tb_cop_usage_segments, on="PersonId"),
    hrvar="CopilotUsageSegment",
    return_type="plot"
)

# Boxplot - Copilot Assisted Hours ---------------------------------------

vi.create_boxplot(
    data=demo_pq,
    hrvar="Organization",
    metric="Copilot_assisted_hours",
    return_type="plot"
)

# Lorenz curve - Total Copilot actions --------------------------------

vi.create_lorenz(
    data=demo_pq,
    metric="Total_Copilot_actions",
    return_type="plot"
)

# Cumulative share table
lorenz_table = vi.create_lorenz(
    data=demo_pq,
    metric="Total_Copilot_actions",
    return_type="table"
)

# Ranked - Total Copilot Actions ----------------------------------------

ranked_data = vi.create_rank(
    data=demo_pq,
    metric="Total_Copilot_actions",
    hrvar=["Organization"],
    return_type="table"
)

# Top 10 - results copied to clipboard
ranked_data.head(10).to_csv("top_10_copilot_actions.csv", index=False)

# Bottom 10 - results copied to clipboard
ranked_data.tail(10).to_csv("bottom_10_copilot_actions.csv", index=False)

# Information value - Heavy Copilot Users ------------------------------
# Identify heavy Copilot users
tb_heavy_copilot_users = (
    demo_pq.groupby("PersonId")["Total_Copilot_actions"]
    .mean()
    .reset_index()
    .assign(HeavyCopilotUsers=lambda df: (df["Total_Copilot_actions"] >= 10).astype(int))
)

# Join user segments with Heavy Copilot users
vi.create_IV(
    data=demo_pq.merge(tb_heavy_copilot_users, on="PersonId"),
    predictors=["Collaboration_hours",
        "Internal_network_size",
        "Emails_sent",
        "Active_connected_hours"],
    outcome="HeavyCopilotUsers",
    exc_sig=False,
    return_type="plot"
    )