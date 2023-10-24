"""
Viva Insights - Python Utility Script
This script shows an example of how to perform pairwise chi-square tests for categorical variables in a dataset. 
One use case of this may be survey results appended to Viva Insights data, and for checking whether there is collinearity prior to using the data for modelling. 
"""

import pandas as pd
import numpy as np
import vivainsights as vi
from itertools import combinations
from scipy.stats import chi2_contingency

# Load sample query data
sample_data = vi.load_pq_data()

# Set random seed for reproducibility
np.random.seed(123)

# Number of unique PersonId in data
n_personid = sample_data['PersonId'].nunique()

# Create fake categorical variables for each PersonId
cat_data = pd.DataFrame({'PersonId': sample_data['PersonId'].unique(),
                         'cat_var1': np.random.choice(['A', 'B', 'C'], size=n_personid),
                         'cat_var2': np.random.choice(['A', 'B', 'C'], size=n_personid),
                         'cat_var3': np.random.choice(['A', 'B', 'C'], size=n_personid)})

sample_data_merged = pd.merge(sample_data, cat_data, on = 'PersonId')

# Separate categorical variables
cat_vars = ['cat_var1', 'cat_var2', 'cat_var3']

# Perform pairwise chi-square tests for all categorical variables
results = []
# Perform pairwise chi-square tests for all categorical variables
for var1, var2 in combinations(cat_vars, 2):
    contingency_table = pd.crosstab(sample_data_merged[var1], sample_data_merged[var2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results.append({'var1': var1, 'var2': var2, 'chi2': chi2, 'p': p})
   
# Create DataFrame of results
results_df = pd.DataFrame(results)

# Copy to clipboard to paste into Excel
vi.export(results_df)

