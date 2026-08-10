"""
Example 1 - Import and validate a Viva Insights query.

Uses the built-in Person Query sample (load_pq_data) so no customer data is
needed. In real work, replace the sample with:
    pq = vi.import_query("your_person_query.csv")

Demonstrates: check_query, extract_hr, extract_date_range, and a privacy-
threshold audit of an HR grouping variable.
"""
import vivainsights as vi

# 1. Load data. Real workflow: pq = vi.import_query("person_query.csv")
pq = vi.load_pq_data()
print(f"Rows x cols: {pq.shape[0]} x {pq.shape[1]}")

# 2. Validate structure and variable types.
vi.check_query(pq)

# 3. List the HR / organisational attributes available for grouping (hrvar).
#    In the Python package extract_hr prints the attributes and returns None,
#    so call it directly rather than assigning its result.
vi.extract_hr(pq)

# 4. Report the period coverage of the query.
print("Date range:")
print(vi.extract_date_range(pq))

# 5. Privacy check: which Organization groups fall below the min group size?
#    Never report groups smaller than the agreed threshold (default 5).
counts = pq.groupby("Organization")["PersonId"].nunique().sort_values()
below = counts[counts < 5]
print(f"\nGroups below privacy threshold (5): {len(below)}")
if len(below):
    print(below.to_dict())
