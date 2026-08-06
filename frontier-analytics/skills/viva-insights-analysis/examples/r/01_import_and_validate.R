# Example 1 - Import and validate a Viva Insights query.
#
# Uses the built-in Person Query sample (pq_data) so no customer data is needed.
# In real work, replace the sample with:
#     pq <- import_query("your_person_query.csv")
#
# Demonstrates: check_query, extract_hr, extract_date_range, and a privacy-
# threshold audit with identify_privacythreshold.

suppressMessages(library(vivainsights))

# 1. Load data. Real workflow: pq <- import_query("person_query.csv")
pq <- pq_data
cat(sprintf("Rows x cols: %d x %d\n", nrow(pq), ncol(pq)))

# 2. Validate structure and variable types.
check_query(pq)

# 3. List the HR / organisational attributes available for grouping (hrvar).
hr_vars <- extract_hr(pq, return = "names")
cat("HR attributes:", paste(hr_vars, collapse = ", "), "\n")

# 4. Report the period coverage of the query.
print(extract_date_range(pq))

# 5. Privacy check: flag Organization groups below the min group size (5).
#    Never report groups smaller than the agreed threshold.
pt <- identify_privacythreshold(pq, hrvar = "Organization",
                                mingroup = 5, return = "table")
cat("\nPrivacy-threshold table (groups and their sizes):\n")
print(head(pt))
