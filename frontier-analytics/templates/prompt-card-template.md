<!-- ==========================================================================
     PROMPT CARD TEMPLATE — How to fill this in
     ==========================================================================

     This template follows the standard prompt card structure used across the
     Frontier Analytics prompt library. Every section is required unless marked
     optional. Good prompt cards share these qualities:

     1. SELF-CONTAINED — A coding agent should be able to execute the prompt
        with no context other than the card itself and a data file.

     2. SPECIFIC — Name exact columns, packages, output formats, and
        thresholds. Vague instructions lead to inconsistent agent behavior.

     3. DEFENSIVE — Anticipate data quirks (missing values, unexpected
        column names, duplicate rows) and tell the agent how to handle them.

     4. TESTED — Before submitting, run the prompt with at least one coding
        agent (GitHub Copilot, Claude Code, etc.) against sample data and
        verify the output matches your expectations.

     Replace every [placeholder] below with your content, then delete this
     comment block before submitting your PR.
     ========================================================================== -->

# [Prompt Title] — [Category Name]

## Purpose

[One sentence describing what this prompt produces. Be specific about the deliverable, e.g., "Generate a self-contained static HTML dashboard showing..." or "Produce a CSV file containing..."]

## Audience

[Who will use this prompt and consume the output, e.g., "People analytics leads, HR business partners" or "IT deployment managers, security teams"]

## When to use

[Specific conditions or scenarios when this prompt is appropriate. Include data prerequisites, e.g., "After exporting a person query with Copilot activity metrics spanning at least 8 weeks of data."]

## Required inputs

- **[Data file 1]**: [Description of the data source, file format (CSV, JSON, etc.), and key columns the prompt depends on. List the most important columns by name.]
- **[Data file 2]** *(if applicable)*: [Description and key columns]
- **Minimum data requirements**: [e.g., "At least 8 weeks of person-week data", "Minimum 50 unique users for meaningful segmentation"]

## Assumptions

- [Data granularity assumption, e.g., "Data is at person-week granularity"]
- [Key column assumptions, e.g., "`PersonId` is a consistent anonymized identifier"]
- [Date handling, e.g., "`MetricDate` is a date field representing the start of each week"]
- [Missing data assumptions, e.g., "Rows with missing Copilot metric values represent unlicensed users"]
- [Package/tool availability, e.g., "The `vivainsights` R or Python package is available in the environment"]

## Recommended output

[Format and description of the expected output. Be specific: "A self-contained static HTML file with embedded charts, suitable for sharing via email or SharePoint" or "A CSV file with one row per user per week, containing columns X, Y, Z."]

## Prompt

```
[Write your full prompt text here. Structure it with clear numbered steps and
section headers (e.g., DATA LOADING, METRIC CALCULATIONS, OUTPUT GENERATION).

The prompt should be detailed enough that a coding agent can execute it without
additional context. Include specific instructions about:

- Data loading and validation (file format, column type parsing, deduplication)
- Key calculations or transformations (formulas, aggregation logic, filters)
- Output format and structure (file type, sections, charts, tables)
- Error handling for common data issues (missing values, unexpected types, duplicates)
- Which packages or libraries to use (and fallbacks if unavailable)
- Privacy thresholds (e.g., suppress groups with fewer than 5 users)

Model your prompt after the existing cards in the library. See
frontier-analytics/prompts/copilot-adoption/dashboard-overview.md for a
well-structured example.

IMPORTANT NOTES
- Remind the agent about key constraints (static output, no server dependencies, etc.)
- Handle missing values explicitly rather than hoping the agent will guess correctly
- Include a step to print or log intermediate results so the analyst can verify correctness]
```

## Adaptation notes

- [How to modify for different data granularity, e.g., "If your data is person-day, add an instruction to aggregate to person-week first"]
- [How to adjust for different HR attributes, e.g., "Replace `Organization` with your actual column name"]
- [How to customize output format, e.g., "For R users, add 'Use R with ggplot2' at the start of the prompt"]
- [Other common modifications, e.g., "For smaller organizations, increase the privacy threshold from 5 to 10"]

## Common failure modes

- **[Failure mode 1]**: [Why it happens and how to prevent it. Example: "Agent assumes all rows have Copilot data — verify that the is_licensed flag is computed before any metric calculations."]
- **[Failure mode 2]**: [Why it happens and how to prevent it.]
- **[Failure mode 3]**: [Why it happens and how to prevent it.]

---

<!-- ==========================================================================
     QUALITY CHECKLIST — Verify before submitting your PR
     ==========================================================================
     [ ] Every section above is filled in (no remaining [placeholders])
     [ ] The prompt is self-contained: a coding agent can run it with only
         the card and a data file
     [ ] Column names referenced in the prompt match the Required Inputs section
     [ ] At least 3 common failure modes are documented
     [ ] The prompt has been tested with at least one coding agent
     [ ] No proprietary data, tenant-specific names, or PII in the card
     [ ] The file is saved in the correct directory under frontier-analytics/prompts/
     ========================================================================== -->
