<!-- ==========================================================================
     STARTER KIT TEMPLATE — How to fill this in
     ==========================================================================

     A starter kit packages multiple prompt cards into an end-to-end analysis
     workflow. Think of it as a guided playbook: an analyst follows the kit
     from data preparation through final deliverable.

     Good starter kits:
     1. Tell the analyst exactly what to prepare before starting
     2. Walk through prompt cards in a logical sequence
     3. Explain what to check between steps
     4. Describe the final deliverable so the analyst knows when they are done

     A starter kit lives in its own directory under frontier-analytics/starter-kits/.
     The directory should contain the files listed below. You can reference
     existing prompt cards from the library instead of duplicating them.

     Replace every [placeholder] below with your content, then delete this
     comment block before submitting your PR.
     ========================================================================== -->

# [Starter Kit Name]

## Recommended file structure

Create a directory under `frontier-analytics/starter-kits/` with the following files:

```
frontier-analytics/starter-kits/[kit-name]/
├── README.md               # Primary overview and workflow guide
├── quickstart.md            # One-page getting-started summary
├── required-inputs.md       # Detailed data requirements
├── recommended-files.md     # Supporting resources and references
└── expected-output.md       # Description of the final deliverable
```

Each file is described below with the sections it should contain.

---

## 1. README.md

The README is the primary document. It should be comprehensive enough that an analyst can understand the full workflow without reading every other file.

### Sections to include

```markdown
# [Kit Name]

## Overview
[2-3 sentences explaining what this kit produces and why it matters.
Example: "This starter kit walks you through a complete Copilot adoption
analysis — from raw data export to executive-ready deliverables. By the end,
you will have a dashboard, an executive summary, and a segmentation report."]

## Use case
[Describe the business scenario this kit addresses. Who asked for this
analysis? What decision does it support? What questions does it answer?]

## Prerequisites
- [Software requirements, e.g., "Python 3.9+ or R 4.x"]
- [Package requirements, e.g., "pandas, matplotlib, or vivainsights"]
- [Data access, e.g., "Analyst role in the Viva Insights portal"]
- [Estimated time to complete]

## Workflow

This kit follows [N] steps:

### Step 1: [Prepare your data]
[Brief description of data export and preparation.
Link to required-inputs.md for details.]

### Step 2: [Run prompt card A]
[Explain what this step produces. Link to the prompt card:
`../../prompts/<category>/<card>.md`
Note any inputs from previous steps.]

### Step 3: [Run prompt card B]
[Same structure. Explain how this step builds on Step 2.]

### Step [N]: [Assemble the final deliverable]
[Describe how to combine outputs into the final package.]

## Included prompt cards

| Step | Prompt Card | What it produces |
|------|-------------|------------------|
| 1 | [Card name](../../prompts/<category>/<card>.md) | [Brief output description] |
| 2 | [Card name](../../prompts/<category>/<card>.md) | [Brief output description] |

## Expected output
[One paragraph summarizing the final deliverable. Link to expected-output.md
for detailed description.]

## Troubleshooting
- [Common issue 1 and resolution]
- [Common issue 2 and resolution]

## Related resources
- [Links to relevant documentation, packages, or external tools]
```

### Guidance

- **Reference existing prompt cards whenever possible.** Do not duplicate prompt text into the README. Link to cards in the `prompts/` directory.
- **If a prompt card does not yet exist for a step**, create one using the [prompt card template](prompt-card-template.md) and submit it alongside the starter kit.
- **Keep the README scannable.** Use tables, numbered steps, and short paragraphs. An analyst should be able to skim the README and understand the full workflow in under 2 minutes.

---

## 2. quickstart.md

A condensed one-page guide for analysts who are already familiar with the tools and data. It should be usable without reading the full README.

### Sections to include

```markdown
# [Kit Name] — Quick Start

## What you need
- [Data file(s) and format]
- [Software and packages]

## Steps

1. Export [data source] from [location].
2. Open your coding agent and paste the prompt from [Card A](../../prompts/<category>/<card>.md). Point it at your data file.
3. Review the output: [what to check].
4. Paste the prompt from [Card B](../../prompts/<category>/<card>.md). Use the output from Step 2 as input.
5. [Continue for each step.]
6. Combine outputs into [final deliverable format].

## Tips
- [One or two key tips, e.g., "Verify column names before pasting the first prompt."]
```

### Guidance

- **Maximum one page when rendered.** If it is longer, you are including too much detail — move it to the README.
- **Every step should be a single sentence with a link** to the relevant prompt card.

---

## 3. required-inputs.md

Detailed documentation of every data file and column the kit depends on.

### Sections to include

```markdown
# [Kit Name] — Required Inputs

## Data sources

### [Data source 1, e.g., "Person Query Export"]
- **Where to export**: [e.g., "Viva Insights Analyst portal → Person queries"]
- **Format**: [e.g., "CSV, UTF-8 encoded"]
- **Granularity**: [e.g., "One row per person per week"]
- **Minimum history**: [e.g., "8 weeks recommended, 4 weeks minimum"]
- **Minimum population**: [e.g., "50 unique users for meaningful segmentation"]

#### Required columns

| Column | Type | Description |
|--------|------|-------------|
| `PersonId` | String | Anonymized unique identifier for each person |
| `MetricDate` | Date | Start date of the measurement week |
| [additional columns] | | |

#### Optional columns

| Column | Type | Description | Used in |
|--------|------|-------------|---------|
| `Organization` | String | Organizational unit | Segmentation charts |
| [additional columns] | | | |

### [Data source 2] *(if applicable)*
[Same structure as above.]

## Data preparation notes
- [Any cleaning or transformation steps required before running the kit]
- [Known data quirks to watch for]
```

### Guidance

- **List every column the prompts reference**, even if the prompt auto-detects them. Analysts need to verify their export contains the expected columns.
- **Separate required columns from optional columns.** The kit should work without optional columns (with reduced output).

---

## 4. recommended-files.md

Supporting resources that enhance the kit but are not strictly required.

### Sections to include

```markdown
# [Kit Name] — Recommended Files and Resources

## Supporting data files
- [e.g., "Organizational hierarchy CSV for enhanced segmentation"]
- [e.g., "Previous period export for trend comparison"]

## Reference materials
- [Link to schema documentation in frontier-analytics/schemas/]
- [Link to relevant Viva Insights documentation]
- [Link to package documentation]

## Sample data
- [If you provide synthetic sample data for testing, describe it here]
- [Note: Do not include real tenant data. Use synthetic or anonymized data only.]

## Related prompt cards
- [Links to prompt cards that complement this kit but are not included in the workflow]
```

### Guidance

- **Everything here is optional.** If the kit has no recommended extras, you can omit this file.
- **Never include real data.** If you provide sample data, it must be synthetic.

---

## 5. expected-output.md

A detailed description of what the analyst should have when the kit is complete.

### Sections to include

```markdown
# [Kit Name] — Expected Output

## Final deliverable

[Describe the complete output package. Example: "A folder containing three
files: a static HTML dashboard, a Markdown executive summary, and a CSV
segmentation report."]

## Output files

### [Output file 1, e.g., "copilot_adoption_dashboard.html"]
- **Format**: [e.g., "Self-contained static HTML"]
- **Contents**: [e.g., "Trend charts, segmentation breakdowns, top users table"]
- **How to share**: [e.g., "Email as attachment or upload to SharePoint"]

### [Output file 2]
[Same structure.]

## How to verify

- [Verification step 1, e.g., "Open the HTML file in a browser and confirm all charts render"]
- [Verification step 2, e.g., "Check that the adoption rate in the summary matches your manual calculation"]
- [Verification step 3, e.g., "Verify no group with fewer than 5 users appears in segmentation charts"]

## Sample screenshots or descriptions
[If possible, describe what each output looks like. You may include placeholder
descriptions like "The dashboard contains 4 sections: summary cards at the top,
trend charts in the middle, segmentation breakdowns below, and a methodology
note at the bottom."]
```

### Guidance

- **Be specific about verification.** Analysts need to know whether the output is correct, not just complete.
- **Describe the format precisely.** "An HTML file" is too vague. "A self-contained static HTML file with base64-encoded chart images, no external dependencies, approximately 2-5 MB" is better.

---

## Quality checklist

Before submitting your starter kit PR, verify:

- [ ] All five files are present in the kit directory (README.md, quickstart.md, required-inputs.md, recommended-files.md, expected-output.md)
- [ ] The README workflow references prompt cards by relative link, not by duplicating prompt text
- [ ] Every prompt card referenced in the kit exists in the `prompts/` directory (or is included in the same PR)
- [ ] Required columns in required-inputs.md match the columns referenced in the prompt cards
- [ ] The quickstart.md fits on one page when rendered
- [ ] expected-output.md includes verification steps
- [ ] No proprietary data, tenant-specific names, or PII anywhere in the kit
- [ ] The full workflow has been tested end-to-end with at least one coding agent
- [ ] The kit directory name uses lowercase kebab-case (e.g., `copilot-adoption-analysis`)
