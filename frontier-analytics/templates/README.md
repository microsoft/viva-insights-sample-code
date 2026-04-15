# Templates

This directory contains **contribution templates** for the Frontier Analytics section of the Viva Insights Sample Code Library. Use these templates to create consistent, high-quality additions to the prompt library, starter kits, and schema documentation.

## Why templates?

Frontier Analytics relies on a consistent structure so that analysts, consultants, and coding agents can quickly find what they need and trust that every prompt card, starter kit, and schema document follows the same pattern. Templates make it easy to contribute without guessing at the expected format.

## Available templates

| Template | Purpose | When to use |
|---|---|---|
| [Prompt Card Template](prompt-card-template.md) | Create a new prompt card for the prompt library | You have a repeatable analytics workflow that can be driven by a coding agent |
| [Starter Kit Template](starter-kit-template.md) | Create a new starter kit (a multi-prompt workflow) | You want to package several prompt cards into an end-to-end analysis workflow |
| [Schema Doc Template](schema-doc-template.md) | Document a new data schema or data source | You need to describe a Viva Insights export format, Purview log format, or other data source |

## How to use a template

1. **Copy the template.** Open the relevant template file and copy its contents into a new file in the appropriate directory:
   - Prompt cards → `frontier-analytics/prompts/<category>/`
   - Starter kits → `frontier-analytics/starter-kits/<kit-name>/`
   - Schema docs → `frontier-analytics/schemas/`

2. **Fill in every section.** Replace all placeholder text (marked with `[brackets]`) with your actual content. Remove any guidance comments once you have addressed them.

3. **Review against the checklist.** Each template includes a quality checklist at the end. Verify your content meets all criteria before submitting.

4. **Submit a pull request.** Open a PR against the `main` branch of [microsoft/viva-insights-sample-code](https://github.com/microsoft/viva-insights-sample-code). In your PR description, note which template you used and summarize what your contribution adds.

## Contribution guidelines

This repository follows the standard Microsoft open-source contribution process. Before submitting:

- Sign the [Microsoft CLA](https://cla.opensource.microsoft.com/) if you haven't already.
- Ensure your content does not include proprietary data, tenant-specific column names, or personally identifiable information.
- Review the [Code of Conduct](../../CODE_OF_CONDUCT.md) and [Support](../../SUPPORT.md) documents.

## Questions?

If you are unsure which template to use or how to structure your contribution, open an issue on the repository and tag it with `frontier-analytics`.
