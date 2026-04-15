# Contributing to Frontier Analytics

Thank you for your interest in improving Frontier Analytics. This guide covers how to add and update content in this section of the repository.

## Before you start

- Read the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
- All contributions require a [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com). A bot will prompt you when you open a pull request.
- Review the parent repository's [contributing guidelines](../README.md#contributing) for general repo policies.

## Adding a new prompt card

1. Copy the prompt card template from [templates/](templates/).
2. Create a new `.md` file in the appropriate subfolder under [prompts/](prompts/) (e.g., `prompts/copilot-adoption/my-new-card.md`).
3. Fill in all sections: Purpose, Audience, When to use, Required inputs, Assumptions, Recommended output, Prompt, Adaptation notes, and Common failure modes.
4. Test the prompt with at least one coding agent (GitHub Copilot, Claude Code, or similar) using realistic data before submitting.
5. Update the prompt library's [README](prompts/README.md) to include your new card.

## Adding a new starter kit

1. Copy the starter kit template from [templates/](templates/).
2. Create a new folder under [starter-kits/](starter-kits/) with a descriptive kebab-case name (e.g., `starter-kits/meeting-culture-report/`).
3. Include at minimum:
   - `README.md` — use case overview, audience, and output description
   - `quickstart.md` — step-by-step instructions
   - `required-inputs.md` — data files and parameters needed
4. Reference existing prompt cards from [prompts/](prompts/) or create new ones.
5. Update [STARTER_KITS.md](STARTER_KITS.md) to list the new kit.

## Improving schema documentation

1. Add or update files in [schemas/](schemas/).
2. Include: column definitions, data types, expected values, granularity, and example rows where helpful.
3. Document common pitfalls (e.g., missing values for unlicensed users, date format variations).
4. If adding a new schema, follow the schema template in [templates/](templates/).

## Style guidelines

- **Be concise.** Write for practitioners who need to get things done, not for marketing audiences.
- **Use practical examples.** Show real column names, realistic metric values, and concrete instructions.
- **No marketing language.** Avoid superlatives, vague claims, and promotional phrasing.
- **Use relative links.** Link to other files within this repository using relative paths (e.g., `prompts/copilot-adoption/dashboard-overview.md`), not absolute URLs.
- **Use Markdown.** All documentation should be in Markdown format.
- **Keep filenames kebab-case.** Use lowercase with hyphens (e.g., `roi-estimation.md`).

## Review criteria

Pull requests to Frontier Analytics are reviewed against the following:

1. **Grounded in real use cases.** Content must address actual Viva Insights analytics scenarios, not hypothetical ones.
2. **Prompts are tested.** Every prompt card must have been tested with at least one coding agent against realistic data.
3. **Complete documentation.** All required sections are filled in. No placeholder text.
4. **Consistent structure.** Follows the relevant template from [templates/](templates/).
5. **No sensitive data.** No real employee data, organization names, or identifiable information in examples.

## Submitting your contribution

1. Fork the repository or create a branch.
2. Make your changes following the guidelines above.
3. Open a pull request against `main`.
4. The CLA bot will check your agreement status.
5. A maintainer will review your contribution.

## Questions?

Open an issue on the [repository](https://github.com/microsoft/viva-insights-sample-code/issues) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).
