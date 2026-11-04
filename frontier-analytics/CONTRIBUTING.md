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

### Generated website content

Prompt cards in `prompts/` are the authored source. Do not edit their generated
`_pages/frontier-analytics-prompt-*.md` mirrors or `_data/prompts.json`.
Site-only titles, badges, preserved routes, navigation, heading aliases and
explicit link mappings live in `scripts/prompt_pages.json`. Register a new card
there, or list it under `repo_only` when it intentionally has no site page.

From the repository root, run:

```shell
python -m pip install -r scripts/requirements-doc-tests.txt -r scripts/requirements-recipes.txt
python scripts/generate_prompt_catalogue.py
python scripts/generate_prompt_catalogue.py --check
python -B -m unittest discover -s scripts -p 'test_*.py'
bundle exec jekyll build --baseurl /viva-insights-sample-code
python scripts/check_rendered_prompts.py --baseurl /viva-insights-sample-code
```

The generation command also publishes the core Skill's
`reference/analysis-policy.md` and `reference/workflow-support.md` as site pages.
Catalogue capability limits come from marked bullet blocks in the support
reference, not the presentation metadata in `_data/agent_skills.yml`.
Keep genuinely workflow-specific instructions with their owner. Prompts must
remain copyable offline, and installed Skills must retain their required
references without a source checkout. The optional dashboard workflow still
requires the explicitly documented runtime checkout.

The marked summary in `reference/analysis-policy.md` is also bundled into full
copyable Person Query, dashboard and adoption prompts by the same command.
Edit the canonical policy block, not generated blocks inside prompt fences.
Quick prompts keep their task-specific short-form instructions.

Check mode never writes files. Regeneration must be deterministic and clean on
a second run. Fenced prompts/code are retained literally, including multiple
prompt blocks. Relative links outside fences require explicit site mappings.
Historical `last_validated` metadata is not evidence that a changed prompt has
been rerun. Record actual commands and observed results separately, distinguishing
synthetic runs, explicit-read agent evaluations, automatic discovery and real-data trials.

The pinned recipe environment uses Python 3.12. Documentation-only environments
may skip optional package smoke tests, but both CI workflows install the recipe
requirements and verify the installed version so those checks execute.
The documentation-contract workflow also runs the matched R segmentation
fixture on pull requests with R 4.6.2 and `vivainsights` 0.7.2.

## Adding a new starter kit

1. Copy the starter kit template from [templates/](templates/).
2. Create a new folder under [starter-kits/](starter-kits/) with a descriptive kebab-case name (e.g., `starter-kits/meeting-culture-report/`).
3. Include at minimum:
   - `README.md`: use case overview, audience, and output description
   - `quickstart.md`: step-by-step instructions
   - `required-inputs.md`: data files and parameters needed
4. Reference existing prompt cards from [prompts/](prompts/) or create new ones.
5. Update [starter-kits/README.md](starter-kits/README.md) to list the new kit.

## Adding a new skill

1. Copy the skill template from [templates/](templates/).
2. Create a new folder under [skills/](skills/) with a descriptive kebab-case name matching the skill's `name` field (e.g., `skills/my-new-skill/`).
3. Fill in `SKILL.md` following the template. Add `reference/` and `examples/` subfolders if the content is long enough to benefit from being split, following the pattern in `skills/viva-insights-analysis/`.
4. Confirm the skill is customer-agnostic. It must not reference a specific organization, engagement, or private data source.
5. Test the skill with at least one Skill-compatible coding agent before submitting.
6. Update [skills/README.md](skills/README.md) to list the new skill.

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
2. **Prompts and skills are tested.** Every prompt card and skill must have been tested with at least one coding agent against realistic data.
3. **Complete documentation.** All required sections are filled in. No placeholder text.
4. **Consistent structure.** Follows the relevant template from [templates/](templates/).
5. **No sensitive data.** No real employee data, organization names, or identifiable information in examples. Skills additionally must be customer-agnostic, with no references to a specific engagement or client repository.

## Submitting your contribution

1. Fork the repository or create a branch.
2. Make your changes following the guidelines above.
3. Open a pull request against `main`.
4. The CLA bot will check your agreement status.
5. A maintainer will review your contribution.

## Questions?

Open an issue on the [repository](https://github.com/microsoft/viva-insights-sample-code/issues) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com).
