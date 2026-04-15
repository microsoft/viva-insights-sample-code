# Example Output Specifications

This directory contains **output specifications** — detailed descriptions of common deliverable formats produced by Viva Insights analytics workflows. These are not actual generated files; they are templates and guidelines that describe what the output should look like.

## How to use these specs

- **As "target output" context for coding agents.** Before running a prompt, share the relevant output spec with your coding agent so it knows the expected format, structure, and quality bar. For example: _"Produce a dashboard following this specification…"_ followed by the spec content.
- **As quality checklists.** After a coding agent produces output, compare it against the spec to verify it meets expectations.
- **As communication tools.** Share specs with stakeholders to align on what a deliverable will look like before investing time in generation.

> **Tip:** Combine an output spec with a [prompt card](../prompts/) and [schema documentation](../schemas/) for the most effective coding agent context. The schema tells the agent about the input data, the prompt tells it what to do, and the output spec tells it what the result should look like.

## Available specs

| Spec | Format | Description |
|------|--------|-------------|
| [HTML Dashboard](sample-output-specs/html-dashboard-spec.md) | Self-contained HTML | Static dashboard with charts, summary metrics, and data tables — suitable for browser viewing and email sharing. |
| [Executive Memo](sample-output-specs/executive-memo-spec.md) | Markdown or HTML | 1–3 page executive summary with key metrics, findings, and recommendations — suitable for VP/C-suite audiences. |
| [Notebook Analysis](sample-output-specs/notebook-analysis-spec.md) | Jupyter (.ipynb) or R Markdown (.Rmd) | Exploratory analysis notebook with code, visualizations, and narrative — suitable for peer review and reproducibility. |

## Notes

- **Actual outputs will vary.** These specs describe the target structure and quality bar. The actual content depends on your data, the prompt used, and the coding agent's implementation.
- **Specs are composable.** A single analysis workflow might produce multiple output types (e.g., a notebook for the detailed analysis plus an executive memo for leadership).
- **Customize for your audience.** Adapt formatting, level of detail, and visual style to match your organization's preferences.

## Related resources

- [Prompt Card Library](../prompts/) — prompts that generate these output types
- [Schema Documentation](../schemas/) — data dictionaries for input data
