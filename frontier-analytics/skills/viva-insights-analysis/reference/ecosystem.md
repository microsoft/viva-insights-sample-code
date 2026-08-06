# Viva Insights open-source ecosystem

Where to find canonical, up-to-date material. When this skill and upstream
disagree, **defer to upstream** and update the skill.

## Function discovery (agent-facing, added by upstream)

Both packages publish machine-readable and human-readable indexes explicitly
designed for coding agents to find an existing function instead of writing new
analysis code:

| | R | Python |
|---|---|---|
| `llms.txt` | https://microsoft.github.io/vivainsights/llms.txt | https://microsoft.github.io/vivainsights-py/llms.txt |
| Function-discovery guide | https://microsoft.github.io/vivainsights/articles/function-discovery.html | https://microsoft.github.io/vivainsights-py/function-discovery.html |

`llms.txt` is generated from the package itself (R: from a `discovery/workflows.yml`
manifest), so it always reflects the exact installed API. Treat it as more
current than this skill's `packages.md`, which is a manually curated snapshot.
See `SKILL.md`'s "Find the right function before writing custom code" for the
recommended lookup order.

## Packages

| | R | Python |
|---|---|---|
| Docs site | https://microsoft.github.io/vivainsights/ | https://microsoft.github.io/vivainsights-py/ |
| Source | https://github.com/microsoft/vivainsights | https://github.com/microsoft/vivainsights-py |
| Distribution | CRAN: `install.packages("vivainsights")` | PyPI: `pip install vivainsights` |
| License | MIT | MIT |

The docs sites carry the function reference and article-style walkthroughs. The R
package is the reference implementation. The Python package mirrors the common
workflows. Confirm signatures against the installed version (see
`packages.md` for the runtime introspection snippets) because the API evolves.

## Sample-code repository

**`microsoft/viva-insights-sample-code`** at https://github.com/microsoft/viva-insights-sample-code
(website: https://microsoft.github.io/viva-insights-sample-code/).

More elaborate, end-to-end scenarios that build on the packages but are not part
of them. Examples ship as R Markdown (`.Rmd`), Jupyter notebooks (`.ipynb`), or
raw `.R` / `.py`, and generally expect a flexible-query export as input. Paired
Python/R utility examples live under `examples/utility-python/` and
`examples/utility-r/`.

## Frontier Analytics toolkit (start here for agent workflows)

Inside the sample-code repo, **`frontier-analytics/`** is an export-first,
self-service toolkit designed for exactly the coding-agent workflow this skill
supports. Prefer pointing users to it rather than reinventing its assets.

| Folder | What it gives you |
|---|---|
| `prompts/` | Ready-to-paste prompt cards for coding agents: Copilot adoption tracking, user segmentation, ROI estimation, Purview audit-log analysis |
| `starter-kits/` | Bundled end-to-end workflows (use case + inputs + prompts + expected outputs) |
| `schemas/` | Data dictionaries for person-query exports, Purview audit logs, join patterns, and common data pitfalls |
| `examples/` | Sample output specifications describing what a finished deliverable looks like |
| `templates/` | Templates for contributing new prompt cards, starter kits, schema docs |
| `mcp/` | Concepts and sample config for Model Context Protocol integration |

Intended flow: export your data, pick a starter kit or prompt card, review the
schema docs, open a coding agent in an R/Python workspace, paste the prompt at
your file, then review the output against the documented failure modes.

**This skill complements Frontier Analytics.** Frontier Analytics owns the
prompt/starter-kit/schema library. This skill owns the package fluency, the
query-schema and data-pitfall knowledge, and the analysis conventions that make
those prompts produce trustworthy output. When a task maps to an existing prompt
card or starter kit, use it and reference upstream rather than duplicating it.

## Legacy note: the `wpa` R package

`wpa` is the predecessor R package for the older Workplace Analytics data model.
`vivainsights` is the current, supported package for Viva Insights query exports.
Prefer `vivainsights`. Only reach for `wpa` when working with legacy Workplace
Analytics outputs, and expect different column schemas.

## Keeping this skill current

- Re-verify function names and signatures against the installed package versions
  when they change (the R and Python versions move independently), or check
  the live `llms.txt` (see "Function discovery" above), which is faster and
  generated directly from the package.
- If upstream adds or renames query columns, update `query-schemas.md`.
- If Frontier Analytics adds a prompt category, mention it here rather than
  copying its content.
