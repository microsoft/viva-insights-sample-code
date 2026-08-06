<!-- ==========================================================================
     SKILL TEMPLATE: How to fill this in
     ==========================================================================

     A Skill is a packaged capability that a compatible coding agent (for
     example GitHub Copilot CLI or Claude Code) loads automatically when a
     task matches its description, without the user pasting anything. This
     differs from a prompt card, which a person copies and pastes manually.

     Use this template when you have Viva Insights domain knowledge,
     conventions, or workflows that should apply consistently across many
     tasks and many sessions, rather than a single one-off analysis prompt.
     If your contribution is a single analysis task with one expected
     output, use the prompt card template instead.

     A good Skill has these qualities:

     1. Self-contained. An agent should be able to use the Skill correctly
        with only its own contents, without requiring another Skill or
        private context to be loaded at the same time.

     2. Customer-agnostic. Do not include organization-specific paths,
        scopes, dates, or data. If your Skill only makes sense for a single
        client or engagement, it does not belong in this public repository.

     3. Verified. Confirm any package function names, signatures, or
        example output against the currently installed package version
        before submitting.

     4. Scoped. Keep the main SKILL.md short and move long or rarely-needed
        material into a reference/ subfolder that the agent reads on demand.
        See frontier-analytics/skills/viva-insights-analysis/ for a working
        example of this pattern.

     Replace every [placeholder] below with your content, then delete this
     comment block before submitting your PR.
     ========================================================================== -->

# [Skill folder name, kebab-case]

Create a folder at `frontier-analytics/skills/[skill-name]/` containing at
least a `SKILL.md` file with the structure below. Add `reference/` and
`examples/` subfolders if the content is long enough to benefit from being
split, as `viva-insights-analysis` does.

## SKILL.md structure

```markdown
---
name: [skill-name, must match the folder name]
description: >
  [One paragraph, third person, describing what this Skill covers and when
  an agent should use it. This is the only part of the file some agents read
  before deciding whether to load the rest, so be specific about the tasks,
  data types, and packages involved.]
---

## What this skill is for

[A short paragraph explaining the purpose, and a bullet list of the specific
tasks that should trigger this skill.]

## [Domain-specific sections as needed]

[The bulk of the file. Cover conventions, common pitfalls, and worked
examples relevant to the domain. Prefer runnable code snippets in both R and
Python where the packages support both.]

## How to use the reference files (if you have a reference/ folder)

[A table mapping each reference file to the situation in which an agent
should read it.]

## Guardrails

[A short list of rules the agent should always follow when using this
skill, for example privacy thresholds, data handling, or preferring
association over causation in written claims.]
```

## Checklist before submitting

- Every section above is filled in and no organization-specific detail remains.
- The `name` in the frontmatter matches the folder name exactly.
- All package function names and signatures are verified against the currently
  installed version of the package.
- Any example data uses the packages' built-in sample datasets rather than real or
  simulated customer data.
- The skill has been tried with at least one Skill-compatible coding agent.
- The file is saved at `frontier-analytics/skills/[skill-name]/SKILL.md`, and
  `frontier-analytics/skills/README.md` lists the new skill.
