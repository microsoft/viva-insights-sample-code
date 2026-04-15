# Starter Kits

A **starter kit** is a bundled workflow that combines a use case, required inputs, prompt cards, and expected outputs into a single package. Each kit gives you everything you need to go from an exported Viva Insights CSV to a finished deliverable using a coding agent.

## Available starter kits

| Starter kit | Description | Complexity | Primary output |
|-------------|-------------|------------|----------------|
| [Copilot Adoption Dashboard](starter-kits/copilot-adoption-dashboard/) | Interactive dashboard tracking Copilot adoption metrics over time, including usage trends, feature-level breakdowns, and user segmentation by HR attributes. | Intermediate | HTML dashboard |
| [Executive Summary Report](starter-kits/executive-summary-report/) | One-page memo summarizing key collaboration and Copilot metrics for executive audiences. Designed for fast turnaround with minimal customization. | Beginner | Markdown / HTML memo |

## What's in a starter kit?

Each starter kit folder contains some or all of the following:

- **README.md** — Overview of the use case, who it's for, and what it produces
- **quickstart.md** — Step-by-step instructions specific to that kit
- **required-inputs.md** — The data files and parameters you need before starting
- **recommended-files.md** — Suggested file organization and naming
- **expected-output.md** — Description or screenshot of what the finished output looks like

Starter kits reference prompt cards from the [prompts/](prompts/) library. The prompts contain the actual text you paste into your coding agent.

## How to use a starter kit

1. **Read the README.** Understand the use case and check that it matches your scenario.
2. **Check the required inputs.** Make sure you have the necessary data exports (person query CSV, Purview audit logs, etc.) and that your R or Python environment is set up.
3. **Follow the quickstart.** The kit's quickstart walks you through the workflow step by step.
4. **Use the prompts with your coding agent.** Open the referenced prompt cards, copy the prompt text, and paste it into your agent with your data context.
5. **Review the expected output.** Compare your results against the documented output to verify correctness.

For general setup instructions, see the [Quickstart guide](QUICKSTART.md).

## How to contribute a new starter kit

We welcome new starter kits. To add one:

1. Check the [templates/](templates/) folder for the starter kit template.
2. Create a new folder under `starter-kits/` with a descriptive kebab-case name (e.g., `starter-kits/meeting-culture-report/`).
3. Include at minimum: `README.md`, `quickstart.md`, and `required-inputs.md`.
4. Reference existing prompt cards from [prompts/](prompts/) or create new ones following the [prompt card template](templates/).
5. Submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Next steps

- [Quickstart guide](QUICKSTART.md) — general getting-started instructions
- [Prompt card library](prompts/) — browse all available prompts
- [Schema documentation](schemas/) — understand your data
- [Back to Frontier Analytics README](README.md)
