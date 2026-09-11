# Quickstart

1. Open this repository checkout in your approved coding agent. Record its
   revision. Keep any customer data in a separate private directory.
2. Paste the [Developer Experience prompt](../../prompts/copilot-dashboards/developer-experience-dashboard.md).
   Choose **demo** and provide a fresh output directory.
3. Have the agent follow the [runner README](../copilot-query-dashboard/README.md),
   create a demo config and run `dashboard.R reproduce <config.json>` using
   Rscript. The helper generates synthetic files in an isolated copied workspace.
4. Open the seven-page HTML. Confirm the whole developer baseline, separate
   eligibility/coverage categories and synthetic labels are retained.
5. Customise the copied source, not generated HTML: for example, "lead with
   focus and coordination, retaining the methodology and all privacy rules."

For real files, use a separate **real** config with
`dashboard.R inspect <config.json>`. Commands are run from the shared runner
directory or with its full path. The correct v1 result is a readiness report
and a missing-evidence list, not an automatically generated developer report.

Never put customer data into the demo's generated-file directories or source
its simulation helper as a real-data loader.
