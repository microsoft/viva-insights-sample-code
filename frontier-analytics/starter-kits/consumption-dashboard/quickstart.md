# Quickstart

1. Open a local checkout of this repository in your approved coding agent.
   Keep customer files outside it. Record the checkout revision.
2. Copy the [Consumption prompt](../../prompts/copilot-dashboards/consumption-dashboard.md)
   and choose **demo** for the first run. Give the checkout and a new output
   directory; no customer data is needed.
3. Let the agent read the [runner's configuration examples](../copilot-query-dashboard/README.md)
   and run `dashboard.R reproduce <config.json>` using Rscript. It checks the
   dependencies and renders copied sources, not the canonical example.
4. Open the HTML. Confirm the synthetic window and labels, seven report pages,
   readable charts and methodology. Keep the copied source and provenance.
5. Ask for a focused change such as "move the consumption distribution before
   the summary, without changing the metrics." The agent edits the copied Rmd.

For real exports, create a separate **real** config and output directory.
Use `dashboard.R inspect <config.json>`, review its supported/excluded analysis
list, then approve explicit mappings before `dashboard.R build <config.json>`.
See the runner README for paths and approval fields; commands are run from
the shared runner directory or by supplying its full path.

Missing token or task-type fields exclude those panels. Do not translate
credits into actions, dollars or tokens. If a required contract is unknown,
finish with a readiness assessment rather than a misleading dashboard.
