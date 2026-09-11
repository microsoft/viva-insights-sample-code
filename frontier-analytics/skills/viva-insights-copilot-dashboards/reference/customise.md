# Customise without rebuilding

Start from the run's configuration, provenance, source copy and existing
aggregate outputs. Ask for the requested change and identify whether it affects
presentation or the meaning of a metric.

Demo reproduction supplies a copied Rmd workspace. A real credits build supplies
released `aggregates.csv`, not a copied Rmd. For its presentation-only changes,
first copy the shared runner's `lib/runtime.R` into the private workspace and
reuse its `dashboard_html` renderer with the released aggregates and recorded
configuration. Keep the original suppressed/empty/ready status; an empty CSV
is not evidence of zero consumption. Record the copy and rerender command.

| Change | Work required |
|---|---|
| Titles, colours, chart order, explanatory text | Edit the copied presentation source; retain labels/limitations; rerender and compare aggregate outputs |
| Period, group or population | Update config, rerun inspection and privacy checks, obtain approval of changed denominators, build in a new output directory |
| New metric, data source or claim | Return to adaptation and verify the contract before computing |
| Switch from synthetic to real | New real config and output directory; never replace fixtures inside the demo workspace |

Use prompt mode when choices change the scope. Do not regenerate unrelated
panels or use the rendered HTML as source. The HTML contains bundled libraries
and image data that are expensive to read and not intended for hand editing.

Run the smallest check covering the change, followed by a local render check.
Check headings, labels, narrow-screen layout, chart readability and disclosures.
Keep privacy suppression and all synthetic/association labels. If the user
asks to remove them, explain why they are necessary.

Record source edits and rerun command beside the modified output. Cached data
may be reused only while input hashes, period, group, mapping, privacy policy
and metric definitions remain unchanged.
