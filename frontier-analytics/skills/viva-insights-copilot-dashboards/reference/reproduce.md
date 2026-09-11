# Reproduce a demo

1. Confirm Consumption or Developer Experience and a new output directory.
   No customer files are needed. Keep the supplied seed and fixed period.
2. Read the selected report manifest and the shared runner README. Create a
   demo config from its example using the user's checkout and output paths.
3. Run `dashboard.R reproduce <config.json>` with Rscript. The runner copies
   the required report, helpers and fixtures into an isolated workspace before
   rendering. It must never render in the canonical `examples/utility-r` folder.
4. If dependencies are missing, report the exact missing packages or Pandoc
   and obtain approval to install them. Do not rebuild the dashboard in another
   framework to bypass setup.
5. Check the HTML opens locally, all report tabs are available, and synthetic
   labels remain visible. Compare key aggregates using the shipped runtime
   tests/reference assertions, not by reading base64 image blobs.
6. Provide the HTML path, rerun command and provenance. Ask about customisation
   only if the user requested it; reproduction itself is a complete result.

Generated synthetic fixture data may be large. Let R read it; inspect source
code only when diagnosing a specific failure. A screenshot is a visual
reference, not a substitute for the source or numeric checks.
