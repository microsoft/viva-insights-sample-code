# Environment and run tips

Practical notes for running vivainsights analyses reliably, especially in a
headless / agent context. These are general good practices. They are not tied
to any machine or customer.

## Prefer script files over long inline one-liners

Running a long analysis as a single inline `-e` / `-c` string is fragile
(quoting, memory, partial state). Prefer writing a small script file and running
it, then removing throwaway probes at the end of the task.

```powershell
Rscript analysis.R
python analysis.py
```

If a specific environment proves unstable running `Rscript -e "library(...)"`
inline, switch to a `.R` file. Write a temporary `_probe.R` / `_probe.py` for
interactive checks and delete it when done. Do not leave debug scripts behind.

## Headless plotting

The `create_*` functions render figures. In a headless or batch context, force a
non-interactive backend so plotting never blocks.

```powershell
# Python: Agg backend, no display needed
$env:MPLBACKEND = "Agg"
python analysis.py
```

```r
# R: write plots to files rather than a screen device
ggplot2::ggsave("rank.png", create_rank(pq, metric = "Collaboration_hours",
                                         hrvar = "Organization", return = "plot"))
```

When you only need the numbers, request `return = "table"` (R) /
`return_type = "table"` (Python) and skip rendering entirely.

## Load only the columns you need

Person Query exports can be very wide. Select columns at read time to keep loads
fast and memory low (see `data-pitfalls.md` for the code). A handful of columns
loads in a fraction of the time of the full width.

## Validate before analysing

Run `check_query()` (and, in R, `validation_report()`) right after import. It is
cheaper to catch a wrong grain, a missing HR variable, or a type problem here
than to debug a misleading chart later.

## Reproducibility

- Pin package versions in the environment and record them in outputs.
- Set a seed for anything stochastic. Network layouts take a `seed=`, and
  simulators and samples should be seeded too.
- State the exact date windows and `mingroup` used, so a cut can be reproduced.

## Verifying the API at runtime

Do not rely on memory for signatures. The R and Python packages version
independently. Introspect the installed version:

```r
packageVersion("vivainsights"); args(vivainsights::create_rank)
```
```python
import vivainsights as vi, inspect
print(vi.__version__); print(inspect.signature(vi.create_rank))
```

## Cleanup

Remove temporary probe scripts, rendered test images, and any intermediate
exports that are not part of the deliverable before finishing. Never commit real
query exports or rendered outputs that embed real data.
