# Behaviour evaluation

Run `cases.json` with a compatible coding agent in a disposable workspace.
Use synthetic fixture inputs only. For the two reproduction cases, supply a
checkout path and a fresh output directory and run the real runner. For blocked
cases, give only the stated evidence; do not let the evaluator invent approvals
or a production schema. The out-of-scope case evaluates routing, not execution.

Record per case: source revision, agent/model, loaded references, commands,
result, observed safety decisions, token counts when exposed, elapsed time
and repairs. A textual promise is not evidence of a successful render.
Use the shared runner's deterministic tests to check numeric and privacy
invariants; the agent evaluation checks instruction-following and routing.

For an efficiency comparison, run the same task and fixtures from clean
workspaces with (a) a plain "build this dashboard" request and (b) this skill.
Keep agent/model/environment and requested output equivalent. Repeat trials;
report median usage and completion/failure counts, not the best successful run.
Do not publish a token/time savings claim until this comparison has been run.

Expected behaviours in `cases.json` are acceptance criteria, not a claim that
all agent/model combinations have passed. Record measured results separately
with their evidence and limitations.
