"""Build the Viva Insights query join guide (Jekyll page plus JSON contract).

One CONTRACT dictionary drives both the human-readable tables and the embedded
machine-readable contract, so they cannot drift apart. validate_contract() fails
the build if any reference does not resolve. Public and synthetic content only:
no tenant data, study results or internal paths.

Output: _pages/query-join-guide.html and assets/data/query-join-contract.json
Run python scripts/build_join_guide.py to regenerate, or add --check to detect drift.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "_pages" / "query-join-guide.html"
CONTRACT_PATH = ROOT / "assets" / "data" / "query-join-contract.json"
DATE = "28 October 2026"

S, D, N, B = "string", "date", "number", "boolean-like text"


def col(name, type_, role, note=""):
    return {"name": name, "type": type_, "role": role, "note": note}


def group(name, note):
    return {"name": name, "headers": "inspect_export", "note": note}


def file_(id_, file, grain, key, columns, *, layout=True, core=True, groups=(), pattern=None, note=""):
    entry = {"id": id_, "file": file, "grain": grain, "native_key": key,
             "required_for_layout_match": layout, "required_for_core_panel": core,
             "columns": columns, "column_groups": list(groups)}
    if pattern:
        entry["file_pattern"] = pattern
    if note:
        entry["note"] = note
    return entry


SURFACE_HEADERS = [
    "Copilot actions taken in Copilot Chat", "Copilot actions taken in Microsoft 365 Copilot app",
    "Days of active Copilot Chat usage", "Days of active Copilot Chat (legacy web) usage",
    "Days of active Copilot usage in Excel", "Days of active Copilot usage in Loop",
    "Days of active Copilot usage in OneNote", "Days of active Copilot usage in Outlook",
    "Days of active Copilot usage in Powerpoint", "Days of active Copilot usage in Teams",
    "Days of active Copilot usage in Word",
]
SURFACE_DAY_HEADERS = [h for h in SURFACE_HEADERS if h.startswith("Days of active")]

CONTRACT = {
    "contract_id": "viva-insights-query-join",
    "version": "0.6.0",
    "date": "2026-10-28",
    "status": "reference",
    "scope": "Joining weekly Viva Insights Person Query exports with Consumption, GitHub and Agent query exports into one person-week panel",
    "spine": "person_query.person",
    "input_requirements": {
        "required_query_ids": ["person_query"],
        "additional_query_ids": ["consumption_query", "github_query", "agent_query"],
        "minimum_additional_queries": 1,
        "recommendation": "Use all four queries for a broader view of ways of working and AI usage across the products and activities they measure.",
        "applicability": "Apply file requirements, joins and source-specific checks only to the selected queries. Record omitted queries as unavailable, without inferring non-use.",
    },
    "naming_conventions": {
        "ids": "File and dataset IDs are references within this guide's contract. They are not exported columns, file paths or required names for objects in your script.",
        "derived_datasets": "Dataset labels describe intermediate summaries and the final panel that your script creates. Choose your own object names.",
        "week_start": "WeekStart is this guide's chosen name for an aligned week-start date. It is not an extra column supplied by the exports. You may keep MetricDate or choose another name, provided every source uses the same week definition.",
    },
    "target_grain": {
        "keys": ["PersonId", "WeekStart"],
        "person_query_cadence_required": "weekly",
        "week_start": "detect_from_person_query",
        "rule": "Export all selected queries grouped by week, with the same week start, wherever possible. This contract requires a weekly Person Query. WeekStart is its MetricDate. Consumption and GitHub default to daily, so change their grouping at setup. Use daily exports only when daily detail or week-start alignment is needed (G03, G03b).",
    },
    "coverage_assumption": {
        "complete_by_default": True,
        "scope": "The configured population, date window and product filters of each supplied Viva Insights query.",
        "missing_activity_rows": "Inactive person-days are excluded by default. Within the query's scope, treat omitted activity rows as zero activity, including a person-week with no activity rows. This does not require separate completeness evidence.",
        "exceptions": "Keep unavailable queries, out-of-scope people or periods, null cells in existing rows and known export problems distinct from inactivity. Partial boundary weeks do not establish full-week totals.",
        "eligibility": "Activity zeros do not establish product eligibility or licence status. Use product-specific evidence for adoption denominators.",
    },
    "header_matching": "Headers are listed in their raw export form, with spaces. The vivainsights R and Python packages replace spaces with underscores when they import a query, so Collaboration hours becomes Collaboration_hours. Treat the two forms as the same column. This applies to headers only, never to identifier values.",
    "query_setup_recommendation": "For all four queries, we recommend the Create analysis route because it offers more customisation parameters than dashboard exports.",
    "queries": [
        {
            "id": "person_query", "name": "Person query", "role": "spine",
            "recommended_path": ["Create analysis", "Customized query", "Person query"],
            "access_context": "The Person query is available only through this route. It gives you access to the full Person query metric list, including behavioural metrics such as Email hours, Meeting hours and Internal network size. Select the metrics you need when setting up the query.",
            "cadence": "Weekly. Daily or monthly groupings are outside this contract.",
            "period_options": ["week"], "default_period": "week",
            "verification": "observed_in_real_exports",
            "files": [file_(
                "person_query.person", "Person_<username>_<date>_<time>Hours.csv", "One row per person per week", ["PersonId", "MetricDate"],
                [col("PersonId", S, "key", "Pseudonymised person identifier. Keep as text."),
                 col("MetricDate", D, "key", "Week start. Detect the weekday from the data.")],
                pattern="A single CSV, for example `Person_jdoe_Oct27_2026_1027Hours.csv`. Localised exports translate the words in the name, and the public sample fixture is `PersonQuery.csv`, so identify the file by its PersonId and MetricDate columns.",
                groups=[group("HR attributes", "Configured at query setup. Observed examples: Organization, FunctionType, SupervisorIndicator, Layer. Values can change between weeks."),
                        group("Metric columns", "Selected at query setup. See the metric families table.")]),
            ],
        },
        {
            "id": "consumption_query", "name": "Consumption query", "role": "fact",
            "recommended_path": ["Create analysis", "Customized query", "Consumption query"],
            "alternative_path": ["Consumption dashboard", "Export data"],
            "cadence": "Daily by default, or weekly when grouped by week at setup. Facts plus metadata files.",
            "period_options": ["day", "week"], "default_period": "day",
            "verification": "observed_in_real_exports",
            "files": [
                file_("consumption.m365_credits", "PersonM365CreditsMetrics.csv",
                      "One row per person, service, spending policy and period (day or week)",
                      ["PersonId", "ServiceId", "SpendingPolicyId", "MetricDate"],
                      [col("PersonId", S, "key"), col("ServiceId", S, "key"), col("ServiceName", S, "attribute"),
                       col("SpendingPolicyId", S, "key"), col("MetricDate", D, "key", "Day or week start, as set at query setup."),
                       col("Session count", N, "measure"), col("Spending policy limit", N, "limit"),
                       col("Total Copilot Credits used", N, "measure"), col("User limit", N, "limit"),
                       col("PeopleHistoricalId", S, "foreign_key", "Links to PeopleMetaData.csv.")]),
                file_("consumption.github_credits", "PersonGitHubCreditsMetrics.csv",
                      "One row per person and period with billable GitHub usage (sparse)", ["PersonId", "MetricDate"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key", "Day or week start, as set at query setup."),
                       col("Total GitHub AI Credits used", N, "measure"),
                       col("PeopleHistoricalId", S, "foreign_key", "Links to PeopleMetaData.csv.")]),
                file_("consumption.people_metadata", "PeopleMetaData.csv", "One row per historical person record",
                      ["PeopleHistoricalId"],
                      [col("PeopleHistoricalId", S, "key", "No PersonId in this file."),
                       col("IsCopilotLicensed", B, "attribute", "Historical Microsoft 365 Copilot licence evidence only. Not evidence of GitHub Copilot provisioning.")],
                      note="Carries no HR attributes. Take organisational attributes from the Person Query."),
                file_("consumption.policy_metadata", "M365SpendingPolicyMetaData.csv", "One row per spending policy",
                      ["SpendingPolicyId"], [col("SpendingPolicyId", S, "key")], core=False,
                      groups=[group("Policy attributes", "Descriptive policy fields. Inspect the headers in each export.")]),
            ],
        },
        {
            "id": "github_query", "name": "GitHub query", "role": "fact",
            "recommended_path": ["Create analysis", "Customized query", "GitHub query"],
            "cadence": "Daily by default, or weekly when grouped by week at setup. Activity facts plus dimensional breakdowns.",
            "period_options": ["day", "week"], "default_period": "day",
            "verification": "reference_schema_not_yet_verified_on_real_exports",
            "files": [
                file_("github.activity", "PersonGitHubActivityMetrics.csv", "One row per person and period",
                      ["PersonId", "MetricDate"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key", "Day or week start, as set at query setup."),
                       col("Agent adoption", N, "indicator", "Adoption indicator or percentage. Excludes code review."),
                       col("Code completions accepted", N, "measure"), col("Code completions suggested", N, "measure"),
                       col("User-initiated chat requests", N, "measure")]),
                file_("github.by_feature", "GitHubActivityBreakdownByFeatureMetrics.csv",
                      "One row per person, period and feature", ["PersonId", "MetricDate", "Feature"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key"), col("Feature", S, "key"),
                       col("Feature Usage Count", N, "measure")], core=False),
                file_("github.by_language_feature", "GitHubActivityBreakdownByLanguageFeatureMetrics.csv",
                      "One row per person, period, language and feature", ["PersonId", "MetricDate", "Language", "Feature"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key"), col("Language", S, "key"),
                       col("Feature", S, "key"), col("Usage count by language and feature", N, "measure")], core=False),
                file_("github.by_language_model", "GitHubActivityBreakdownByLanguageModelMetrics.csv",
                      "One row per person, period, language and model", ["PersonId", "MetricDate", "Language", "Model"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key"), col("Language", S, "key"),
                       col("Model", S, "key"), col("Usage count by language and model", N, "measure")], core=False),
                file_("github.by_model_feature", "GitHubActivityBreakdownByModelFeatureMetrics.csv",
                      "One row per person, period, model and feature", ["PersonId", "MetricDate", "Model", "Feature"],
                      [col("PersonId", S, "key"), col("MetricDate", D, "key"), col("Model", S, "key"),
                       col("Feature", S, "key"), col("Usage count by model and feature", N, "measure")], core=False),
            ],
        },
        {
            "id": "agent_query", "name": "Agent query", "role": "fact",
            "recommended_path": ["Create analysis", "Customized query", "Agent query"],
            "alternative_path": ["Agent Dashboard", "Export data"],
            "cadence": "Weekly facts with an optional metadata file.",
            "period_options": ["week"], "default_period": "week",
            "verification": "observed_in_real_exports",
            "files": [
                file_("agent.activity", "PersonAgentActivityMetrics.csv", "One row per person, agent and week",
                      ["PersonId", "AgentId", "MetricDate"],
                      [col("PersonId", S, "key"), col("AgentId", S, "key"),
                       col("MetricDate", D, "key", "Weekly. Sunday-start in observed exports."),
                       col("Conversational agent sessions", N, "measure"),
                       col("Returning Agent 365 user (most recent and prior 7 days)", N, "indicator", "0 or 1 over a 7-day window."),
                       col("Returning Agent 365 user (most recent and prior 28 days)", N, "indicator", "0 or 1 over a 28-day window.")]),
                file_("agent.metadata", "AgentMetadata.csv", "One row per agent", ["AgentId"],
                      [col("AgentId", S, "key")], layout=False, core=False,
                      groups=[group("Agent attributes", "Descriptive agent fields. Inspect the headers in each export. Never use them to identify people.")],
                      note="Present in some exports and absent in others."),
            ],
        },
    ],
    "derived_datasets": [
        {"id": "summary.m365_credits_person_week", "label": "Microsoft 365 credits summary", "from_file_ids": ["consumption.m365_credits"], "grain": ["PersonId", "WeekStart"], "build_steps": [5, 6], "built_by": "Sum credits across services and spending policies after G05, and across days if the export is daily. Include sessions summed across policies only when G05b passes."},
        {"id": "summary.github_credits_person_week", "label": "GitHub credits summary", "from_file_ids": ["consumption.github_credits"], "grain": ["PersonId", "WeekStart"], "build_steps": [6], "built_by": "Sum credits across days if the export is daily. A weekly export is already one row per person-week."},
        {"id": "summary.github_activity_person_week", "label": "GitHub activity summary", "from_file_ids": ["github.activity"], "grain": ["PersonId", "WeekStart"], "build_steps": [6], "built_by": "Apply the measure rules, aggregating across days only if the export is daily."},
        {"id": "summary.agent_person_week", "label": "Agent activity summary", "from_file_ids": ["agent.activity"], "grain": ["PersonId", "WeekStart"], "build_steps": [6], "built_by": "Sum sessions across agents. Derive any person-level returning flag explicitly."},
        {"id": "panel.person_week", "label": "Combined person-week panel", "from_file_ids": ["person_query.person"], "grain": ["PersonId", "WeekStart"], "build_steps": [7], "built_by": "Start with the Person Query and left join each summary. Keep every Person Query row."},
    ],
    "person_query_metric_families": [
        {"family": "Collaboration", "kind": "weekly_total",
         "observed_real_export_headers": ["Collaboration hours", "External collaboration hours"],
         "sample_fixture_headers": ["Collaboration_hours", "Meeting_hours", "Email_hours", "Chat_hours", "After_hours_collaboration_hours", "External_collaboration_hours"]},
        {"family": "Focus", "kind": "weekly_total", "observed_real_export_headers": [],
         "sample_fixture_headers": ["Available_to_focus_hours", "Uninterrupted_hours"]},
        {"family": "Network", "kind": "stock",
         "observed_real_export_headers": ["Internal network size", "External network size"],
         "sample_fixture_headers": ["Internal_network_size", "External_network_size", "Strong_ties", "Diverse_ties", "Network_outside_organization"]},
        {"family": "Copilot eligibility", "kind": "day_count",
         "observed_real_export_headers": ["Total Copilot enabled days"], "sample_fixture_headers": ["Total_Copilot_enabled_days"]},
        {"family": "Copilot activity, totals", "kind": "weekly_total_or_day_count",
         "observed_real_export_headers": ["Total Copilot actions taken", "Total Copilot active days"], "sample_fixture_headers": []},
        {"family": "Copilot activity, by surface", "kind": "weekly_total_or_day_count",
         "observed_real_export_headers": SURFACE_HEADERS, "sample_fixture_headers": []},
        {"family": "Cowork", "kind": "weekly_total_or_day_count",
         "observed_real_export_headers": ["Cowork tasks submitted", "User-initiated Cowork tasks submitted", "Days of active Cowork usage"],
         "sample_fixture_headers": []},
        {"family": "Unlicensed Copilot Chat", "kind": "weekly_total_or_day_count", "observed_real_export_headers": [],
         "sample_fixture_headers": [],
         "note": "A separate metric family for Copilot Chat use without a Microsoft 365 Copilot licence. Inspect the export for its headers and keep it separate from licensed Copilot metrics."},
    ],
    "relationships": [
        {"id": "J1", "from_id": "consumption.m365_credits", "to_id": "consumption.people_metadata", "on": ["PeopleHistoricalId"], "cardinality": "many_to_one", "purpose": "Microsoft 365 Copilot licence evidence, before aggregation"},
        {"id": "J2", "from_id": "consumption.github_credits", "to_id": "consumption.people_metadata", "on": ["PeopleHistoricalId"], "cardinality": "many_to_one", "purpose": "Check that each historical person record exists. IsCopilotLicensed describes Microsoft 365 only and cannot confirm GitHub eligibility."},
        {"id": "J3", "from_id": "consumption.m365_credits", "to_id": "consumption.policy_metadata", "on": ["SpendingPolicyId"], "cardinality": "many_to_one", "purpose": "Check that each spending policy exists in the metadata."},
        {"id": "J4", "from_id": "agent.activity", "to_id": "agent.metadata", "on": ["AgentId"], "cardinality": "many_to_one", "purpose": "Optional enrichment, only when the metadata file is supplied"},
        {"id": "J5", "from_id": "summary.m365_credits_person_week", "to_id": "panel.person_week", "on": ["PersonId", "WeekStart"], "cardinality": "one_to_one", "purpose": "Left join onto the spine"},
        {"id": "J6", "from_id": "summary.github_credits_person_week", "to_id": "panel.person_week", "on": ["PersonId", "WeekStart"], "cardinality": "one_to_one", "purpose": "Left join onto the spine"},
        {"id": "J7", "from_id": "summary.github_activity_person_week", "to_id": "panel.person_week", "on": ["PersonId", "WeekStart"], "cardinality": "one_to_one", "purpose": "Left join onto the spine"},
        {"id": "J8", "from_id": "summary.agent_person_week", "to_id": "panel.person_week", "on": ["PersonId", "WeekStart"], "cardinality": "one_to_one", "purpose": "Left join onto the spine"},
        {"id": "J9", "from_ids": ["github.by_feature", "github.by_language_feature", "github.by_language_model", "github.by_model_feature"], "to_id": "panel.person_week", "on": ["PersonId", "WeekStart"], "cardinality": "linked_detail", "purpose": "Never joined wide. Keep as separate detail tables keyed to the panel"},
    ],
    "measures": [
        {"id": "M01", "file_id": "consumption.m365_credits", "columns": ["Total Copilot Credits used"], "kind": "additive", "person_week_rule": "Sum across services and spending policies after G05 passes, and across days when the export is daily."},
        {"id": "M02", "file_id": "consumption.m365_credits", "columns": ["Session count"], "kind": "conditionally_additive", "person_week_rule": "Sum across services, and across days when the export is daily. Sum across SpendingPolicyId rows only when G05b passes. Sessions are not people."},
        {"id": "M03", "file_id": "consumption.m365_credits", "columns": ["Spending policy limit", "User limit"], "kind": "limit", "person_week_rule": "Never sum. Keep per policy as linked detail."},
        {"id": "M04", "file_id": "consumption.github_credits", "columns": ["Total GitHub AI Credits used"], "kind": "additive", "person_week_rule": "Sum across days when the export is daily. Keep separate from Microsoft 365 credits."},
        {"id": "M05", "file_id": "github.activity", "columns": ["Code completions accepted", "Code completions suggested", "User-initiated chat requests"], "kind": "additive", "person_week_rule": "Sum across days when the export is daily. Acceptance rate is summed accepted divided by summed suggested, with a stated zero-denominator rule."},
        {"id": "M06", "file_id": "github.activity", "columns": ["Agent adoption"], "kind": "indicator", "person_week_rule": "Do not sum. For a daily export, state a derived weekly rule, for example any positive day in the week. For a weekly export, use the exported value."},
        {"id": "M07", "file_id": "github.by_feature", "columns": ["Feature Usage Count"], "kind": "additive_within_file", "person_week_rule": "Sum days within this file when daily, retaining Feature for each person-week. Keep as linked detail. Never add to another breakdown."},
        {"id": "M08", "file_id": "github.by_language_feature", "columns": ["Usage count by language and feature"], "kind": "additive_within_file", "person_week_rule": "Sum days within this file when daily, retaining Language and Feature for each person-week. Keep as linked detail. Never add to another breakdown."},
        {"id": "M09", "file_id": "github.by_language_model", "columns": ["Usage count by language and model"], "kind": "additive_within_file", "person_week_rule": "Sum days within this file when daily, retaining Language and Model for each person-week. Keep as linked detail. Never add to another breakdown."},
        {"id": "M10", "file_id": "github.by_model_feature", "columns": ["Usage count by model and feature"], "kind": "additive_within_file", "person_week_rule": "Sum days within this file when daily, retaining Model and Feature for each person-week. Keep as linked detail. Never add to another breakdown."},
        {"id": "M11", "file_id": "agent.activity", "columns": ["Conversational agent sessions"], "kind": "additive", "person_week_rule": "Sum across agents to person-week."},
        {"id": "M12", "file_id": "agent.activity", "columns": ["Returning Agent 365 user (most recent and prior 7 days)"], "kind": "window_indicator", "person_week_rule": "Do not sum across agents or weeks. A person-level flag, such as returning for any agent, is a stated derived definition."},
        {"id": "M13", "file_id": "agent.activity", "columns": ["Returning Agent 365 user (most recent and prior 28 days)"], "kind": "window_indicator", "person_week_rule": "Do not sum across agents or weeks. A person-level flag, such as returning for any agent, is a stated derived definition."},
        {"id": "M14", "file_id": "person_query.person", "columns": ["Internal network size", "External network size"], "kind": "stock", "person_week_rule": "Detect and record the measurement window. Values can repeat within a month. Average or deduplicate snapshots according to that window, never sum."},
        {"id": "M15", "file_id": "person_query.person", "columns": SURFACE_DAY_HEADERS, "kind": "day_count", "person_week_rule": "Never add across surfaces to make total active days, because one day can include several surfaces. Use Total Copilot active days."},
    ],
    "missingness_states": [
        {"id": "observed_positive", "meaning": "A row exists with a positive value"},
        {"id": "observed_zero", "meaning": "A row exists with a recorded zero"},
        {"id": "absent_inactive_zero", "meaning": "No activity row within a supplied query's configured population and date window. Assume complete Viva Insights data and record zero activity"},
        {"id": "absent_unknown", "meaning": "The query is unavailable, the person or period is outside its scope, a known export problem applies, or an absent non-activity measure has no defined zero interpretation. Keep as unknown"},
        {"id": "observed_null", "meaning": "A row exists but the measure is null. Do not apply the omitted-inactive-row default to this cell"},
        {"id": "ineligible", "meaning": "Not eligible for that product in the week"},
        {"id": "outside_cohort", "meaning": "Not selected by the analysis cohort rule"},
    ],
    "gates": [
        {"id": "G01", "level": "MUST", "rule": "Require the Person Query plus at least one of Consumption, GitHub or Agent. Record which queries are supplied and identify each export's layout. Within each selected query, files with `required_for_layout_match` set to true must all be present. If files or key columns differ from this contract, stop and record the difference."},
        {"id": "G02", "level": "MUST", "rule": "Read identifiers as text. Do not change their case, trim them silently or fuzzy-match people."},
        {"id": "G03", "level": "MUST", "rule": "Confirm weekly grouping from the Person Query configuration or equivalent export metadata, then validate one row per PersonId and MetricDate and detect the weekday of MetricDate. Stop if grouping is unknown, daily or monthly. Do not assume Monday or Sunday."},
        {"id": "G03b", "level": "MUST", "rule": "Record each fact export's configured date grouping from query setup or equivalent export metadata, then validate MetricDate against it. Sparse dates alone cannot establish cadence. Recommend weekly grouping for all queries. Daily and weekly exports are supported. A weekly export must start on the same weekday as the Person Query. Stop for unknown grouping, any other period such as monthly, or a different weekly start."},
        {"id": "G04", "level": "MUST", "rule": "Assert uniqueness of each file's native key. Report duplicate counts and stop. Never keep an arbitrary first row."},
        {"id": "G05", "level": "MUST", "rule": "Before summing Microsoft 365 credits across SpendingPolicyId, count person-service-period keys with more than one positive-credit row. If any exist, obtain a documented confirmation that the rows record disjoint consumption."},
        {"id": "G05b", "level": "MUST", "rule": "Sum Session count across SpendingPolicyId rows only when the documented confirmation explicitly covers sessions. Otherwise keep sessions per policy."},
        {"id": "G06", "level": "MUST", "rule": "Join metadata many-to-one on unique keys. Report unmatched key counts without identifiers. Check that each PeopleHistoricalId maps to one PersonId."},
        {"id": "G07", "level": "MUST", "rule": "Bring only panel-bound summaries to one row per PersonId and WeekStart before joining. Apply the measure rules: sum days only for configured daily exports, credits across policies only after G05, and sessions across policies only after G05b. Otherwise retain sessions per policy outside the panel. Keep GitHub breakdown dimensions as linked detail under G11. Never expand weekly rows into invented daily rows."},
        {"id": "G08", "level": "MUST", "rule": "Left join each person-week summary onto the Person Query. Assert one-to-one, preserve the Person Query row count, and reconcile additive totals before and after."},
        {"id": "G09", "level": "MUST", "rule": "Assume supplied Viva Insights queries are complete within their configured population, date window and product filters. Omitted activity rows represent inactivity by default, so record zero activity within that scope without requiring separate completeness evidence. Do not zero-fill unavailable queries, out-of-scope periods or people, null cells in existing rows, rolling indicators, limits, or known export problems. Record partial boundary weeks separately. Eligibility is a separate attribute: Microsoft 365 licence evidence never confirms GitHub eligibility."},
        {"id": "G10", "level": "MUST", "rule": "Keep Microsoft 365 and GitHub credits in separate columns. Never derive tokens or currency from credits."},
        {"id": "G11", "level": "MUST", "rule": "Do not join GitHub breakdown files wide. Keep them as linked detail tables."},
        {"id": "G12", "level": "SHOULD", "rule": "Treat integration and cohort selection as separate, declared steps. An unavailable query does not establish non-use."},
        {"id": "G13", "level": "SHOULD", "rule": "Record each variable's measurement window. Network sizes can repeat within a month and are averaged rather than summed."},
        {"id": "G14", "level": "MUST", "rule": "Coverage, licence and additivity confirmations apply only to the export they were given for."},
        {"id": "G15", "level": "MUST", "rule": "Check PersonId overlap between sources before joining. Investigate low overlap against activity filters, omitted inactive people and configured populations before concluding that pseudonymisation differs. Do not repair identifiers by matching on attributes."},
        {"id": "G16", "level": "MUST", "rule": "Publish a group only when it contains at least 10 distinct people, counted with count_distinct(PersonId), in every filter and time slice. Withhold an entire related breakdown if remaining cells or totals would reveal a suppressed cell by subtraction. Never output identifiers."},
        {"id": "G17", "level": "MUST", "rule": "Describe joined results as associations. Credits, sessions and actions do not measure value, quality or productivity."},
    ],
}

PROCEDURE = [
    ("Match the layout", "G01", "Confirm that the Person Query and at least one of Consumption, GitHub or Agent are supplied. List the files for each selected query and compare their names and key columns with the schema tables. Record the export date, configured population, date window and product filters, which layout matched and which queries are unavailable."),
    ("Load safely", "G02", "Read every identifier column as text and parse dates explicitly. Headers may arrive in raw form with spaces, or with underscores after import through the vivainsights packages."),
    ("Fix the target grain", "G03, G03b", "Recommend weekly grouping for every selected query. Record each configured grouping from query setup or equivalent export metadata before checking dates. Confirm the Person Query is weekly with one row per PersonId and MetricDate, and detect its weekday as WeekStart. Validate daily fact dates or matching weekly starts. Stop if grouping is unknown, rather than inferring it from sparse dates."),
    ("Validate each file", "G04, G06", "Assert each native key. Join metadata files many-to-one and report unmatched counts. Check that each PeopleHistoricalId maps to one PersonId."),
    ("Check policy additivity", "G05, G05b", "In PersonM365CreditsMetrics.csv, count person-service-period keys with more than one positive-credit SpendingPolicyId row. Zero means credits can be summed. Otherwise stop for a documented decision, and confirm separately whether it covers sessions."),
    ("Bring each source to person-week", "G07, G10, G11", "Build only panel-bound summaries at one row per person-week. For a configured daily export, map MetricDate to WeekStart and sum permitted measures across days. For a weekly export, use MetricDate as WeekStart. Follow the measure-specific service, policy and agent rules, including G05b for sessions. Keep unconfirmed per-policy sessions and dimension-preserving GitHub breakdowns as linked detail. Keep Microsoft 365 and GitHub credits separate."),
    ("Join onto the spine", "G08, G15", "Left join each summary to the Person Query on PersonId and WeekStart. Assert one-to-one, check PersonId overlap, and reconcile row counts and additive totals."),
    ("Label missing values", "G09", "Assume supplied Viva Insights data are complete within the recorded query scope. For activity measures, treat omitted inactive person-days, or an entire inactive person-week, as zero. Label unavailable or out-of-scope data, known export problems and null cells separately. Record partial boundary weeks and keep licence eligibility separate from activity zeros."),
    ("Select the cohort", "G12, G13", "Choose the population for the question, for example everyone eligible for an adoption view, and record the rule and the measurement windows."),
    ("Publish safely", "G14, G16, G17", "Publish only groups of at least 10 distinct people in every slice, withhold related breakdowns that would allow subtraction, keep identifiers out of every output and describe results as associations."),
]


def _merge_headers(contract: dict) -> None:
    """Show one header list per family: the union of real-export and fixture headers, in raw export form."""
    for fam in contract["person_query_metric_families"]:
        merged: list[str] = []
        for header in fam.pop("observed_real_export_headers") + fam.pop("sample_fixture_headers"):
            raw = header.replace("_", " ")
            if raw.lower() not in {h.lower() for h in merged}:
                merged.append(raw)
        fam["headers"] = merged


_merge_headers(CONTRACT)

# Single words that are also ordinary English are left unmarked in prose.
_GENERIC = {"Feature", "Language", "Model"}


def _code_names(contract: dict) -> list[str]:
    names = {"WeekStart"}
    for q in contract["queries"]:
        for f in q["files"]:
            names.add(f["file"])
            names.update(x["name"] for x in f["columns"])
    for fam in contract["person_query_metric_families"]:
        names.update(fam["headers"])
    return sorted(names - _GENERIC, key=len, reverse=True)


def backtick(text: str, names: list[str]) -> str:
    """Wrap known file and column names in backticks, longest first, without double wrapping."""
    import re
    pattern = re.compile(r"(?<![\w`.])(" + "|".join(re.escape(n) for n in names) + r")(?![\w`])")
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else pattern.sub(r"`\1`", p) for p in parts)


def _mark_code(contract: dict) -> None:
    names = _code_names(contract)
    contract["header_matching"] = backtick(contract["header_matching"], names)
    contract["target_grain"]["rule"] = backtick(contract["target_grain"]["rule"], names)
    for q in contract["queries"]:
        for f in q["files"]:
            for key in ("note", "file_pattern"):
                if key in f:
                    f[key] = backtick(f[key], names)
            for x in f["columns"]:
                x["note"] = backtick(x["note"], names)
            for g in f["column_groups"]:
                g["note"] = backtick(g["note"], names)
    for d in contract["derived_datasets"]:
        d["built_by"] = backtick(d["built_by"], names)
    for r in contract["relationships"]:
        r["purpose"] = backtick(r["purpose"], names)
    for m in contract["measures"]:
        m["person_week_rule"] = backtick(m["person_week_rule"], names)
    for g in contract["gates"]:
        g["rule"] = backtick(g["rule"], names)


_mark_code(CONTRACT)

ERRORS = [
    ("Row count grows after a join", "A fact file was joined before aggregation, or GitHub breakdowns were joined wide", "Aggregate to one row per PersonId and WeekStart first. Keep breakdowns as linked detail."),
    ("Duplicate key on PersonId, ServiceId and MetricDate", "The export includes SpendingPolicyId in the native key", "Use the four-part key, then run the additivity check before summing."),
    ("Credits double after joining metadata", "A metadata key is not unique", "Assert a unique PeopleHistoricalId or SpendingPolicyId before the join."),
    ("Most Consumption or Agent rows do not match the Person Query", "Week starts differ, daily dates were not mapped to WeekStart, or PersonId values come from different populations", "Align to the detected WeekStart. Check PersonId overlap. Never match on attributes."),
    ("No HR attributes for grouping", "PeopleMetaData.csv carries only PeopleHistoricalId and IsCopilotLicensed", "Take attributes from the Person Query with a stated assignment rule."),
    ("Network size totals look implausibly large", "Repeated monthly snapshots were summed", "Average snapshots and record the measurement window."),
    ("Total active days exceed seven per week", "Per-surface active days were added together", "Use Total Copilot active days for the total."),
    ("Weekly Consumption or GitHub rows match no Person Query weeks", "The weekly export starts on a different weekday from the Person Query", "Export the query daily and aggregate the days to the Person Query weeks. Weekly totals cannot be shifted to another week start."),
    ("Out-of-scope GitHub periods treated as zero", "The query's configured date window or population differs from the Person Query", "Apply the inactivity default only within the supplied query's scope. Keep unavailable or out-of-scope data unknown. Microsoft 365 licence evidence does not establish GitHub eligibility."),
]


def validate_contract(c: dict) -> None:
    query_ids = {q["id"] for q in c["queries"]}
    inputs = c["input_requirements"]
    assert set(inputs["required_query_ids"]) <= query_ids, "unknown required query"
    assert set(inputs["additional_query_ids"]) <= query_ids, "unknown additional query"
    assert not set(inputs["required_query_ids"]) & set(inputs["additional_query_ids"]), "overlapping query requirements"
    assert 1 <= inputs["minimum_additional_queries"] <= len(inputs["additional_query_ids"]), "invalid minimum query count"
    files = {f["id"]: f for q in c["queries"] for f in q["files"]}
    derived = {d["id"]: d for d in c["derived_datasets"]}
    nodes = set(files) | set(derived)
    gates = {g["id"] for g in c["gates"]}
    for d in derived.values():
        assert set(d["from_file_ids"]) <= set(files), d["id"]
        assert d["label"] and d["build_steps"], d["id"]
        assert all(1 <= step <= len(PROCEDURE) for step in d["build_steps"]), d["id"]
    for r in c["relationships"]:
        ends = r.get("from_ids", [r.get("from_id")]) + [r["to_id"]]
        assert all(x in nodes for x in ends), r["id"]
    for m in c["measures"]:
        f = files[m["file_id"]]
        assert all(name in {x["name"] for x in f["columns"]} or f["id"] == "person_query.person" for name in m["columns"]), m["id"]
    pq_headers = {h for fam in c["person_query_metric_families"] for h in fam["headers"]}
    for m in c["measures"]:
        if m["file_id"] == "person_query.person":
            assert set(m["columns"]) <= pq_headers, m["id"]
    for _, refs, _ in PROCEDURE:
        assert all(x.strip() in gates for x in refs.split(",")), refs
    ids = [m["id"] for m in c["measures"]] + [r["id"] for r in c["relationships"]] + list(gates) + list(nodes)
    assert len(ids) == len(set(ids)), "duplicate ids"


def e(text: str) -> str:
    return html.escape(text, quote=True)


def rich(text: str) -> str:
    """Escape text, then render backticked names as inline code."""
    import re
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", e(text))


def code_list(items) -> str:
    return ", ".join(f"<code>{e(i)}</code>" for i in items) if items else '<span class="none">None confirmed</span>'


def table(caption: str, headers: list[str], rows: list[list[str]], table_id: str = "",
          widths: list[int] | None = None) -> str:
    head = "".join(f'<th scope="col">{h}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    tid = f' id="{table_id}"' if table_id else ""
    cols = ("<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>") if widths else ""
    cls = ' class="fixed"' if widths else ""
    return (f'<div class="table-wrap"><table{tid}{cls}><caption>{caption}</caption>{cols}'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


ROLE_LABEL = {"key": "Key", "foreign_key": "Join key", "attribute": "Attribute", "measure": "Measure",
              "limit": "Limit", "indicator": "Indicator"}


def diagram() -> str:
    boxes = [
        (20, 30, "Consumption query", "4 files, day or week", "#0078D4", "#E8F2FB", False),
        (20, 150, "GitHub query", "5 files, day or week", "#5C2D91", "#F1EAF7", False),
        (20, 270, "Agent query", "2 files, week", "#B35C00", "#FFF4E0", False),
    ]
    parts = ['<svg viewBox="0 0 960 400" role="img" aria-labelledby="dgm-title dgm-desc" class="diagram">',
             '<title id="dgm-title">Join architecture</title>',
             '<desc id="dgm-desc">Consumption, GitHub and Agent files are validated and aggregated to person-week '
             'summaries, then left joined onto the weekly Person Query on PersonId and WeekStart. A cohort is '
             'selected after integration. GitHub breakdowns stay as linked detail.</desc>',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1E1E1E"/></marker></defs>']
    for x, y, title, sub, stroke, fill, dashed in boxes:
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        parts.append(f'<rect x="{x}" y="{y}" width="210" height="90" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{x + 16}" y="{y + 36}" class="dt">{title}</text>')
        parts.append(f'<text x="{x + 16}" y="{y + 62}" class="ds">{sub}</text>')
        parts.append(f'<rect x="270" y="{y + 15}" width="220" height="60" rx="8" fill="#FFFFFF" stroke="#505A64" stroke-width="1.5"/>')
        parts.append(f'<text x="284" y="{y + 40}" class="ds">Validate keys, check additivity</text>')
        parts.append(f'<text x="284" y="{y + 62}" class="ds">Bring to person-week</text>')
        parts.append(f'<path d="M230,{y + 45} L268,{y + 45}" stroke="#1E1E1E" stroke-width="2" marker-end="url(#arr)"{dash}/>')
        parts.append(f'<path d="M490,{y + 45} L548,{min(max(y + 45, 120), 250)}" stroke="#1E1E1E" stroke-width="2" marker-end="url(#arr)"{dash}/>')
    parts += [
        '<rect x="550" y="100" width="200" height="170" rx="10" fill="#FEE2E2" stroke="#B91C1C" stroke-width="2.5"/>',
        '<text x="566" y="136" class="dt">Person query</text>',
        '<text x="566" y="162" class="ds">Weekly spine: one row per</text>',
        '<text x="566" y="182" class="ds">PersonId and WeekStart</text>',
        '<text x="566" y="214" class="ds">LEFT JOIN each summary,</text>',
        '<text x="566" y="234" class="ds">one-to-one</text>',
        '<path d="M750,185 L788,185" stroke="#1E1E1E" stroke-width="2" marker-end="url(#arr)"/>',
        '<rect x="790" y="120" width="150" height="130" rx="10" fill="#E9F5E9" stroke="#107C10" stroke-width="2"/>',
        '<text x="804" y="154" class="dt">Select cohort</text>',
        '<text x="804" y="182" class="ds">Then analyse and</text>',
        '<text x="804" y="202" class="ds">publish groups of 10</text>',
        '<text x="804" y="222" class="ds">or more people</text>',
        '<rect x="550" y="300" width="390" height="70" rx="10" fill="#F3F4F6" stroke="#505A64" stroke-width="1.5" stroke-dasharray="4 4"/>',
        '<text x="566" y="330" class="ds">Linked detail, outside the panel: service,</text>',
        '<text x="566" y="352" class="ds">spending policy, agent, GitHub breakdowns</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def file_section(f: dict) -> str:
    rows = [[f"<code>{e(x['name'])}</code>", e(x["type"]), e(ROLE_LABEL[x["role"]]), rich(x["note"])] for x in f["columns"]]
    rows += [[f'<em>{e(g["name"])}</em>', "varies", "Column group", rich(g["note"])] for g in f["column_groups"]]
    tags = []
    if not f["required_for_layout_match"]:
        tags.append("optional in the export")
    if not f["required_for_core_panel"]:
        tags.append("outside the core panel")
    tag = f' <span class="tag">{e(", ".join(tags))}</span>' if tags else ""
    meta = (f'<p class="meta"><strong>Grain:</strong> {e(f["grain"])}. '
            f'<strong>Native key:</strong> {code_list(f["native_key"])}.'
            + (f' {rich(f["file_pattern"])}' if f.get("file_pattern") else "")
            + (f' {rich(f["note"])}' if f.get("note") else "") + "</p>")
    return (f'<h4 id="{e(f["id"])}"><code>{e(f["file"])}</code>{tag}</h4>{meta}'
            + table(f"Columns in {e(f['file'])}", ["Column", "Type", "Role", "Notes"], rows, widths=[38, 12, 13, 37]))


def build_html() -> str:
    c = CONTRACT
    validate_contract(c)
    names = _code_names(c)
    files = {f["id"]: f for q in c["queries"] for f in q["files"]}
    derived = {d["id"]: d for d in c["derived_datasets"]}

    def dataset_link(id_):
        label = f"<code>{e(files[id_]['file'])}</code>" if id_ in files else e(derived[id_]["label"])
        return f'<a href="#{e(id_)}">{label}</a>'

    def files_cell(q):
        n = len(q["files"])
        listing = ", ".join(dataset_link(f["id"]) + (" (optional)" if not f["required_for_layout_match"] else "")
                            for f in q["files"])
        return f'{n} CSV {"file" if n == 1 else "files"}<br>{listing}'

    def period_cell(q):
        return "Week (Recommended)" + (", Day (default)" if q["default_period"] == "day" else "")

    query_rows = [[f'<a href="#{q["id"]}">{e(q["name"])}</a>', files_cell(q), e(period_cell(q)),
                   e({"spine": "Spine (left side of every join)", "fact": "Joined onto the spine"}[q["role"]])]
                  for q in c["queries"]]

    sections = []
    for q in c["queries"]:
        path = " &gt; ".join(f"<strong>{e(part)}</strong>" for part in q["recommended_path"])
        block = [f'<section id="{q["id"]}" class="query-section"><h3 id="{q["id"]}-heading">{e(q["name"])}</h3>',
                 f'<p class="query-access">Recommended route: {path}.</p>']
        if q.get("access_context"):
            block.append(f'<p>{rich(backtick(q["access_context"], names))}</p>')
        if q.get("alternative_path"):
            dashboard, button = q["alternative_path"]
            block.append(f'<p>You can also use the <strong>{e(button)}</strong> button in the {e(dashboard)}.</p>')
        block.append(f'<p>{e(q["cadence"])}</p>')
        breakdowns = [f for f in q["files"] if f["id"].startswith("github.by_")]
        block += [file_section(f) for f in q["files"] if f not in breakdowns]
        if breakdowns:
            rows = [[f'<span id="{e(f["id"])}"><code>{e(f["file"])}</code></span>', code_list(f["native_key"]),
                     code_list([x["name"] for x in f["columns"] if x["role"] == "measure"])] for f in breakdowns]
            block.append('<h4 id="github.breakdowns">Breakdown files <span class="tag">part of the export, outside the core panel</span></h4>')
            block.append('<p class="meta">Each adds one or two dimensions to the person-period grain, so the four files '
                         'do not share a grain with each other or with the activity file. Keep them as linked detail '
                         '(<a href="#G11">G11</a>).</p>')
            block.append(table("GitHub breakdown files", ["File", "Native key", "Measure"], rows, widths=[42, 33, 25]))
        if q["id"] == "person_query":
            fam_rows = [[e(m["family"]), e(m["kind"].replace("_", " ")),
                         (code_list(m["headers"]) if m["headers"] else '<span class="none">Inspect the export</span>')
                         + (f'<br><span class="meta">{rich(m["note"])}</span>' if m.get("note") else "")]
                        for m in c["person_query_metric_families"]]
            block.append('<h4 id="person_query.metric_families">Metric families</h4>')
            block.append("<p>Metric columns are chosen when the query is set up, so each list is representative "
                         "rather than complete, and wording or capitalisation can vary by query version. "
                         f"{rich(c['header_matching'])} See the <a href=\"https://learn.microsoft.com/en-us/viva/insights/"
                         "advanced/reference/metrics\">metric definitions on Microsoft Learn</a> for meanings.</p>")
            block.append(table("Person Query metric families", ["Family", "Kind", "Headers"],
                               fam_rows, widths=[18, 14, 68]))
        block.append("</section>")
        sections.append("\n".join(block))

    key_rows = [
        ["<code>PersonId</code>", "Every person-level file in all four queries", "Pseudonymised person and the panel join key. Verify overlap between sources (<a href=\"#G15\">G15</a>)."],
        ["<code>MetricDate</code>", "Every fact file and the Person Query", "Period start. Weekly in the Person and Agent queries. Daily by default in Consumption and GitHub, or weekly when grouped by week at setup. Real exports and the sample fixtures use Sunday-start weeks, while some documentation describes Monday, so detect it (<a href=\"#G03\">G03</a>, <a href=\"#G03b\">G03b</a>)."],
        ["<code>WeekStart</code>", "Chosen name in this guide", rich(backtick(c["naming_conventions"]["week_start"], names))],
        ["<code>PeopleHistoricalId</code>", "Both Consumption fact files and PeopleMetaData.csv", "Historical person record. Read it from each fact row. It cannot be derived from PersonId."],
        ["<code>ServiceId</code>", "PersonM365CreditsMetrics.csv", "Part of the native key. Keep service detail as linked detail."],
        ["<code>SpendingPolicyId</code>", "PersonM365CreditsMetrics.csv and M365SpendingPolicyMetaData.csv", "Part of the native key. Unique keys do not establish additivity (<a href=\"#G05\">G05</a>)."],
        ["<code>AgentId</code>", "Both Agent files", "Stable agent key. Agent names can change."],
        ["<code>Feature</code>, <code>Language</code>, <code>Model</code>", "GitHub breakdown files", "Extra dimensions that change the grain (<a href=\"#G11\">G11</a>)."],
    ]

    metadata_rows = [[dataset_link(r["from_id"]), dataset_link(r["to_id"]), code_list(r["on"]),
                      f'<span id="{e(r["id"])}">{rich(r["purpose"])}</span>']
                     for r in c["relationships"] if r["cardinality"] == "many_to_one"]
    panel_rows = [[dataset_link(r["from_id"]), code_list(r["on"]),
                   f'<span id="{e(r["id"])}">Keep every Person Query row. Match at most one summary row per person and week.</span>']
                  for r in c["relationships"] if r["cardinality"] == "one_to_one"]
    detail_rows = [[", ".join(dataset_link(id_) for id_ in r["from_ids"]), rich(r["purpose"]),
                    f'<span id="{e(r["id"])}">Use <code>PersonId</code> and the aligned week to relate detail to the panel. '
                    'Retain each file&#x27;s additional dimensions.</span>']
                   for r in c["relationships"] if r["cardinality"] == "linked_detail"]

    def step_link(step):
        return f'<a href="#step-{step}">Step {step}: {e(PROCEDURE[step - 1][0])}</a>'

    derived_rows = [[f'<span id="{e(d["id"])}">{e(d["label"])}</span>',
                     ", ".join(dataset_link(id_) for id_ in d["from_file_ids"]),
                     rich(d["built_by"]) + "<br>" + "<br>".join(step_link(step) for step in d["build_steps"])]
                    for d in c["derived_datasets"]]
    measure_rows = [[m["id"], code_list(m["columns"]), dataset_link(m["file_id"]), e(m["kind"].replace("_", " ")),
                     rich(m["person_week_rule"])] for m in c["measures"]]

    def gate_links(refs: str) -> str:
        return ", ".join(f'<a href="#{r.strip()}">{r.strip()}</a>' for r in refs.split(","))

    proc_items = "".join(f'<li id="step-{i}"><strong>{e(t)}</strong> <span class="gate-ref">{gate_links(g)}</span><br>{rich(backtick(d, names))}</li>'
                         for i, (t, g, d) in enumerate(PROCEDURE, 1))
    gate_rows = [[f'<span id="{g["id"]}">{g["id"]}</span>', f'<span class="level {g["level"].lower()}">{g["level"]}</span>',
                  rich(g["rule"])] for g in c["gates"]]
    miss_rows = [[f"<code>{m['id']}</code>", e(m["meaning"])] for m in c["missingness_states"]]
    err_rows = [[rich(backtick(a, names)), rich(backtick(b, names)), rich(backtick(d, names))] for a, b, d in ERRORS]
    contract_json = json.dumps(c, indent=2, ensure_ascii=False)
    script_json = contract_json.replace("</", "<\\/")
    n_gates = len(c["gates"])

    return f"""---
layout: page
title: "Joining Viva Insights queries"
eyebrow: "Advanced analytics"
description: "Combine Person, Consumption, GitHub and Agent query exports into a person-week panel for analysis and reporting."
permalink: /query-join-guide/
css: "/assets/css/query-join-guide.css"
lang: en-GB
---
<!-- Generated by scripts/build_join_guide.py. Edit the builder, then regenerate. -->
<div class="query-join-guide">
<h2 id="overview">Overview</h2>
<p>This tutorial walks through how to join the four custom queries in Viva Insights (Person, Consumption, GitHub and Agent) to bring together behavioural and AI usage metrics, alongside organisational attributes, in a single panel data file for analysis and reporting. The resulting file has one row per person per week.</p>
<p>This is the current reference for these cross-query joins and takes precedence over older schema and join guidance in this repository. Historical layouts and synthetic fixture manifests remain available for reproducing earlier examples. This guide does not extend the existing dashboard runners' supported inputs.</p>
<p>You need the <strong>Person Query</strong> plus at least one of the Consumption, GitHub or Agent queries. We recommend using all four for a broader view of a population's ways of working and AI usage across the products and activities they measure.</p>
<p>All four queries are available under <strong>Create analysis &gt; Customized query</strong> in <a href="https://analysis.insights.cloud.microsoft/">Viva Insights</a>, where you can adjust parameters such as date grouping and population filters. We recommend this route because it offers more customisation parameters than dashboard exports.</p>
<p>We recommend exporting <strong>all selected queries grouped by Week</strong>, with the same week start. Consumption and GitHub default to Day, so change their grouping at setup. Weekly exports avoid daily-to-weekly aggregation and sparse-day handling, although service, spending-policy and agent rules still apply. If you need daily detail or must align different week starts, export those queries by Day and aggregate to the Person Query weeks.</p>
<p>Record the configured grouping for every query before validating its dates. Sparse observed dates alone cannot distinguish daily from weekly or monthly data.</p>
{table("The four queries", ["Query", "Files", "Period", "Role"], query_rows, "query-summary", widths=[18, 40, 22, 20])}

<h2 id="architecture">Join architecture</h2>
<p>The weekly Person Query is the spine. Validate each other selected query, bring it to one row per person per week, and left join it onto the Person Query. The diagram shows all four queries. Follow the paths for the queries you have, then select your analysis cohort from the joined population.</p>
{diagram()}

<h2 id="queries">Query schemas</h2>
<p>Export layouts vary between tenants and over time, so check each supplied export against the schemas below. {e(c["query_setup_recommendation"])}</p>
{''.join(sections)}

<h2 id="keys">Join keys</h2>
{table("Keys and where they appear", ["Key", "Appears in", "Use"], key_rows, widths=[22, 30, 48])}

<h2 id="derived">Summaries you create</h2>
<p>The names below describe intermediate tables you create in your script. They are not extra downloads or required object names. Choose your own names for data frames, SQL views or other objects. Each summary, and the final combined panel, has one row per person per week.</p>
<p>{rich(backtick(c["naming_conventions"]["week_start"], names))} The source filenames below link to their schemas. The step links take you to the numbered <a href="#procedure">Join procedure</a> later on this page.</p>
{table("Preparing summaries and the final panel", ["Table you create", "Source export", "What to do"], derived_rows, widths=[22, 32, 46])}

<h2 id="relationships">How to connect the data</h2>
<h3>Attach metadata before summarising</h3>
<p>In <a href="#step-4">step 4: Validate each file</a>, match each fact row to at most one metadata row using the exported key shown below. First check that the key is unique in the metadata file. Report unmatched keys and ensure the join does not multiply fact rows.</p>
{table("Metadata lookups on exported columns", ["Fact file", "Metadata file", "Match on", "Use"], metadata_rows, widths=[29, 27, 18, 26])}
<h3>Add summaries to the Person Query</h3>
<p>In <a href="#step-7">step 7: Join onto the spine</a>, start with the weekly Person Query on the left and add each summary on the right with a left join. The summary names below refer to the tables you created above. Match on <code>PersonId</code> and the aligned week-start date, called <code>WeekStart</code> in this guide. Both sides must have at most one row per person per week.</p>
{table("Left joins that preserve the Person Query rows", ["Summary to add (right side)", "Match on", "Expected result"], panel_rows, widths=[30, 25, 45])}
<h3>Keep GitHub breakdowns separate</h3>
<p>These files have multiple rows per person and period. Joining them directly to the person-week panel would repeat panel rows and inflate totals. Keep each as a separate detail table, aligning its dates to the Person Query weeks when needed.</p>
{table("Detail outside the person-week panel", ["Source files", "Treatment", "How to relate them"], detail_rows, widths=[40, 25, 35])}

<h2 id="measures">Aggregation rules</h2>
<p>Apply these rules to panel-bound summaries and when rolling person-weeks up to longer periods. Weekly grouping removes daily aggregation, but does not make every measure additive across services, spending policies or agents. GitHub breakdowns remain separate, retaining their dimensions.</p>
{table("Person-week rule for each measure", ["ID", "Columns", "File", "Kind", "Person-week rule"], measure_rows, widths=[6, 28, 19, 12, 35])}

<h2 id="procedure">Join procedure</h2>
<p>Follow the procedure for the queries you have. Apply source-specific checks only when that source is supplied, for example <a href="#step-5">step 5: Check policy additivity</a> applies to Consumption. Record omitted queries as unavailable, without treating their absence as evidence of non-use.</p>
<ol class="procedure">{proc_items}</ol>

<h2 id="gates">Gates</h2>
{table(f"{n_gates} checks before and during the join", ["ID", "Level", "Rule"], gate_rows, widths=[7, 10, 83])}

<h2 id="missingness">Missing values</h2>
<p><strong>Default:</strong> assume supplied Viva Insights queries are complete. {e(c["coverage_assumption"]["missing_activity_rows"])}</p>
<p><strong>Scope:</strong> {e(c["coverage_assumption"]["scope"])} {e(c["coverage_assumption"]["exceptions"])}</p>
<p>{e(c["coverage_assumption"]["eligibility"])} Every joined measure should carry a state below, judged separately for each product. Use <code>absent_inactive_zero</code> for omitted activity rows under this default. Do not fill null cells, rolling indicators or limits merely because omitted activity is zero.</p>
{table("Missingness states", ["State", "Meaning"], miss_rows, widths=[25, 75])}

<h2 id="errors">Common errors</h2>
{table("Symptoms, causes and fixes", ["Symptom", "Likely cause", "Fix"], err_rows, widths=[28, 36, 36])}

<h2 id="for-agents">For coding agents</h2>
<details class="agent-box"><summary>How coding agents should use this page</summary>
<ul>
<li>This page is designed to be read by coding agents as well as people. The complete contract is embedded as JSON in the script element with id <code>join-contract</code> and type <code>application/json</code>. Parse it instead of scraping the tables. The tables are generated from the same JSON.</li>
<li>Use <code>input_requirements</code> to check that the Person Query and at least one other query are supplied. Apply file requirements, joins and source-specific gates only to the selected queries. Record the unavailable queries explicitly.</li>
<li>Read <code>coverage_assumption</code> before handling absent rows. Assume complete Viva Insights exports within their configured scope, with omitted inactive person-days contributing zero activity. This is separate from product eligibility and does not apply to unavailable queries or null cells.</li>
<li>Every file, derived dataset, relationship (<code>J1</code> to <code>J9</code>), measure (<code>M01</code> to <code>M15</code>) and gate has a stable ID. These are contract references, not exported columns or required script object names. Resolve file IDs to <code>file</code> and derived IDs to their <code>label</code>, then map them to your own objects. Cite contract IDs in logs and summaries.</li>
<li><code>WeekStart</code> is a chosen name for the aligned week date. Map it to your implementation, including <code>MetricDate</code> if you keep that header. See <code>naming_conventions</code> in the contract. <code>build_steps</code> refers to the numbered Join procedure on this page.</li>
<li><code>columns</code> lists literal headers to find. <code>column_groups</code> describes configurable sets whose headers must be inspected in each export.</li>
<li><strong>MUST</strong> gates block the join when they fail. Stop and report the gate ID, a bounded count and the next required input. <strong>SHOULD</strong> gates require a recorded decision.</li>
<li>Never print identifiers or employee rows. Report counts only.</li>
<li>If a supplied export does not match a layout here, do not adapt the contract silently. Report the difference (<a href="#G01">G01</a>).</li>
</ul>
</details>
<p><a href="{{{{ '/assets/data/query-join-contract.json' | relative_url }}}}">Download the JSON contract</a>.</p>\n<details><summary>Show the JSON contract</summary><pre>{e(contract_json)}</pre></details>
<script type="application/json" id="join-contract">
{script_json}
</script>

<footer>
<p>Sources: <a href="https://learn.microsoft.com/en-us/viva/insights/advanced/reference/metrics">Viva Insights metric definitions</a>, <a href="https://learn.microsoft.com/en-us/viva/insights/advanced/analyst/ai-cost-query">Consumption query documentation</a>, and the query contracts and synthetic sample exports in <a href="https://github.com/microsoft/viva-insights-sample-code">microsoft/viva-insights-sample-code</a>. Contract version {e(c["version"])}, {DATE}.</p>
</footer>
</div>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated files are stale.")
    args = parser.parse_args()
    outputs = {
        PAGE: build_html(),
        CONTRACT_PATH: json.dumps(CONTRACT, indent=2, ensure_ascii=False) + "\n",
    }
    stale = []
    for path, content in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(path.relative_to(ROOT))
    if stale:
        parser.exit(1, "Regenerate with python scripts/build_join_guide.py: " + ", ".join(stale) + "\n")


if __name__ == "__main__":
    main()
