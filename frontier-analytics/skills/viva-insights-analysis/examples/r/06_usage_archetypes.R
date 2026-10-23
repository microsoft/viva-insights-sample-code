# Example 6 - Experimental usage archetypes (reference rule v1).
#
# EXPERIMENTAL, SYNTHETIC DATA ONLY. The script generates its own seeded,
# synthetic person-week panel. It reads no files and contains no real export,
# customer data or published result. The printed numbers illustrate the
# mechanics only and say nothing about real Copilot use. See
# reference/usage-archetypes.md for the rule, evidence states, context record
# and publication rules.
#
# Steps:
# 1. Generate a synthetic 12-week person-week panel (Sunday-start weeks).
# 2. Assign every person a named evidence state before any label.
# 3. For full profiles, assign Consistency (C/E), Surface mix (S/B, integer
#    rule on group-weeks) and Metered use (L/I, median split of positive M365
#    credits, ties Lean).
# 4. Print the context record, a privacy-safe summary that withholds groups
#    under 10, and leave-one-week and split-half sensitivity diagnostics.
#
# The Power / Habitual / Novice / Low / Non-user ladder (example 3) is
# unchanged and should be reported alongside these descriptive labels. Labels
# describe a period. Never use them to rank or monitor individuals.
#
# The random generator is a minimal-standard (Park-Miller) generator written
# out identically in python/06_usage_archetypes.py, so both scripts build the
# same synthetic panel and the same classifications for the same seed.
# Base R only.

SEED <- 20250105
N_PEOPLE <- 600
N_WEEKS <- 12
WINDOW_START <- as.Date("2025-01-05")  # a Sunday
RULE_VERSION <- "usage archetypes, reference rule v1 (experimental)"
MIN_GROUP_SIZE <- 10
CONSISTENT_MIN_WEEKS <- 9

# Synthetic stand-ins for real export fields:
#   copilot_enabled_days       -> Person Query `Total Copilot enabled days`
#   copilot_actions            -> Person Query `Total Copilot actions taken`
#   chat/work_apps/cowork_*    -> tool-specific Person Query indicators
#   m365_credits               -> Consumption `Total Copilot Credits used`,
#                                 aggregated from daily facts to Sunday weeks
#   github_credits             -> `Total GitHub AI Credits used`
#   multi_policy_positive_keys -> person-service-date keys with more than one
#                                 positive spending-policy row
GROUPS <- c("chat", "work_apps", "cowork", "github")
GROUP_LABELS <- c(
  chat = "Chat and the Copilot app",
  work_apps = "Copilot in work apps",
  cowork = "Cowork",
  github = "GitHub (positive GitHub credits)"
)

# Synthetic analysis decisions recorded in the context record. In a real
# application these need evidence from the export owner.
COVERAGE_CONFIRMED <- c(A = TRUE, B = FALSE)  # per synthetic export partition
POLICY_ADDITIVITY_CONFIRMED <- FALSE

STATES <- c(
  "Full profile",
  "Confirmed zero M365 credits (No M365 metered use)",
  "No positive observed credits, coverage unconfirmed",
  "Unresolved credit additivity",
  "Incomplete evidence or eligibility",
  "Outside the analysis cohort"
)
grid <- expand.grid(m = c("L", "I"), s = c("S", "B"), c = c("C", "E"),
                    stringsAsFactors = FALSE)
PROFILE_CODES <- paste(grid$c, grid$s, grid$m, sep = "-")
PROFILE_LABELS <- paste(
  c(C = "Consistent", E = "Episodic")[grid$c],
  c(S = "Specialised", B = "Broad")[grid$s],
  c(L = "Lean", I = "Intensive")[grid$m],
  sep = ", "
)

# Park-Miller generator. Exact in R doubles and Python integers.
make_generator <- function(seed) {
  state <- seed %% 2147483647
  if (state == 0) state <- 1
  function(k) {
    out <- numeric(k)
    for (j in seq_len(k)) {
      state <<- (16807 * state) %% 2147483647
      out[j] <- state / 2147483647
    }
    out
  }
}

# Returns a list(roster, pw) for a SYNTHETIC population.
generate_panel <- function(seed) {
  draw <- make_generator(seed)
  n <- N_PEOPLE * N_WEEKS
  pid <- character(N_PEOPLE); partition <- character(N_PEOPLE)
  in_cohort <- logical(N_PEOPLE)
  w_pid <- character(n); w_week <- integer(n); enabled <- integer(n)
  actions_v <- numeric(n); chat_v <- numeric(n); apps_v <- numeric(n)
  cowork_v <- numeric(n); github_v <- numeric(n); m365_v <- numeric(n)
  multi_v <- integer(n)
  r <- 0
  for (i in seq_len(N_PEOPLE)) {
    u <- draw(20)
    pid[i] <- sprintf("SYN%04d", i)
    partition[i] <- if (u[1] < 0.7) "A" else "B"
    in_cohort[i] <- u[2] >= 0.03
    short_week <- if (u[4] < 0.05) floor(u[3] * N_WEEKS) else -1
    missing_week <- if (u[6] < 0.02) floor(u[5] * N_WEEKS) else -1
    p_active <- if (u[8] < 0.12) 0.35 + 0.40 * u[7] else 0.85 + 0.15 * u[7]
    p_group <- c(0.88 + 0.12 * u[9], 0.88 + 0.12 * u[10], 0.15 + 0.60 * u[11],
                 if (u[13] < 0.30) 0.20 + 0.75 * u[12] else 0.0)
    if (u[14] < 0.15) {
      focus <- floor(u[15] * 4) + 1
      p_group <- ifelse(seq_along(p_group) == focus, 0.97, 0.3 * p_group)
    }
    zero_credits <- u[16] < 0.08
    m365_scale <- 40 + floor(960 * u[17])
    overlap_week <- if (u[18] < 0.03) floor(u[19] * N_WEEKS) else -1
    github_scale <- 5 + floor(195 * u[20])
    for (w in 0:(N_WEEKS - 1)) {
      v <- draw(8)
      r <- r + 1
      active <- v[1] < p_active
      flags <- active & (v[2:5] < p_group)
      group_actions <- ifelse(flags[1:3], 1 + floor(20 * v[2:4]), 0)
      m365 <- if (zero_credits || !active) 0 else 1 + floor(v[6] * m365_scale)
      w_pid[r] <- pid[i]
      w_week[r] <- w
      enabled[r] <- if (w == short_week) 4L else 7L
      actions_v[r] <- if (w == missing_week) NA else
        sum(group_actions) + (if (active) 1 + floor(10 * v[8]) else 0)
      chat_v[r] <- group_actions[1]
      apps_v[r] <- group_actions[2]
      cowork_v[r] <- group_actions[3]
      github_v[r] <- if (flags[4]) 1 + floor(v[7] * github_scale) else 0
      m365_v[r] <- m365
      multi_v[r] <- if (w == overlap_week && m365 > 0) 1L else 0L
    }
  }
  list(
    roster = data.frame(PersonId = pid, partition = partition,
                        in_cohort = in_cohort, stringsAsFactors = FALSE),
    pw = data.frame(
      PersonId = w_pid, MetricDate = WINDOW_START + 7 * w_week, week = w_week,
      copilot_enabled_days = enabled, copilot_actions = actions_v,
      chat_actions = chat_v, work_apps_actions = apps_v,
      cowork_actions = cowork_v, github_credits = github_v,
      m365_credits = m365_v, multi_policy_positive_keys = multi_v,
      stringsAsFactors = FALSE
    )
  )
}

# Reference rule v1 with integers: share >= 40% and lead >= 5 points.
surface_mix <- function(counts) {
  total <- sum(counts)
  if (total == 0) return(NA_character_)
  top <- sort(counts, decreasing = TRUE)
  c1 <- top[1]
  c2 <- top[2]
  if (sum(counts == c1) == 1 && 5 * c1 >= 2 * total && 20 * (c1 - c2) >= total) {
    "Specialised"
  } else {
    "Broad"
  }
}

# Withhold cells under MIN_GROUP_SIZE. If exactly one cell is withheld and the
# total is published, also withhold the next smallest cell so the first cannot
# be recovered by subtraction. Real releases must check all related outputs
# jointly rather than one table at a time.
protect <- function(counts) {
  shown <- ifelse(counts >= MIN_GROUP_SIZE, counts, NA)
  if (sum(is.na(shown)) == 1) {
    visible <- which(!is.na(shown))
    if (length(visible) > 0) {
      shown[visible[order(shown[visible], visible)][1]] <- NA
    }
  }
  ifelse(is.na(shown), "withheld", as.character(shown))
}

share <- function(numerator, denominator) {
  if (numerator < MIN_GROUP_SIZE || denominator < MIN_GROUP_SIZE) {
    return("withheld (fewer than 10)")
  }
  sprintf("%d of %d (%.1f%%)", as.integer(numerator), as.integer(denominator),
          100 * numerator / denominator)
}

# 1. Synthetic panel ------------------------------------------------------------
panel <- generate_panel(SEED)
roster <- panel$roster
pw <- panel$pw
pw <- pw[order(pw$PersonId, pw$week), ]

# 2. Person-week flags ----------------------------------------------------------
pw$chat <- pw$chat_actions > 0
pw$work_apps <- pw$work_apps_actions > 0
pw$cowork <- pw$cowork_actions > 0
pw$github <- pw$github_credits > 0
pw$active <- (ifelse(is.na(pw$copilot_actions), 0, pw$copilot_actions) > 0) |
  pw$chat | pw$work_apps | pw$cowork | pw$github

person <- factor(pw$PersonId, levels = sort(roster$PersonId))
by_person <- function(x, f) as.vector(tapply(x, person, f))
people <- data.frame(
  PersonId = levels(person),
  active_weeks = by_person(pw$active, sum),
  chat = by_person(pw$chat, sum),
  work_apps = by_person(pw$work_apps, sum),
  cowork = by_person(pw$cowork, sum),
  github = by_person(pw$github, sum),
  m365_total = by_person(pw$m365_credits, sum),
  min_enabled_days = by_person(pw$copilot_enabled_days, min),
  missing_indicators = by_person(pw$copilot_actions, function(s) sum(is.na(s))),
  multi_policy_keys = by_person(pw$multi_policy_positive_keys, sum),
  stringsAsFactors = FALSE
)
people <- merge(people, roster, by = "PersonId", sort = TRUE)
people$group_weeks <- rowSums(people[, GROUPS])

# 3. Evidence states before any label -------------------------------------------
evidence_state <- function(p) {
  if (!p$in_cohort) return(STATES[6])
  if (p$min_enabled_days < 7 || p$missing_indicators > 0) return(STATES[5])
  if (p$multi_policy_keys > 0 && !POLICY_ADDITIVITY_CONFIRMED) return(STATES[4])
  if (p$m365_total == 0) {
    return(if (COVERAGE_CONFIRMED[[p$partition]]) STATES[2] else STATES[3])
  }
  if (p$group_weeks == 0) return(STATES[5])
  STATES[1]
}
people$state <- vapply(seq_len(nrow(people)),
                       function(k) evidence_state(people[k, ]), character(1))
full <- people$state == STATES[1]

# 4. Labels for full profiles only ----------------------------------------------
reference_median <- median(people$m365_total[full])
people$consistency <- NA_character_
people$surface_mix <- NA_character_
people$metered_use <- NA_character_
people$consistency[full] <- ifelse(people$active_weeks[full] >= CONSISTENT_MIN_WEEKS,
                                   "Consistent", "Episodic")
people$surface_mix[full] <- apply(as.matrix(people[full, GROUPS]), 1, surface_mix)
people$metered_use[full] <- ifelse(people$m365_total[full] <= reference_median,
                                   "Lean", "Intensive")
people$profile <- NA_character_
people$profile[full] <- paste(substr(people$consistency[full], 1, 1),
                              substr(people$surface_mix[full], 1, 1),
                              substr(people$metered_use[full], 1, 1), sep = "-")

# 5. Context record -------------------------------------------------------------
cat("SYNTHETIC DATA. Illustrates the mechanics of an EXPERIMENTAL framework.\n")
cat("\n=== Context record ===\n")
cat(sprintf("Rule version: %s\n", RULE_VERSION))
cat(sprintf("Window: %s to %s (%d Sunday-start weeks)\n", WINDOW_START,
            WINDOW_START + 7 * N_WEEKS - 1, N_WEEKS))
cat("Cohort rule: synthetic roster members present in all required queries\n")
cat("Eligibility: 7 enabled days in every week, no missing required indicators\n")
cat("Tool groups (person-weeks with positive observations, capture assumption):\n")
cohort_pw <- pw[pw$PersonId %in% people$PersonId[people$in_cohort], ]
for (g in GROUPS) {
  status <- if (sum(cohort_pw[[g]]) > 0) "observed" else
    "NO POSITIVE OBSERVATIONS, report as such"
  cat(sprintf("  %-36s %s, synthetic capture assumed complete\n",
              GROUP_LABELS[[g]], status))
}
cat(sprintf("Credit reference: full-profile population, median M365 credits %.1f, ties Lean\n",
            reference_median))
cat("Coverage confirmation: synthetic partition A confirmed, partition B unconfirmed\n")
cat(sprintf("Policy-row additivity confirmed: %s\n",
            if (POLICY_ADDITIVITY_CONFIRMED) "True" else "False"))
cat("GitHub credits flag the GitHub group only and are never added to M365 credits\n")

# 6. Privacy-safe summaries -----------------------------------------------------
cat(sprintf("\n=== Evidence states (groups under %d withheld) ===\n", MIN_GROUP_SIZE))
state_counts <- vapply(STATES, function(s) sum(people$state == s), numeric(1))
state_shown <- protect(state_counts)
for (k in seq_along(STATES)) {
  cat(sprintf("  %-52s %9s\n", STATES[k], state_shown[k]))
}
cat(sprintf("  %-52s %9d\n", "Supplied synthetic roster", nrow(people)))

cat(sprintf("\n=== Profiles among full profiles (groups under %d withheld) ===\n",
            MIN_GROUP_SIZE))
profile_counts <- vapply(PROFILE_CODES, function(code) sum(people$profile == code, na.rm = TRUE),
                         numeric(1))
profile_shown <- protect(profile_counts)
for (k in seq_along(PROFILE_CODES)) {
  cat(sprintf("  %s  %-36s %9s\n", PROFILE_CODES[k], PROFILE_LABELS[k], profile_shown[k]))
}
cat(sprintf("  %-42s %9d\n", "Full profiles", sum(full)))
cat("  'withheld' means fewer than 10 people, or withheld so that another cell\n")
cat("  cannot be recovered by subtraction. Withholding itself conveys information.\n")

# 7. Sensitivity diagnostics (Surface mix) --------------------------------------
flags_by_person <- lapply(split(pw[, GROUPS], person),
                          function(d) matrix(as.integer(as.matrix(d)), ncol = length(GROUPS)))
people$loo_surface_changed <- NA
split_eligible <- 0
split_disagree <- 0
for (k in which(full)) {
  weeks <- flags_by_person[[people$PersonId[k]]]
  base <- people$surface_mix[k]
  changed <- FALSE
  for (drop in seq_len(N_WEEKS)) {
    if (!identical(surface_mix(colSums(weeks[-drop, , drop = FALSE])), base)) {
      changed <- TRUE
      break
    }
  }
  people$loo_surface_changed[k] <- changed
  first <- colSums(weeks[1:6, , drop = FALSE])
  last <- colSums(weeks[7:12, , drop = FALSE])
  if (sum(first) > 0 && sum(last) > 0) {
    split_eligible <- split_eligible + 1
    split_disagree <- split_disagree +
      !identical(surface_mix(first), surface_mix(last))
  }
}

specialised <- full & people$surface_mix %in% "Specialised"
cat("\n=== Sensitivity: Surface mix (stability diagnostics rather than observed change) ===\n")
cat("Label changed in at least one leave-one-week recalculation:\n")
cat(sprintf("  all full profiles: %s\n",
            share(sum(people$loo_surface_changed[full]), sum(full))))
cat(sprintf("  Specialised:       %s\n",
            share(sum(people$loo_surface_changed[specialised]), sum(specialised))))
cat("First versus last six weeks, label disagreement among people with\n")
cat(sprintf("group-weeks in both halves: %s\n", share(split_disagree, split_eligible)))
