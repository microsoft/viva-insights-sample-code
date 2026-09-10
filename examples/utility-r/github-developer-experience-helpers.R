# Developer Experience and Copilot: extended illustrative synthetic schema.
# Source from the Rmd. No input data files or live systems are read.
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)

SEED <- 20260910L
MIN_GROUP_N <- 10L
BASELINE_WEEKS <- 8L
AFTER_HOURS_CONVENTION <- 3 # illustrative hours/person/week, not a health threshold
PERSISTENCE_WEEKS <- 4L    # at or above the convention in at least 4 of 8 weeks
stopifnot(AFTER_HOURS_CONVENTION >= 0, PERSISTENCE_WEEKS >= 2,
          PERSISTENCE_WEEKS <= BASELINE_WEEKS)
set.seed(SEED)
options(OutDec = '.')
weeks <- seq(as.Date('2026-01-04'), by = 'week', length.out = 26)
baseline_start <- tail(weeks, BASELINE_WEEKS)[1]
period_end <- max(weeks) + 6
period_label <- paste(format(baseline_start, '%d %b %Y'), 'to', format(period_end, '%d %b %Y'))
history_label <- paste(format(min(weeks), '%d %b %Y'), 'to', format(period_end, '%d %b %Y'))
teams <- c('Data Engineering', 'Developer Experience', 'Identity', 'Mobile', 'Payments', 'Platform')
roles <- c('Application engineering', 'Infrastructure engineering', 'Engineering management')
seniority <- c('Early career', 'Experienced', 'Senior / lead')
tenures <- c('Under 2 years', '2 to 5 years', 'Over 5 years')
team_effects <- tibble(
  Team = teams,
  meeting_delta = c(0.6, -0.8, 0.0, -3.0, 1.6, 5.0),
  focus_delta = c(0.3, 2.1, 0.0, 3.7, -1.0, -4.1),
  after_delta = c(0.0, -0.4, 0.0, -0.6, 2.4, 0.4),
  scheduled_call_delta = c(0.0, -0.1, 0.0, -0.4, 0.2, 0.6),
  email_delta = c(0.2, -0.1, 0.0, -0.5, 0.8, 1.5),
  chat_delta = c(0.1, 0.0, 0.0, -0.3, 0.6, 1.2),
  recurring_delta = c(0.02, -0.02, 0.00, -0.05, 0.03, 0.10),
  conflict_delta = c(0.00, -0.01, 0.00, -0.02, 0.11, 0.03),
  short_notice_delta = c(0.00, -0.01, 0.00, -0.02, 0.12, 0.02),
  gh_prop_delta = c(0.06, 0.24, 0.00, -0.18, 0.02, 0.04),
  m365_prop_delta = c(0.24, 0.05, 0.00, -0.17, 0.02, 0.05),
  gh_intensity_mult = c(1.10, 1.85, 1.00, 0.72, 1.05, 1.08),
  m365_intensity_mult = c(1.85, 1.05, 1.00, 0.72, 1.05, 1.08)
)

# A roster defines the population, never the presence of a tool-activity record.
people <- tibble(
  PersonId = sprintf('SYN%05d', 1:1080),
  IsDeveloper = seq_len(1080) <= 900,
  Team = c(sample(rep(teams, each = 150)), sample(teams, 180, TRUE)),
  Role = c(sample(roles, 900, TRUE, c(.48,.32,.20)), rep('Other job families', 180)),
  Seniority = sample(seniority, 1080, TRUE, c(.26,.43,.31)),
  Tenure = sample(tenures, 1080, TRUE, c(.34,.42,.24)),
  GH_eligible = sample(c(TRUE,FALSE), 1080, TRUE, c(.92,.08)),
  M365_eligible = sample(c(TRUE,FALSE), 1080, TRUE, c(.90,.10))
)
latent <- people |>
  left_join(team_effects, by = 'Team', relationship = 'many-to-one') |>
  mutate(
    meeting_base = 7 + 3 * (Role == 'Engineering management') +
      .9 * (Seniority == 'Senior / lead') + meeting_delta + rnorm(n(), 0, 1.8),
    focus_base = focus_delta + rnorm(n(), 0, 1.7),
    after_base = pmax(.05, rlnorm(n(), .35, .70) + after_delta),
    # Separate product propensities. Team signatures are cross-sectional only.
    gh_base_prop = if_else(runif(n()) < .25, 0, runif(n(), .2, .85)),
    m365_base_prop = if_else(runif(n()) < .28, 0, runif(n(), .18, .8)),
    gh_prop = pmin(.96, pmax(0, gh_base_prop + gh_prop_delta)),
    m365_prop = pmin(.96, pmax(0, m365_base_prop + m365_prop_delta)),
    gh_intensity = rlnorm(n(), 2.5, .65) * gh_intensity_mult,
    m365_intensity = rlnorm(n(), 1.8, .6) * m365_intensity_mult,
    # Missing coverage is a reference status, not a zero in an activity feed.
    gh_gap = runif(n()) < .035,
    m365_gap = runif(n()) < .035
  )
stopifnot(sum(table(people$Team[people$IsDeveloper]) == 150L) == length(teams))
coverage <- expand_grid(PersonId = people$PersonId, Week = weeks) |>
  left_join(select(latent, PersonId, GH_eligible, M365_eligible, gh_gap, m365_gap),
            by = 'PersonId', relationship = 'many-to-one') |>
  mutate(
    PQ_eligible = TRUE, PQ_complete = TRUE,
    GH_complete = !gh_gap & runif(n()) > .003,
    M365_complete = !m365_gap & runif(n()) > .003,
    # Complete means all seven calendar days covered. Activity uses weekdays.
    GH_covered_days = if_else(GH_complete, 7L, 0L),
    M365_covered_days = if_else(M365_complete, 7L, 0L)
  ) |>
  select(PersonId, Week, PQ_eligible, PQ_complete, GH_eligible, GH_complete,
         GH_covered_days, M365_eligible, M365_complete, M365_covered_days)

pq <- coverage |>
  select(PersonId, MetricDate = Week) |>
  left_join(latent, by = 'PersonId', relationship = 'many-to-one') |>
  mutate(
    # Smooth variation is descriptive, with no intervention date or treatment.
    seasonal = .6 * sin(2 * pi * as.numeric(MetricDate - min(weeks)) / 182),
    Meeting_hours = pmin(26, pmax(.5, meeting_base + seasonal + rnorm(n(),0,1.5))),
    Scheduled_call_hours = pmin(3.5, pmax(.25, runif(n(), .5, 2) + scheduled_call_delta)),
    Email_collaboration_hours = pmax(.2, rlnorm(n(), .55, .35) + .03 * Meeting_hours + email_delta),
    Chat_instant_message_hours = pmax(.1, rlnorm(n(), .20, .40) + .02 * Meeting_hours + chat_delta),
    Collaboration_hours = pmin(40, Meeting_hours + Scheduled_call_hours +
      Email_collaboration_hours + Chat_instant_message_hours),
    Available_to_focus_hours = 40 - Meeting_hours - Scheduled_call_hours,
    Uninterrupted_hours = pmin(Available_to_focus_hours,
      pmax(0, 16 - .55 * Meeting_hours + focus_base + rnorm(n(),0,1.8))),
    Interrupted_hours = Available_to_focus_hours - Uninterrupted_hours,
    Open_1_hour_block = pmin(floor(Available_to_focus_hours),
      pmax(0, round(22 - .6 * Meeting_hours + rnorm(n(),0,2)))),
    Recurring_meeting_hours = Meeting_hours * pmin(.90, pmax(.05, runif(n(), .35, .75) + recurring_delta)),
    Conflicting_meeting_hours = Meeting_hours * pmin(.45, pmax(.00, runif(n(), .01, .18) + conflict_delta)),
    Meeting_hours_with_six_or_fewer_hours_of_advanced_notice =
      Meeting_hours * pmin(.50, pmax(.00, runif(n(), .03, .23) + short_notice_delta)),
    After_hours_collaboration_hours = pmin(16, pmax(0, after_base + rnorm(n(),0,.65)))
  ) |>
  select(PersonId, MetricDate, Meeting_hours, Scheduled_call_hours, Collaboration_hours,
         Available_to_focus_hours, Uninterrupted_hours, Interrupted_hours,
         Open_1_hour_block, Recurring_meeting_hours, Conflicting_meeting_hours,
         Meeting_hours_with_six_or_fewer_hours_of_advanced_notice,
         After_hours_collaboration_hours) |>
  mutate(across(where(is.numeric), ~round(.x, 3)))

# Sparse activity feeds contain only positive observed days. A missing record is
# resolved to zero ONLY after eligibility and complete reference coverage pass.
daily_grid <- expand_grid(PersonId = people$PersonId, MetricDate = seq(min(weeks), period_end, by='day')) |>
  mutate(Week = weeks[findInterval(MetricDate, weeks)],
         weekday = as.POSIXlt(MetricDate)$wday) |>
  filter(weekday %in% 1:5) |>
  left_join(coverage, by=c('PersonId','Week'), relationship='many-to-one') |>
  left_join(select(latent, PersonId, gh_prop, m365_prop, gh_intensity, m365_intensity),
            by='PersonId', relationship='many-to-one')
gh_daily <- daily_grid |>
  filter(GH_eligible, GH_complete) |>
  mutate(active = rbinom(n(),1,gh_prop),
         suggested = active * rpois(n(),gh_intensity),
         accepted = rbinom(n(),suggested,.53),
         chat = active * rpois(n(),gh_intensity * .25),
         agent = as.integer(active == 1 & runif(n()) < .32)) |>
  filter(suggested + chat > 0) |>
  transmute(PersonId, MetricDate, Week,
            `Code completions suggested` = suggested,
            `Code completions accepted` = accepted,
            `User-initiated chat requests` = chat,
            `Agent use recorded` = agent)
m365_daily <- daily_grid |>
  filter(M365_eligible, M365_complete) |>
  mutate(active = rbinom(n(),1,m365_prop),
         drafting = active * rpois(n(),m365_intensity * .45),
         meetings = active * rpois(n(),m365_intensity * .30),
         search = active * rpois(n(),m365_intensity * .25)) |>
  filter(drafting + meetings + search > 0) |>
  transmute(PersonId, MetricDate, Week,
            Drafting_actions = drafting, Meeting_assistance_actions = meetings,
            Search_and_summarisation_actions = search,
            M365_actions = drafting + meetings + search)

gh_weekly <- gh_daily |>
  group_by(PersonId, Week) |>
  summarise(GH_active_days=n(), GH_suggestions=sum(`Code completions suggested`),
            GH_accepted=sum(`Code completions accepted`),
            GH_chats=sum(`User-initiated chat requests`),
            GH_agent_days=sum(`Agent use recorded`), .groups='drop')
m365_weekly <- m365_daily |>
  group_by(PersonId, Week) |>
  summarise(M365_active_days=n(), M365_actions=sum(M365_actions),
            M365_drafting=sum(Drafting_actions),
            M365_meetings=sum(Meeting_assistance_actions),
            M365_search=sum(Search_and_summarisation_actions), .groups='drop')
panel <- coverage |>
  left_join(people |> select(-GH_eligible,-M365_eligible), by='PersonId', relationship='many-to-one') |>
  left_join(pq, by=c('PersonId','Week'='MetricDate'), relationship='one-to-one') |>
  left_join(gh_weekly, by=c('PersonId','Week'), relationship='one-to-one') |>
  left_join(m365_weekly, by=c('PersonId','Week'), relationship='one-to-one') |>
  mutate(
    GH_valid = GH_eligible & GH_complete,
    M365_valid = M365_eligible & M365_complete,
    across(c(GH_active_days,GH_suggestions,GH_accepted,GH_chats,GH_agent_days),
           ~if_else(GH_valid, coalesce(.x,0), NA_real_)),
    across(c(M365_active_days,M365_actions,M365_drafting,M365_meetings,M365_search),
           ~if_else(M365_valid, coalesce(.x,0), NA_real_)))

metric_labels <- c(Meeting_hours='Meetings', Uninterrupted_hours='Uninterrupted time',
                   Collaboration_hours='Collaboration hours',
                   After_hours_collaboration_hours='After-hours collaboration',
                   Available_to_focus_hours='Available to focus',
                   Recurring_meeting_hours='Recurring meetings',
                   Conflicting_meeting_hours='Conflicting meetings',
                   Meeting_hours_with_six_or_fewer_hours_of_advanced_notice='Short-notice meetings')
work_metrics <- names(metric_labels)
baseline_panel <- panel |> filter(IsDeveloper, PQ_eligible, PQ_complete, Week >= baseline_start)
baseline <- baseline_panel |>
  group_by(PersonId,Team,Role,Seniority,Tenure) |>
  summarise(Weeks_observed=n(),
    Weeks_above=sum(After_hours_collaboration_hours >= AFTER_HOURS_CONVENTION),
    across(all_of(work_metrics),mean),
    Both_eligible=all(GH_eligible & M365_eligible),
    Both_complete=all(GH_complete & M365_complete),
    GH_all_valid=all(GH_valid), M365_all_valid=all(M365_valid),
    GH_active_days=if(all(GH_valid)) sum(GH_active_days) else NA_real_,
    M365_active_days=if(all(M365_valid)) sum(M365_active_days) else NA_real_,
    GH_suggestions=if(all(GH_valid)) sum(GH_suggestions) else NA_real_,
    GH_accepted=if(all(GH_valid)) sum(GH_accepted) else NA_real_,
    GH_chats=if(all(GH_valid)) sum(GH_chats) else NA_real_,
    GH_agent_days=if(all(GH_valid)) sum(GH_agent_days) else NA_real_,
    M365_actions=if(all(M365_valid)) sum(M365_actions) else NA_real_,
    M365_drafting=if(all(M365_valid)) sum(M365_drafting) else NA_real_,
    M365_meetings=if(all(M365_valid)) sum(M365_meetings) else NA_real_,
    M365_search=if(all(M365_valid)) sum(M365_search) else NA_real_,
    .groups='drop') |>
  mutate(Joint=case_when(
    !Both_eligible ~ 'Not both eligible',
    !Both_complete ~ 'Coverage unresolved',
    GH_active_days > 0 & M365_active_days > 0 ~ 'Both recorded',
    GH_active_days > 0 ~ 'GitHub only recorded',
    M365_active_days > 0 ~ 'M365 only recorded',
    TRUE ~ 'Neither recorded'))
joint_levels <- c('Both recorded','GitHub only recorded','M365 only recorded',
                  'Neither recorded','Not both eligible','Coverage unresolved')
baseline$Joint <- factor(baseline$Joint, levels=joint_levels)
matched <- baseline |> filter(Both_eligible, Both_complete)
# "Matched" means common eligibility/coverage only, not statistical matching.

# Secondary exports describe suggestion attribution across the full 26 weeks.
# They are illustrative allocations, not claims about a real query's schema.
make_mix <- function(labels, field) {
  users <- gh_daily |> group_by(PersonId) |>
    summarise(Suggestions=sum(`Code completions suggested`), .groups='drop') |>
    filter(Suggestions > 0)
  out <- lapply(seq_len(nrow(users)), function(i) {
    counts <- as.vector(rmultinom(1, users$Suggestions[i], rep(1,length(labels))))
    tibble(PersonId=users$PersonId[i], Category=labels,
           Suggestions=counts, Share=counts/sum(counts),
           PeriodStart=min(weeks), PeriodEnd=period_end)
  }) |> bind_rows()
  names(out)[names(out)=='Category'] <- field
  out
}
model_mix <- make_mix(c('Model A','Model B','Model C'), 'Model')
language_mix <- make_mix(c('TypeScript','Python','Java','Go','C#','SQL'), 'Language')

# Fail before exporting or rendering if source or population contracts break.
assert_key <- function(data, keys) stopifnot(!anyDuplicated(data[keys]), !anyNA(data[keys]))
assert_key(people,'PersonId')
for (d in list(pq,gh_daily,m365_daily)) assert_key(d,c('PersonId','MetricDate'))
for (d in list(coverage,panel,gh_weekly,m365_weekly)) assert_key(d,c('PersonId','Week'))
assert_key(model_mix,c('PersonId','Model'))
assert_key(language_mix,c('PersonId','Language'))
stopifnot(sum(people$IsDeveloper)==900L, n_distinct(baseline$Team)==6L,
          nrow(coverage)==1080L*26L, nrow(panel)==nrow(pq), nrow(baseline)==900L,
          nrow(baseline_panel)==900L*BASELINE_WEEKS,
          all(baseline$Weeks_observed==BASELINE_WEEKS),
          sum(table(baseline$Joint))==900L,
          nrow(matched)==sum(table(baseline$Joint)[1:4]),
          all(gh_daily$`Code completions accepted` <= gh_daily$`Code completions suggested`),
          all(panel$GH_active_days[panel$GH_valid] <= 5),
          all(panel$M365_active_days[panel$M365_valid] <= 5),
          all(is.na(panel$GH_active_days[!panel$GH_valid])),
          all(is.na(panel$M365_active_days[!panel$M365_valid])),
          all(pq$Uninterrupted_hours <= pq$Available_to_focus_hours),
          max(abs(pq$Available_to_focus_hours - pq$Uninterrupted_hours - pq$Interrupted_hours)) <= .0011,
          all(pq$Collaboration_hours + .0011 >= pq$Meeting_hours + pq$Scheduled_call_hours),
          all(pq$Collaboration_hours <= 40))
for(d in list(pq,gh_daily,m365_daily)) {
  nums <- d |> select(where(is.numeric))
  stopifnot(all(is.finite(as.matrix(nums))), all(as.matrix(nums)>=0))
}
stopifnot(all(table(baseline$Joint) >= MIN_GROUP_N),
          all(table(baseline$Team)>=MIN_GROUP_N))

stopifnot(all(m365_daily$M365_actions == m365_daily$Drafting_actions +
                m365_daily$Meeting_assistance_actions + m365_daily$Search_and_summarisation_actions),
          max(abs(pq$Meeting_hours + pq$Scheduled_call_hours + pq$Available_to_focus_hours - 40)) <= .0011,
          all(pq$Open_1_hour_block == floor(pq$Open_1_hour_block)),
          all(pq$Recurring_meeting_hours <= pq$Meeting_hours),
          all(pq$Conflicting_meeting_hours <= pq$Meeting_hours),
          all(pq$Meeting_hours_with_six_or_fewer_hours_of_advanced_notice <= pq$Meeting_hours))
for (mix in list(model_mix,language_mix)) {
  totals <- mix |> group_by(PersonId) |> summarise(Share=sum(Share),Suggestions=sum(Suggestions),.groups='drop')
  expected <- gh_daily |> group_by(PersonId) |>
    summarise(Expected=sum(`Code completions suggested`),.groups='drop') |> filter(Expected>0)
  check <- left_join(totals,expected,by='PersonId',relationship='one-to-one')
  stopifnot(nrow(check)==nrow(expected),all(abs(check$Share-1)<1e-10),
            all(check$Suggestions==check$Expected))
}
DATA_DIR <- file.path('_data','github')
export_csv <- function(data, folder, filename) {
  dest <- file.path(DATA_DIR,folder)
  dir.create(dest, recursive=TRUE, showWarnings=FALSE)
  write.csv(data, file.path(dest,filename), row.names=FALSE, na='NA', fileEncoding='UTF-8')
}
export_csv(people,'reference','people-snapshot.csv')
export_csv(coverage,'reference','coverage-weekly.csv')
export_csv(pq,'person-query','person-query-weekly.csv')
export_csv(gh_daily,'github-query','activity-daily.csv')
export_csv(m365_daily,'m365-query','activity-daily.csv')
export_csv(model_mix,'github-query','model-mix.csv')
export_csv(language_mix,'github-query','language-mix.csv')

# Presentation helpers use group-level aggregates only. No point per person.
# The package boxplot table performs person averaging and mingroup suppression.
# Custom visuals are needed for percentile intervals, explicit denominators and
# joint coverage. Do not use a single-product segment function for joint status.
wrap <- function(x, width=100) stringr::str_wrap(x,width=width)
f1 <- function(x) formatC(x,format='f',digits=1,big.mark=',',decimal.mark='.')
pct <- function(x) scales::percent(x,accuracy=.1,decimal.mark='.',big.mark=',')
num <- function(x) format(x,big.mark=',',scientific=FALSE,trim=TRUE)
PAL <- c('#2563eb','#0f766e','#7c3aed','#db2777','#f97316','#64748b')
joint_pal <- setNames(PAL,joint_levels)
theme_report <- function() theme_minimal(base_size=12) + theme(
  panel.grid.minor=element_blank(), panel.grid.major.y=element_blank(),
  plot.title.position='plot', plot.title=element_text(face='bold',colour='#20243b',size=15),
  plot.subtitle=element_text(colour='#475569',size=11,margin=margin(b=12)),
  plot.caption=element_text(hjust=0,colour='#475569',size=9,margin=margin(t=14)),
  plot.margin=margin(14,24,14,16), legend.position='bottom', legend.title=element_blank(),
  strip.text=element_text(face='bold',size=11), axis.title=element_text(size=11),
  legend.text=element_text(size=10))
chart_labs <- function(title,subtitle,caption,x=NULL,y=NULL) labs(
  title=wrap(title,88),subtitle=wrap(subtitle,110),caption=wrap(caption,115),x=x,y=y)
base_caption <- function(n=nrow(baseline), unit='hours/person/week') paste0(
  period_label,' | ',num(n),' distinct developers | ',BASELINE_WEEKS,
  ' complete weeks | ',unit,'. Dates are week-aligned, UTC convention.')

interval_data <- function(data, metrics, group='Team') {
  bind_rows(lapply(metrics, function(m) {
    source <- data |> mutate(MetricDate=baseline_start)
    result <- vivainsights::create_boxplot(source,metric=m,hrvar=group,
                  mingroup=MIN_GROUP_N,return='table')
    result |> transmute(Group=as.character(group),Metric=unname(metric_labels[m]),
                        p25,p50,p75)
  }))
}
interval_plot <- function(data, metrics, group='Team',title,subtitle,caption=base_caption(nrow(data))) {
  d <- interval_data(data,metrics,group)
  first_metric <- unname(metric_labels[metrics[1]])
  group_order <- if(group=='Joint') joint_levels[1:4] else d |>
    filter(Metric == first_metric) |> arrange(desc(p50)) |> pull(Group)
  d$Group <- factor(d$Group, levels=rev(group_order))
  d$Metric <- factor(d$Metric,levels=unname(metric_labels[metrics]))
  ggplot(d,aes(x=p50,y=Group)) +
    geom_linerange(aes(xmin=p25,xmax=p75),linewidth=2.4,colour='#bfdbfe') +
    geom_point(size=2.6,colour='#2563eb') +
    facet_wrap(~Metric,nrow=1,scales='free_x',labeller=label_wrap_gen(22)) +
    scale_x_continuous(expand=expansion(mult=c(.12,.14))) + theme_report() +
    chart_labs(title,subtitle,caption,x='Hours per person per week')
}

# Suppress the entire composition if any segment is below the disclosure limit.
# This also avoids inferring small cells by subtraction from a published total.
composition <- function(data, group, category) {
  counts <- data |> count(Group=.data[[group]],Category=.data[[category]],name='People',.drop=TRUE)
  counts <- counts |> group_by(Group) |> filter(all(People>=MIN_GROUP_N)) |> ungroup()
  stopifnot(nrow(counts)>0,all(counts$People>=MIN_GROUP_N))
  counts |> group_by(Group) |> mutate(Share=People/sum(People)) |> ungroup()
}
stack_plot <- function(d,title,subtitle,caption,palette=PAL,sort_category=NULL,group_order=NULL) {
  d <- d |> mutate(Group=as.character(Group), Category=as.character(Category))
  d$Category <- factor(d$Category,levels=unique(d$Category))
  target <- if(is.null(sort_category)) as.character(d$Category)[1] else sort_category
  if(is.null(group_order)) {
    group_order <- d |>
      filter(as.character(Category) == target) |>
      arrange(desc(Share)) |>
      pull(Group) |> as.character()
    group_order <- c(group_order, setdiff(unique(d$Group), group_order))
    # Name the sort key so the reader can verify the ranking against a segment.
    subtitle <- paste0(subtitle,' Groups are ranked by the share of ',target,'.')
  }
  d$Group <- factor(d$Group,levels=rev(group_order))
  colours <- setNames(rep(palette,length.out=nlevels(d$Category)),levels(d$Category))
  dark <- c('#2563eb','#0f766e','#7c3aed','#db2777','#64748b')
  d$TextColour <- ifelse(colours[as.character(d$Category)] %in% dark,'white','#182033')
  ggplot(d,aes(x=Share,y=Group,fill=Category,group=Category)) + geom_col(width=.6) +
    geom_text(aes(label=ifelse(Share>=.06,pct(Share),''),colour=TextColour),
              position=position_stack(vjust=.5),size=3.2) +
    scale_colour_identity() + scale_fill_manual(values=colours,labels=function(x) wrap(x,25)) +
    scale_x_continuous(labels=label_percent(),expand=expansion(mult=c(0,.01))) +
    guides(fill=guide_legend(nrow=2,byrow=TRUE)) + theme_report() +
    chart_labs(title,subtitle,caption,x='Share of developers')
}
html_table <- function(data, caption=NULL, escape=TRUE) {
  stopifnot(!is.null(data), nrow(data) > 0)
  cat('<div class="table-scroll">')
  print(knitr::kable(data,format='html',row.names=FALSE,escape=escape,caption=caption))
  cat('</div>')
}
composition_table <- function(d) d |> transmute(Group,Category,`Developers (count)`=People,`Share (%)`=pct(Share))
metric_count <- function(x,unit) {
  display <- if(x>=1e6) paste0(f1(x/1e6),'M') else if(x>=1e3) paste0(f1(x/1e3),'K') else num(x)
  sprintf('<span title="%s %s">%s %s</span>',num(x),unit,display,unit)
}
card <- function(label,value,detail='') cat(sprintf(
  '<div class="kpi"><span>%s</span><strong>%s</strong><small>%s</small></div>',label,value,detail))

joint_counts <- baseline |> count(Joint,name='People',.drop=TRUE) |>
  mutate(Share=People/nrow(baseline))
team_context <- composition(baseline,'Team','Role')
role_context <- composition(matched,'Joint','Role')
team_joint_context <- composition(matched,'Joint','Team')
work_summary <- bind_rows(lapply(c('Collaboration_hours','Meeting_hours','Uninterrupted_hours','After_hours_collaboration_hours'),function(m) tibble(
  Metric=metric_labels[[m]],`25th percentile`=f1(quantile(baseline[[m]],.25)),
  Median=f1(median(baseline[[m]])),`75th percentile`=f1(quantile(baseline[[m]],.75)))))

product_summary <- bind_rows(lapply(c('GH','M365'), function(product) {
  gh <- product=='GH'
  valid <- baseline |> filter(if(gh) GH_all_valid else M365_all_valid)
  active <- valid |> filter(if(gh) GH_active_days>0 else M365_active_days>0)
  volume <- if(gh) 'GH_suggestions' else 'M365_actions'
  stopifnot(nrow(valid)>=MIN_GROUP_N,nrow(active)>=MIN_GROUP_N,
            ceiling(.1*nrow(active))>=MIN_GROUP_N)
  # Only the aggregate share is published, never the identities or ranking.
  top_n <- ceiling(.1*nrow(active))
  share <- sum(sort(active[[volume]],decreasing=TRUE)[seq_len(top_n)])/sum(active[[volume]])
  tibble(Product=if(gh) 'GitHub Copilot' else 'M365 Copilot',
    `Valid developers (count)`=nrow(valid), `Recorded active (count)`=nrow(active),
    `Recorded active (%)`=pct(nrow(active)/nrow(valid)),
    `Active days / developer / week (mean)`=f1(mean(if(gh) valid$GH_active_days else valid$M365_active_days)/BASELINE_WEEKS),
    `Intensity / developer / week (mean)`=f1(mean(valid[[volume]])/BASELINE_WEEKS),
    `Intensity unit`=if(gh) 'Suggestions' else 'Illustrative actions',
    `Top 10% active share (%)`=pct(share), `Top group (count)`=top_n)
}))
team_league <- baseline |>
  group_by(Team) |>
  summarise(
    Developers = n(),
    Collaboration_hours = median(Collaboration_hours),
    Meeting_hours = median(Meeting_hours),
    Uninterrupted_hours = median(Uninterrupted_hours),
    After_hours_collaboration_hours = median(After_hours_collaboration_hours),
    GH_valid_n = sum(GH_all_valid),
    GH_active_n = sum(GH_all_valid & coalesce(GH_active_days, 0) > 0),
    GH_active_share = GH_active_n / GH_valid_n,
    GH_intensity = mean(GH_suggestions[GH_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
    M365_valid_n = sum(M365_all_valid),
    M365_active_n = sum(M365_all_valid & coalesce(M365_active_days, 0) > 0),
    M365_active_share = M365_active_n / M365_valid_n,
    M365_intensity = mean(M365_actions[M365_all_valid], na.rm = TRUE) / BASELINE_WEEKS,
    .groups = 'drop') |>
  mutate(across(c(GH_active_share, GH_intensity, M365_active_share, M365_intensity),
                ~if_else(Developers >= MIN_GROUP_N & GH_valid_n >= MIN_GROUP_N &
                           M365_valid_n >= MIN_GROUP_N, .x, NA_real_))) |>
  arrange(desc(Collaboration_hours)) |>
  mutate(Rank = row_number(), .before = Team)
stopifnot(all(team_league$Developers >= MIN_GROUP_N), all(team_league$GH_valid_n >= MIN_GROUP_N),
          all(team_league$M365_valid_n >= MIN_GROUP_N),
          team_league$Team[which.max(team_league$Collaboration_hours)] == 'Platform',
          team_league$Team[which.max(team_league$Meeting_hours)] == 'Platform',
          team_league$Team[which.min(team_league$Uninterrupted_hours)] == 'Platform',
          team_league$Team[which.max(team_league$After_hours_collaboration_hours)] == 'Payments',
          team_league$Team[which.max(team_league$GH_active_share)] == 'Developer Experience',
          team_league$Team[which.max(team_league$GH_intensity)] == 'Developer Experience',
          team_league$Team[which.max(team_league$M365_active_share)] == 'Data Engineering',
          team_league$Team[which.max(team_league$M365_intensity)] == 'Data Engineering',
          team_league$Team[which.min(team_league$Collaboration_hours)] == 'Mobile',
          team_league$Team[which.min(team_league$Meeting_hours)] == 'Mobile',
          team_league$Team[which.max(team_league$Uninterrupted_hours)] == 'Mobile',
          team_league$Team[which.min(team_league$GH_active_share)] == 'Mobile',
          team_league$Team[which.min(team_league$M365_active_share)] == 'Mobile')
# Shares are carried as percentage points so every panel shares one axis
# formatter. Workload panels are ordered before product panels so the reader
# meets the working week before the tooling.
league_metric_levels <- c('Collaboration hours','Meeting hours','Uninterrupted hours',
  'After-hours collaboration','GitHub recorded-active share (%)','GitHub intensity',
  'M365 recorded-active share (%)','M365 intensity')
team_league_plot <- bind_rows(
  team_league |> transmute(Team, Metric='Collaboration hours', Value=Collaboration_hours, Label=f1(Value)),
  team_league |> transmute(Team, Metric='Meeting hours', Value=Meeting_hours, Label=f1(Value)),
  team_league |> transmute(Team, Metric='Uninterrupted hours', Value=Uninterrupted_hours, Label=f1(Value)),
  team_league |> transmute(Team, Metric='After-hours collaboration', Value=After_hours_collaboration_hours, Label=f1(Value)),
  team_league |> transmute(Team, Metric='GitHub recorded-active share (%)', Value=GH_active_share*100, Label=pct(GH_active_share)),
  team_league |> transmute(Team, Metric='GitHub intensity', Value=GH_intensity, Label=f1(Value)),
  team_league |> transmute(Team, Metric='M365 recorded-active share (%)', Value=M365_active_share*100, Label=pct(M365_active_share)),
  team_league |> transmute(Team, Metric='M365 intensity', Value=M365_intensity, Label=f1(Value))) |>
  mutate(Metric=factor(Metric, levels=league_metric_levels))
stopifnot(!anyNA(team_league_plot$Metric),
          setequal(levels(team_league_plot$Metric), unique(as.character(team_league_plot$Metric))))
team_effect_register <- team_effects |>
  transmute(Team, `Planted illustrative characteristics`=case_when(
    Team == 'Platform' ~ 'Coordination-heavy. Highest meeting and collaboration hours, lowest uninterrupted time.',
    Team == 'Payments' ~ 'Highest after-hours collaboration, conflicting meetings and short-notice meeting hours.',
    Team == 'Developer Experience' ~ 'Highest GitHub Copilot recorded-active share and intensity, with healthy uninterrupted time.',
    Team == 'Data Engineering' ~ 'Highest M365 Copilot recorded-active share and intensity, with mid-range working-condition metrics.',
    Team == 'Mobile' ~ 'Focus-rich. Lowest meeting and collaboration hours, highest uninterrupted time and lowest recorded AI adoption.',
    TRUE ~ 'Population-median reference profile across workload, focus and recorded AI use.'
  ))
trend <- panel |> filter(IsDeveloper,PQ_eligible,PQ_complete) |>
  group_by(Week) |> summarise(PQ_n=n_distinct(PersonId),
    across(all_of(c('Collaboration_hours','Meeting_hours','Uninterrupted_hours','After_hours_collaboration_hours')),median),
    GH_n=sum(GH_valid), GH_active=sum(GH_active_days>0,na.rm=TRUE),
    M365_n=sum(M365_valid), M365_active=sum(M365_active_days>0,na.rm=TRUE),
    Joint_n=sum(GH_valid & M365_valid), .groups='drop') |>
  mutate(GH_rate=GH_active/GH_n,M365_rate=M365_active/M365_n)
stopifnot(all(trend$PQ_n==900),all(trend$GH_n>=MIN_GROUP_N),all(trend$M365_n>=MIN_GROUP_N),
          all(trend$GH_rate>=0 & trend$GH_rate<=1),all(trend$M365_rate>=0 & trend$M365_rate<=1))
validation_summary <- tibble(
  Check=c('Roster developers','Teams','Person Query developer-weeks','Baseline developer-weeks',
          'Collaboration-hours contract','Planted team signatures','Source keys and joins',
          'Non-negative metrics and acceptance bounds','Joint reconciliation','Privacy floor'),
  Result=c('900','6','23,400','7,200',
           'Collaboration hours are at least meeting plus scheduled-call hours','Passed',
           'Unique keys, no row amplification','Passed',
           paste(sum(joint_counts$People),'=',nrow(matched),'+',900-nrow(matched)),
           paste('At least',MIN_GROUP_N,'distinct people in every published group')))
