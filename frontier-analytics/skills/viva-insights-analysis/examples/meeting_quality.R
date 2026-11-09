# Reviewed seven-rule MQ example. Thresholds are analytical choices.
normalise_flag <- function(values) {
  tokens <- tolower(trimws(as.character(values)))
  tokens[tokens %in% c("", "na", "n/a", "#n/a", "null")] <- NA_character_
  yes <- c("true", "yes", "y", "1", "1.0")
  no <- c("false", "no", "n", "0", "0.0")
  if (any(!is.na(tokens) & !tokens %in% c(yes, no)))
    stop("Unrecognised boolean flag; inspect the source before filtering")
  out <- rep(NA, length(tokens))
  out[!is.na(tokens) & tokens %in% yes] <- TRUE
  out[!is.na(tokens) & tokens %in% no] <- FALSE
  out
}

quality_mask <- function(frame, max_intended = 150,
                         min_duration_hours = 1 / 60, max_duration_hours = 8) {
  numeric <- c("Number_of_attendees", "Intended_participant_count", "Duration")
  required <- c(numeric, "Cancelled", "All_Day_Meeting")
  if (anyDuplicated(names(frame)) || !all(required %in% names(frame)))
    stop("Seven-rule example requires unique quality columns, including Duration")
  if (!all(is.finite(c(max_intended, min_duration_hours, max_duration_hours))) ||
      max_intended <= 1 || min_duration_hours <= 0 ||
      min_duration_hours > max_duration_hours)
    stop("Invalid reviewed quality thresholds")
  for (column in numeric) {
    if (!is.numeric(frame[[column]]))
      stop("Parse the numeric quality column explicitly: ", column)
    if (any(!is.na(frame[[column]]) & !is.finite(frame[[column]])))
      stop("Nonfinite quality value: ", column)
  }
  keep <- frame$Number_of_attendees >= 2 &
    frame$Intended_participant_count > 1 &
    frame$Intended_participant_count <= max_intended &
    !normalise_flag(frame$Cancelled) & !normalise_flag(frame$All_Day_Meeting) &
    frame$Duration >= min_duration_hours & frame$Duration <= max_duration_hours
  keep[is.na(keep)] <- FALSE
  keep
}
