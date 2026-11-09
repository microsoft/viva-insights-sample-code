"""Reviewed seven-rule MQ example; thresholds are analytical choices."""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def normalise_flag(values):
    """Keep missing flags unknown and reject unrecognised nonmissing tokens."""
    tokens = values.astype("string").str.strip().str.casefold()
    tokens = tokens.mask(tokens.isin(["", "na", "n/a", "#n/a", "null"]))
    mapping = {
        "true": True, "yes": True, "y": True, "1": True, "1.0": True,
        "false": False, "no": False, "n": False, "0": False, "0.0": False,
    }
    if (tokens.notna() & ~tokens.isin(mapping)).any():
        raise ValueError("Unrecognised boolean flag; inspect the source before filtering")
    return tokens.map(mapping).astype("boolean")


def quality_mask(frame, *, max_intended=150, min_duration_hours=1 / 60, max_duration_hours=8):
    """Return a mask without mutating raw rows; missing quality inputs fail."""
    numeric = ("Number_of_attendees", "Intended_participant_count", "Duration")
    required = {*numeric, "Cancelled", "All_Day_Meeting"}
    if not frame.columns.is_unique or not required.issubset(frame.columns):
        raise ValueError("Seven-rule example requires unique quality columns, including Duration")
    if not (np.isfinite([max_intended, min_duration_hours, max_duration_hours]).all()
            and max_intended > 1 and 0 < min_duration_hours <= max_duration_hours):
        raise ValueError("Invalid reviewed quality thresholds")
    for column in numeric:
        if is_bool_dtype(frame[column]) or not is_numeric_dtype(frame[column]):
            raise ValueError(f"Parse the numeric quality column explicitly: {column}")
        if not np.isfinite(frame[column].dropna()).all():
            raise ValueError(f"Nonfinite quality value: {column}")
    cancelled = normalise_flag(frame["Cancelled"])
    all_day = normalise_flag(frame["All_Day_Meeting"])
    return (
        frame["Number_of_attendees"].ge(2)
        & frame["Intended_participant_count"].gt(1)
        & frame["Intended_participant_count"].le(max_intended)
        & cancelled.eq(False) & all_day.eq(False)
        & frame["Duration"].ge(min_duration_hours)
        & frame["Duration"].le(max_duration_hours)
    ).fillna(False)
