"""Optional in-memory reconstruction, not an authoritative current MQ snapshot.

Inputs are normalised pandas frames with identical schemas. The caller confirms
comparable keys/settings and supplies every organiser/HR text column.
See reference/meeting-query.md for assumptions, privacy and production limits.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from meeting_quality import quality_mask


KEY = "MeetingId"
TIME = "Collaboration_start_time"
NUMERIC = ("Number_of_attendees", "Intended_participant_count", "Duration")
FLAGS = ("Cancelled", "All_Day_Meeting")
REQUIRED = {KEY, TIME, *NUMERIC, *FLAGS}
BLANK = {"", "na", "n/a", "#n/a", "null"}


def _blank(values):
    return values.isna() | values.astype("string").str.strip().str.casefold().isin(BLANK)


def _validate(frame, attributes):
    if not frame.columns.is_unique or not REQUIRED.issubset(frame.columns):
        raise ValueError("Duplicate columns or missing required MQ columns")
    if "source_export" in frame.columns:
        raise ValueError("Expected raw exports without source_export")
    ids = frame[KEY]
    canonical = all(isinstance(v, str) and v == v.strip() for v in ids)
    if not canonical or _blank(ids).any() or ids.duplicated().any():
        raise ValueError("MeetingId must be unique, nonblank, canonical text")
    if not is_datetime64_any_dtype(frame[TIME]) or frame[TIME].isna().any():
        raise ValueError("Collaboration_start_time must contain nonmissing datetimes")
    for col in NUMERIC:
        if is_bool_dtype(frame[col]) or not is_numeric_dtype(frame[col]):
            raise ValueError(f"Expected numeric quality column: {col}")
        if not np.isfinite(frame[col].dropna()).all():
            raise ValueError(f"Nonfinite quality value: {col}")
    for col in FLAGS:
        if not is_bool_dtype(frame[col]):
            raise ValueError(f"Normalise boolean flags explicitly: {col}")
    for col in attributes:
        if col not in frame or not all(isinstance(v, str) for v in frame[col].dropna()):
            raise ValueError(f"Missing or nontext organiser/HR column: {col}")


def merge_mq(old, new, *, attribute_columns):
    """Return a full raw union and aggregate audit, without mutating inputs."""
    if isinstance(attribute_columns, str):
        raise ValueError("Supply an explicit sequence of organiser/HR columns")
    attributes = tuple(attribute_columns)
    if (not all(isinstance(col, str) for col in attributes)
            or len(set(attributes)) != len(attributes)
            or set(attributes) & (REQUIRED | {"source_export"})):
        raise ValueError("Invalid or duplicate organiser/HR columns")
    for frame in (old, new):
        _validate(frame, attributes)
    if set(old.columns) != set(new.columns):
        raise ValueError("Column sets differ, reconcile the schema explicitly")
    if any(old[col].dtype != new[col].dtype for col in old.columns):
        raise ValueError("Column dtypes differ, normalise both schemas explicitly")

    old_ids, new_ids = set(old[KEY]), set(new[KEY])
    shared = old_ids & new_ids
    old_only, new_only = old_ids - new_ids, new_ids - old_ids
    prior, current = old.set_index(KEY), new.set_index(KEY)
    shared_index = sorted(shared)
    left, right = prior.loc[shared_index], current.loc[shared_index]
    if not left[TIME].equals(right[TIME]):
        raise ValueError("Shared-event timestamps differ, check identity and vintage")
    for col in attributes:
        populated = ~_blank(left[col]) & ~_blank(right[col])
        unequal = left[col].astype("string").str.strip().ne(right[col].astype("string").str.strip())
        if (populated & unequal).any():
            raise ValueError(f"Populated organiser/HR conflict: {col}")

    old_ok = set(old.loc[quality_mask(old), KEY])
    new_ok = set(new.loc[quality_mask(new), KEY])
    prefer_old = shared & (old_ok - new_ok)
    keep_old = old_only | prefer_old
    combined = pd.concat([
        new.loc[~new[KEY].isin(prefer_old)].assign(source_export="new"),
        old.loc[old[KEY].isin(keep_old)].assign(source_export="old"),
    ], ignore_index=True)
    backfills = {}
    backfill_sources = {}
    for col in attributes:
        counts = {}
        for source, alternate in (("old", prior), ("new", current)):
            candidate = combined[KEY].map(alternate[col])
            fill = (combined["source_export"].ne(source)
                    & _blank(combined[col]) & ~_blank(candidate))
            combined.loc[fill, col] = candidate.loc[fill]
            counts[source] = int(fill.sum())
        backfills[col] = sum(counts.values())
        backfill_sources[col] = counts

    expected_ids = old_ids | new_ids
    if (combined[KEY].duplicated().any() or set(combined[KEY]) != expected_ids
            or len(combined) != len(new) + len(old_only)):
        raise AssertionError("Output is not the unique full MeetingId union")
    audit = {
        "old_rows": len(old), "new_rows": len(new), "shared_ids": len(shared),
        "old_only_ids": len(old_only), "new_only_ids": len(new_only),
        "older_valid_preferred_ids": len(prefer_old),
        "old_rows_written": len(keep_old), "new_rows_written": len(new) - len(prefer_old),
        "combined_rows": len(combined), "backfilled_cells_by_column": backfills,
        "backfilled_cells_by_source": backfill_sources, "true_conflicts": 0,
    }
    return combined, audit
