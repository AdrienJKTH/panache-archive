"""Daily statistics of hourly fields, by the rules Panache applies to Open-Meteo's series.

These are `daily_means` and `daily_max_8h_means` of Panache's `climatology` module, on whole
grids at once. Days are UTC days. Missing values (NaN) are skipped, never read as 0, and a
statistic needs 75 % of its data (EU Directive 2008/50/EC, Annexes I and VII): 18 hours for a
daily mean; 6 hours for an 8-hour mean, and 18 such means for the day's maximum.

Hourly arrays are `(hours, *cells)`, start at 00:00 UTC and hold whole days; daily arrays are
`(days, *cells)`.
"""

from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

HOURS_PER_DAY: Final = 24
MIN_HOURS_PER_DAY: Final = 18
"""Valid hours needed for a daily mean."""
WINDOW_HOURS: Final = 8
MIN_HOURS_PER_8H: Final = 6
"""Valid hours needed for an 8-hour mean."""
MIN_8H_MEANS_PER_DAY: Final = 18
"""Valid 8-hour means needed for a day's maximum."""
EVENING_HOURS: Final = WINDOW_HOURS - 1
"""Hours before a day that its earliest 8-hour means cover (17:00 to 23:00 the day before)."""


def daily_means(hourly: FloatArray) -> FloatArray:
    """UTC daily means; NaN where a day has fewer than 18 valid hours."""
    days = _whole_days(hourly)
    by_day = hourly.reshape(days, HOURS_PER_DAY, *hourly.shape[1:])
    valid = ~np.isnan(by_day)
    count = valid.sum(axis=1)
    total = np.where(valid, by_day, 0.0).sum(axis=1)
    enough = count >= MIN_HOURS_PER_DAY
    return np.where(enough, total / np.maximum(count, 1), np.nan)


def daily_max_8h_means(hourly: FloatArray, evening: FloatArray | None = None) -> FloatArray:
    """UTC daily maxima of the running 8-hour means; NaN where a day has too few valid means.

    As in EU Directive 2008/50/EC (Annex VII), each 8-hour mean belongs to the day on which it
    ends: from the one ending at 01:00 (17:00 the day before to 01:00) to the one ending at
    24:00. `evening` holds the hours just before `hourly` (its last 7 are used). Without it, as
    for the first day of a Panache series, the first day's earliest means are incomplete; an
    ozone maximum, in the afternoon, is not affected.
    """
    days = _whole_days(hourly)
    cells = hourly.shape[1:]
    before = np.full((EVENING_HOURS, *cells), np.nan)
    if evening is not None:
        if evening.shape[1:] != cells:
            raise ValueError(f"evening cells {evening.shape[1:]} differ from the day's {cells}")
        tail = evening[-EVENING_HOURS:]
        before[EVENING_HOURS - len(tail) :] = tail
    series = np.concatenate([before, hourly])
    valid = ~np.isnan(series)
    sums = _cumulative(np.where(valid, series, 0.0))
    counts = _cumulative(valid.astype(np.float64))

    # The mean ending at hour e (1, 2, … to the last) covers the 8 hours before it: rows e - 1
    # to e + 6 of `series`, since `before` shifts every hour by 7 rows.
    hours = hourly.shape[0]
    window_sums = sums[WINDOW_HOURS : hours + WINDOW_HOURS] - sums[:hours]
    window_counts = counts[WINDOW_HOURS : hours + WINDOW_HOURS] - counts[:hours]
    means_valid = window_counts >= MIN_HOURS_PER_8H
    means = window_sums / np.maximum(window_counts, 1)

    by_day = means.reshape(days, HOURS_PER_DAY, *cells)
    valid_by_day = means_valid.reshape(days, HOURS_PER_DAY, *cells)
    highest = np.where(valid_by_day, by_day, -np.inf).max(axis=1)
    enough = valid_by_day.sum(axis=1) >= MIN_8H_MEANS_PER_DAY
    return np.where(enough, highest, np.nan)


def _whole_days(hourly: FloatArray) -> int:
    hours = int(hourly.shape[0])
    if hours == 0 or hours % HOURS_PER_DAY:
        raise ValueError(f"{hours} hours are not whole days")
    return hours // HOURS_PER_DAY


def _cumulative(values: FloatArray) -> FloatArray:
    """Cumulative sums along time, with a leading 0, so that a window's sum is a difference."""
    zero = np.zeros((1, *values.shape[1:]))
    return np.concatenate([zero, np.cumsum(values, axis=0)])
