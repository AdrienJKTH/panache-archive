import math

import numpy as np
import pytest
from numpy.typing import NDArray

from panache_archive.daily import daily_max_8h_means, daily_means

FloatArray = NDArray[np.float64]


# Panache's own rules (`panache.climatology`), hour by hour, as references for one cell.


def panache_daily_means(hourly: list[float]) -> list[float]:
    means = []
    for day in range(len(hourly) // 24):
        hours = [v for v in hourly[day * 24 : day * 24 + 24] if not math.isnan(v)]
        means.append(sum(hours) / len(hours) if len(hours) >= 18 else math.nan)
    return means


def panache_daily_max_8h_means(hourly: list[float]) -> list[float]:
    maxima = []
    for day in range(len(hourly) // 24):
        means = []
        for end in range(day * 24 + 1, day * 24 + 25):
            window = [v for v in hourly[max(0, end - 8) : end] if not math.isnan(v)]
            if len(window) >= 6:
                means.append(sum(window) / len(window))
        maxima.append(max(means) if len(means) >= 18 else math.nan)
    return maxima


@pytest.fixture
def hourly() -> FloatArray:
    """Ten days of noisy hourly values on 2 x 3 cells, with gaps of every length."""
    generator = np.random.default_rng(31)
    values = generator.gamma(4.0, 10.0, size=(240, 2, 3))
    values[generator.random(values.shape) < 0.15] = np.nan
    values[30:40, 0, 0] = np.nan  # ten missing hours: a day short of data, means short too
    values[100:124, 1, 2] = np.nan  # a whole day missing
    return values


def test_daily_means_follow_panache(hourly: FloatArray) -> None:
    means = daily_means(hourly)

    assert means.shape == (10, 2, 3)
    for lat in range(2):
        for lon in range(3):
            expected = panache_daily_means(list(hourly[:, lat, lon]))
            np.testing.assert_allclose(means[:, lat, lon], expected, rtol=1e-12)


def test_daily_maxima_of_8_hour_means_follow_panache(hourly: FloatArray) -> None:
    maxima = daily_max_8h_means(hourly)

    assert maxima.shape == (10, 2, 3)
    for lat in range(2):
        for lon in range(3):
            expected = panache_daily_max_8h_means(list(hourly[:, lat, lon]))
            np.testing.assert_allclose(maxima[:, lat, lon], expected, rtol=1e-12)


def test_a_daily_mean_needs_18_valid_hours() -> None:
    hourly = np.full((48,), 10.0)
    hourly[:6] = np.nan  # 18 valid hours left on the first day
    hourly[24:31] = np.nan  # 17 on the second

    np.testing.assert_array_equal(daily_means(hourly), [10.0, np.nan])


def test_missing_hours_are_skipped_never_read_as_zero() -> None:
    hourly = np.array([np.nan] + [8.0] * 12 + [2.0] * 11, dtype=np.float64)

    assert daily_means(hourly)[0] == pytest.approx((12 * 8 + 11 * 2) / 23)
    assert daily_max_8h_means(hourly)[0] == pytest.approx(8.0)


def test_the_evening_before_completes_the_first_day() -> None:
    evening = np.full((10,), 90.0)  # only the last 7 hours count
    day = np.full((24,), 30.0)

    without = daily_max_8h_means(day)
    with_evening = daily_max_8h_means(day, evening)

    assert without[0] == pytest.approx(30.0)
    # The mean ending at 01:00 covers 17:00 to 01:00: 7 evening hours at 90 and one at 30.
    assert with_evening[0] == pytest.approx((7 * 90 + 30) / 8)


def test_a_short_evening_fills_the_latest_hours() -> None:
    day = np.full((24, 1), 30.0)

    maxima = daily_max_8h_means(day, np.full((2, 1), 110.0))

    # The highest mean is the first with 6 valid hours: 22:00 to 04:00, two of them at 110.
    assert maxima[0, 0] == pytest.approx((2 * 110 + 4 * 30) / 6)


def test_the_evening_must_cover_the_same_cells() -> None:
    with pytest.raises(ValueError, match="evening cells"):
        daily_max_8h_means(np.zeros((24, 2)), np.zeros((7, 3)))


@pytest.mark.parametrize("hours", [0, 23, 25])
def test_only_whole_days_are_summed_up(hours: int) -> None:
    with pytest.raises(ValueError, match="not whole days"):
        daily_means(np.zeros((hours, 1)))
    with pytest.raises(ValueError, match="not whole days"):
        daily_max_8h_means(np.zeros((hours, 1)))


def test_a_day_with_too_few_8_hour_means_has_no_maximum() -> None:
    hourly = np.full((24,), 50.0)
    hourly[:13] = np.nan  # means ending before 19:00 have fewer than 6 hours

    assert np.isnan(daily_max_8h_means(hourly)[0])
