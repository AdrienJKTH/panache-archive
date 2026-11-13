# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chronos-forecasting>=2.2",
#     "numpy>=2.0",
#     "pandas>=2.2",
#     "pyarrow>=20.0",
#     "tabpfn-time-series>=1.3",
#     "torch>=2.6",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Panache's D89: two foundation models forecast PM2.5 from J+1 to J+7 without any training.

The table is Panache's `panache-forecast-experiment build`: per place and day, the CAMS
reanalysis's PM2.5, ERA5's daily weather and ECMWF IFS's archived runs at each lead. On each issue
day `t`, a model reads the place's past up to `t - 1` (ln(PM2.5 + 0.1), with ERA5's weather as
past covariates) and the run of day `t` as known future covariates (the day `t + j` from lead
`j`), and forecasts the eight days `t` to `t + 7`. Its quantiles at 20 equal levels are the
members Panache scores, from J+1 to J+7.

- Chronos-2 (Amazon, arXiv 2510.15821): code and weights under Apache 2.0.
- TabPFN-TS (Prior Labs, arXiv 2501.02945): code under Apache 2.0; its default TabPFN-3.5 weights
  are non-commercial and need an account, so it runs here on the TabPFN-2 regressor's weights
  (Prior Labs License: Apache 2.0 with an attribution requirement), on the CPU, nothing sent to a
  cloud. Telemetry is off.

Run as `uv run --script experiments/forecast_foundation.py --help`: torch and the models stay
out of every project's dependencies.
"""

import argparse
import logging
import math
import os
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

FLOOR: Final = 0.1
TEST_START: Final = date(2024, 3, 1)
TEST_END: Final = date(2025, 12, 31)
LEADS: Final = range(1, 8)
HORIZON: Final = 8
"""The days `t` to `t + 7`."""
MEMBERS: Final = 20
LEVELS: Final = tuple((i + 0.5) / MEMBERS for i in range(MEMBERS))
WEATHER: Final = (
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "precipitation_sum",
    "shortwave_radiation_sum",
    "cloud_cover_mean",
)
COVARIATES: Final = (
    "temperature",
    "humidity",
    "wind_speed",
    "wind_from_east",
    "wind_from_north",
    "pressure",
    "precipitation",
    "radiation",
    "cloud_cover",
)
CHRONOS_CONTEXT: Final = 512
CHRONOS_BATCH: Final = 64
TABPFN_CONTEXT: Final = 365
TABPFN_BATCH: Final = 50

logger = logging.getLogger("forecast_foundation")


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    first: date
    log_pm: np.ndarray
    reanalysis: np.ndarray
    """(day, covariate): ERA5."""
    runs: np.ndarray
    """(lead, day, covariate): IFS's run of `day - lead`, for `day`."""

    def index(self, day: date) -> int:
        return (day - self.first).days


def covariates(columns: dict[str, np.ndarray]) -> np.ndarray:
    """The daily weather as Panache's `features.weather` sees it: the direction as components."""
    speed = columns["wind_speed_10m_mean"]
    direction = np.radians(columns["wind_direction_10m_dominant"])
    return np.stack(
        [
            columns["temperature_2m_mean"],
            columns["relative_humidity_2m_mean"],
            speed,
            speed * np.sin(direction),
            speed * np.cos(direction),
            columns["pressure_msl_mean"],
            columns["precipitation_sum"],
            columns["shortwave_radiation_sum"],
            columns["cloud_cover_mean"],
        ],
        axis=-1,
    )


def places(path: Path, country: str) -> list[Place]:
    table = pq.read_table(path).filter(pc.field("country") == country)
    frame = table.to_pandas()
    out = []
    for name, rows in frame.groupby("place", sort=False):
        rows = rows.sort_values("day")
        first = rows["day"].iloc[0]
        era5 = covariates({v: rows[f"era5_{v}"].to_numpy(float) for v in WEATHER})
        runs = np.stack(
            [
                covariates({v: rows[f"ifs{lead}_{v}"].to_numpy(float) for v in WEATHER})
                for lead in range(HORIZON)
            ]
        )
        log_pm = np.log(rows["pm2_5"].to_numpy(float) + FLOOR)
        out.append(Place(str(name), first, log_pm, era5, runs))
    if not out:
        raise SystemExit(f"no place of {country} in {path}")
    return out


@dataclass(frozen=True, slots=True)
class Issue:
    place: Place
    day: date

    @property
    def at(self) -> int:
        return self.place.index(self.day)

    def future(self) -> np.ndarray:
        """(day, covariate) for `t` to `t + 7`, each from the run of `t`."""
        return np.stack([self.place.runs[j, self.at + j] for j in range(HORIZON)])


def issues(selected: Sequence[Place], every: int) -> list[Issue]:
    """Issue days whose J+1 to J+7 reach the test, whose run and context are whole: the last week's
    short leads are lost, and Panache scores every method on the days left."""
    # The run of the last issue day must reach t + 7 inside the table, which ends with the test.
    first, last = TEST_START - timedelta(days=LEADS[-1]), TEST_END - timedelta(days=HORIZON - 1)
    out = []
    for place in selected:
        day = first
        while day <= last:
            issue = Issue(place, day)
            context = place.log_pm[issue.at - CHRONOS_CONTEXT : issue.at]
            if np.isfinite(issue.future()).all() and np.isfinite(context).all():
                out.append(issue)
            day += timedelta(days=every)
    return out


def rows(issue: Issue, quantiles: np.ndarray, method: str) -> Iterator[dict[str, Any]]:
    """`quantiles` (level, day `t` to `t + 7`) as one row per lead and target."""
    for lead in LEADS:
        target = issue.day + timedelta(days=lead)
        if TEST_START <= target <= TEST_END:
            yield {
                "method": method,
                "place": issue.place.name,
                "lead": lead,
                "target": target,
                "members": np.sort(quantiles[:, lead]).astype(float).tolist(),
            }


def batches(items: Sequence[Issue], size: int) -> Iterator[Sequence[Issue]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chronos(selected: Sequence[Issue]) -> Iterator[dict[str, Any]]:
    import torch
    from chronos import Chronos2Pipeline

    torch.set_num_threads(os.cpu_count() or 1)
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    for batch in batches(selected, CHRONOS_BATCH):
        inputs = []
        for issue in batch:
            past = slice(issue.at - CHRONOS_CONTEXT, issue.at)
            future = issue.future()
            inputs.append(
                {
                    "target": issue.place.log_pm[past],
                    "past_covariates": {
                        name: issue.place.reanalysis[past, j] for j, name in enumerate(COVARIATES)
                    },
                    "future_covariates": {name: future[:, j] for j, name in enumerate(COVARIATES)},
                }
            )
        quantiles, _mean = pipeline.predict_quantiles(
            inputs, prediction_length=HORIZON, quantile_levels=list(LEVELS)
        )
        for issue, made in zip(batch, quantiles, strict=True):
            # (variate, day, level) for the target, its only variate.
            values = np.asarray(made, dtype=float)[0].T
            yield from rows(issue, values, "chronos2")


def tabpfn(selected: Sequence[Issue]) -> Iterator[dict[str, Any]]:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline, TimeSeriesDataFrame

    weights = TabPFNRegressor.create_default_for_version(ModelVersion.V2).model_path
    pipeline = TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_model_config={"model_path": weights},
        compile=False,
    )
    for batch in batches(selected, TABPFN_BATCH):
        context, future = [], []
        for n, issue in enumerate(batch):
            days = pd.date_range(
                issue.day - timedelta(days=TABPFN_CONTEXT), periods=TABPFN_CONTEXT + HORIZON
            )
            past = slice(issue.at - TABPFN_CONTEXT, issue.at)
            weather = np.vstack([issue.place.reanalysis[past], issue.future()])
            frame = pd.DataFrame(weather, columns=list(COVARIATES))
            frame.insert(0, "timestamp", days)
            frame.insert(0, "item_id", n)
            target = np.concatenate([issue.place.log_pm[past], np.full(HORIZON, np.nan)])
            frame.insert(2, "target", target)
            context.append(frame.iloc[:TABPFN_CONTEXT])
            future.append(frame.iloc[TABPFN_CONTEXT:])
        made = pipeline.predict(
            TimeSeriesDataFrame(pd.concat(context)),
            TimeSeriesDataFrame(pd.concat(future)),
            quantiles=list(LEVELS),
        )
        for n, issue in enumerate(batch):
            mine = made.loc[n]
            values = np.stack([_quantile_column(mine, level) for level in LEVELS])
            yield from rows(issue, values, "tabpfn_ts")


def _quantile_column(frame: pd.DataFrame, level: float) -> np.ndarray:
    for name in (level, str(level), f"{level:g}"):
        if name in frame.columns:
            return frame[name].to_numpy(float)
    raise KeyError(f"no column for the quantile {level} among {list(frame.columns)}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Panache's experiment table")
    parser.add_argument("--method", choices=("chronos2", "tabpfn_ts"), required=True)
    parser.add_argument("--country", required=True, help="the country whose places to forecast")
    parser.add_argument("--every", type=int, default=1, help="one issue day in this many")
    parser.add_argument("--out", type=Path, required=True, help="the forecasts, as Parquet")
    args = parser.parse_args()
    selected = issues(places(args.data, args.country), args.every)
    logger.info("%s: %d issue days in %s", args.method, len(selected), args.country)
    started = time.monotonic()
    forecast = chronos if args.method == "chronos2" else tabpfn
    made = list(forecast(selected))
    elapsed = time.monotonic() - started
    logger.info(
        "%d forecasts in %.0f s (%.2f s an issue)", len(made), elapsed, elapsed / len(selected)
    )
    if any(not all(math.isfinite(v) for v in row["members"]) for row in made):
        raise SystemExit("a forecast is not finite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(made), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
