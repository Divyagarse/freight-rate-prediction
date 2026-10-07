"""Time-based validation of the development data.

The final predictions are for the two months *after* the development data ends
(train: Jan-Oct 2025, predict: Nov-Dec 2025). A random split would leak the
future into training and hide how the model degrades with forecast horizon, so
validation uses rolling-origin folds that each mimic that setup: train on
everything before a cutoff, test on the next two months.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class TimeFold:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive

    def split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Train on every row before ``test_start``; test on [test_start, test_end)."""
        train = frame[frame["date"] < self.test_start]
        test = frame[(frame["date"] >= self.test_start) & (frame["date"] < self.test_end)]
        return train, test


def rolling_origin_folds(
    dates: pd.Series,
    n_folds: int = 3,
    horizon_months: int = 2,
) -> list[TimeFold]:
    """Folds whose test windows are consecutive ``horizon_months`` blocks ending at the last date.

    With Jan-Oct data and the defaults: test May-Jun, Jul-Aug and Sep-Oct, trained
    on Jan-Apr, Jan-Jun and Jan-Aug respectively.
    """
    last_month = dates.max().to_period("M")
    folds = []
    for index in range(n_folds, 0, -1):
        start = (last_month - index * horizon_months + 1).to_timestamp()
        end = (last_month - (index - 1) * horizon_months + 1).to_timestamp()
        label = f"{start:%b}-{(end - pd.Timedelta(days=1)):%b}"
        folds.append(TimeFold(name=label, test_start=start, test_end=end))
    if folds[0].test_start <= dates.min():
        raise ValueError("not enough history for the requested folds")
    return folds


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    pct_error = np.abs(error) / actual
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * np.mean(pct_error)),
        "median_ape_pct": float(100 * np.median(pct_error)),
        "bias_pct": float(100 * (np.exp(np.median(np.log(predicted / actual))) - 1)),
    }


def plausible_label_mask(actual, predicted, bounds=config.LABEL_RATIO_BOUNDS) -> np.ndarray:
    """Rows whose label is within the plausible ratio band around the prediction."""
    ratio = np.asarray(actual, dtype=float) / np.asarray(predicted, dtype=float)
    return (ratio >= bounds[0]) & (ratio <= bounds[1])


def evaluate_fold(model_factory: Callable[[], object], frame: pd.DataFrame, fold: TimeFold) -> list[dict]:
    """Fit on the fold's past, score on its future.

    Two rows of metrics are returned: on every test load ("all") and on test
    loads whose label is plausible ("clean"). Corrupted labels (~1.4%) are off by
    2-5x and would otherwise dominate MAE/RMSE, hiding real model differences.
    """
    train, test = fold.split(frame)
    model = model_factory().fit(train)
    predicted = model.predict(test)
    actual = test[config.TARGET].to_numpy()
    clean = plausible_label_mask(actual, predicted)
    base = {"fold": fold.name, "n_train": len(train)}
    return [
        {**base, "subset": "all", **regression_metrics(actual, predicted)},
        {**base, "subset": "clean", **regression_metrics(actual[clean], predicted[clean])},
    ]


def backtest(model_factory: Callable[[], object], frame: pd.DataFrame, folds: list[TimeFold]) -> pd.DataFrame:
    rows = [row for fold in folds for row in evaluate_fold(model_factory, frame, fold)]
    return pd.DataFrame(rows)


def unseen_city_check(
    model_factory: Callable[[], object],
    frame: pd.DataFrame,
    fold: TimeFold,
    n_cities: int = 8,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Simulate cities that never appear in training (8 validation cities are new).

    Every training load touching ``n_cities`` randomly chosen cities is dropped;
    test-window error is then compared for loads that do and do not touch them.
    """
    cities = sorted(set(frame["pickup"]) | set(frame["delivery"]))
    held_out = set(np.random.default_rng(seed).choice(cities, n_cities, replace=False))

    def touches(rows: pd.DataFrame) -> np.ndarray:
        return (rows["pickup"].isin(held_out) | rows["delivery"].isin(held_out)).to_numpy()

    train, test = fold.split(frame)
    model = model_factory().fit(train[~touches(train)])
    predicted = model.predict(test)
    actual = test[config.TARGET].to_numpy()
    clean = plausible_label_mask(actual, predicted)
    rows = []
    for name, subset in [("unseen city", touches(test)), ("seen cities", ~touches(test))]:
        mask = clean & subset
        rows.append({"fold": fold.name, "lanes": name, **regression_metrics(actual[mask], predicted[mask])})
    return pd.DataFrame(rows)
