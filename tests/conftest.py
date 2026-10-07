"""Shared fixtures: a small synthetic load dataset with known structure.

The generator mirrors the real data closely enough to exercise every code path
(cities with fixed coordinates, three equipment types, daily market index with
per-load noise, an upward time trend, missing / sign-flipped weights, missing
market_index) while being small enough for fast tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freight_rate import config
from freight_rate.features import haversine_miles

TRUE_TREND_PER_DAY = 0.0003
EQUIPMENT_EFFECT = {"Dry Van": 0.0, "Flatbed": 0.08, "Reefer": 0.12}


def make_loads(
    n_loads: int = 4000,
    start: str = "2025-01-01",
    end: str = "2025-10-31",
    seed: int = 0,
    noise: float = 0.02,
    trend_per_day: float = TRUE_TREND_PER_DAY,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city_rng = np.random.default_rng(1234)  # same cities for every call
    cities = pd.DataFrame(
        {"lat": city_rng.uniform(28, 45, 20), "lon": city_rng.uniform(-120, -70, 20)},
        index=[f"City{i:02d}" for i in range(20)],
    )

    dates = pd.date_range(start, end, freq="D")
    date = pd.Series(rng.choice(dates, n_loads))
    pickup = rng.choice(cities.index, n_loads)
    delivery = np.array([rng.choice(cities.index.drop(p)) for p in pickup])
    p_lat, p_lon = cities.loc[pickup, "lat"].to_numpy(), cities.loc[pickup, "lon"].to_numpy()
    d_lat, d_lon = cities.loc[delivery, "lat"].to_numpy(), cities.loc[delivery, "lon"].to_numpy()
    distance = np.maximum(70.0, haversine_miles(p_lat, p_lon, d_lat, d_lon) * 1.18).round(1)

    equipment = rng.choice(list(EQUIPMENT_EFFECT), n_loads, p=[0.55, 0.2, 0.25])
    weight = rng.uniform(10_000, 45_000, n_loads).round()
    days = (date - config.TREND_ORIGIN).dt.days.to_numpy()
    market_day = 1.0 + 0.2 * np.sin(2 * np.pi * days / 120)
    market_index = market_day * (1 + rng.normal(0, 0.02, n_loads))
    quote_signal = rng.normal(2.05, 0.2, n_loads)

    log_rate = (
        0.75
        + 0.87 * np.log(distance)
        + 0.15 * np.log(market_index)
        + 0.08 * np.log(weight / 30_000)
        + trend_per_day * days
        + np.array([EQUIPMENT_EFFECT[e] for e in equipment])
        + rng.normal(0, noise, n_loads)
    )

    return pd.DataFrame(
        {
            "load_id": [f"TR-{i:06d}" for i in range(1, n_loads + 1)],
            "pickup": pickup,
            "delivery": delivery,
            "pickup_lat": p_lat,
            "pickup_lon": p_lon,
            "delivery_lat": d_lat,
            "delivery_lon": d_lon,
            "distance": distance,
            "equipment": equipment,
            "weight": weight,
            "date": date,
            "market_index": market_index,
            "quote_signal": quote_signal,
            config.TARGET: np.exp(log_rate).round(2),
        }
    )


def corrupt(frame: pd.DataFrame, fraction: float = 0.02, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject the real data's quality issues. Returns the frame and the corrupted-label mask."""
    rng = np.random.default_rng(seed)
    result = frame.copy()
    n = len(result)

    bad_label = rng.random(n) < fraction
    factor = np.where(rng.random(n) < 0.5, rng.uniform(0.2, 0.45, n), rng.uniform(1.9, 4.9, n))
    result.loc[bad_label, config.TARGET] = (result.loc[bad_label, config.TARGET] * factor[bad_label]).round(2)

    result.loc[rng.random(n) < 0.01, "weight"] = np.nan
    flip = rng.random(n) < 0.01
    result.loc[flip, "weight"] = -result.loc[flip, "weight"]
    result.loc[rng.random(n) < 0.01, "market_index"] = np.nan
    return result, bad_label


@pytest.fixture(scope="session")
def synthetic_loads() -> pd.DataFrame:
    return make_loads()


@pytest.fixture(scope="session")
def fast_lgbm_params() -> dict:
    return {**config.LGBM_PARAMS, "n_estimators": 200, "learning_rate": 0.08}
