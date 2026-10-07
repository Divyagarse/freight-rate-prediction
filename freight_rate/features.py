"""Feature engineering on cleaned load data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EARTH_RADIUS_MILES = 3958.8

# Columns used by the linear trend model (the only stage that sees calendar time,
# so it is the only stage that extrapolates into Nov/Dec).
LINEAR_NUMERIC = [
    "days_since_origin",
    "quarter_end_ramp",
    "log_market_index",
    "log_distance",
    "log_weight",
    "log_quote_signal",
]
LINEAR_EQUIPMENT = [f"equipment_{name}" for name in config.EQUIPMENT_TYPES[1:]]
LINEAR_FEATURES = LINEAR_NUMERIC + LINEAR_EQUIPMENT

# Columns used by the gradient-boosting residual model. Calendar time enters only
# through cyclical features (weekday, position in quarter) that repeat within the
# training window, never through an absolute date the trees could not extrapolate.
GBM_FEATURES = [
    "distance",
    "log_distance",
    "great_circle_miles",
    "circuity",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "equipment",
    "weight",
    "weight_missing",
    "market_index",
    "quote_signal",
    "quote_signal_day",
    "day_of_week",
    "days_to_quarter_end",
]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    half_chord = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(half_chord))


def days_to_quarter_end(dates: pd.Series) -> pd.Series:
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates.dt.normalize()).dt.days


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a frame already passed through ``clean_features``."""
    result = frame.copy()
    dates = result["date"]

    result["days_since_origin"] = (dates - config.TREND_ORIGIN).dt.days.astype(float)
    result["day_of_week"] = dates.dt.dayofweek
    result["days_to_quarter_end"] = days_to_quarter_end(dates)
    result["quarter_end_ramp"] = np.clip(
        1 - result["days_to_quarter_end"] / config.QUARTER_END_RAMP_DAYS, 0, None
    )

    result["log_distance"] = np.log(result["distance"])
    result["great_circle_miles"] = haversine_miles(
        result["pickup_lat"], result["pickup_lon"], result["delivery_lat"], result["delivery_lon"]
    )
    # Short lanes can have near-zero great-circle distance in this data, so floor it.
    result["circuity"] = result["distance"] / np.maximum(result["great_circle_miles"], 1.0)
    result["delta_lat"] = result["delivery_lat"] - result["pickup_lat"]
    result["delta_lon"] = result["delivery_lon"] - result["pickup_lon"]

    result["log_weight"] = np.log(np.maximum(result["weight"], 1.0))
    result["log_market_index"] = np.log(result["market_index"])
    result["log_quote_signal"] = np.log(result["quote_signal"])

    result["equipment"] = pd.Categorical(result["equipment"], categories=config.EQUIPMENT_TYPES)
    for name, column in zip(config.EQUIPMENT_TYPES[1:], LINEAR_EQUIPMENT):
        result[column] = (result["equipment"] == name).astype(float)

    if result["equipment"].isna().any():
        raise ValueError(f"unknown equipment type; expected one of {config.EQUIPMENT_TYPES}")
    return result
