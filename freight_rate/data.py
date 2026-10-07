"""Loading and cleaning of the raw CSV files.

Data-quality issues handled here (see README "Data quality" for the evidence):

* ``weight`` has ~0.6% negative values whose magnitudes match the normal weight
  distribution -> sign flips, fixed with ``abs``. ~0.6% are missing -> filled
  with the equipment-type median and flagged.
* ``market_index`` is missing for ~0.8% of loads. It is a date-level market
  reading plus small per-load noise, so gaps are filled with that day's mean.
* ``posted_rate`` has ~1.4% corrupted labels; those are handled in
  :mod:`freight_rate.outliers` because detecting them needs a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV and parse its ``date`` column when present."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; copy the assessment CSVs into data/ (see README)")
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_development(path: Path = config.TRAIN_PATH) -> pd.DataFrame:
    frame = read_csv(path)
    _require_columns(frame, [config.ID_COLUMN, *FEATURE_COLUMNS, config.TARGET], path)
    return frame


def load_validation(path: Path = config.VALIDATION_PATH) -> pd.DataFrame:
    frame = read_csv(path)
    _require_columns(frame, [config.ID_COLUMN, *FEATURE_COLUMNS], path)
    return frame


def _require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def build_daily_context(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date market context from the feature columns of every available load.

    Only features (never labels) are used, and each date's value comes from loads
    on that same date, so this introduces no look-ahead or target leakage.
    """
    stacked = pd.concat([frame[["date", "market_index", "quote_signal"]] for frame in frames])
    daily = stacked.groupby("date").agg(
        market_index_day=("market_index", "mean"),
        quote_signal_day=("quote_signal", "mean"),
    )
    return daily.sort_index()


def build_city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup. Each city has exactly one coordinate pair in the data."""
    pickups = pd.concat(
        frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for frame in frames
    )
    deliveries = pd.concat(
        frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for frame in frames
    )
    return pd.concat([pickups, deliveries]).groupby("city")[["lat", "lon"]].first()


def clean_features(
    frame: pd.DataFrame,
    daily_context: pd.DataFrame,
    weight_medians: pd.Series | None = None,
) -> pd.DataFrame:
    """Return a cleaned copy of the feature columns. Labels are left untouched.

    ``weight_medians`` (per equipment type) should come from the training data so
    the same fill values are used at train and predict time.
    """
    result = frame.copy()

    result["weight_missing"] = result["weight"].isna().astype(int)
    result["weight"] = result["weight"].abs().clip(upper=config.MAX_WEIGHT)
    if weight_medians is None:
        weight_medians = equipment_weight_medians(result)
    fill = result["equipment"].map(weight_medians).fillna(weight_medians.median())
    result["weight"] = result["weight"].fillna(fill)

    context = daily_context.reindex(result["date"])
    market_day = pd.Series(context["market_index_day"].to_numpy(), index=result.index)
    quote_day = pd.Series(context["quote_signal_day"].to_numpy(), index=result.index)
    result["market_index"] = result["market_index"].fillna(market_day)
    result["quote_signal_day"] = quote_day.fillna(result["quote_signal"])

    if result[["weight", "market_index"]].isna().any().any():
        raise ValueError("cleaning left missing weight/market_index values")
    return result


def equipment_weight_medians(frame: pd.DataFrame) -> pd.Series:
    return frame.assign(weight=frame["weight"].abs()).groupby("equipment")["weight"].median()


def december_chart_frame(
    december: pd.DataFrame,
    daily_context: pd.DataFrame,
    city_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Turn the 7-column December chart inputs into full model-ready rows.

    The chart file has no coordinates, ``market_index`` or ``quote_signal``.
    Coordinates come from the city lookup; the two market signals are set to
    that day's mean across the validation loads, so the only thing that varies
    between rows is the date and the market conditions observed on that date.
    """
    result = december.copy()
    result["date"] = pd.to_datetime(result["date"])
    for side in ("pickup", "delivery"):
        coordinates = city_coordinates.reindex(result[side])
        if coordinates.isna().any().any():
            raise ValueError(f"unknown {side} city in December inputs")
        result[f"{side}_lat"] = coordinates["lat"].to_numpy()
        result[f"{side}_lon"] = coordinates["lon"].to_numpy()

    context = daily_context.reindex(result["date"])
    if context.isna().any().any():
        raise ValueError("no market context available for some December dates")
    result["market_index"] = context["market_index_day"].to_numpy()
    result["quote_signal"] = context["quote_signal_day"].to_numpy()
    result["distance"] = result["distance"].astype(float)
    result["weight"] = result["weight"].astype(float)
    return result
