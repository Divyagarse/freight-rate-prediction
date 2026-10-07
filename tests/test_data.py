import numpy as np
import pandas as pd
import pytest

from freight_rate import config, data


@pytest.fixture
def raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]),
            "equipment": ["Dry Van", "Reefer", "Dry Van", "Reefer"],
            "weight": [30_000.0, -20_000.0, np.nan, 60_000.0],
            "market_index": [1.0, np.nan, 0.9, 1.1],
            "quote_signal": [2.0, 2.2, 1.8, 2.0],
            config.TARGET: [500.0, 600.0, 700.0, 800.0],
        }
    )


def test_daily_context_is_per_date_mean(raw):
    context = data.build_daily_context(raw)
    assert context.loc["2025-01-01", "market_index_day"] == pytest.approx(1.0)  # NaN ignored
    assert context.loc["2025-01-02", "market_index_day"] == pytest.approx(1.0)
    assert context.loc["2025-01-02", "quote_signal_day"] == pytest.approx(1.9)


def test_clean_features_fixes_weight_and_market_index(raw):
    context = data.build_daily_context(raw)
    medians = pd.Series({"Dry Van": 25_000.0, "Reefer": 35_000.0})
    cleaned = data.clean_features(raw, context, medians)

    assert cleaned["weight"].tolist() == [30_000.0, 20_000.0, 25_000.0, config.MAX_WEIGHT]
    assert cleaned["weight_missing"].tolist() == [0, 0, 1, 0]
    assert cleaned.loc[1, "market_index"] == pytest.approx(1.0)  # filled with that day's mean
    assert not cleaned[["weight", "market_index"]].isna().any().any()


def test_clean_features_leaves_input_and_labels_untouched(raw):
    before = raw.copy()
    cleaned = data.clean_features(raw, data.build_daily_context(raw))
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_series_equal(cleaned[config.TARGET], raw[config.TARGET])


def test_city_coordinates_lookup(synthetic_loads):
    coordinates = data.build_city_coordinates(synthetic_loads)
    row = synthetic_loads.iloc[0]
    assert coordinates.loc[row["pickup"]].tolist() == [row["pickup_lat"], row["pickup_lon"]]
    assert coordinates.loc[row["delivery"]].tolist() == [row["delivery_lat"], row["delivery_lon"]]


def test_december_chart_frame_fills_missing_inputs(synthetic_loads):
    context = data.build_daily_context(synthetic_loads)
    coordinates = data.build_city_coordinates(synthetic_loads)
    city_a, city_b = coordinates.index[:2]
    chart = pd.DataFrame(
        {
            "pickup": [city_a] * 2,
            "delivery": [city_b] * 2,
            "distance": ["360", "360"],
            "equipment": ["Dry Van"] * 2,
            "weight": ["32000", "32000"],
            "date": ["2025-03-01", "2025-03-02"],
        }
    )
    frame = data.december_chart_frame(chart, context, coordinates)

    assert frame["pickup_lat"].eq(coordinates.loc[city_a, "lat"]).all()
    assert frame["delivery_lon"].eq(coordinates.loc[city_b, "lon"]).all()
    assert frame["market_index"].tolist() == context.loc[frame["date"], "market_index_day"].tolist()
    assert frame["quote_signal"].tolist() == context.loc[frame["date"], "quote_signal_day"].tolist()


def test_december_chart_frame_rejects_unknown_city(synthetic_loads):
    context = data.build_daily_context(synthetic_loads)
    coordinates = data.build_city_coordinates(synthetic_loads)
    chart = pd.DataFrame(
        {"pickup": ["Atlantis"], "delivery": [coordinates.index[0]], "distance": [100],
         "equipment": ["Dry Van"], "weight": [30000], "date": ["2025-03-01"]}
    )
    with pytest.raises(ValueError, match="unknown pickup"):
        data.december_chart_frame(chart, context, coordinates)
