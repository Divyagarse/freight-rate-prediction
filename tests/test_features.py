import numpy as np
import pandas as pd
import pytest

from freight_rate import data
from freight_rate.features import (
    GBM_FEATURES,
    LINEAR_FEATURES,
    build_features,
    days_to_quarter_end,
    haversine_miles,
)


def test_haversine_one_degree_of_latitude():
    assert haversine_miles(40.0, -90.0, 41.0, -90.0) == pytest.approx(69.09, abs=0.05)
    assert haversine_miles(35.0, -80.0, 35.0, -80.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "date, expected",
    [("2025-03-31", 0), ("2025-03-01", 30), ("2025-04-01", 90), ("2025-12-31", 0), ("2025-12-01", 30), ("2025-10-01", 91)],
)
def test_days_to_quarter_end(date, expected):
    assert days_to_quarter_end(pd.Series(pd.to_datetime([date]))).iloc[0] == expected


def test_build_features_produces_complete_model_inputs(synthetic_loads):
    cleaned = data.clean_features(synthetic_loads, data.build_daily_context(synthetic_loads))
    features = build_features(cleaned)

    for column in set(LINEAR_FEATURES) | set(GBM_FEATURES):
        assert column in features.columns, column
        assert not features[column].isna().any(), column

    ramp = features["quarter_end_ramp"]
    assert ramp.between(0, 1).all()
    assert (ramp[features["days_to_quarter_end"] == 0] == 1).all()
    assert (ramp[features["days_to_quarter_end"] >= 28] == 0).all()
    assert np.allclose(features["log_distance"], np.log(features["distance"]))


def test_build_features_rejects_unknown_equipment(synthetic_loads):
    cleaned = data.clean_features(synthetic_loads.head(5), data.build_daily_context(synthetic_loads))
    cleaned.loc[cleaned.index[0], "equipment"] = "Tanker"
    with pytest.raises(ValueError, match="unknown equipment"):
        build_features(cleaned)
