import numpy as np
import pandas as pd
import pytest

from freight_rate import config, data
from freight_rate.features import GBM_FEATURES, LINEAR_FEATURES
from freight_rate.model import FreightRateModel
from freight_rate.validation import regression_metrics

from conftest import corrupt, make_loads


@pytest.fixture(scope="module")
def train_and_future():
    """Jan-Oct training data and Nov-Dec 'future' loads, like the real task."""
    train = make_loads(n_loads=5000, start="2025-01-01", end="2025-10-31", seed=0)
    future = make_loads(n_loads=1500, start="2025-11-01", end="2025-12-31", seed=1)
    return train, future, data.build_daily_context(train, future)


def test_fit_predict_contract(train_and_future, fast_lgbm_params):
    train, future, context = train_and_future
    model = FreightRateModel(context, lgbm_params=fast_lgbm_params).fit(train)
    predictions = model.predict(future.drop(columns=[config.TARGET]))

    assert predictions.shape == (len(future),)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    assert model.n_label_outliers_ + model.n_training_rows_ == len(train)
    assert model.feature_importance().sum() == pytest.approx(1.0)


def test_linear_stage_recovers_true_trend(train_and_future, fast_lgbm_params):
    from conftest import TRUE_TREND_PER_DAY

    train, _, context = train_and_future
    model = FreightRateModel(context, lgbm_params=fast_lgbm_params).fit(train)
    assert model.linear_coefficients()["days_since_origin"] == pytest.approx(TRUE_TREND_PER_DAY, rel=0.15)


def test_trend_is_extrapolated_where_trees_alone_fall_flat(train_and_future, fast_lgbm_params):
    """The core design choice: the linear stage carries the trend past the training window."""
    train, future, context = train_and_future
    actual = future[config.TARGET].to_numpy()

    final = FreightRateModel(context, lgbm_params=fast_lgbm_params).fit(train)
    trees_only = FreightRateModel(
        context,
        lgbm_params=fast_lgbm_params,
        linear_features=[f for f in LINEAR_FEATURES if f != "days_since_origin"],
        gbm_features=[*GBM_FEATURES, "days_since_origin"],
    ).fit(train)

    final_metrics = regression_metrics(actual, final.predict(future))
    trees_metrics = regression_metrics(actual, trees_only.predict(future))
    # Trees alone predict Nov-Dec at roughly the late-October level, so they under-shoot.
    assert trees_metrics["bias_pct"] < -0.4
    assert abs(final_metrics["bias_pct"]) < abs(trees_metrics["bias_pct"]) / 3
    assert final_metrics["mape_pct"] < trees_metrics["mape_pct"]


def test_label_outlier_filter_protects_the_fit(train_and_future, fast_lgbm_params):
    train, future, context = train_and_future
    dirty, bad = corrupt(train, fraction=0.03)
    actual = future[config.TARGET].to_numpy()

    filtered = FreightRateModel(context, lgbm_params=fast_lgbm_params).fit(dirty)
    unfiltered = FreightRateModel(context, lgbm_params=fast_lgbm_params, filter_label_outliers=False).fit(dirty)

    assert filtered.n_label_outliers_ == pytest.approx(bad.sum(), rel=0.05)
    assert (
        regression_metrics(actual, filtered.predict(future))["mape_pct"]
        < regression_metrics(actual, unfiltered.predict(future))["mape_pct"]
    )


def test_predictions_handle_dirty_feature_rows(train_and_future, fast_lgbm_params):
    train, future, context = train_and_future
    model = FreightRateModel(context, lgbm_params=fast_lgbm_params).fit(train)
    dirty = future.head(3).copy()
    dirty.loc[dirty.index[0], "weight"] = np.nan
    dirty.loc[dirty.index[1], "weight"] = -abs(dirty.loc[dirty.index[1], "weight"])
    dirty.loc[dirty.index[2], "market_index"] = np.nan

    predictions = model.predict(dirty)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    # A sign-flipped weight must give the same prediction as the corrected one.
    fixed = dirty.iloc[[1]].assign(weight=lambda f: f["weight"].abs())
    assert predictions[1] == pytest.approx(model.predict(fixed)[0])
