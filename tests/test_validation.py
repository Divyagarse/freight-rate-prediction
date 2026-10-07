import numpy as np
import pandas as pd
import pytest

from freight_rate.validation import plausible_label_mask, regression_metrics, rolling_origin_folds


@pytest.fixture
def jan_to_oct() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.date_range("2025-01-01", "2025-10-31", freq="D")})


def test_rolling_origin_folds_match_two_month_horizon(jan_to_oct):
    folds = rolling_origin_folds(jan_to_oct["date"])
    assert [fold.name for fold in folds] == ["May-Jun", "Jul-Aug", "Sep-Oct"]
    assert folds[-1].test_start == pd.Timestamp("2025-09-01")
    assert folds[-1].test_end == pd.Timestamp("2025-11-01")


def test_folds_never_train_on_the_future(jan_to_oct):
    folds = rolling_origin_folds(jan_to_oct["date"])
    covered = []
    for fold in folds:
        train, test = fold.split(jan_to_oct)
        assert train["date"].max() < test["date"].min()
        assert train["date"].min() == pd.Timestamp("2025-01-01")  # expanding window
        covered.extend(test["date"])
    assert len(covered) == len(set(covered))  # test windows do not overlap
    assert max(covered) == pd.Timestamp("2025-10-31")


def test_rolling_origin_folds_require_history(jan_to_oct):
    with pytest.raises(ValueError):
        rolling_origin_folds(jan_to_oct["date"], n_folds=5)


def test_regression_metrics_known_values():
    metrics = regression_metrics([100.0, 200.0], [110.0, 180.0])
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics["mape_pct"] == pytest.approx(10.0)


def test_plausible_label_mask():
    mask = plausible_label_mask([100, 100, 100, 100], [100, 30, 250, 150])
    assert mask.tolist() == [True, False, False, True]
