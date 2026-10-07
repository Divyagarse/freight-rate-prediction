"""Two-stage freight rate model: linear trend model + gradient-boosted residuals.

Why two stages
--------------
Rates drift upward by ~0.02% per day (about +7.5% a year) on top of what
``market_index`` explains, and the final predictions are for Nov-Dec, after
the end of the training data. Tree models cannot extrapolate a trend: past the
last training date they predict as if it were the last training date. So:

1. A linear model on ``log(rate)`` captures the trend plus the main log-linear
   effects (distance, market index, weight, equipment, quote signal). It is the
   only stage that sees absolute time, and it extrapolates smoothly.
2. LightGBM fits the residual using non-linear, interaction and cyclical
   features (lane geometry, weekday, days to quarter end, ...). Every value
   those features take in Nov-Dec also occurs in the training window.

Predictions are ``exp(linear + gbm)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config
from .data import clean_features, equipment_weight_medians
from .features import GBM_FEATURES, LINEAR_FEATURES, build_features
from .outliers import label_outlier_mask


@dataclass
class FreightRateModel:
    daily_context: pd.DataFrame
    lgbm_params: dict = field(default_factory=lambda: dict(config.LGBM_PARAMS))
    linear_features: list[str] = field(default_factory=lambda: list(LINEAR_FEATURES))
    gbm_features: list[str] = field(default_factory=lambda: list(GBM_FEATURES))
    filter_label_outliers: bool = True

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Clean raw rows and add model features."""
        return build_features(clean_features(frame, self.daily_context, self.weight_medians_))

    def fit(self, frame: pd.DataFrame) -> "FreightRateModel":
        self.weight_medians_ = equipment_weight_medians(frame)
        features = self.prepare(frame)
        log_target = np.log(frame[config.TARGET].astype(float))

        outliers = np.zeros(len(features), dtype=bool)
        if self.filter_label_outliers:
            outliers = label_outlier_mask(features, log_target)
        self.n_training_rows_ = int((~outliers).sum())
        self.n_label_outliers_ = int(outliers.sum())
        features, log_target = features[~outliers], log_target[~outliers]

        self.linear_model_ = LinearRegression().fit(features[self.linear_features], log_target)
        residual = log_target - self._linear_log_prediction(features)

        self.gbm_model_ = None
        if self.gbm_features:
            self.gbm_model_ = lgb.LGBMRegressor(**self.lgbm_params)
            self.gbm_model_.fit(features[self.gbm_features], residual)
        return self

    def _linear_log_prediction(self, features: pd.DataFrame) -> np.ndarray:
        return self.linear_model_.predict(features[self.linear_features])

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.prepare(frame)
        log_prediction = self._linear_log_prediction(features)
        if self.gbm_model_ is not None:
            log_prediction = log_prediction + self.gbm_model_.predict(features[self.gbm_features])
        return log_prediction

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))

    def linear_coefficients(self) -> pd.Series:
        return pd.Series(self.linear_model_.coef_, index=self.linear_features)

    def feature_importance(self) -> pd.Series:
        """LightGBM gain importance, normalised to sum to 1."""
        if self.gbm_model_ is None:
            return pd.Series(dtype=float)
        gain = self.gbm_model_.booster_.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.gbm_features).sort_values(ascending=False)
