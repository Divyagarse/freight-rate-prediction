"""Detection of corrupted ``posted_rate`` labels.

About 1.4% of development labels are off by a multiplicative factor (roughly
0.19-0.45x or 1.85-4.9x the rate their features imply). Clean loads sit within a
few percent of a model estimate, so the two groups are separated by a wide empty
gap and a simple ratio threshold isolates the corrupted rows.

Each row is scored against a model that never saw it (K-fold out-of-fold
predictions), with a Huber loss so the corrupted rows barely move the fit.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config
from .features import GBM_FEATURES

OUTLIER_FEATURES = [*GBM_FEATURES, "days_since_origin"]


def out_of_fold_log_predictions(
    features: pd.DataFrame,
    log_target: pd.Series,
    n_folds: int = config.OUTLIER_CV_FOLDS,
) -> np.ndarray:
    predictions = np.empty(len(features))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)
    for fit_index, score_index in splitter.split(features):
        model = lgb.LGBMRegressor(**config.OUTLIER_LGBM_PARAMS)
        model.fit(features.iloc[fit_index][OUTLIER_FEATURES], log_target.iloc[fit_index])
        predictions[score_index] = model.predict(features.iloc[score_index][OUTLIER_FEATURES])
    return predictions


def label_outlier_mask(
    features: pd.DataFrame,
    log_target: pd.Series,
    bounds: tuple[float, float] = config.LABEL_RATIO_BOUNDS,
) -> np.ndarray:
    """Boolean array, True where the label is implausible for its features."""
    ratio = np.exp(log_target.to_numpy() - out_of_fold_log_predictions(features, log_target))
    low, high = bounds
    return (ratio < low) | (ratio > high)
