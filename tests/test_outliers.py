import numpy as np

from freight_rate import config, data
from freight_rate.features import build_features
from freight_rate.outliers import label_outlier_mask

from conftest import corrupt


def test_corrupted_labels_are_recovered(synthetic_loads):
    dirty, truly_bad = corrupt(synthetic_loads, fraction=0.02)
    features = build_features(data.clean_features(dirty, data.build_daily_context(dirty)))

    flagged = label_outlier_mask(features, np.log(dirty[config.TARGET]))

    recall = (flagged & truly_bad).sum() / truly_bad.sum()
    precision = (flagged & truly_bad).sum() / flagged.sum()
    assert recall >= 0.95, recall
    assert precision >= 0.95, precision


def test_clean_data_is_not_flagged(synthetic_loads):
    features = build_features(data.clean_features(synthetic_loads, data.build_daily_context(synthetic_loads)))
    flagged = label_outlier_mask(features, np.log(synthetic_loads[config.TARGET]))
    assert flagged.mean() < 0.001
