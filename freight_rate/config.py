"""Central configuration: file locations, column names and model hyper-parameters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
ID_COLUMN = "load_id"
PREDICTION_COLUMN = "predicted_rate"
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]

# Day zero for the linear time-trend feature. Any fixed date works; this is the
# first day of the development data so the coefficient reads as "per day since".
TREND_ORIGIN = pd.Timestamp("2025-01-01")

# Rates climb over roughly the last four weeks of every quarter (~+4% at the
# peak) and reset when the next quarter starts. The linear stage models this as
# a ramp from 0 (28 days out) to 1 (quarter end); LightGBM refines the shape.
QUARTER_END_RAMP_DAYS = 28

# Weights in the raw data are capped at this value (it is also the max abs value
# of the sign-flipped negatives), so it is used as the upper clip after cleaning.
MAX_WEIGHT = 47_500.0

# Label-noise filter. Clean loads sit within ~±5% of an out-of-fold estimate,
# while corrupted labels are 0.19-0.45x or 1.85-4.9x it, with nothing in between.
# Any bound inside that empty gap gives the same result.
LABEL_RATIO_BOUNDS = (0.6, 1.6)
OUTLIER_CV_FOLDS = 5

RANDOM_SEED = 42

# Gradient boosting stage, fitted on the residual of the linear trend model.
LGBM_PARAMS: dict = {
    "objective": "regression",
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Small, fast model used only to flag corrupted labels.
OUTLIER_LGBM_PARAMS: dict = {
    "objective": "huber",
    "alpha": 0.3,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "num_leaves": 31,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
