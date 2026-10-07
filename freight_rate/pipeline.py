"""End-to-end run: validate, train on all development data, write submission files.

Usage::

    python -m freight_rate                      # backtest + train + predict
    python -m freight_rate --skip-backtest      # train + predict only
    python -m freight_rate --compare-baselines  # also backtest alternative models
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config, data
from .features import GBM_FEATURES, LINEAR_FEATURES
from .model import FreightRateModel
from .validation import backtest, rolling_origin_folds, unseen_city_check


def baseline_factories(daily_context: pd.DataFrame) -> dict:
    """Alternatives the final model is compared against in the backtest."""
    without_trend = [f for f in LINEAR_FEATURES if f != "days_since_origin"]
    return {
        "linear model only": lambda: FreightRateModel(daily_context, gbm_features=[]),
        "GBM with date feature (no linear trend)": lambda: FreightRateModel(
            daily_context,
            linear_features=without_trend,
            gbm_features=[*GBM_FEATURES, "days_since_origin"],
        ),
        "final model without label-outlier filter": lambda: FreightRateModel(
            daily_context, filter_label_outliers=False
        ),
    }


def run_backtest(dev: pd.DataFrame, daily_context: pd.DataFrame, compare_baselines: bool) -> pd.DataFrame:
    folds = rolling_origin_folds(dev["date"])
    factories = {"final model": lambda: FreightRateModel(daily_context)}
    if compare_baselines:
        factories.update(baseline_factories(daily_context))

    results = []
    for name, factory in factories.items():
        started = time.time()
        result = backtest(factory, dev, folds)
        result.insert(0, "model", name)
        results.append(result)
        print(f"  backtested {name!r} in {time.time() - started:.0f}s")
    return pd.concat(results, ignore_index=True)


def summarise_backtest(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "rmse", "mape_pct", "median_ape_pct", "bias_pct"]
    return results.groupby(["model", "subset"], sort=False)[metrics].mean()


def fill_validation_template(template: pd.DataFrame, load_ids: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    """Fill the template's ``predicted_rate`` by ``load_id``, keeping its row order."""
    lookup = pd.Series(predictions, index=load_ids.to_numpy())
    if lookup.index.duplicated().any():
        raise ValueError("duplicate load_id in validation data")
    filled = template[[config.ID_COLUMN]].copy()
    filled[config.PREDICTION_COLUMN] = filled[config.ID_COLUMN].map(lookup).round(2)
    if filled[config.PREDICTION_COLUMN].isna().any():
        raise ValueError("some template load_ids have no prediction")
    return filled


def fill_december_inputs(model: FreightRateModel, daily_context, city_coordinates) -> pd.DataFrame:
    """Predict the 31 December chart rows, leaving the other six columns byte-identical."""
    raw = pd.read_csv(config.DECEMBER_PATH, dtype=str)
    features = data.december_chart_frame(raw.drop(columns=[config.PREDICTION_COLUMN]), daily_context, city_coordinates)
    raw[config.PREDICTION_COLUMN] = np.round(model.predict(features), 2)
    return raw


def model_summary(model: FreightRateModel) -> dict:
    coefficients = model.linear_coefficients()
    return {
        "training_rows_used": model.n_training_rows_,
        "label_outliers_removed": model.n_label_outliers_,
        "trend_pct_per_year": float(100 * (np.exp(365 * coefficients["days_since_origin"]) - 1)),
        "quarter_end_peak_uplift_pct": float(100 * (np.exp(coefficients["quarter_end_ramp"]) - 1)),
        "market_index_elasticity": float(coefficients["log_market_index"]),
        "linear_coefficients": coefficients.round(6).to_dict(),
        "gbm_gain_importance": model.feature_importance().round(4).to_dict(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-backtest", action="store_true", help="skip time-based validation")
    parser.add_argument("--compare-baselines", action="store_true", help="also backtest alternative models")
    args = parser.parse_args(argv)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    dev = data.load_development()
    validation = data.load_validation()
    daily_context = data.build_daily_context(dev, validation)
    city_coordinates = data.build_city_coordinates(dev, validation)
    print(f"Loaded {len(dev):,} development loads ({dev['date'].min():%Y-%m-%d} to {dev['date'].max():%Y-%m-%d})")
    print(f"Loaded {len(validation):,} validation loads ({validation['date'].min():%Y-%m-%d} to {validation['date'].max():%Y-%m-%d})")

    if not args.skip_backtest:
        print("\nRolling-origin backtest (train on all prior months, test on the next two):")
        results = run_backtest(dev, daily_context, args.compare_baselines)
        results.to_csv(config.REPORTS_DIR / "backtest_folds.csv", index=False)
        summary = summarise_backtest(results)
        summary.to_csv(config.REPORTS_DIR / "backtest_summary.csv")
        with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
            print(results.drop(columns=["n_train"]).to_string(index=False))
            print("\nMean over folds:")
            print(summary.to_string())

        if args.compare_baselines:
            print("\nUnseen-city check (8 cities removed from training, last fold):")
            cities = unseen_city_check(lambda: FreightRateModel(daily_context), dev, rolling_origin_folds(dev["date"])[-1])
            cities.to_csv(config.REPORTS_DIR / "unseen_city_check.csv", index=False)
            with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
                print(cities.to_string(index=False))

    print("\nTraining final model on all development data...")
    model = FreightRateModel(daily_context).fit(dev)
    summary = model_summary(model)
    (config.REPORTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  removed {summary['label_outliers_removed']} corrupted labels, trained on {summary['training_rows_used']:,} loads")
    print(f"  learned trend: {summary['trend_pct_per_year']:+.1f}%/year; quarter-end peak uplift: {summary['quarter_end_peak_uplift_pct']:+.1f}%")

    template = pd.read_csv(config.TEMPLATE_PATH)
    predictions = fill_validation_template(template, validation[config.ID_COLUMN], model.predict(validation))
    predictions.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"\nWrote {len(predictions):,} predictions to {config.PREDICTIONS_PATH.name}")

    december = fill_december_inputs(model, daily_context, city_coordinates)
    december.to_csv(config.DECEMBER_PATH, index=False)
    rates = december[config.PREDICTION_COLUMN]
    print(f"Filled {len(december)} December chart rows in {config.DECEMBER_PATH.relative_to(config.PROJECT_ROOT)} "
          f"(range ${rates.min():,.2f} - ${rates.max():,.2f})")


if __name__ == "__main__":
    main()
