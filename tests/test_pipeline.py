"""Pipeline helpers, plus integration checks on the real submission files.

The integration tests run the provided ``score.py`` validators against the files
written by ``python -m freight_rate``; they are skipped if those files have not
been generated yet.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import score
from freight_rate import config
from freight_rate.pipeline import fill_validation_template


def test_fill_validation_template_keeps_template_order():
    template = pd.DataFrame({"load_id": ["B", "A", "C"], "predicted_rate": [np.nan] * 3})
    filled = fill_validation_template(template, pd.Series(["A", "B", "C"]), np.array([1.234, 2.0, 3.0]))
    assert filled.columns.tolist() == ["load_id", "predicted_rate"]
    assert filled["load_id"].tolist() == ["B", "A", "C"]
    assert filled["predicted_rate"].tolist() == [2.0, 1.23, 3.0]


def test_fill_validation_template_rejects_missing_ids():
    template = pd.DataFrame({"load_id": ["A", "Z"], "predicted_rate": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="no prediction"):
        fill_validation_template(template, pd.Series(["A"]), np.array([1.0]))


@pytest.fixture(scope="module")
def submission():
    if not config.PREDICTIONS_PATH.exists():
        pytest.skip("run `python -m freight_rate` first to generate submission files")
    return pd.read_csv(config.PREDICTIONS_PATH)


@pytest.fixture(scope="module")
def december():
    frame = pd.read_csv(config.DECEMBER_PATH)
    if frame[config.PREDICTION_COLUMN].isna().all():
        pytest.skip("run `python -m freight_rate` first to fill the December inputs")
    return frame


@pytest.fixture(scope="module")
def raw_data():
    paths = [config.TRAIN_PATH, config.VALIDATION_PATH, config.TEMPLATE_PATH]
    if not all(path.exists() for path in paths):
        pytest.skip("assessment CSVs are not in data/ (they are not committed; see README)")


@pytest.mark.integration
def test_submission_passes_official_validator(submission):
    score.validate_predictions(submission)  # raises SystemExit on any problem


@pytest.mark.integration
def test_submission_matches_validation_loads(submission, raw_data):
    validation = pd.read_csv(config.VALIDATION_PATH)
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert submission["load_id"].tolist() == template["load_id"].tolist()
    assert set(submission["load_id"]) == set(validation["load_id"])


@pytest.mark.integration
def test_submission_values_are_plausible(submission, raw_data):
    validation = pd.read_csv(config.VALIDATION_PATH).merge(submission, on="load_id")
    rate_per_mile = validation["predicted_rate"] / validation["distance"]
    assert rate_per_mile.between(1.0, 5.0).all()
    development = pd.read_csv(config.TRAIN_PATH)
    assert validation["predicted_rate"].median() == pytest.approx(development[config.TARGET].median(), rel=0.1)


@pytest.mark.integration
def test_december_passes_official_validator(december):
    validated = score.validate_december(december)
    assert len(validated) == 31


@pytest.mark.integration
def test_backtest_meets_accuracy_bar():
    path = config.REPORTS_DIR / "backtest_summary.csv"
    if not path.exists():
        pytest.skip("backtest summary not generated")
    summary = pd.read_csv(path).set_index(["model", "subset"])
    assert summary.loc[("final model", "clean"), "mape_pct"] < 2.5


@pytest.mark.integration
def test_score_script_end_to_end(submission, december, tmp_path):
    result = subprocess.run(
        [sys.executable, "score.py", "--predictions", str(config.PREDICTIONS_PATH),
         "--december-predictions", str(config.DECEMBER_PATH), "--output-dir", str(tmp_path)],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 12,000 final predictions." in result.stdout
    assert (tmp_path / "candidate_december.png").stat().st_size > 0
