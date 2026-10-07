# Freight Rate Prediction

Solution to the Spotter ML Engineer freight-rate assessment (the brief is summarised at the end of this file).

The model is a **linear trend model plus LightGBM on its residuals**, trained on `log(posted_rate)` after
corrupted labels are filtered out. On the backtest fold that best mimics the real task (train on Jan–Aug,
predict the next two months, Sep–Oct), it reaches **1.23% MAPE / $28 MAE** on loads with plausible labels,
with −0.08% bias.

| Deliverable | Location |
|---|---|
| Predictions for the 12,000 validation loads | [`validation_predictions.csv`](validation_predictions.csv) |
| Filled December chart inputs | [`data/december_chart_inputs.csv`](data/december_chart_inputs.csv) |
| December chart (from `score.py`) | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| PDF report (validation, split approach, chart) | [`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf) |
| Backtest metrics / model summary | [`reports/`](reports/) |

## Run it

Python 3.10+.

**Data.** The assessment's input files are not committed to this public repository. Copy them into `data/`
with these names (the paths used in the brief):

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
```

`data/december_chart_inputs.csv` is committed with its `predicted_rate` column already filled. Re-running
the pipeline overwrites it.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt

python -m freight_rate --compare-baselines   # backtest, train, write both prediction files (~2 min)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python scripts/build_report.py               # rebuild reports/Freight_Rate_Report.pdf
python -m pytest -v                          # 35 tests; the 2 that read the raw CSVs skip without them
```

`python -m freight_rate` without flags backtests only the final model. `--skip-backtest` goes straight to
training and prediction. Every run is deterministic (fixed seeds).

## Repository layout

```
freight_rate/            the solution package
  config.py              paths, constants, hyper-parameters
  data.py                loading, cleaning, per-date market context, December input preparation
  features.py            feature engineering (trend, quarter-end ramp, lane geometry, ...)
  outliers.py            out-of-fold detection of corrupted labels
  model.py               FreightRateModel: linear trend stage + LightGBM residual stage
  validation.py          rolling-origin time folds, metrics, backtest, unseen-city check
  pipeline.py            end-to-end CLI (python -m freight_rate)
scripts/build_report.py  builds the PDF report from the artifacts in reports/
tests/                   pytest suite: unit tests on synthetic data + integration tests on the real outputs
data/                    input CSVs (not committed, see "Data" above) + filled December inputs
score.py                 provided scorer (unchanged)
```

## Approach

### Key findings

* **The task is a forecast.** Development data covers Jan 1 – Oct 31, 2025. The validation loads are all from
  Nov 1 – Dec 31, 2025.
* **Distance dominates.** `log(rate) ≈ 0.87·log(distance)`, so rate per mile falls as trips get longer. Reefer costs
  about 13% more than Dry Van, and Flatbed about 8% more.
* **Market index explains only part of the time variation** (an elasticity of about 0.15). On top of it:
  * rates drift up by **about 7.5% a year**;
  * rates climb **about 4% over the last four weeks of every quarter**, then reset (visible in Mar, Jun and Sep).
    December is a quarter-end month.
* **`market_index` and `quote_signal` are mostly date-level signals**: a per-day mean plus small per-load noise.
* **Coordinates are synthetic but consistent** (one pair per city). Eight validation cities never appear in
  training (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego).

### Data-quality issues

| Issue | Dev / Val rows | Handling |
|---|---|---|
| Negative `weight` (sign flips: same magnitude range as valid weights) | 292 / 145 | `abs()` |
| Missing `weight` | 300 / 165 | median for the equipment type, plus a missing-value flag |
| Missing `market_index` | 374 / 249 | that day's mean `market_index` |
| Corrupted `posted_rate`: 0.19–0.45× or 1.85–4.9× the rate implied by the features | 677 (1.4%) / – | flagged by an out-of-fold Huber LightGBM and dropped from training |

Clean labels fall within about 5% of the out-of-fold estimate, and nothing falls between 0.45× and 1.85×.
That means the 0.6–1.6× cutoff needs no tuning.

### Model

1. **Linear stage** (OLS on `log(rate)`): time trend, quarter-end ramp, log market index, log distance,
   log weight, log quote signal, equipment. It is the only stage that sees absolute time, so it carries
   the trend smoothly into Nov–Dec.
2. **LightGBM stage** fits the remaining residual using lane geometry (coordinates, great-circle distance,
   circuity), weight, equipment, quote signal (per load and per day), weekday and days to quarter end.
   Every value those features take in Nov–Dec also appears in the training data.

Tree models cannot extrapolate. Given a date feature directly, a GBM predicts Nov–Dec as if it were still
late October. In the backtest that variant was biased by −1.1%.

### Validation and split

The split is **rolling-origin (expanding window) with a 2-month horizon**, mirroring the real Jan–Oct → Nov–Dec
setup. There are three folds: train Jan–Apr → test May–Jun, Jan–Jun → Jul–Aug, and Jan–Aug → Sep–Oct. The
label-outlier filter is re-fitted inside each fold's training window. Metrics are reported on all test loads,
and on "clean" loads whose label is within 0.6–1.6× of the prediction. The ~1.4% of corrupted labels cannot be
predicted, and they push RMSE to about $625 for every model.

Mean over the three folds, clean loads:

| Model | MAE | MAPE | Bias |
|---|---|---|---|
| **Final: linear trend + LightGBM** | **$41.1** | **1.69%** | +0.58% |
| LightGBM with date feature (no linear trend) | $43.3 | 1.93% | −1.13% |
| Linear model only | $60.5 | 2.83% | +0.08% |
| Final model without the label-outlier filter | $79.0 | 3.48% | +0.78% |

Per fold, the final model scores 1.27% (May–Jun), 2.59% (Jul–Aug) and 1.23% (Sep–Oct). Jul–Aug is the hardest
fold: its training window ends on the June quarter-end spike, and during those months market index and time rise
together, which inflates the trend slope. With the full Jan–Oct window the slope estimate is stable.

In an unseen-city check, 8 cities were removed from training. MAPE on lanes touching them was 1.29%, against
1.23% for the other lanes.

### December chart

The chart inputs fix the lane (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) and vary only the date.
Coordinates come from the city lookup. `market_index` and `quote_signal` are set to each date's mean across the
validation loads. The weekly wave in the chart follows the weekly cycle of the December market index, and the
rise into late December is the quarter-end effect plus the trend.

## Original brief

1. Train and validate using `data/train_test.csv`.
2. Predict every load in `data/validation.csv` (unique `load_id`).
3. Fill `predicted_rate` in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Fill `predicted_rate` for every row of `data/december_chart_inputs.csv`.
5. Run `python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv`,
   which validates both files and creates `scorer_results/candidate_december.png`.

Submit: GitHub repository (code, dependencies, run instructions), `validation_predictions.csv`, a PDF/DOCX report
covering validation, the data split approach and `candidate_december.png`, and a 2–3 minute Loom.
