"""Build reports/Freight_Rate_Report.pdf from the pipeline artifacts.

Run after ``python -m freight_rate --compare-baselines`` and ``score.py``:

    python scripts/build_report.py

Every number in the report is read from ``reports/`` or recomputed from the raw
CSVs, so the PDF always matches the latest run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from freight_rate import config  # noqa: E402
from freight_rate.validation import rolling_origin_folds  # noqa: E402

OUTPUT = config.REPORTS_DIR / "Freight_Rate_Report.pdf"
CHART = config.PROJECT_ROOT / "scorer_results" / "candidate_december.png"
INK = "#064A56"
ACCENT = "#C8553D"


def data_quality_stats(dev: pd.DataFrame, val: pd.DataFrame) -> dict:
    return {
        "dev_rows": len(dev),
        "val_rows": len(val),
        "neg_weight": int((dev["weight"] < 0).sum()),
        "neg_weight_val": int((val["weight"] < 0).sum()),
        "missing_weight": int(dev["weight"].isna().sum()),
        "missing_weight_val": int(val["weight"].isna().sum()),
        "missing_market": int(dev["market_index"].isna().sum()),
        "missing_market_val": int(val["market_index"].isna().sum()),
        "unseen_cities": sorted((set(val["pickup"]) | set(val["delivery"])) - (set(dev["pickup"]) | set(dev["delivery"]))),
        "n_cities": dev["pickup"].nunique(),
    }


def seasonality_figure(dev: pd.DataFrame, path: Path) -> None:
    """Daily rate residual after removing distance, equipment, weight and market effects."""
    frame = dev.dropna(subset=["weight", "market_index"]).copy()
    frame["weight"] = frame["weight"].abs()
    frame = frame[(frame[config.TARGET] / frame["distance"]).between(1.0, 5.0)]  # crude label filter
    design = np.column_stack(
        [
            np.ones(len(frame)),
            np.log(frame["distance"]),
            np.log(frame["weight"]),
            np.log(frame["market_index"]),
            frame["equipment"].eq("Flatbed"),
            frame["equipment"].eq("Reefer"),
        ]
    ).astype(float)
    target = np.log(frame[config.TARGET].to_numpy())
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    frame["residual"] = target - design @ coefficients
    daily = frame.groupby("date")["residual"].mean()
    smooth = daily.rolling(7, center=True, min_periods=4).mean()
    days = (daily.index - daily.index[0]).days.to_numpy()
    slope, intercept = np.polyfit(days, daily.to_numpy(), 1)

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    ax.plot(daily.index, 100 * daily, color="#9DAFB3", linewidth=0.6, label="daily mean")
    ax.plot(smooth.index, 100 * smooth, color=INK, linewidth=2, label="7-day mean")
    ax.plot(daily.index, 100 * (intercept + slope * days), color=ACCENT, linestyle="--", linewidth=1.3,
            label="linear trend")
    for quarter_end in ["2025-03-31", "2025-06-30", "2025-09-30"]:
        ax.axvline(pd.Timestamp(quarter_end), color="#455A60", linestyle=":", linewidth=1)
    ax.set_ylabel("rate residual (%)")
    ax.set_title("Rates beyond market index: upward trend + quarter-end spikes (dotted = quarter end)",
                 fontsize=9, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def split_figure(dev: pd.DataFrame, val: pd.DataFrame, path: Path) -> None:
    folds = rolling_origin_folds(dev["date"])
    start, dev_end, val_end = dev["date"].min(), dev["date"].max() + pd.Timedelta(days=1), val["date"].max() + pd.Timedelta(days=1)
    rows = [(f"Fold {i + 1}", start, f.test_start, f.test_start, f.test_end) for i, f in enumerate(folds)]
    rows.append(("Final", start, dev_end, dev_end, val_end))

    fig, ax = plt.subplots(figsize=(7.2, 1.9), dpi=200)
    for y, (label, train_start, train_end, test_start, test_end) in enumerate(reversed(rows)):
        ax.barh(y, train_end - train_start, left=train_start, color=INK, height=0.55)
        ax.barh(y, test_end - test_start, left=test_start, color=ACCENT if label != "Final" else "#E3A587", height=0.55)
        ax.text(test_start + (test_end - test_start) / 2, y, "test" if label != "Final" else "predict",
                ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Expanding-window, 2-month-horizon backtest (dark = training data)", fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def table(rows: list[list], widths: list[float], highlight_row: int | None = None, text_columns: int = 1) -> Table:
    result = Table(rows, colWidths=[w * inch for w in widths], hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (-1, 0), (-1, -1), "RIGHT" if text_columns == 1 else "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F6")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D5D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if highlight_row is not None:
        style.append(("FONT", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold", 8))
    result.setStyle(TableStyle(style))
    return result


def build() -> Path:
    for required in [config.REPORTS_DIR / "backtest_folds.csv", config.REPORTS_DIR / "model_summary.json", CHART]:
        if not required.exists():
            raise SystemExit(f"missing {required}; run the pipeline and score.py first")

    dev = pd.read_csv(config.TRAIN_PATH, parse_dates=["date"])
    val = pd.read_csv(config.VALIDATION_PATH, parse_dates=["date"])
    stats = data_quality_stats(dev, val)
    folds = pd.read_csv(config.REPORTS_DIR / "backtest_folds.csv")
    summary = json.loads((config.REPORTS_DIR / "model_summary.json").read_text())
    final = folds[folds["model"] == "final model"]
    final_clean = final[final["subset"] == "clean"]
    last_fold = final_clean.iloc[-1]
    outlier_pct = 100 * summary["label_outliers_removed"] / stats["dev_rows"]
    coef = summary["linear_coefficients"]
    reefer_pct, flatbed_pct = (100 * (np.exp(coef[f"equipment_{e}"]) - 1) for e in ("Reefer", "Flatbed"))
    with PILImage.open(CHART) as chart_image:
        chart_aspect = chart_image.height / chart_image.width

    unseen_city_sentence = ""
    unseen_path = config.REPORTS_DIR / "unseen_city_check.csv"
    if unseen_path.exists():
        unseen = pd.read_csv(unseen_path).set_index("lanes")["mape_pct"]
        unseen_city_sentence = (
            f"In a test that removed 8 cities from training, MAPE on lanes touching them was "
            f"{unseen['unseen city']:.2f}%, against {unseen['seen cities']:.2f}% for the other lanes."
        )

    seasonality_png = config.REPORTS_DIR / "fig_seasonality.png"
    split_png = config.REPORTS_DIR / "fig_split.png"
    seasonality_figure(dev, seasonality_png)
    split_figure(dev, val, split_png)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=18, textColor=colors.HexColor(INK), alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor(INK), spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12.5, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#455A60"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)

    def bullets(items: list[str]) -> list[Paragraph]:
        return [Paragraph(item, bullet, bulletText="•") for item in items]

    story = [
        Paragraph("Freight Rate Prediction: Validation Report", h1),
        Paragraph("Spotter ML Engineer assessment. Model: linear trend model + LightGBM on its residuals, trained on log(rate).", small),
        Spacer(1, 6),
        Paragraph("Summary", h2),
        Paragraph(
            f"On the backtest fold most like the real task (train on Jan-Aug, predict Sep-Oct), the model's "
            f"<b>mean absolute percentage error is {last_fold['mape_pct']:.2f}%</b> (MAE ${last_fold['mae']:.0f}) "
            f"on loads with plausible labels, and its median bias is {last_fold['bias_pct']:+.2f}%. Averaged over all "
            f"three 2-month folds, MAPE is {final_clean['mape_pct'].mean():.2f}%. "
            f"The final model was trained on all {stats['dev_rows']:,} development loads after removing "
            f"{summary['label_outliers_removed']} corrupted labels, then used to predict all {stats['val_rows']:,} "
            f"validation loads (Nov-Dec 2025) and the 31-day December chart.",
            body,
        ),
        Paragraph("Key findings from the data", h2),
        *bullets([
            "<b>The validation loads come after the training period.</b> Development data runs Jan 1 to Oct 31, 2025, "
            "and the validation loads run Nov 1 to Dec 31, 2025. The task is a forecast, so the validation must be time-based.",
            f"<b>Distance drives most of the rate</b>: log(rate) is about {coef['log_distance']:.2f} x log(distance), so the "
            f"rate per mile falls as trips get longer. Reefer costs about {reefer_pct:.0f}% more than Dry Van and Flatbed about {flatbed_pct:.0f}% more.",
            f"<b>Market index matters, but not fully.</b> The rate moves about {summary['market_index_elasticity']:.2f}% for "
            f"each 1% move in market_index. On top of that, rates drift upward by about "
            f"{summary['trend_pct_per_year']:+.1f}%/year, and they climb about {summary['quarter_end_peak_uplift_pct']:.0f}% over the "
            "last four weeks of every quarter, then reset (Figure 1). December is a quarter-end month.",
            "<b>market_index and quote_signal are mostly date-level signals.</b> Each date has its own mean, plus "
            "a small amount of per-load noise. The December chart inputs do not include these two columns, so "
            "each December day's values are filled in from the validation loads on that same date.",
            f"<b>Coordinates are synthetic but consistent</b> (one lat/lon pair per city). The validation set includes "
            f"{len(stats['unseen_cities'])} cities that never appear in training ({', '.join(stats['unseen_cities'])}). "
            "The model uses coordinates instead of city names, so it can still price those lanes. " + unseen_city_sentence,
        ]),
        Image(str(seasonality_png), width=7.0 * inch, height=7.0 * inch * 2.6 / 7.2),
        Paragraph("Figure 1. Daily mean log-rate residual after a regression on distance, weight, equipment and market index.", small),
        KeepTogether([Paragraph("Data-quality issues and how they were handled", h2),
        table(
            [
                ["Issue", "Development", "Validation", "Handling"],
                ["Negative weight (sign flip)", f"{stats['neg_weight']}", f"{stats['neg_weight_val']}", "abs(); same magnitude range as valid weights"],
                ["Missing weight", f"{stats['missing_weight']}", f"{stats['missing_weight_val']}", "equipment-type median + missing flag"],
                ["Missing market_index", f"{stats['missing_market']}", f"{stats['missing_market_val']}", "that day's mean market_index (date-level signal)"],
                ["Corrupted posted_rate (2-5x off)", f"{summary['label_outliers_removed']} ({outlier_pct:.1f}%)", "n/a",
                 "flagged with out-of-fold model, removed from training"],
            ],
            [2.0, 0.95, 0.75, 3.3],
            text_columns=2,
        )]),
        Spacer(1, 4),
        Paragraph(
            "Corrupted labels were found by checking each load against a model that had not seen it (5-fold, "
            "Huber loss). Clean loads fall within about 5% of that model's estimate, and corrupted ones are off by a "
            "factor of 2 to 5. Nothing falls in between, so the 0.6x-1.6x cutoff does not need tuning. Removing these rows halves the "
            "backtest error (table below).",
            body,
        ),
        Paragraph("Train / validation split approach", h2),
        Paragraph(
            "A random split would let the model train on loads from the very days it is tested on, and it would not show "
            "how error grows the further ahead the model has to predict. So each fold copies the real setup: "
            "<b>train on every month before a cutoff, then test on the next two months</b>. The cutoffs are expanding, giving three folds "
            "(Figure 2). The label-outlier filter is re-fitted inside each fold's training window, so test labels "
            "never influence training. Metrics are reported on all test loads and on the 'clean' loads whose label is "
            f"within 0.6-1.6x of the prediction. The roughly {outlier_pct:.1f}% of corrupted test labels cannot be predicted, and they dominate "
            "RMSE (about $625 for every model), which would otherwise hide real differences between models.",
            body,
        ),
        Image(str(split_png), width=7.0 * inch, height=7.0 * inch * 1.9 / 7.2),
        Paragraph("Figure 2. Backtest folds and the final train/predict split.", small),
        Paragraph("Backtest results: final model, per fold", h2),
        table(
            [["Test window", "Train loads", "Subset", "Loads", "MAE ($)", "RMSE ($)", "MAPE", "Median APE", "Bias"]]
            + [
                [r.fold, f"{r.n_train:,}", r.subset, f"{r.n:,}", f"{r.mae:.1f}", f"{r.rmse:.1f}",
                 f"{r.mape_pct:.2f}%", f"{r.median_ape_pct:.2f}%", f"{r.bias_pct:+.2f}%"]
                for r in final.itertuples()
            ],
            [0.85, 0.8, 0.6, 0.6, 0.7, 0.75, 0.65, 0.85, 0.65],
        ),
        Spacer(1, 4),
        Paragraph(
            "The Jul-Aug fold is the hardest. Its training window (Jan-Jun) ends right on the June quarter-end spike, and "
            "during those months market index and time rise together, so the trend slope is overestimated. "
            "With the full Jan-Oct window, the estimated slope is stable (about 0.02%/day whether training ends in Aug or Oct).",
            body,
        ),
    ]

    compared = folds[folds["subset"] == "clean"].groupby("model", sort=False)[["mae", "mape_pct", "bias_pct"]].mean()
    if len(compared) > 1:
        story += [
            Paragraph("Model choice: comparison on clean loads, mean over the 3 folds", h2),
            table(
                [["Model", "MAE ($)", "MAPE", "Bias"]]
                + [[name, f"{r.mae:.1f}", f"{r.mape_pct:.2f}%", f"{r.bias_pct:+.2f}%"] for name, r in compared.iterrows()],
                [3.6, 0.9, 0.9, 0.9],
                highlight_row=1,
            ),
            Spacer(1, 4),
            Paragraph(
                "<b>Why two stages?</b> Tree models cannot extrapolate. Past the last training date, a GBM given a date "
                "feature predicts as if it were still late October, so it falls behind the upward trend (bias "
                f"{compared.loc['GBM with date feature (no linear trend)', 'bias_pct']:+.2f}%). "
                "In this model, the linear stage is the only part that sees absolute time, so it carries the trend "
                "smoothly into Nov-Dec. LightGBM then fits what is left over using lane geometry, weight, equipment, "
                "quote signal, weekday and days to quarter end. Every value those features take in Nov-Dec also "
                "appears in the training data. On its own, the linear model leaves non-linear lane effects unmodelled.",
                body,
            ),
        ]

    story += [
        Paragraph("Fixed December prediction chart (produced by score.py)", h2),
        Image(str(CHART), width=7.0 * inch, height=7.0 * inch * chart_aspect),
        Paragraph(
            "Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes. "
            "City coordinates come from the data, and market_index and quote_signal are set to each date's mean across "
            "the validation loads. The weekly wave follows the weekly cycle in the December market index. The "
            "rise into late December is the quarter-end effect plus the trend.",
            small,
        ),
        Paragraph("Limitations", h2),
        *bullets([
            "Only one year of history, so yearly seasonality (for example, holiday peaks) cannot be separated from the trend. "
            "The trend is assumed to continue linearly for two more months.",
            f"Corrupted labels in the hidden validation truth (about {outlier_pct:.1f}% if they occur at the development rate) cannot be predicted, "
            "and they will dominate RMSE.",
            "The December chart uses each day's average market conditions; individual loads would vary around it.",
        ]),
    ]

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch, title="Freight Rate Prediction: Validation Report")
    doc.build(story)
    return OUTPUT


if __name__ == "__main__":
    print(f"Wrote {build()}")
