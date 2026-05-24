import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    EXTENDED_FEATURES,
    FREQ,
    FORECAST_SCHEME,
    HAR_FEATURES,
    HORIZON,
    OUTPUT_FIGURE_DIR,
    OUTPUT_TABLE_DIR,
    RANDOM_SEED,
    RESULT_DIR,
    ROLLING_REFIT_FREQUENCY,
    TARGET_TRANSFORM,
    TICKERS,
)
from data_loader import load_intraday
from features import build_features
from models import fit_predict_models
from plots import (
    plot_feature_importance,
    plot_feature_importance_average,
    plot_relative_mse,
    plot_relative_mse_by_stock,
    plot_rv_series,
)
from realized_measures import compute_daily_measures
from utils import ensure_directories, experiment_suffix


def run(args: argparse.Namespace) -> None:
    ensure_directories(DATA_PROCESSED_DIR, RESULT_DIR, OUTPUT_TABLE_DIR, OUTPUT_FIGURE_DIR)
    result_figure_dir = RESULT_DIR / "figures"
    ensure_directories(result_figure_dir)
    suffix = experiment_suffix(args.horizon, args.target_transform, args.forecast_scheme)

    all_results = []
    all_importance = []
    all_predictions = []
    daily_by_ticker = {}

    for ticker in args.tickers:
        print(f"\n=== {ticker} ===", flush=True)
        intraday = load_intraday(ticker, DATA_RAW_DIR)
        daily = compute_daily_measures(intraday, freq=args.freq)
        features = build_features(daily, horizon=args.horizon)

        daily.to_csv(DATA_PROCESSED_DIR / f"{ticker}_daily_measures.csv")
        features.to_csv(DATA_PROCESSED_DIR / f"{ticker}_features_{suffix}.csv")
        daily_by_ticker[ticker] = daily

        print(
            f"observations: intraday={len(intraday):,}, daily={len(daily):,}, "
            f"model={len(features):,}",
            flush=True,
        )
        if "n_intraday_returns" in daily:
            print(
                "average intraday returns per day: "
                f"{daily['n_intraday_returns'].mean():.2f}",
                flush=True,
            )

        for setting, feature_cols in {
            "har_only": HAR_FEATURES,
            "extended": EXTENDED_FEATURES,
        }.items():
            result, importance, predictions = fit_predict_models(
                features,
                feature_cols=list(feature_cols),
                setting=setting,
                ticker=ticker,
                horizon=args.horizon,
                random_state=args.seed,
                target_transform=args.target_transform,
                forecast_scheme=args.forecast_scheme,
                rolling_refit_frequency=args.rolling_refit_frequency,
                include_mlp=args.include_mlp,
                mlp_refit_frequency=args.mlp_refit_frequency,
                mlp_architecture=args.mlp_architecture,
            )
            all_results.append(result)
            all_predictions.append(predictions)
            if importance is not None:
                all_importance.append(importance)

    results = pd.concat(all_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    model_comparison_path = (
        RESULT_DIR / f"model_comparison_{args.target_transform}_{args.forecast_scheme}.csv"
    )
    prediction_path = (
        RESULT_DIR / f"predictions_{args.target_transform}_{args.forecast_scheme}.csv"
    )
    results.to_csv(OUTPUT_TABLE_DIR / f"model_results_{suffix}.csv", index=False)
    results.to_csv(model_comparison_path, index=False)
    predictions.to_csv(OUTPUT_TABLE_DIR / f"predictions_{suffix}.csv", index=False)
    predictions.to_csv(prediction_path, index=False)

    avg = (
        results.groupby(
            ["target_transform", "forecast_scheme", "feature_set", "model"],
            as_index=False,
        )
        .agg(
            avg_rel_mse=("rel_mse", "mean"),
            avg_rel_mae=("rel_mae", "mean"),
            avg_qlike=("qlike", "mean"),
        )
        .sort_values(["feature_set", "avg_rel_mse"])
    )
    summary_path = (
        RESULT_DIR / f"relative_mse_summary_{args.target_transform}_{args.forecast_scheme}.csv"
    )
    avg.to_csv(OUTPUT_TABLE_DIR / f"avg_relative_mse_{suffix}.csv", index=False)
    avg.to_csv(summary_path, index=False)

    plot_rv_series(daily_by_ticker, OUTPUT_FIGURE_DIR / f"rv_timeseries_{suffix}.png")
    plot_relative_mse(results, OUTPUT_FIGURE_DIR / f"relative_mse_{suffix}.png")
    plot_relative_mse_by_stock(
        predictions,
        OUTPUT_FIGURE_DIR
        / f"relative_mse_by_stock_{args.target_transform}_{args.forecast_scheme}.png",
        result_figure_dir
        / f"relative_mse_by_stock_{args.target_transform}_{args.forecast_scheme}.png",
        target_transform=args.target_transform,
        forecast_scheme=args.forecast_scheme,
    )

    if all_importance:
        importance = pd.concat(all_importance, ignore_index=True)
        importance.to_csv(OUTPUT_TABLE_DIR / f"feature_importance_{suffix}.csv", index=False)
        importance.to_csv(
            RESULT_DIR / f"feature_importance_{args.target_transform}_{args.forecast_scheme}.csv",
            index=False,
        )
        plot_feature_importance(importance, OUTPUT_FIGURE_DIR)
        plot_feature_importance_average(
            importance, result_figure_dir, args.target_transform
        )
    else:
        for candidate in [
            RESULT_DIR / f"feature_importance_{args.target_transform}_{args.forecast_scheme}.csv",
            RESULT_DIR / f"feature_importance_{args.target_transform}.csv",
        ]:
            if candidate.exists():
                prior_importance = pd.read_csv(candidate)
                plot_feature_importance_average(
                    prior_importance, result_figure_dir, args.target_transform
                )
                break

    print("\nSaved tables:")
    print(f"  {model_comparison_path}")
    print(f"  {prediction_path}")
    print(f"  {summary_path}")
    print(f"  {OUTPUT_TABLE_DIR / f'model_results_{suffix}.csv'}")
    print(f"  {OUTPUT_TABLE_DIR / f'avg_relative_mse_{suffix}.csv'}")
    if all_importance:
        print(f"  {OUTPUT_TABLE_DIR / f'feature_importance_{suffix}.csv'}")

    print("\nAverage relative MSE:")
    print(avg.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--freq", default=FREQ)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--target-transform", choices=["level", "log"], default=TARGET_TRANSFORM)
    parser.add_argument("--forecast-scheme", choices=["fixed", "rolling"], default=FORECAST_SCHEME)
    parser.add_argument("--rolling-refit-frequency", type=int, default=ROLLING_REFIT_FREQUENCY)
    parser.add_argument(
        "--include-mlp",
        action="store_true",
        help="include MLP in the rolling model comparison",
    )
    parser.add_argument(
        "--mlp-refit-frequency",
        type=int,
        default=None,
        help=(
            "refit frequency for rolling MLP; defaults to "
            "--rolling-refit-frequency, use 1 for daily MLP refits"
        ),
    )
    parser.add_argument(
        "--mlp-architecture",
        choices=["nn1", "nn2", "nn3", "nn4", "all"],
        default="nn2",
        help="MLP architecture to run when --include-mlp is set",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
