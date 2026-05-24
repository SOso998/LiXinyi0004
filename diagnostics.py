import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import RESULT_DIR
from diagnostics_tools import (
    compute_posthoc_ale,
    lightweight_mcs,
    load_prediction_file,
    loss_matrix,
    pairwise_matrices,
    starred_relative_mse,
    validate_real_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-transform", choices=["log", "level"], default="log")
    parser.add_argument("--forecast-scheme", choices=["rolling", "fixed"], default="rolling")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--mcs-bootstrap", type=int, default=500)
    parser.add_argument("--mcs-block-length", type=int, default=20)
    parser.add_argument("--ale-models", nargs="*", default=[])
    parser.add_argument("--ale-features", nargs="*", default=[])
    parser.add_argument("--ale-bins", type=int, default=20)
    parser.add_argument("--ale-tickers", nargs="*", default=["AAPL", "AMZN", "JPM"])
    parser.add_argument("--skip-ale", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        predictions = load_prediction_file(args.target_transform, args.forecast_scheme)
        validate_real_predictions(predictions)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    rel, pvals = pairwise_matrices(predictions)
    starred = starred_relative_mse(rel, pvals)
    rel_path = RESULT_DIR / f"relative_mse_matrix_{args.target_transform}_{args.forecast_scheme}.csv"
    pval_path = RESULT_DIR / f"dm_pvalue_matrix_{args.target_transform}_{args.forecast_scheme}.csv"
    star_path = RESULT_DIR / f"relative_mse_matrix_starred_{args.target_transform}_{args.forecast_scheme}.csv"
    rel.to_csv(rel_path)
    pvals.to_csv(pval_path)
    starred.to_csv(star_path)

    benchmark, losses = loss_matrix(predictions)
    mcs = lightweight_mcs(
        losses,
        benchmark=benchmark,
        alpha=args.alpha,
        bootstrap=args.mcs_bootstrap,
        block_length=args.mcs_block_length,
        random_state=args.seed,
    )
    mcs_path = RESULT_DIR / f"mcs_{args.target_transform}_{args.forecast_scheme}.csv"
    mcs.to_csv(mcs_path, index=False)

    print("Saved diagnostics:")
    print(f"  {rel_path}")
    print(f"  {pval_path}")
    print(f"  {star_path}")
    print(f"  {mcs_path}")

    if not args.skip_ale and args.ale_models and args.ale_features:
        ale = compute_posthoc_ale(
            tickers=args.ale_tickers,
            model_names=args.ale_models,
            feature_names=args.ale_features,
            target_transform=args.target_transform,
            forecast_scheme=args.forecast_scheme,
            bins=args.ale_bins,
            random_state=args.seed,
        )
        if not ale.empty:
            for model_name, sub in ale.groupby("model"):
                safe_model = model_name.replace(":", "_")
                ale_path = RESULT_DIR / f"ale_{safe_model}_{args.target_transform}_{args.forecast_scheme}.csv"
                sub.to_csv(ale_path, index=False)
                print(f"  {ale_path}")
        else:
            print("  ALE skipped: no compatible fitted post-hoc model/features were available.")


if __name__ == "__main__":
    main()
