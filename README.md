# RV Forecasting Replication

Coursework replication framework for Christensen, Siggaard and Veliyev (2023), "A Machine Learning Approach to Volatility Forecasting".

## Project Structure

```text
data/raw/          raw 1-minute OHLCV files: AAPL.txt, AMZN.txt, JPM.txt
data/processed/    daily realized measures and model feature matrices
src/               modular Python implementation
notebooks/         optional result exploration notebook
results/           main CSV files for the report
outputs/tables/    CSV result tables
outputs/figures/   generated figures
main.py            one-command replication entry point
```

## Sample Split

The train/validation/test split follows the original paper: 70% training, 10% validation and 20% test. The split is strictly time ordered and does not use random shuffling.

## Forecast Schemes

The project supports two forecasting schemes:

- `--forecast-scheme fixed`: fit models once using the initial train/validation split, then forecast the whole test period.
- `--forecast-scheme rolling`: re-estimate models through the out-of-sample period using only past data.

The default is `--forecast-scheme rolling`, because this is closer to the rolling-window / validation-tuning design in `rv1.pdf`. The reported rolling results use `--rolling-refit-frequency 20`, so Random Forest, Bagging and Gradient Boosting are periodically re-estimated while the time-ordered rolling design is preserved.

This project is a simplified coursework replication of the main empirical design of Christensen, Siggaard and Veliyev, not an exact numerical reproduction of the original paper. The dataset remains the provided three-stock AAPL, AMZN and JPM OHLCV sample, so the code does not add external IV, VIX, announcement or macro data. The main differences are the smaller stock universe, the simplified OHLCV-derived extended predictor set, the focus on log-RV as the main specification, and the computationally simplified rolling refit scheme for tree-based models.

For the first test forecast, the rolling windows are:

- training window: initial 70% of observations
- validation window: next 10% of observations
- forecast date: first observation in the final 20%

For each subsequent forecast, both windows move forward by one observation, while keeping fixed lengths. No random shuffling is used, no test observation is used in validation tuning, and no future data are used for any forecast.

HAR/LogHAR and HAR-X/LogHAR-X are re-estimated using rolling train + validation data. The rolling pipeline also includes simplified SHAR/LogSHAR and HARQ/LogHARQ extensions using the available semivariance and realized-quarticity variables. Ridge and Lasso tune only penalty strength on the rolling validation window, while ElasticNet tunes both penalty strength and the L1/L2 mixing ratio. The selected regularized model is then re-fit on rolling train + validation data. Random Forest and Bagging use fixed tree settings, while Gradient Boosting uses a small validation grid; all three are re-fit on rolling train + validation data. Passing `--include-mlp` includes sklearn ReLU MLP models in the same rolling output table.

The fixed-window regularization search uses a finer alpha grid. The rolling-window search uses 20 alpha values over the same range to keep the coursework replication computationally feasible while avoiding an overly coarse validation grid. ElasticNet additionally searches over `l1_ratio` values `[0.1, 0.25, 0.5, 0.75, 0.9]`, which is closer to the original paper's validation tuning of both penalty strength and mixing parameter, although the grid is smaller for runtime reasons. Tree models also use coursework-scale estimator counts. These choices do not force the results in any direction; they reduce repeated rolling tuning cost.

For the reported rolling results, Random Forest, Bagging and Gradient Boosting are refitted every 20 trading days for computational feasibility, while HAR-type and regularized linear models are refitted daily. MLP refit frequency is controlled separately with `--mlp-refit-frequency`; the main report command uses daily MLP refits. All forecasts still use only information available before the relevant test date, so the rolling design avoids future leakage.

The reported rolling command is:

```bash
python3 main.py --horizon 1 --target-transform log --forecast-scheme rolling --rolling-refit-frequency 20 --include-mlp --mlp-architecture nn2 --mlp-refit-frequency 1 --seed 42
```

This command reproduces the final reported log-RV table with Bagging and MLP included. It writes the standard rolling output files directly; no manual merge step is needed.

To run appendix-style neural-network robustness with all pyramid MLP sizes:

```bash
python3 main.py --horizon 1 --target-transform log --forecast-scheme rolling --rolling-refit-frequency 20 --include-mlp --mlp-architecture all --mlp-refit-frequency 20 --seed 42
```

The `--mlp-architecture` choices are `nn1`, `nn2`, `nn3`, `nn4` and `all`. They map to sklearn ReLU MLP hidden layers `(2,)`, `(4, 2)`, `(8, 4, 2)` and `(16, 8, 4, 2)`. These are lightweight approximations of the paper's NN1-NN4 pyramid idea, not exact Leaky-ReLU ensemble neural networks.

## Target Settings

The main specification forecasts log realized variance. This follows the LogHAR-style specification in the paper and helps reduce the influence of extreme realized variance observations.

For `--target-transform log`, models predict `log(RV_t)`. MSE, MAE, Relative MSE and Relative MAE are computed on the log scale. QLIKE is computed after converting predictions back to RV levels, so log-scale MSE is not mixed with RV-level MSE.

The original paper mainly reports realized variance forecast MSE in RV levels. Therefore, I also run a level-RV specification as a supplementary check. However, the report focuses on log-RV results because they are more stable for the provided AAPL, AMZN and JPM sample.

For `--target-transform level`, models predict `RV_t` directly. MSE, MAE, Relative MSE and Relative MAE are computed on the RV level scale. Non-positive or unrealistically small level-RV forecasts are floored at the minimum realized variance observed in the current rolling train-validation sample, rather than at a fixed `1e-12` constant.

## Realized Variance Construction

Daily realized variance is built from intraday five-minute returns. For each day, the price path starts with the first valid opening price and then appends the five-minute bar closing prices. Log returns are computed from this open-plus-closes path, so a complete U.S. trading day contributes about 78 five-minute returns rather than losing the first interval.

## Main Replication Command: Log RV

```bash
python3 main.py --horizon 1 --target-transform log --forecast-scheme rolling --rolling-refit-frequency 20 --include-mlp --mlp-architecture nn2 --mlp-refit-frequency 1 --seed 42
```

This runs the main one-day-ahead log realized variance experiment and compares LogHAR with SHAR/HARQ-style extensions, regularized linear models, Bagging, Random Forest, Gradient Boosting and MLP.

Main CSV outputs:

- `results/model_comparison_log_rolling.csv`
- `results/relative_mse_summary_log_rolling.csv`

## Supplementary Check: Level RV

```bash
python3 main.py --horizon 1 --target-transform level --forecast-scheme rolling --rolling-refit-frequency 20 --seed 42
```

Supplementary CSV outputs:

- `results/model_comparison_level_rolling.csv`
- `results/relative_mse_summary_level_rolling.csv`

To run the older fixed-window version:

```bash
python3 main.py --horizon 1 --target-transform log --forecast-scheme fixed --seed 42
```

Fixed-window outputs use names such as:

- `results/model_comparison_log_fixed.csv`
- `results/relative_mse_summary_log_fixed.csv`

All model comparison tables contain `target_transform`, `evaluation_scale`, `forecast_scheme`, `refit_scheme`, `stock`, `feature_set`, `model`, `mse`, `mae`, and `relative_mse`/`rel_mse` fields, so log-RV, level-RV, rolling-window and fixed-window results are not mixed.

## Regularization Note

Ridge/ElasticNet may produce predictions almost identical to LogHAR-X when the selected penalty is close to zero or when regularization has little effect. This is not necessarily a bug. In the report, this should be interpreted as regularization providing little additional improvement over the linear HAR-X specification for this dataset.

The `selected_alpha` column in the model comparison CSV records the validation-selected regularization strength for Ridge, Lasso and ElasticNet. For ElasticNet, `selected_params` also records the selected or median validation-selected `l1_ratio`.

## Figures

- `outputs/figures/relative_mse_h1_log_rolling.png`
- `outputs/figures/relative_mse_by_stock_log_rolling.png`
- `outputs/figures/rv_timeseries_h1_log_rolling.png`
- `outputs/figures/relative_mse_h1_level_rolling.png`
- `outputs/figures/rv_timeseries_h1_level_rolling.png`

`main.py` also regenerates the stock-level relative MSE figure from the saved
prediction-level results, using log-scale squared error for log targets and
level-scale squared error for level targets. The same figure is saved to both
`outputs/figures/relative_mse_by_stock_log_rolling.png` and
`results/figures/relative_mse_by_stock_log_rolling.png` for the main log rolling
run.

## Additional Paper-Style Diagnostics

After running `main.py`, the project saves prediction-level files such as
`results/predictions_log_rolling.csv`. These files support optional diagnostics
without rerunning the rolling forecast loop. `diagnostics.py` does not create
fake data: it requires predictions generated by `main.py`, and it stops if the
prediction file is missing or looks incomplete.

Pairwise relative-MSE and Diebold-Mariano matrices approximate the paper's
forecast comparison tables. Rows are benchmark models, columns are comparison
models, and values below one indicate that the column model has lower MSE than
the row benchmark. The one-sided DM p-value matrix uses the alternative that the
row benchmark has larger squared forecast loss than the column model.

The diagnostics script also includes a lightweight coursework MCS-style routine.
It uses squared-error losses and a simple block bootstrap to iteratively remove
clearly inferior average-loss models. This is a practical approximation for the
coursework replication, not a full replacement for the paper's full MCS
procedure.

ALE is implemented as a post-hoc interpretability diagnostic. It refits selected
nonlinear models on the final train-validation window and computes one-dimensional
accumulated local effects on the final test window. This moves the interpretation
section closer to the paper's accumulated local effects analysis while keeping
per-forecast rolling evaluation unchanged. Permutation importance is retained as
a supplementary diagnostic.

Full workflow:

Step 1, run the full rolling forecast:

```bash
python3 main.py --horizon 1 --target-transform log --forecast-scheme rolling --rolling-refit-frequency 20 --include-mlp --mlp-architecture nn2 --mlp-refit-frequency 1 --seed 42
```

Step 2, run diagnostics from real saved predictions:

```bash
python3 diagnostics.py --target-transform log --forecast-scheme rolling --alpha 0.10 --mcs-bootstrap 500 --ale-models RandomForest GradientBoosting Bagging --ale-features rv_d rv_w rv_m rsv_neg_d harq_interaction
```

Pairwise DM matrices, relative MSE matrices, MCS output and ALE output are based
only on full saved predictions. If `results/predictions_log_rolling.csv` is
missing, run `main.py` first.
