"""
Unified CLI for the forex-ml-hf project.

Replaces the individual ``download_finance_data.py``, ``process_data.py``,
``train_model.py``, ``tune_model.py`` and ``test_model.py`` scripts with a
single entry-point backed by Typer.

Usage examples::

    python cli.py download
    python cli.py process
    python cli.py train
    python cli.py train --use-mlflow
    python cli.py tune --n-trials 100
    python cli.py backtest
    python cli.py run-all
"""
from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
import typer

app = typer.Typer(
    name="forex-ml-hf",
    help="ML-based trading strategy CLI",
    add_completion=False,
)


# --------------------------------------------------------------------------- #
# download                                                                     #
# --------------------------------------------------------------------------- #


@app.command()
def download() -> None:
    """Download raw market data from Yahoo Finance."""
    from src.data import download_universe

    typer.echo("📥 Downloading market data (universe + SPY + VIX) …")
    tickers = download_universe()
    typer.echo(f"✅ Download complete: {len(tickers)} tickers.")


# --------------------------------------------------------------------------- #
# process                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def process() -> None:
    """Process raw data and engineer features."""
    from config import DATA_CONFIG
    from src.data import FeatureEngineer
    from src.data.features import build_market_frame
    from src.exceptions import DataNotFoundError, ForexMLError
    from src.utils import load_csv_safe, validate_ohlcv_data

    typer.echo("⚙️  Processing data …")
    output_file = DATA_CONFIG.features_file
    ohlcv = ["Open", "High", "Low", "Close", "Volume"]

    try:
        market = build_market_frame(
            load_csv_safe(DATA_CONFIG.raw_path(DATA_CONFIG.BENCHMARK_TICKER), required_columns=ohlcv),
            load_csv_safe(DATA_CONFIG.raw_path(DATA_CONFIG.VIX_TICKER), required_columns=["Close"]),
        )

        frames = []
        for ticker in DATA_CONFIG.TICKERS:
            raw_file = DATA_CONFIG.raw_path(ticker)
            if not raw_file.exists():
                typer.echo(f"⚠️  Skipping {ticker}: {raw_file} not found")
                continue
            data = load_csv_safe(raw_file, required_columns=ohlcv)
            validate_ohlcv_data(data)
            features = FeatureEngineer(data, market=market).create_all_features()
            features.insert(0, "Ticker", ticker)
            frames.append(features)

        if not frames:
            raise DataNotFoundError("No raw ticker data found")

        panel = pd.concat(frames).sort_index(kind="stable")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_file)

        typer.echo(
            f"✅ Panel saved to: {output_file}  "
            f"(shape: {panel.shape}, tickers: {len(frames)})"
        )
    except DataNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py download", err=True)
        raise typer.Exit(code=1) from None
    except ForexMLError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #


@app.command()
def train(
    use_mlflow: Annotated[bool, typer.Option("--use-mlflow", help="Log run to MLflow")] = False,
) -> None:
    """Train the ML model."""
    from config import BACKTEST_CONFIG, DATA_CONFIG
    from src.exceptions import DataNotFoundError, ForexMLError
    from src.model import ModelEvaluator, ModelTrainer
    from src.utils import Plotter, load_csv_safe, validate_features_data

    typer.echo("🏋️  Training model …")
    try:
        df = load_csv_safe(DATA_CONFIG.features_file)
        validate_features_data(df)

        trainer = ModelTrainer(df)
        trainer.prepare_data()
        model = trainer.train(use_mlflow=use_mlflow)

        evaluator = ModelEvaluator(model, trainer.X_test, trainer.y_test)
        evaluator.evaluate_validation(trainer.X_val, trainer.y_val)
        evaluator.print_evaluation()

        importances = trainer.get_feature_importances()
        Plotter.print_feature_importance_analysis(importances)
        Plotter.plot_feature_importance(
            importances,
            save_path=BACKTEST_CONFIG.feature_importance_plot_file,
        )

        trainer.save_model()
        typer.echo("✅ Model trained and saved.")
    except DataNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py process", err=True)
        raise typer.Exit(code=1) from None
    except ForexMLError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
# tune                                                                         #
# --------------------------------------------------------------------------- #


@app.command()
def tune(
    n_trials: Annotated[int, typer.Option("--n-trials", help="Number of Optuna trials")] = 50,
    use_mlflow: Annotated[
        bool, typer.Option("--use-mlflow", help="Log best run to MLflow")
    ] = False,
) -> None:
    """Tune hyperparameters with Optuna, then retrain with best parameters."""
    from config import BACKTEST_CONFIG, DATA_CONFIG, MODEL_CONFIG
    from src.exceptions import DataNotFoundError, ForexMLError
    from src.model import HyperparameterTuner, ModelEvaluator, ModelTrainer
    from src.model.model_factory import create_model
    from src.utils import Plotter, load_csv_safe, validate_features_data

    typer.echo(f"🔍 Starting Optuna study ({n_trials} trials) …")
    try:
        df = load_csv_safe(DATA_CONFIG.features_file)
        validate_features_data(df)

        tuner = HyperparameterTuner(df)
        best_params = tuner.run(n_trials=n_trials, show_progress_bar=True)

        typer.echo("\nBest hyperparameters found:")
        for k, v in best_params.items():
            typer.echo(f"  {k}: {v}")

        typer.echo("\nRetraining model with best parameters …")
        trainer = ModelTrainer(df)
        trainer.prepare_data()

        model = create_model(MODEL_CONFIG.MODEL_TYPE, **best_params)
        model.fit(trainer.X_tr, trainer.y_tr)
        trainer.model = model

        if use_mlflow:
            try:
                import mlflow
                import mlflow.sklearn
                from sklearn.metrics import roc_auc_score

                with mlflow.start_run(run_name="optuna-best"):
                    mlflow.log_params(best_params)
                    val_proba = model.predict_proba(trainer.X_val)[:, 1]
                    mlflow.log_metric("val_auc", roc_auc_score(trainer.y_val, val_proba))
                    mlflow.sklearn.log_model(model, artifact_path="model")
            except ImportError:
                typer.echo("⚠️  MLflow not installed – skipping experiment logging.", err=True)

        evaluator = ModelEvaluator(model, trainer.X_test, trainer.y_test)
        evaluator.evaluate_validation(trainer.X_val, trainer.y_val)
        evaluator.print_evaluation()

        importances = trainer.get_feature_importances()
        Plotter.print_feature_importance_analysis(importances)
        Plotter.plot_feature_importance(
            importances,
            save_path=BACKTEST_CONFIG.feature_importance_plot_file,
        )

        trainer.save_model()
        typer.echo("✅ Tuned model saved.")
    except DataNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py process", err=True)
        raise typer.Exit(code=1) from None
    except ForexMLError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
# backtest                                                                     #
# --------------------------------------------------------------------------- #


@app.command()
def backtest() -> None:
    """Run backtesting on the trained model."""
    from config import BACKTEST_CONFIG, DATA_CONFIG, MODEL_CONFIG
    from src.backtest import RiskMetrics
    from src.backtest.panel import panel_strategy_returns
    from src.exceptions import DataNotFoundError, ForexMLError, ModelNotFoundError
    from src.model import ModelTrainer
    from src.utils import Plotter, load_csv_safe, validate_features_data

    typer.echo("📊 Running backtest (equal-weight portfolio across tickers) …")
    try:
        model = ModelTrainer.load_model()
        df = load_csv_safe(DATA_CONFIG.features_file)
        validate_features_data(df)

        feature_cols = [
            col for col in df.columns if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
        ]
        df_test = df[df.index >= MODEL_CONFIG.SPLIT_DATE].copy()
        proba = model.predict_proba(df_test[feature_cols])[:, 1]
        threshold = BACKTEST_CONFIG.PREDICTION_THRESHOLD

        returns_no_filter, returns_market = panel_strategy_returns(df_test, proba, threshold)
        proba_adx = np.where(df_test["ADX"] > BACKTEST_CONFIG.ADX_THRESHOLD, proba, 0.0)
        returns_adx, _ = panel_strategy_returns(df_test, proba_adx, threshold)

        cum_market = (1 + returns_market).cumprod()
        cum_no_filter = (1 + returns_no_filter).cumprod()
        cum_adx = (1 + returns_adx).cumprod()

        market_close = load_csv_safe(
            DATA_CONFIG.raw_path(DATA_CONFIG.BENCHMARK_TICKER), required_columns=["Close"]
        )["Close"]
        spy_returns = market_close.pct_change().reindex(returns_market.index).fillna(0)
        cum_spy = (1 + spy_returns).cumprod()

        typer.echo(f"\nEqual-weight Buy & Hold: {cum_market.iloc[-1] - 1:.2%}")
        typer.echo(f"{DATA_CONFIG.BENCHMARK_TICKER} Buy & Hold: {cum_spy.iloc[-1] - 1:.2%}")
        typer.echo(f"Strategy Return (no filter): {cum_no_filter.iloc[-1] - 1:.2%}")
        typer.echo(f"Strategy Return (ADX filter): {cum_adx.iloc[-1] - 1:.2%}")

        RiskMetrics.print_comparison(
            [
                RiskMetrics(spy_returns, cum_spy),
                RiskMetrics(returns_market, cum_market),
                RiskMetrics(returns_no_filter, cum_no_filter),
                RiskMetrics(returns_adx, cum_adx),
            ],
            [DATA_CONFIG.BENCHMARK_TICKER, "Equal-weight B&H", "without filter", "with ADX filter"],
        )

        curves = {
            f"{DATA_CONFIG.BENCHMARK_TICKER} Buy & Hold": cum_spy,
            "Equal-weight Buy & Hold": cum_market,
            "ML Strategy (without filter)": cum_no_filter,
            "ML Strategy (with ADX filter)": cum_adx,
        }
        Plotter.plot_backtest(
            cum_market.index, curves, save_path=BACKTEST_CONFIG.backtest_plot_file
        )
        Plotter.plot_portfolio_with_adx(
            cum_market.index,
            curves,
            df_test.groupby(level=0)["ADX"].mean(),
            save_path=BACKTEST_CONFIG.portfolio_plot_file,
        )

        typer.echo(f"\n✅ Charts saved to {BACKTEST_CONFIG.backtest_plot_file.parent}")
    except ModelNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py train", err=True)
        raise typer.Exit(code=1) from None
    except DataNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py process", err=True)
        raise typer.Exit(code=1) from None
    except ForexMLError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1) from None


# --------------------------------------------------------------------------- #
# compare                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def compare() -> None:
    """Compare the main model with simple baselines (walk-forward AUC + test returns)."""
    from sklearn.metrics import roc_auc_score

    from config import BACKTEST_CONFIG, DATA_CONFIG, MODEL_CONFIG
    from src.backtest.panel import panel_strategy_returns
    from src.exceptions import DataNotFoundError, ForexMLError
    from src.model import (
        create_logistic_baseline,
        create_model,
        pre_split_data,
        sma_trend_signals,
        walk_forward_evaluate,
    )
    from src.utils import load_csv_safe, validate_features_data

    typer.echo("⚖️  Comparing model with baselines …")
    try:
        df = load_csv_safe(DATA_CONFIG.features_file)
        validate_features_data(df)
    except DataNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("Run first: python cli.py process", err=True)
        raise typer.Exit(code=1) from None
    except ForexMLError as e:
        typer.echo(f"❌ {e}", err=True)
        raise typer.Exit(code=1) from None

    feature_cols = [c for c in df.columns if c not in MODEL_CONFIG.EXCLUDED_COLUMNS]
    x_pre, y_pre = pre_split_data(df[feature_cols], df["Target"])
    df_test = df[df.index >= MODEL_CONFIG.SPLIT_DATE]
    labelled = df_test["Target"].notna().to_numpy()
    threshold = BACKTEST_CONFIG.PREDICTION_THRESHOLD
    _, market_daily = panel_strategy_returns(df_test, np.zeros(len(df_test)), threshold)
    typer.echo(f"Equal-weight Buy & Hold (test period): {(1 + market_daily).prod() - 1:.2%}\n")

    builders = {
        "GradientBoosting (config)": lambda: create_model(
            MODEL_CONFIG.MODEL_TYPE,
            n_estimators=MODEL_CONFIG.N_ESTIMATORS,
            max_depth=MODEL_CONFIG.MAX_DEPTH,
            learning_rate=MODEL_CONFIG.LEARNING_RATE,
            subsample=MODEL_CONFIG.SUBSAMPLE,
            min_samples_leaf=MODEL_CONFIG.MIN_SAMPLES_LEAF,
            random_state=MODEL_CONFIG.RANDOM_STATE,
        ),
        "LogisticRegression": create_logistic_baseline,
    }

    header = f"{'Model':28} {'WF AUC':>14} {'Test AUC':>9} {'Exposure':>9} {'Return':>9}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, builder in builders.items():
        wf = walk_forward_evaluate(builder, x_pre, y_pre)
        model = builder()
        model.fit(x_pre, y_pre)
        proba = model.predict_proba(df_test[feature_cols])[:, 1]
        test_auc = roc_auc_score(df_test["Target"].to_numpy()[labelled], proba[labelled])
        strat, _ = panel_strategy_returns(df_test, proba, threshold)
        typer.echo(
            f"{name:28} {wf.mean_auc:6.3f}±{wf.std_auc:5.3f} {test_auc:9.3f} "
            f"{(proba > threshold).mean():9.0%} {(1 + strat).prod() - 1:9.2%}"
        )

    trend = sma_trend_signals(df_test).to_numpy()
    trend_ret, _ = panel_strategy_returns(df_test, trend, 0.5)
    typer.echo(
        f"{'Close > SMA_200 rule':28} {'-':>14} {'-':>9} "
        f"{trend.mean():9.0%} {(1 + trend_ret).prod() - 1:9.2%}"
    )


# --------------------------------------------------------------------------- #
# run-all                                                                      #
# --------------------------------------------------------------------------- #


@app.command(name="run-all")
def run_all(
    use_mlflow: Annotated[bool, typer.Option("--use-mlflow")] = False,
) -> None:
    """Run the full pipeline: download → process → train → backtest."""
    typer.echo("🚀 Running full pipeline …\n")
    download()
    process()
    train(use_mlflow=use_mlflow)
    backtest()
    typer.echo("\n🎉 Pipeline complete.")


# --------------------------------------------------------------------------- #
# entry-point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    app()
