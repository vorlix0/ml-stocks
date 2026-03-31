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
    from src.data import DataDownloader

    typer.echo("📥 Downloading market data …")
    downloader = DataDownloader()
    downloader.download_and_save()
    typer.echo("✅ Download complete.")


# --------------------------------------------------------------------------- #
# process                                                                      #
# --------------------------------------------------------------------------- #


@app.command()
def process() -> None:
    """Process raw data and engineer features."""
    from config import DATA_CONFIG
    from src.data import FeatureEngineer
    from src.exceptions import DataNotFoundError, ForexMLError
    from src.utils import load_csv_safe, validate_ohlcv_data

    typer.echo("⚙️  Processing data …")
    input_file = DATA_CONFIG.data_file
    output_file = DATA_CONFIG.features_file

    try:
        data = load_csv_safe(
            input_file,
            required_columns=["Open", "High", "Low", "Close", "Volume"],
        )
        validate_ohlcv_data(data)

        feature_engineer = FeatureEngineer(data)
        data_features = feature_engineer.create_all_features()
        data_features.to_csv(output_file)

        typer.echo(f"✅ Features saved to: {output_file}  (shape: {data_features.shape})")
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
    from src.backtest import RiskMetrics, TradingSimulator, TradingStrategy
    from src.exceptions import DataNotFoundError, ForexMLError, ModelNotFoundError
    from src.model import ModelTrainer
    from src.utils import Plotter, load_csv_safe, validate_features_data

    typer.echo("📊 Running backtest …")
    try:
        model = ModelTrainer.load_model()
        df = load_csv_safe(DATA_CONFIG.features_file)
        validate_features_data(df)

        split_date = MODEL_CONFIG.SPLIT_DATE
        feature_cols = [
            col for col in df.columns if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
        ]
        df_test = df[df.index >= split_date].copy()
        x_test = df_test[feature_cols]

        y_pred_proba = model.predict_proba(x_test)[:, 1]
        strategy = TradingStrategy(df_test, y_pred_proba)

        signals_no_filter = strategy.generate_signals(use_adx_filter=False)
        signals_adx = strategy.generate_signals(use_adx_filter=True)

        returns_no_filter = strategy.calculate_strategy_returns(signals_no_filter)
        returns_adx = strategy.calculate_strategy_returns(signals_adx)

        cum_market = strategy.get_market_returns()
        cum_no_filter = strategy.calculate_cumulative_returns(returns_no_filter)
        cum_adx = strategy.calculate_cumulative_returns(returns_adx)

        typer.echo(f"\nMarket Return: {cum_market.iloc[-1] - 1:.2%}")
        typer.echo(f"Strategy Return (no filter): {cum_no_filter.iloc[-1] - 1:.2%}")
        typer.echo(f"Strategy Return (ADX filter): {cum_adx.iloc[-1] - 1:.2%}")

        simulator = TradingSimulator(df_test)
        results_market = simulator.simulate_buy_and_hold()
        results_no_filter = simulator.simulate(signals_no_filter)
        results_adx = simulator.simulate(signals_adx)
        simulator.compare_strategies(
            [results_market, results_no_filter, results_adx],
            ["Buy & Hold", "No filter", "ADX filter"],
        )

        metrics_market = RiskMetrics(df_test["Returns"], cum_market)
        metrics_no_filter = RiskMetrics(returns_no_filter, cum_no_filter)
        metrics_adx = RiskMetrics(returns_adx, cum_adx)
        RiskMetrics.print_comparison(
            [metrics_market, metrics_no_filter, metrics_adx],
            ["Market", "without filter", "with ADX filter"],
        )

        Plotter.plot_backtest(
            df_test.index,
            {
                "Buy & Hold": cum_market,
                "ML Strategy (without filter)": cum_no_filter,
                "ML Strategy (with ADX filter)": cum_adx,
            },
            save_path=BACKTEST_CONFIG.backtest_plot_file,
        )
        Plotter.plot_portfolio_with_adx(
            df_test.index,
            {
                "Buy & Hold": cum_market,
                "ML Strategy (without filter)": cum_no_filter,
                "ML Strategy (with ADX filter)": cum_adx,
            },
            df_test["ADX"],
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
