# tune_model.py
"""Script for hyperparameter tuning using Optuna."""

import sys

import pandas as pd

from config import BACKTEST_CONFIG, DATA_CONFIG, MODEL_CONFIG
from src.exceptions import DataNotFoundError, ForexMLError
from src.model import HyperparameterTuner, ModelEvaluator, ModelTrainer
from src.model.model_factory import create_model
from src.utils import Plotter, load_csv_safe, validate_features_data


def load_data() -> pd.DataFrame:
    """Loads data with features."""
    df = load_csv_safe(DATA_CONFIG.features_file)
    validate_features_data(df)
    return df


def main() -> None:
    """Run Optuna hyperparameter search and retrain with best parameters."""
    try:
        df = load_data()

        print(f"Data shape: {df.shape}")

        # ------------------------------------------------------------------ #
        # Hyperparameter search                                                #
        # ------------------------------------------------------------------ #
        n_trials = 50
        print(f"\nStarting Optuna study ({n_trials} trials) …")
        tuner = HyperparameterTuner(df)
        best_params = tuner.run(n_trials=n_trials, show_progress_bar=True)

        print("\nBest hyperparameters found:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # ------------------------------------------------------------------ #
        # Retrain with best parameters                                         #
        # ------------------------------------------------------------------ #
        print("\nRetraining model with best parameters …")
        trainer = ModelTrainer(df)
        trainer.prepare_data()

        # Override config values with Optuna results (in-place on trainer)
        model = create_model(
            MODEL_CONFIG.MODEL_TYPE,
            **best_params,
        )
        model.fit(trainer.X_tr, trainer.y_tr)
        trainer.model = model

        # Evaluation
        evaluator = ModelEvaluator(model, trainer.X_test, trainer.y_test)
        evaluator.evaluate_validation(trainer.X_val, trainer.y_val)
        evaluator.print_evaluation()

        # Feature importance
        importances = trainer.get_feature_importances()
        Plotter.print_feature_importance_analysis(importances)
        Plotter.plot_feature_importance(
            importances,
            save_path=BACKTEST_CONFIG.feature_importance_plot_file,
        )

        # Save tuned model
        trainer.save_model()
        print("\n✅ Tuned model saved.")

    except DataNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run first: python process_data.py")
        sys.exit(1)
    except ForexMLError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
