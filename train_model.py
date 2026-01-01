# train_model.py
"""Script for training ML model."""

import sys
import pandas as pd

from config import DATA_CONFIG, BACKTEST_CONFIG
from src.model import ModelTrainer, ModelEvaluator
from src.utils import Plotter, load_csv_safe, validate_features_data
from src.exceptions import ForexMLError, DataNotFoundError


def load_data() -> pd.DataFrame:
    """Loads data with features."""
    df = load_csv_safe(DATA_CONFIG.features_file)
    validate_features_data(df)
    return df


def main():
    """Main function for model training."""
    try:
        # Load data
        df = load_data()
    
        print(f"Data shape: {df.shape}")
        print(f"NaN in data: {df.isna().sum().sum()}")
        print(f"\nTarget distribution:")
        print(df['Target'].value_counts())
        print(f"% UP: {(df['Target'] == 1).sum() / len(df) * 100:.2f}%")
        
        # Initialize trainer
        trainer = ModelTrainer(df)
        trainer.prepare_data()
        
        # Train model
        model = trainer.train()
        
        # Validation evaluation
        evaluator = ModelEvaluator(model, trainer.X_test, trainer.y_test)
        evaluator.evaluate_validation(trainer.X_val, trainer.y_val)
        
        # Test evaluation
        evaluator.print_evaluation()
        
        # Feature importance
        importances = trainer.get_feature_importances()
        Plotter.print_feature_importance_analysis(importances)
        Plotter.plot_feature_importance(
            importances,
            save_path=BACKTEST_CONFIG.feature_importance_plot_file
        )
        
        # Save model
        trainer.save_model()
    
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
