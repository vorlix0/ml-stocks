"""
Tests for ModelTrainer class.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.model.trainer import ModelTrainer
from src.exceptions import EmptyDataError, InvalidDataError


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def sample_training_data(self, sample_ohlcv_data) -> pd.DataFrame:
        """Creates sample data with features and target."""
        from src.data.features import FeatureEngineer
        
        engineer = FeatureEngineer(sample_ohlcv_data)
        return engineer.create_all_features()
    
    def test_init_with_valid_data(self, sample_training_data):
        """Should initialize with valid training data."""
        trainer = ModelTrainer(sample_training_data)
        
        assert trainer.df is not None
        assert trainer.model is None
        assert len(trainer.feature_cols) > 0
    
    def test_init_raises_on_empty_data(self):
        """Should raise EmptyDataError for empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(EmptyDataError, match="empty"):
            ModelTrainer(empty_df)
    
    def test_init_raises_on_missing_target(self, sample_ohlcv_data):
        """Should raise InvalidDataError when Target column missing."""
        df_no_target = sample_ohlcv_data.copy()
        
        with pytest.raises(InvalidDataError, match="Target"):
            ModelTrainer(df_no_target)
    
    def test_get_feature_columns_excludes_target(self, sample_training_data):
        """Feature columns should not include Target."""
        trainer = ModelTrainer(sample_training_data)
        
        assert 'Target' not in trainer.feature_cols
    
    def test_get_feature_columns_excludes_ohlcv(self, sample_training_data):
        """Feature columns should not include OHLCV columns."""
        trainer = ModelTrainer(sample_training_data)
        
        excluded = ['Close', 'Open', 'High', 'Low', 'Volume']
        for col in excluded:
            assert col not in trainer.feature_cols
    
    def test_feature_columns_not_empty(self, sample_training_data):
        """Feature columns should not be empty."""
        trainer = ModelTrainer(sample_training_data)
        
        assert len(trainer.feature_cols) > 0
        assert len(trainer.feature_cols) > 50  # Should have many features


class TestModelTrainerIntegration:
    """Integration tests for ModelTrainer with real config."""
    
    @pytest.fixture
    def large_training_data(self) -> pd.DataFrame:
        """Creates larger sample data for integration tests."""
        from src.data.features import FeatureEngineer
        
        np.random.seed(42)
        # Need data that covers SPLIT_DATE (2023-01-01) in config
        dates = pd.date_range(start='2021-01-01', periods=1000, freq='D')
        
        base_price = 100
        returns = np.random.randn(1000) * 0.02
        close = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'Open': close * (1 + np.random.randn(1000) * 0.005),
            'High': close * (1 + np.abs(np.random.randn(1000) * 0.01)),
            'Low': close * (1 - np.abs(np.random.randn(1000) * 0.01)),
            'Close': close,
            'Volume': np.random.randint(1_000_000, 10_000_000, size=1000)
        }, index=dates)
        
        engineer = FeatureEngineer(df)
        return engineer.create_all_features()
    
    def test_prepare_data_creates_splits(self, large_training_data):
        """prepare_data should create train/test splits."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        
        assert trainer.X_train is not None
        assert trainer.X_test is not None
        assert trainer.y_train is not None
        assert trainer.y_test is not None
        assert len(trainer.X_train) > 0
        assert len(trainer.X_test) > 0
    
    def test_prepare_data_chronological_split(self, large_training_data):
        """Train data should be before test data chronologically."""
        from config import MODEL_CONFIG
        
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        
        # All train dates should be before split
        assert (trainer.X_train.index < MODEL_CONFIG.SPLIT_DATE).all()
        # All test dates should be >= split
        assert (trainer.X_test.index >= MODEL_CONFIG.SPLIT_DATE).all()
    
    def test_train_creates_model(self, large_training_data):
        """train should create a model."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        trainer.train()
        
        assert trainer.model is not None
    
    def test_predict_returns_probabilities(self, large_training_data):
        """predict should return probabilities in [0, 1]."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        trainer.train()
        
        predictions = trainer.model.predict_proba(trainer.X_test)[:, 1]
        
        assert predictions.min() >= 0
        assert predictions.max() <= 1
    
    def test_predict_returns_correct_length(self, large_training_data):
        """Predictions should have same length as test data."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        trainer.train()
        
        predictions = trainer.model.predict_proba(trainer.X_test)[:, 1]
        
        assert len(predictions) == len(trainer.X_test)
    
    def test_save_and_load_model(self, large_training_data):
        """Model should be saveable and loadable."""
        trainer = ModelTrainer(large_training_data)
        trainer.prepare_data()
        trainer.train()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
        
        try:
            trainer.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Load and compare predictions
            loaded_model = ModelTrainer.load_model(temp_path)
            original_preds = trainer.model.predict_proba(trainer.X_test)[:, 1]
            loaded_preds = loaded_model.predict_proba(trainer.X_test)[:, 1]
            
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
