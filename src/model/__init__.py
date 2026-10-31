"""
Model module - training and evaluation of ML models.
"""
from .base_trainer import BaseTrainer
from .baselines import create_logistic_baseline, sma_trend_signals
from .data_splitter import ChronologicalSplitter, DataSplit, DataSplitter
from .evaluator import ModelEvaluator
from .model_factory import MODEL_REGISTRY, create_model
from .model_repository import JoblibModelRepository, ModelRepository
from .trainer import ModelTrainer
from .tuner import HyperparameterTuner
from .walk_forward import WalkForwardResult, pre_split_data, walk_forward_evaluate

__all__ = [
    'WalkForwardResult',
    'create_logistic_baseline',
    'pre_split_data',
    'sma_trend_signals',
    'walk_forward_evaluate',
    'BaseTrainer',
    'ChronologicalSplitter',
    'DataSplit',
    'DataSplitter',
    'JoblibModelRepository',
    'MODEL_REGISTRY',
    'ModelEvaluator',
    'ModelRepository',
    'ModelTrainer',
    'HyperparameterTuner',
    'create_model',
]
