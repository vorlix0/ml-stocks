"""
Model module - training and evaluation of ML models.
"""
from .base_trainer import BaseTrainer
from .evaluator import ModelEvaluator
from .model_factory import MODEL_REGISTRY, create_model
from .trainer import ModelTrainer

__all__ = ['BaseTrainer', 'ModelTrainer', 'ModelEvaluator', 'create_model', 'MODEL_REGISTRY']
