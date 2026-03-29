"""
Factory / Registry for scikit-learn classifier construction.

Add new model types here without touching any other module.

Usage::

    from src.model.model_factory import create_model
    from config import MODEL_CONFIG

    model = create_model(MODEL_CONFIG.MODEL_TYPE, **hyperparams)
"""
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Registry maps configuration keys to estimator classes.
# Extend this dict to support additional algorithms.
MODEL_REGISTRY: dict[str, type[ClassifierMixin]] = {
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
}


def create_model(name: str, **kwargs: object) -> ClassifierMixin:
    """Instantiate a classifier by its registry name.

    Args:
        name: Key from :data:`MODEL_REGISTRY` (e.g. ``"gradient_boosting"``).
        **kwargs: Keyword arguments forwarded to the estimator constructor.

    Returns:
        An unfitted scikit-learn classifier instance.

    Raises:
        ValueError: When ``name`` is not found in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)
