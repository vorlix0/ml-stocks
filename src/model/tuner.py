"""
Hyperparameter tuning module using Optuna.

Provides :class:`HyperparameterTuner` which runs a Bayesian optimisation
study over the model hyperparameter space defined in ``config.py`` and
returns the best parameters found.

Usage::

    from src.model.tuner import HyperparameterTuner
    import pandas as pd

    df: pd.DataFrame = ...           # feature-engineered DataFrame
    tuner = HyperparameterTuner(df)
    best_params = tuner.run(n_trials=50)
    print(best_params)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("forex_ml.model.tuner")


class HyperparameterTuner:
    """Bayesian hyperparameter search using Optuna.

    Args:
        df: Feature-engineered DataFrame (must contain a ``Target`` column).
        model_type: Key from :data:`src.model.model_factory.MODEL_REGISTRY`.
            Defaults to the value in ``MODEL_CONFIG``.
        direction: Optuna optimisation direction (``"maximize"`` or
            ``"minimize"``).  Defaults to ``"maximize"`` (AUC).
        study_name: Optional name forwarded to :func:`optuna.create_study`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_type: str | None = None,
        direction: str = "maximize",
        study_name: str = "forex-ml-hf",
    ) -> None:
        try:
            import optuna  # noqa: F401 – validate availability at construction time
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install it with: pip install optuna"
            ) from exc

        from config import MODEL_CONFIG

        self.df = df
        self.model_type = model_type or MODEL_CONFIG.MODEL_TYPE
        self.direction = direction
        self.study_name = study_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_trials: int = 50,
        timeout: float | None = None,
        show_progress_bar: bool = False,
        **study_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the hyperparameter optimisation study.

        Args:
            n_trials: Number of Optuna trials.
            timeout: Optional wall-clock timeout in seconds.
            show_progress_bar: Show tqdm progress bar if *True*.
            **study_kwargs: Extra kwargs forwarded to
                :func:`optuna.create_study`.

        Returns:
            Dictionary with best hyperparameter values.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            **study_kwargs,
        )
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
        )

        best = study.best_params
        logger.info(
            f"Optuna study finished – best value={study.best_value:.4f}, "
            f"best_params={best}"
        )
        return best

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _objective(self, trial: Any) -> float:
        """Optuna objective: mean walk-forward AUC minus fold-to-fold spread."""
        from config import MODEL_CONFIG
        from src.model.model_factory import create_model
        from src.model.walk_forward import pre_split_data, walk_forward_evaluate

        # ---- model-specific search space ----
        params: dict[str, Any] = self._suggest_params(trial)
        params["random_state"] = MODEL_CONFIG.RANDOM_STATE

        # ---- data preparation (pre-test period only) ----
        feature_cols = [
            col for col in self.df.columns
            if col not in MODEL_CONFIG.EXCLUDED_COLUMNS
        ]
        x = self.df[feature_cols].dropna()
        y = self.df.loc[x.index, "Target"]
        x, y = pre_split_data(x, y)

        result = walk_forward_evaluate(
            lambda: create_model(self.model_type, **params), x, y
        )
        if not result.fold_aucs:
            return 0.0
        return result.mean_auc - result.std_auc

    def _suggest_params(self, trial: Any) -> dict[str, Any]:
        """Return model-specific hyperparameter suggestions for *trial*."""
        common: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        }

        if self.model_type == "gradient_boosting":
            common["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-3, 0.1, log=True
            )
            common["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        elif self.model_type == "random_forest":
            common["max_features"] = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            )

        return common
