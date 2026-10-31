"""
Walk-forward (expanding window) validation for time-series models.

Each fold trains on all data before a cut-off and validates on the next
block, with a gap of ``TARGET_HORIZON_DAYS`` samples between them so that
forward-looking labels never overlap the validation block.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config import FEATURE_CONFIG, MODEL_CONFIG

logger = logging.getLogger("forex_ml.model.walk_forward")


@dataclass(frozen=True)
class WalkForwardResult:
    """AUC per fold plus summary statistics."""

    fold_aucs: tuple[float, ...]

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_aucs)) if self.fold_aucs else float("nan")

    @property
    def std_auc(self) -> float:
        return float(np.std(self.fold_aucs)) if self.fold_aucs else float("nan")


def pre_split_data(
    x: pd.DataFrame, y: pd.Series, split_date: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns labelled rows strictly before the train/test boundary.

    The last ``TARGET_HORIZON_DAYS`` rows are dropped because their labels
    look into the test period.
    """
    split_date = split_date or MODEL_CONFIG.SPLIT_DATE
    mask = (x.index < split_date) & y.notna().to_numpy()
    gap = FEATURE_CONFIG.TARGET_HORIZON_DAYS
    return x[mask].iloc[:-gap], y[mask].iloc[:-gap]


def walk_forward_evaluate(
    model_builder: Callable[[], ClassifierMixin],
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int | None = None,
) -> WalkForwardResult:
    """Evaluates a model with expanding-window walk-forward validation.

    Args:
        model_builder: Callable returning a fresh unfitted classifier.
        x: Features (chronologically ordered, labelled rows only).
        y: Binary target aligned with ``x``.
        n_splits: Number of folds (default ``MODEL_CONFIG.WALK_FORWARD_SPLITS``).

    Returns:
        Per-fold ROC AUC (folds with a single validation class are skipped).
    """
    n_splits = n_splits or MODEL_CONFIG.WALK_FORWARD_SPLITS
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=FEATURE_CONFIG.TARGET_HORIZON_DAYS)

    aucs: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x), start=1):
        y_val = y.iloc[val_idx]
        if y_val.nunique() < 2 or y.iloc[train_idx].nunique() < 2:
            logger.warning(f"Fold {fold}: single class, skipped")
            continue
        model = model_builder()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(x.iloc[val_idx])[:, 1]
        auc = float(roc_auc_score(y_val, proba))
        logger.info(f"Fold {fold}: AUC={auc:.3f} (train={len(train_idx)}, val={len(val_idx)})")
        aucs.append(auc)

    return WalkForwardResult(fold_aucs=tuple(aucs))
