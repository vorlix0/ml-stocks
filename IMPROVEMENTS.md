# Improvements: Complexity & Design Patterns – ml-stocks (forex-ml-hf)

**Prepared:** 2026-04-04  
**Scope:** Code complexity, design patterns, architecture, testing, and configuration  
**Note:** Items already tracked in `TODO.md` or resolved in `CODE_REVIEW.md` are excluded here.

---

## Table of Contents

1. [Design Pattern Gaps](#1-design-pattern-gaps)
2. [Code Complexity Issues](#2-code-complexity-issues)
3. [Architecture Concerns](#3-architecture-concerns)
4. [Code Quality Issues](#4-code-quality-issues)
5. [Testing Gaps](#5-testing-gaps)
6. [Configuration & Dependencies](#6-configuration--dependencies)
7. [Priority Matrix](#7-priority-matrix)

---

## 1. Design Pattern Gaps

### 1.1 Builder Pattern for `FeatureEngineer`

- **Problem:** `FeatureEngineer.create_all_features()` always computes all 70+ indicators. There is no way to selectively include/exclude feature groups without modifying the class source.
- **Impact:** Slower experimentation, wasted computation when testing specific feature subsets.
- **Recommendation:** Introduce a `FeatureEngineerBuilder` that accumulates feature-step callbacks and executes only what is needed.

```python
# sketch
class FeatureEngineerBuilder:
    def __init__(self, df: pd.DataFrame) -> None:
        self._engineer = FeatureEngineer.__new__(FeatureEngineer)
        self._engineer.df = df.copy()
        self._steps: list[Callable] = []

    def with_moving_averages(self) -> "FeatureEngineerBuilder":
        self._steps.append(self._engineer._add_moving_averages)
        return self

    def with_momentum_indicators(self) -> "FeatureEngineerBuilder":
        self._steps.append(self._engineer._add_momentum_indicators)
        return self

    def build(self) -> pd.DataFrame:
        self._engineer._add_basic_features()  # always required
        for step in self._steps:
            step()
        self._engineer._add_target()
        return self._engineer.df.dropna()
```

- **Files:** `src/data/features.py`

---

### 1.2 Repository Pattern for Data Access

- **Problem:** I/O logic is split across `src/utils/helpers.py` (CSV loading), entry-point scripts, and `DataDownloader`. Swapping the data storage layer (e.g., CSV → SQLite → S3) requires changes in multiple places.
- **Impact:** Tight coupling to the filesystem makes unit testing and future storage migration difficult.
- **Recommendation:** Define a `DataRepository` ABC with a `CsvDataRepository` implementation.

```python
# src/data/repository.py
from abc import ABC, abstractmethod
import pandas as pd

class DataRepository(ABC):
    @abstractmethod
    def load_raw(self, ticker: str) -> pd.DataFrame: ...

    @abstractmethod
    def save_raw(self, ticker: str, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_features(self, ticker: str) -> pd.DataFrame: ...

    @abstractmethod
    def save_features(self, ticker: str, df: pd.DataFrame) -> None: ...
```

- **Files:** New `src/data/repository.py`; update `src/utils/helpers.py`, entry-point scripts

---

### 1.3 Pipeline / Chain of Responsibility for Workflow Orchestration

- **Problem:** The four entry-point scripts (`download_finance_data.py`, `process_data.py`, `train_model.py`, `test_model.py`) must be run manually in the correct order. There is no caching, skip-if-cached logic, or recovery.
- **Impact:** Re-downloading and re-processing is required even when only the model needs retraining.
- **Recommendation:** Add a `Pipeline` class with `Step` objects and `skip_if_exists` callbacks.

```python
# src/pipeline.py (sketch)
from dataclasses import dataclass
from typing import Callable
from pathlib import Path

@dataclass
class Step:
    name: str
    run: Callable[[], None]
    output_path: Path | None = None  # skip step if path exists

class Pipeline:
    def __init__(self, steps: list[Step]) -> None:
        self._steps = steps

    def execute(self, force: bool = False) -> None:
        for step in self._steps:
            if not force and step.output_path and step.output_path.exists():
                continue
            step.run()
```

- **Files:** New `src/pipeline.py`; update entry-point scripts

---

### 1.4 Observer / Event Pattern for Experiment Tracking

- **Problem:** There are no hooks for monitoring pipeline events (e.g., "model trained", "backtest complete"). Integrating MLflow or Weights & Biases currently requires modifying core class internals.
- **Impact:** Core logic must be changed every time a new monitoring tool is introduced.
- **Recommendation:** Add a lightweight `EventEmitter` and emit named events from `ModelTrainer` and `TradingSimulator`.

```python
# src/utils/events.py (sketch)
from collections import defaultdict
from typing import Callable, Any

class EventEmitter:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Callable) -> None:
        self._listeners[event].append(callback)

    def emit(self, event: str, **payload: Any) -> None:
        for cb in self._listeners[event]:
            cb(**payload)
```

- **Files:** New `src/utils/events.py`; emit events in `ModelTrainer.train()` and `TradingSimulator.simulate()`

---

### 1.5 Abstract Data Source for `DataDownloader`

- **Problem:** `DataDownloader` is tightly coupled to `yfinance`. Switching to a different provider (Alpaca, Polygon.io, Alpha Vantage) requires rewriting the class.
- **Recommendation:** Extract a `DataSource` ABC and wrap `yfinance` in `YfinanceDataSource`.

```python
# src/data/sources.py (sketch)
from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):
    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...

class YfinanceDataSource(DataSource):
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        import yfinance as yf
        return yf.download(ticker, start=start, end=end, auto_adjust=True)
```

- **Files:** New `src/data/sources.py`; refactor `src/data/downloader.py`

---

## 2. Code Complexity Issues

### 2.1 `FeatureEngineer` Has 18 Private Methods (High Complexity)

- **Problem:** `FeatureEngineer` (327 lines, 18 `_add_*` methods) is a large class. `create_all_features()` always runs all 18 steps in a fixed sequence. Adding a new indicator group requires understanding the full method list and insertion order.
- **Cyclomatic complexity:** Low per method, but the class as a whole is a God Object for feature computation.
- **Recommendation:**
  - Group related methods into inner classes or separate feature-category modules (e.g., `momentum.py`, `volatility.py`).
  - Or implement the Builder pattern (see §1.1) to make step selection explicit.

- **Files:** `src/data/features.py`

---

### 2.2 `ModelTrainer.train()` Is Implicitly Coupled to `MODEL_CONFIG`

- **Problem:** `train()` reads hyperparameters directly from the module-level `MODEL_CONFIG` singleton. To train with different hyperparameters, callers must mutate module state or mock the module.
- **Recommendation:** Accept hyperparameters (or a `ModelConfig` instance) as constructor or method arguments, with `MODEL_CONFIG` as the default.

```python
# current
def train(self) -> ClassifierMixin:
    self.model = create_model(
        MODEL_CONFIG.MODEL_TYPE,
        n_estimators=MODEL_CONFIG.N_ESTIMATORS,
        ...
    )

# improved
def train(self, config: ModelConfig = MODEL_CONFIG) -> ClassifierMixin:
    self.model = create_model(
        config.MODEL_TYPE,
        n_estimators=config.N_ESTIMATORS,
        ...
    )
```

- **Files:** `src/model/trainer.py`

---

### 2.3 `Plotter` Is a Namespace Disguised as a Class

- **Problem:** All 5 methods in `Plotter` are `@staticmethod`. The class has no instance state, constructor, or subclassing intent. Using it as a class adds unnecessary boilerplate (`Plotter.plot_backtest_results(...)`) without any benefit.
- **Options:**
  - Convert to module-level functions in `visualization.py` (simplest).
  - Keep as class but add `@classmethod`-based factory state (e.g., output directory) if shared state is needed later.
  - Mark with `@final` and override `__new__` to prevent instantiation if the namespace pattern is intentional.
- **Files:** `src/utils/visualization.py`

---

### 2.4 Long Parameter Lists in `ModelTrainer.prepare_data()`

- **Problem:** `prepare_data()` uses 8 instance attributes to pass data between methods (`X_train`, `X_test`, `y_train`, `y_test`, `X_tr`, `X_val`, `y_tr`, `y_val`). These are initialized to `None` in `__init__` and set by `prepare_data()`, which is a form of temporal coupling.
- **Recommendation:** Introduce a `TrainTestSplit` dataclass to encapsulate the split state, and return it from `prepare_data()`.

```python
@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_tr: pd.DataFrame
    X_val: pd.DataFrame
    y_tr: pd.Series
    y_val: pd.Series
```

- **Files:** `src/model/trainer.py`

---

### 2.5 Magic Constant `top_n=10` in `Plotter.plot_feature_importance()`

- **Problem:** The default `top_n=10` for feature importance plots is not tied to any config value.
- **Recommendation:** Add `FEATURE_IMPORTANCE_TOP_N: int = 10` to `ModelConfig` or `BacktestConfig` and reference it as the default.
- **Files:** `src/utils/visualization.py`, `config.py`

---

## 3. Architecture Concerns

### 3.1 Global Configuration Coupling (All Modules)

- **Problem:** Every module imports config singletons directly:
  ```python
  from config import DATA_CONFIG, MODEL_CONFIG
  ```
  This creates an implicit global dependency that:
  - Makes unit testing harder (must mock at module level).
  - Prevents running multiple configurations in the same process (e.g., multi-ticker training).
- **Recommendation:** Pass config objects as constructor parameters with module-level singletons as defaults.

```python
# current
class DataDownloader:
    def __init__(self, ticker: str | None = None):
        self.ticker = ticker or DATA_CONFIG.TICKER

# improved
class DataDownloader:
    def __init__(self, ticker: str | None = None, config: DataConfig = DATA_CONFIG):
        self.ticker = ticker or config.TICKER
```

- **Files:** `src/data/downloader.py`, `src/data/features.py`, `src/model/trainer.py`, `src/backtest/strategy.py`, `src/backtest/simulator.py`, `src/backtest/metrics.py`

---

### 3.2 No End-to-End Pipeline Test

- **Problem:** Each module is tested in isolation, but there is no test that exercises the full flow: `download → process → train → backtest`. A regression in the interface between modules would go undetected.
- **Recommendation:** Add `tests/test_e2e.py` with a smoke test using synthetic data (no network calls), verifying that the pipeline completes and returns plausible results.
- **Files:** New `tests/test_e2e.py`

---

### 3.3 No Environment Variable Support in Configuration

- **Problem:** `config.py` has no way to override values (e.g., `TICKER`, `SPLIT_DATE`) from environment variables or CLI flags. Changing the ticker requires editing source code.
- **Recommendation:** Add env-var overrides using `os.environ.get()` or move to a Pydantic `BaseSettings` model.

```python
import os

@dataclass(frozen=True)
class DataConfig:
    TICKER: str = os.environ.get("TICKER", "AAPL")
    START_DATE: str = os.environ.get("START_DATE", "2015-01-01")
```

- **Files:** `config.py`

---

### 3.4 Entry-Point Scripts Lack a CLI Interface

- **Problem:** The four entry-point scripts accept no arguments. Users cannot specify a different ticker or date range without editing `config.py`.
- **Recommendation:** Add `argparse` or `typer`-based CLI parsing to each script, overriding config defaults from command-line flags.
- **Files:** `download_finance_data.py`, `process_data.py`, `train_model.py`, `test_model.py`

---

### 3.5 `ModelEvaluator` Is Not Covered by Tests

- **Problem:** `src/model/evaluator.py` (94 lines) containing accuracy, ROC AUC, and confusion matrix metrics has no corresponding test file.
- **Recommendation:** Add `tests/test_evaluator.py` covering:
  - Perfect prediction accuracy (1.0)
  - Random prediction baseline
  - `evaluate_model()` integration test with a real `GradientBoostingClassifier`
- **Files:** New `tests/test_evaluator.py`

---

## 4. Code Quality Issues

### 4.1 `load_model_safe()` Duplicates Logic from `ModelTrainer.load_model()`

- **Problem:** Both `src/utils/helpers.py::load_model_safe()` and `src/model/trainer.py::ModelTrainer.load_model()` check for file existence, call `joblib.load()`, and wrap errors in custom exceptions. The logic is nearly identical.
- **Recommendation:** Have `ModelTrainer.load_model()` delegate to `load_model_safe()`, or remove `load_model_safe()` and call `ModelTrainer.load_model()` from entry-point scripts.
- **Files:** `src/utils/helpers.py`, `src/model/trainer.py`

---

### 4.2 Date-Dependent Integration Test in `test_trainer.py`

- **Problem:** `TestModelTrainerIntegration` creates synthetic data from `"2021-01-01"` to `"2024-12-31"`. `MODEL_CONFIG.SPLIT_DATE = "2023-01-01"` is hardcoded in the global config. If the split date is changed in config, the test may silently produce an empty training set.
- **Recommendation:** Parametrize the split date or mock `MODEL_CONFIG` within the test to remove the coupling.
- **Files:** `tests/test_trainer.py`

---

### 4.3 `str` Paths Instead of `Path` Objects in Some Method Signatures

- **Problem:** Several methods accept `str` for file paths but internally convert to `Path`. The public API should be typed as `str | Path` (or `os.PathLike`) to be consistent with Python best practices.
- **Files:** `src/model/trainer.py` (`save_model`, `load_model`), `src/utils/helpers.py`

---

### 4.4 Missing `__slots__` on Data-Holder Classes

- **Problem:** Classes like `RiskMetrics`, `TradingSimulator`, and `TradingStrategy` store a fixed set of instance attributes. Without `__slots__`, each instance carries a `__dict__` that adds memory overhead and allows accidental attribute creation.
- **Impact:** Minor for a research tool; notable if running many simulations in a loop.
- **Recommendation:** Add `__slots__` to classes that have a stable, fixed attribute set.
- **Files:** `src/backtest/metrics.py`, `src/backtest/simulator.py`, `src/backtest/strategy.py`

---

### 4.5 `print_signal_stats()` Should Be a `__repr__` or Property

- **Problem:** `TradingStrategy.print_signal_stats()` uses the logger to output formatted text. Its name starts with `print_*`, which typically implies side effects only, yet it is used as the primary diagnostic method.
- **Recommendation:** Rename to `log_signal_stats()` to match the actual behavior (logging, not printing). Additionally, expose the stats as a `dict` or `pd.Series` return value for testability.
- **Files:** `src/backtest/strategy.py`

---

## 5. Testing Gaps

### 5.1 `DataDownloader` Has No Unit Tests

- **Problem:** `src/data/downloader.py` is entirely untested. The `download()` method calls `yf.download()` directly with no mockable seam.
- **Recommendation:**
  - Extract the `yfinance` call behind a `DataSource` ABC (see §1.5).
  - Add `tests/test_downloader.py` with a mock data source that returns a controlled DataFrame.
- **Files:** New `tests/test_downloader.py`

---

### 5.2 `Plotter` Has No Tests

- **Problem:** `src/utils/visualization.py` (`Plotter`, 203 lines, 5 chart methods) is not tested. Chart rendering bugs go undetected.
- **Recommendation:** Add `tests/test_visualization.py` using `matplotlib`'s `Agg` backend to verify that each method runs without error and returns a `Figure`/`Axes` object.
- **Files:** New `tests/test_visualization.py`

---

### 5.3 No Smoke Test for Complete Pipeline

- **Problem:** Individual module tests pass, but a bug in how modules connect to each other would not be caught. (See also §3.2 for the architecture context.)
- **Recommendation:** Add `tests/test_e2e.py` using synthetic data (no network calls) that flows through `FeatureEngineer → ModelTrainer → TradingStrategy → TradingSimulator → RiskMetrics`. The test should assert that final Sharpe ratio and max drawdown values are within reasonable bounds.
- **Files:** New `tests/test_e2e.py`

---

### 5.4 Signal Filter Tests Don't Cover Edge Cases

- **Problem:** `tests/test_signal_filter.py` doesn't test:
  - All-zero signals through `ADXFilter` (should stay all-zero).
  - ADX column missing from DataFrame (should raise `KeyError` or a custom exception).
  - `ADXFilter` with `threshold=0` (should pass all signals).
- **Files:** `tests/test_signal_filter.py`

---

## 6. Configuration & Dependencies

### 6.1 No Reproducibility Lock File

- **Problem:** `pyproject.toml` specifies only version lower bounds (e.g., `pandas >= 2.0.3`). A fresh install may pick up newer (potentially breaking) versions.
- **Recommendation:** Add a `requirements-lock.txt` or use `pip-tools` / `uv` to generate a pinned lock file for CI and reproducible environments.
- **Files:** Root directory (new lock file), `.github/workflows/test.yml`

---

### 6.2 `ta` Library Version Constraint Is Too Loose

- **Problem:** The technical analysis library `ta` is pinned only as `>= 0.10.0`. This library has had breaking API changes between minor versions.
- **Recommendation:** Pin to a specific minor version range (e.g., `ta >= 0.10.0, < 0.12.0`) and test on the minimum and maximum allowed versions in CI.
- **Files:** `pyproject.toml`

---

### 6.3 Dependency Vulnerability Scanning Not in CI

- **Problem:** There is no automated check for known vulnerabilities in `pyproject.toml` dependencies. A new CVE in `scikit-learn`, `pandas`, or `yfinance` would go unnoticed until a developer manually checks.
- **Recommendation:** Add a `pip-audit` or `safety` step to `.github/workflows/test.yml` to fail the build when vulnerable dependency versions are detected.

```yaml
# .github/workflows/test.yml — add after `pytest tests/`
- name: Security audit
  run: pip install pip-audit && pip-audit
```

- **Files:** `.github/workflows/test.yml`

---

## 7. Priority Matrix

| # | Item | Category | Effort | Impact | Priority |
|---|------|----------|--------|--------|----------|
| 1 | Global config coupling (§3.1) | Architecture | Medium | High | 🔴 High |
| 2 | `ModelTrainer` coupled to `MODEL_CONFIG` (§2.2) | Complexity | Low | High | 🔴 High |
| 3 | Repository Pattern for data access (§1.2) | Design Pattern | Medium | High | 🔴 High |
| 4 | Pipeline orchestration with skip-if-cached (§1.3) | Design Pattern | High | High | 🔴 High |
| 5 | Builder Pattern for `FeatureEngineer` (§1.1) | Design Pattern | Medium | Medium | 🟠 Medium |
| 6 | Abstract `DataSource` for `DataDownloader` (§1.5) | Design Pattern | Low | Medium | 🟠 Medium |
| 7 | `TrainTestSplit` dataclass (§2.4) | Complexity | Low | Medium | 🟠 Medium |
| 8 | `load_model_safe()` duplication (§4.1) | Code Quality | Low | Medium | 🟠 Medium |
| 9 | `ModelEvaluator` tests missing (§3.5) | Testing | Low | Medium | 🟠 Medium |
| 10 | `DataDownloader` unit tests (§5.1) | Testing | Medium | Medium | 🟠 Medium |
| 11 | End-to-end pipeline test (§3.2 / §5.3) | Testing | Medium | Medium | 🟠 Medium |
| 12 | Environment variable support in config (§3.3) | Config | Low | Medium | 🟠 Medium |
| 13 | CLI interface for entry-point scripts (§3.4) | Architecture | Medium | Low | 🟡 Low |
| 14 | Observer pattern for experiment tracking (§1.4) | Design Pattern | High | Low | 🟡 Low |
| 15 | `Plotter` namespace refactor (§2.3) | Complexity | Low | Low | 🟡 Low |
| 16 | `str | Path` in method signatures (§4.3) | Code Quality | Low | Low | 🟡 Low |
| 17 | `__slots__` on data-holder classes (§4.4) | Code Quality | Low | Low | 🟡 Low |
| 18 | `print_signal_stats()` rename (§4.5) | Code Quality | Low | Low | 🟡 Low |
| 19 | `Plotter` tests (§5.2) | Testing | Low | Low | 🟡 Low |
| 20 | Signal filter edge-case tests (§5.4) | Testing | Low | Low | 🟡 Low |
| 21 | Reproducibility lock file (§6.1) | Config | Low | Low | 🟡 Low |
| 22 | `ta` library version pin (§6.2) | Config | Low | Low | 🟡 Low |
| 23 | Dependency vulnerability scanning in CI (§6.3) | Config | Low | Medium | 🟠 Medium |
| 24 | Magic constant `top_n=10` (§2.5) | Code Quality | Low | Low | 🟡 Low |

---

*Legend: 🔴 High priority — fix soon; 🟠 Medium priority — fix in next iteration; 🟡 Low priority — nice to have*
