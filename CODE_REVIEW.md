# Code Review – ml-stocks (forex-ml-hf)

**Reviewer:** GitHub Copilot  
**Date:** 2026-03-21  
**Repository:** `vorlix0/ml-stocks`  
**Language:** Python 3.10+  
**Framework:** scikit-learn, pandas, yfinance, ta

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Patterns](#2-design-patterns)
   - [2.1 Patterns in Use](#21-patterns-in-use)
   - [2.2 Missing Patterns & Recommendations](#22-missing-patterns--recommendations)
3. [Architecture](#3-architecture)
4. [Code Quality Findings](#4-code-quality-findings)
5. [Testing](#5-testing)
6. [Security & Data Integrity](#6-security--data-integrity)
7. [Performance](#7-performance)
8. [Summary & Priority Matrix](#8-summary--priority-matrix)

---

## 1. Project Overview

The project is an end-to-end **ML-based stock trading pipeline** that:

1. Downloads OHLCV data from Yahoo Finance (`DataDownloader`)
2. Engineers 70+ technical indicators as ML features (`FeatureEngineer`)
3. Trains a `GradientBoostingClassifier` to predict price direction (`ModelTrainer`)
4. Backtests generated signals against a Buy & Hold benchmark (`TradingStrategy`, `TradingSimulator`)
5. Computes risk metrics and renders charts (`RiskMetrics`, `Plotter`)

The code is clean, well-documented, and clearly the result of deliberate refactoring (see `TODO.md`). The findings below aim to help the project mature further.

---

## 2. Design Patterns

### 2.1 Patterns in Use

#### ✅ Value Object / Immutable Configuration (`config.py`)

`DataConfig`, `FeatureConfig`, `ModelConfig`, and `BacktestConfig` are **frozen dataclasses** with computed `@property` paths. This is the _Value Object_ pattern: instances carry no identity—only their data matters—and they are effectively immutable.

```python
@dataclass(frozen=True)
class BacktestConfig:
    PREDICTION_THRESHOLD: float = 0.38
    ADX_THRESHOLD: int = 25
    INITIAL_CAPITAL: float = 10000.0
    ...
```

**Strength:** Prevents accidental mutation of shared configuration. All four config objects are module-level singletons (`DATA_CONFIG`, `FEATURE_CONFIG`, etc.).

**Improvement opportunity:** The singleton instances live at module scope, making unit-test isolation harder. Consider a lightweight `AppConfig` container that groups them for easier injection/mocking.

---

#### ✅ Strategy Pattern (`src/backtest/strategy.py`)

`TradingStrategy.generate_signals()` selects an algorithm branch at runtime via the `use_adx_filter` flag:

```python
def generate_signals(self, threshold=None, adx_threshold=None, use_adx_filter=False):
    if use_adx_filter:
        signals = (self.df['Prediction'] > threshold) & (self.df['ADX'] > adx_threshold)
    else:
        signals = self.df['Prediction'] > threshold
    return signals.astype(int)
```

This is a _boolean-flag Strategy_ variant. It works, but see §2.2 for a stronger implementation.

---

#### ✅ Template Method Pattern (`src/data/features.py`)

`FeatureEngineer.create_all_features()` defines a fixed algorithm skeleton; each step delegates to a protected method:

```python
def create_all_features(self) -> pd.DataFrame:
    self._add_basic_features()
    self._add_moving_averages()
    self._add_bollinger_bands()
    ...          # ← fixed order, variable steps
    self._add_target()
    return self.df.dropna()
```

This is the _Template Method_ pattern. The public method is the template; each `_add_*` method is a step that could be overridden in a subclass.

---

#### ✅ Facade Pattern (entry-point scripts)

`train_model.py` and `test_model.py` act as **Facades**: they wire together multiple subsystems (`FeatureEngineer`, `ModelTrainer`, `TradingStrategy`, `RiskMetrics`, `Plotter`) behind a single, sequential script. Callers only need to run one file.

---

#### ✅ Custom Exception Hierarchy (`src/exceptions.py`)

```
ForexMLError
├── DataNotFoundError
├── EmptyDataError
├── InvalidDataError
├── ModelNotFoundError
├── ModelNotTrainedError
└── DownloadError
```

This is the _Type Object_ / layered exception pattern, enabling fine-grained `except` clauses while sharing a common base for broad catches.

---

#### ✅ Static Factory Methods (`src/utils/helpers.py`)

`load_csv_safe()` and `load_pickle_safe()` are **Static Factory** helpers: they encapsulate construction logic (open file → validate → return object) behind a named function, hiding `pd.read_csv` / `joblib.load` internals.

---

#### ✅ Composition over Inheritance

No deep inheritance chains exist. Classes are wired together in scripts rather than coupled by parent/child relationships. `TradingSimulator`, `RiskMetrics`, and `TradingStrategy` each receive data as a constructor argument—a form of **Dependency Injection** by composition.

---

### 2.2 Missing Patterns & Recommendations

#### ❌ Strategy Pattern — use a proper interface instead of a boolean flag

**Current code (strategy.py):**
```python
def generate_signals(self, ..., use_adx_filter: bool = False) -> pd.Series:
    if use_adx_filter:
        ...
    else:
        ...
```

A boolean flag that selects an algorithm is a code smell. Adding a third filter (e.g. RSI-based) requires editing this method.

**Recommended refactoring:**

```python
# src/backtest/signal_filter.py
from abc import ABC, abstractmethod
import pandas as pd

class SignalFilter(ABC):
    @abstractmethod
    def apply(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series: ...

class NoFilter(SignalFilter):
    def apply(self, signals, df):
        return signals

class ADXFilter(SignalFilter):
    def __init__(self, threshold: int = 25):
        self.threshold = threshold

    def apply(self, signals, df):
        return signals & (df['ADX'] > self.threshold)
```

`TradingStrategy.generate_signals()` then accepts a `SignalFilter` object, making it **Open/Closed**: open for extension (new filter subclass), closed for modification.

---

#### ❌ Abstract Base Class for models (`src/model/trainer.py`)

`ModelTrainer` is tightly coupled to `GradientBoostingClassifier`. Swapping to `RandomForestClassifier` or `XGBClassifier` requires modifying the class internals.

**Recommended refactoring:**

```python
# src/model/base_trainer.py
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import ClassifierMixin

class BaseTrainer(ABC):
    @abstractmethod
    def train(self) -> ClassifierMixin: ...

    @abstractmethod
    def get_feature_importances(self) -> pd.DataFrame: ...

    @abstractmethod
    def save_model(self, path: str = None) -> None: ...

    @staticmethod
    @abstractmethod
    def load_model(path: str = None) -> ClassifierMixin: ...
```

`ModelTrainer` would implement `BaseTrainer`. A future `XGBoostTrainer` could do the same with zero changes to backtest code.

---

#### ❌ Factory Method / Registry for model selection

Currently the model type (GradientBoosting) is hard-coded in `ModelTrainer.train()`. A **Factory** would centralise model construction:

```python
# src/model/model_factory.py
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

MODEL_REGISTRY = {
    "gradient_boosting": GradientBoostingClassifier,
    "random_forest": RandomForestClassifier,
}

def create_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
```

Combined with a `model_type` key in `ModelConfig`, the model can be switched via configuration alone.

---

#### ❌ Builder Pattern for `FeatureEngineer`

`FeatureEngineer.create_all_features()` always creates all 70+ features. This makes it impossible to create a subset for experimentation or a lighter production model.

**Recommended refactoring:**

```python
class FeatureEngineerBuilder:
    def __init__(self, df: pd.DataFrame):
        self._engineer = FeatureEngineer(df)

    def with_moving_averages(self):
        self._engineer._add_moving_averages()
        return self

    def with_momentum(self):
        self._engineer._add_momentum_indicators()
        return self

    def with_target(self):
        self._engineer._add_target()
        return self

    def build(self) -> pd.DataFrame:
        return self._engineer.df.dropna()

# Usage:
df = (
    FeatureEngineerBuilder(raw_df)
    .with_moving_averages()
    .with_momentum()
    .with_target()
    .build()
)
```

This preserves the existing `create_all_features()` as a convenience method while enabling flexible feature selection.

---

#### ❌ Repository Pattern for data access

File I/O is scattered across helper functions, `DataDownloader`, and top-level scripts. A **Repository** abstraction would decouple the rest of the code from the storage layer:

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

class CsvDataRepository(DataRepository):
    """Concrete implementation using local CSV files."""
    ...
```

Benefits: easy to swap for a database, S3, or an in-memory stub in tests.

---

#### ❌ Pipeline / Chain of Responsibility for the ML workflow

The four entry-point scripts (`download_finanse_data.py → process_data.py → train_model.py → test_model.py`) form a manual pipeline with no failure recovery, progress tracking, or skip-if-cached logic.

A **Pipeline** pattern would model each step as a composable unit:

```python
# src/pipeline.py
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Step:
    name: str
    run: Callable[[], Any]
    skip_if: Callable[[], bool] = lambda: False

class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps

    def execute(self) -> None:
        for step in self.steps:
            if step.skip_if():
                logger.info(f"Skipping step: {step.name}")
                continue
            logger.info(f"Running step: {step.name}")
            step.run()
```

This enables caching (skip download if CSV already exists), easy re-running of individual steps, and better observability.

---

#### ❌ Observer / Event Pattern for monitoring

There is currently no way to hook into pipeline events (e.g. "model trained", "backtest complete") without modifying source files. An **Observer** pattern would allow plugging in experiment-tracking tools (MLflow, W&B) or alerting without touching core logic.

---

## 3. Architecture

### Positive findings

| Aspect | Assessment |
|--------|------------|
| Module separation | ✅ Excellent – `data/`, `model/`, `backtest/`, `utils/` |
| Single Responsibility | ✅ Each class has one clear job |
| Centralized config | ✅ `config.py` with frozen dataclasses |
| Logging | ✅ Hierarchical `forex_ml.*` namespace |
| Type hints | ✅ Present on all public methods |
| Custom exceptions | ✅ Well-structured hierarchy |
| Dependency management | ✅ `pyproject.toml` with optional groups |

### Architectural concern – coupling to `config.py` module globals

All classes import `DATA_CONFIG`, `FEATURE_CONFIG`, etc. directly from `config`:

```python
# src/data/downloader.py
from config import DATA_CONFIG

class DataDownloader:
    def __init__(self, ticker=None, ...):
        self.ticker = ticker or DATA_CONFIG.TICKER  # ← module-level global
```

This is **implicit dependency injection**. The class cannot be used without the global being available, making testing and multi-configuration runs harder. Prefer passing config objects explicitly:

```python
class DataDownloader:
    def __init__(self, config: DataConfig = DATA_CONFIG):
        self.ticker = config.TICKER
```

---

## 4. Code Quality Findings

### 4.1 Bug – side-effects in `_validate_input` (`src/model/trainer.py`, lines 52–60)

`_validate_input()` is a validation method, but it also **initialises instance attributes** as a side effect:

```python
def _validate_input(self, df: pd.DataFrame) -> None:
    if df.empty:
        raise EmptyDataError(...)
    if 'Target' not in df.columns:
        raise InvalidDataError(...)

    # ← side effect: attribute initialisation in a validation method
    self.X_train = None
    self.X_test = None
    ...
```

These initialisations should be in `__init__`, not in the validator. If validation raises before reaching those lines, the attributes never exist, causing `AttributeError` later.

**Fix:**
```python
def __init__(self, df: pd.DataFrame):
    self._validate_input(df)
    self.df = df
    self.model = None
    self.feature_cols = self._get_feature_columns()
    # initialise split attributes here
    self.X_train = self.X_test = self.y_train = self.y_test = None
    self.X_tr = self.X_val = self.y_tr = self.y_val = None

def _validate_input(self, df: pd.DataFrame) -> None:
    if df.empty:
        raise EmptyDataError("Input DataFrame is empty")
    if 'Target' not in df.columns:
        raise InvalidDataError("Missing 'Target' column in data.")
```

---

### 4.2 Typo in filename – `download_finanse_data.py`

The script is named `download_finanse_data.py` (Polish spelling). It should be `download_finance_data.py`. This is a visible entry point documented in the README.

---

### 4.3 Mixed language – Polish docstring in `src/exceptions.py`

```python
class DownloadError(ForexMLError):
    """Wyjątek rzucany gdy pobieranie danych się nie powiedzie."""
```

All other docstrings are in English. This line is Polish and should be translated:

```python
class DownloadError(ForexMLError):
    """Exception raised when downloading data fails."""
```

---

### 4.4 Polish comment in `src/utils/logger.py`

```python
# Domyślny logger dla projektu
logger = setup_logger()
```

Should be:

```python
# Default logger for the project
logger = setup_logger()
```

---

### 4.5 `load_pickle_safe` is not loading pickle files (`src/utils/helpers.py`)

The function is named `load_pickle_safe` but uses `joblib.load` internally. After migrating from pickle to joblib (noted in `TODO.md`), the function name was not updated. This is misleading.

**Rename to:** `load_model_safe` or `load_joblib_safe`.

---

### 4.6 `Optional` import could use `X | None` syntax (`src/utils/helpers.py`)

The project targets Python ≥ 3.10. The `Optional[str]` and `Optional[List[str]]` type hints can use the modern union syntax:

```python
# Before (Python 3.9 style)
from typing import List, Optional
def load_csv_safe(
    path: str,
    required_columns: Optional[List[str]] = None,
    # ... other params
) -> pd.DataFrame: ...

# After (Python 3.10+ style)
def load_csv_safe(
    path: str,
    required_columns: list[str] | None = None,
    # ... other params
) -> pd.DataFrame: ...
```

`ruff` rule `UP` (pyupgrade) would flag this; it is already enabled in `pyproject.toml`.

---

### 4.7 `Plotter` methods are all `@staticmethod` – consider a plain module

`Plotter` has no instance state; every method is `@staticmethod`. This is a sign the class is a **namespace**, not an object. Consider converting to a module with top-level functions, which avoids the `Plotter.method()` boilerplate:

```python
# src/utils/visualization.py  (module-level functions)
def plot_feature_importance(importances, top_n=10, save_path=None, show=True): ...
def plot_backtest(index, cumulative_returns, save_path=None, show=True): ...
```

Alternatively, keep the class but add `__init_subclass__` to prevent instantiation (no-instance pattern).

---

### 4.8 Magic number in `Plotter.plot_feature_importance` – `top_n` default

The default `top_n=10` is repeated in two methods (`plot_feature_importance` and `print_feature_importance_analysis`). This value should be sourced from config or a named constant to keep it consistent.

---

### 4.9 `ModelConfig.EXCLUDED_COLUMNS` excludes raw OHLCV columns

```python
EXCLUDED_COLUMNS: tuple = ('Target', 'Close', 'Open', 'High', 'Low', 'Volume')
```

`Close`, `Open`, `High`, `Low`, `Volume` appear in both raw data and as lag/interaction features. Excluding `Close` is intentional (avoid data leakage), but the comment should document this reasoning clearly.

---

## 5. Testing

### Positive findings

- 4 test files with good coverage of core functionality
- Pytest fixtures in `conftest.py` are reusable and well-structured
- Tests exercise edge cases (empty data, missing columns)

### Missing tests

| Area | Missing test |
|------|-------------|
| `DataDownloader` | No test for network error simulation |
| `TradingSimulator` | `simulate()` and `simulate_buy_and_hold()` have no tests |
| `Plotter` | No rendering tests (even `show=False` path) |
| `helpers.py` | `load_pickle_safe` / `validate_ohlcv_data` / `validate_features_data` not tested |
| Pipeline scripts | `train_model.py`, `test_model.py` have no integration tests |

### Test isolation concern

`ModelTrainer` tests that call `trainer.prepare_data()` depend on `MODEL_CONFIG.SPLIT_DATE = "2023-01-01"`. The synthetic fixtures in `conftest.py` generate 200 rows starting from today; if the test environment date falls before 2023-01-01 or after a date where all rows end up in the test set, the split produces an empty training set.

**Fix:** Pass explicit `split_date` or mock `MODEL_CONFIG.SPLIT_DATE` in `test_trainer.py`.

---

## 6. Security & Data Integrity

### 6.1 Potential look-ahead bias in feature engineering

The target is computed as:

```python
future_return = (self.df['Close'].shift(-horizon) - self.df['Close']) / self.df['Close']
self.df['Target'] = (future_return > threshold).astype(int)
```

`shift(-horizon)` correctly shifts future prices back. However, some features (e.g. `ZScore_{window}`, `Returns_std_{window}`) use `rolling()` without `min_periods`, so early rows may be computed from fewer than `window` observations. After `dropna()` these rows are removed, which is correct, but this should be made explicit in comments.

### 6.2 No input sanitization for the `ticker` parameter

```python
class DataDownloader:
    def __init__(self, ticker: str = None, ...):
        self.ticker = ticker or DATA_CONFIG.TICKER
```

The `ticker` value is passed directly to `yf.download()`. An attacker providing a crafted ticker string could potentially cause unexpected yfinance behaviour. A simple allowlist or regex check (e.g. `^[A-Z0-9\.\-\^]{1,10}$`) would harden this.

### 6.3 Model file path traversal (`src/model/trainer.py`)

```python
@staticmethod
def load_model(path: str = None):
    path = Path(path) if path else MODEL_CONFIG.model_file
    return joblib.load(path)
```

If `path` comes from user input, `joblib.load` of an untrusted file can execute arbitrary code (pickle-based deserialization). This is a **low severity** finding for an internal research tool but should be noted if the API is ever exposed externally.

---

## 7. Performance

### 7.1 `iterrows()` in `TradingSimulator.simulate()` (HIGH impact)

```python
for idx, row in df_sim.iterrows():   # ← O(n) Python loop
    signal = row['Signal']
    close_price = row['Close']
    ...
    portfolio_values.append(current_value)
```

`iterrows()` is the slowest way to iterate a DataFrame. For 2500 rows it is fine; for tick-level data it will be a bottleneck.

**Vectorized replacement:**

```python
def simulate(self, signals: pd.Series) -> dict:
    df = self.df[['Close']].copy()
    df['Signal'] = signals
    df['Position'] = signals.shift(1).fillna(0)           # yesterday's signal
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    df['Portfolio'] = self.initial_capital * (1 + df['Strategy_Return']).cumprod()
    ...
```

### 7.2 Feature engineering modifies `self.df` in place

Each `_add_*` method writes directly to `self.df`. If `create_all_features()` fails mid-way, the DataFrame is left in a partially-mutated state. Consider building features in a separate dict and merging at the end, or at least deep-copying at the start of each public method.

### 7.3 `FeatureEngineer` holds a copy of the full DataFrame

```python
self.df = df.copy()
```

For large datasets (many tickers, long histories), this doubles RAM usage. Document this trade-off; for production use, consider operating on a view or processing in chunks.

---

## 8. Summary & Priority Matrix

| # | Finding | Severity | Category | Effort |
|---|---------|----------|----------|--------|
| 1 | Replace boolean Strategy flag with ABC + concrete filters | Medium | Design Pattern | Medium |
| 2 | Add `BaseTrainer` ABC for model pluggability | Medium | Design Pattern | Low |
| 3 | Add model Factory/Registry | Low | Design Pattern | Low |
| 4 | Add Builder for selective feature engineering | Low | Design Pattern | Medium |
| 5 | Add Repository pattern for data access | Low | Design Pattern | High |
| 6 | Bug: side effects in `_validate_input` (`trainer.py`) | High | Bug | Low |
| 7 | Rename `download_finanse_data.py` → `download_finance_data.py` | Low | Quality | Trivial |
| 8 | Translate Polish docstring in `exceptions.py` | Low | Quality | Trivial |
| 9 | Translate Polish comment in `logger.py` | Low | Quality | Trivial |
| 10 | Rename `load_pickle_safe` → `load_model_safe` | Low | Quality | Trivial |
| 11 | Use Python 3.10+ union type hints (`X | None`) | Low | Quality | Low |
| 12 | Convert `Plotter` static class to module functions | Low | Design | Low |
| 13 | Config objects passed as globals – prefer injection | Medium | Architecture | Medium |
| 14 | Add tests for `TradingSimulator`, `Plotter`, `helpers` | Medium | Testing | Medium |
| 15 | Fix `ModelTrainer` test date-dependency | Medium | Testing | Low |
| 16 | Vectorize `TradingSimulator.simulate()` (remove `iterrows`) | Medium | Performance | Low |
| 17 | Add ticker input validation in `DataDownloader` | Low | Security | Low |
| 18 | Document pickle-deserialization risk in `load_model` | Low | Security | Trivial |

### Top 3 Immediate Actions (highest ROI)

1. **Fix `_validate_input` side effect** (Finding #6) – latent `AttributeError` bug, one-line fix.
2. **Replace Strategy boolean flag with ABC** (Finding #1) – the most impactful design-pattern improvement; enables future signal filters without modifying existing code.
3. **Vectorize `TradingSimulator.simulate()`** (Finding #16) – `iterrows()` is an anti-pattern in pandas; the fix is straightforward and pays off as data grows.
