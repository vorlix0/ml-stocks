# TODO: Remaining optimizations for forex-ml-hf project

## ✅ Completed

- [x] **Hardcoded values** - moved to `config.py`
- [x] **Lack of SRP (Single Responsibility Principle)** - refactored to `src/` modules
- [x] **Missing `if __name__ == "__main__"`** - added to all scripts
- [x] **File names vs comments** - fixed file headers
- [x] **Missing error handling** - added custom exceptions and data validation
- [x] **Print debugging → Logging** - added `src/utils/logger.py` module with hierarchical logging (`forex_ml.*`)
- [x] **Code language** - all comments, docstrings and logs translated to English
- [x] **Pickle → Joblib** - replaced pickle with joblib for faster serialization (with compression)
- [x] **Complete type hints** - added full type hints and docstrings to all private methods
- [x] **requirements.txt → pyproject.toml** - modern dependency management with optional groups (dev, notebook)
- [x] **Remove unused dependencies** - removed tensorflow, seaborn; moved jupyter to optional [notebook]
- [x] **Unit tests** - added `tests/` folder with 53 tests for FeatureEngineer, RiskMetrics, TradingStrategy, ModelTrainer
- [x] **Directory structure** - organized project into `data/`, `models/`, `outputs/` directories
- [x] **Add pre-commit hooks** - added `.pre-commit-config.yaml` with ruff, ruff-format, and mypy hooks

---

## ✅ Additional Completions

### 1. Add CI/CD (GitHub Actions) ✅
File `.github/workflows/test.yml` (already implemented with Python 3.10/3.11/3.12 matrix):

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e ".[dev]"
      - run: pytest tests/
```

---

# to extend this list

## 🟢 Nice-to-have (future)

- [x] Add input data validation with Pydantic
- [x] Experiment tracking with MLflow/Weights&Biases
- [x] Hyperparameter tuning with Optuna
- [ ] Feature store (e.g., Feast)
- [x] Containerization with Docker
- [x] CLI with Click/Typer instead of separate scripts
