# Forex ML HF - ML Trading Strategy

Machine learning-based trading strategy for stock market prediction using technical indicators.

## 🚀 Quick Start

### Installation

```bash
# Clone and enter project
cd forex-ml-hf

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .           # Core only
pip install -e ".[dev]"    # With dev tools (pytest, black, ruff)
pip install -e ".[notebook]"  # With Jupyter
```

### Running the Pipeline

```bash
# 1. Download financial data
python download_finanse_data.py

# 2. Process data and create features
python process_data.py

# 3. Train ML model
python train_model.py

# 4. Backtest strategy
python test_model.py
```

Or run all at once:
```bash
python download_finanse_data.py && python process_data.py && python train_model.py && python test_model.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_metrics.py -v
```

## 📁 Project Structure

```
forex-ml-hf/
├── data/
│   ├── raw/              # Raw downloaded data (AAPL_data.csv)
│   └── processed/        # Processed features (AAPL_features.csv)
├── models/               # Trained models (model.joblib)
├── outputs/
│   └── charts/           # Generated charts (backtest.png, etc.)
├── src/
│   ├── backtest/         # Backtesting modules
│   ├── data/             # Data downloading & feature engineering
│   ├── model/            # ML model training & evaluation
│   └── utils/            # Helpers, logging, visualization
├── tests/                # Unit tests (53 tests)
├── config.py             # Central configuration
├── pyproject.toml        # Dependencies & tool config
└── README.md
```

## ⚙️ Configuration

Edit `config.py` to change:
- **Ticker symbol** (default: AAPL)
- **Date range** for data
- **Model hyperparameters** (n_estimators, max_depth, etc.)
- **Strategy thresholds** (prediction threshold, ADX filter)

## 📊 Features

- 70+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ADX, etc.)
- GradientBoosting classifier
- Backtesting with Buy & Hold comparison
- ADX trend filter for signal generation
- Risk metrics (Sharpe ratio, max drawdown)

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black .
ruff check . --fix

# Type checking
mypy src/
```

## 📈 Output Example

After running the pipeline:
- `outputs/charts/feature_importance.png` - Top features
- `outputs/charts/backtest.png` - Strategy vs Buy & Hold
- `outputs/charts/portfolio_value.png` - Portfolio with ADX indicator
