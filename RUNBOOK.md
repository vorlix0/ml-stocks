# ML Stocks – Runbook

Step-by-step guide for running the ML trading strategy pipeline on **macOS** and **Windows**.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
   - [macOS](#macos)
   - [Windows](#windows)
3. [Configuration](#3-configuration)
4. [How to Train the Model](#4-how-to-train-the-model)
5. [How to Run Predictions (Backtest)](#5-how-to-run-predictions-backtest)
6. [Outputs](#6-outputs)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | latest |
| Git | any recent |
| Internet access | required (to download stock data via yfinance) |

### Python dependencies (installed automatically)

| Package | Minimum version | Purpose |
|---|---|---|
| numpy | 1.26.0 | Numerical computing |
| pandas | 2.0.3 | Data manipulation |
| matplotlib | 3.8.0 | Charting |
| yfinance | 0.2.0 | Downloading stock data from Yahoo Finance |
| scikit-learn | 1.3.0 | Machine learning (GradientBoosting) |
| joblib | 1.3.0 | Model serialisation |
| ta | 0.10.0 | Technical analysis indicators |

---

## 2. Installation

### macOS

**1. Verify Python version**

Open **Terminal** and run:

```bash
python3 --version
```

If Python 3.10 or newer is not installed, download it from [python.org](https://www.python.org/downloads/) or use Homebrew:

```bash
brew install python@3.12
```

**2. Clone the repository**

```bash
git clone https://github.com/vorlix0/ml-stocks.git
cd ml-stocks
```

**3. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)`.

**4. Install dependencies**

```bash
pip install -e .
```

To also install developer tools (linting, testing):

```bash
pip install -e ".[dev]"
```

---

### Windows

**1. Verify Python version**

Open **Command Prompt** or **PowerShell** and run:

```cmd
python --version
```

If Python 3.10 or newer is not installed, download it from [python.org](https://www.python.org/downloads/).  
During installation, tick **"Add Python to PATH"**.

**2. Clone the repository**

```cmd
git clone https://github.com/vorlix0/ml-stocks.git
cd ml-stocks
```

**3. Create and activate a virtual environment**

Command Prompt:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Note (PowerShell only):** If you see an error about execution policy, run the following once and then retry activation:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Your prompt should now show `(.venv)`.

**4. Install dependencies**

```cmd
pip install -e .
```

To also install developer tools (linting, testing):

```cmd
pip install -e ".[dev]"
```

---

## 3. Configuration

All settings are stored in `config.py`. Edit this file to customise the pipeline.

| Setting | Default | Description |
|---|---|---|
| `DataConfig.TICKER` | `AAPL` | Stock ticker symbol (e.g. `MSFT`, `GOOGL`) |
| `DataConfig.START_DATE` | `2015-01-01` | Start of the historical data range |
| `DataConfig.END_DATE` | `2024-12-21` | End of the historical data range |
| `ModelConfig.SPLIT_DATE` | `2023-01-01` | Date separating training and test sets |
| `ModelConfig.N_ESTIMATORS` | `300` | Number of trees in GradientBoosting |
| `ModelConfig.MAX_DEPTH` | `5` | Maximum depth of each tree |
| `ModelConfig.LEARNING_RATE` | `0.05` | Learning rate for GradientBoosting |
| `BacktestConfig.PREDICTION_THRESHOLD` | `0.38` | Minimum predicted probability to generate a buy signal |
| `BacktestConfig.ADX_THRESHOLD` | `25` | ADX value above which a trend is considered strong |
| `BacktestConfig.INITIAL_CAPITAL` | `10000.0` | Starting capital (USD) for the simulation |

---

## 4. How to Train the Model

Run the following four scripts **in order**. Each step depends on the previous one.

### Step 1 – Download financial data

Downloads raw OHLCV data from Yahoo Finance and saves it to `data/raw/`.

```bash
# macOS / Linux
python download_finanse_data.py

# Windows
python download_finanse_data.py
```

Expected output (example):
```
[*********************100%***********************]  1 of 1 completed
Data saved to data/raw/AAPL_data.csv
```

### Step 2 – Process data and create features

Computes 70+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ADX, …) and saves the enriched dataset to `data/processed/`.

```bash
python process_data.py
```

Expected output (example):
```
Loading data from: data/raw/AAPL_data.csv
Features: 87
Features saved to: data/processed/AAPL_features.csv
```

### Step 3 – Train the model

Trains a GradientBoosting classifier on data before `SPLIT_DATE`, evaluates it on the validation set, prints metrics, generates a feature-importance chart, and saves the model to `models/model.joblib`.

```bash
python train_model.py
```

Expected output (example):
```
Data shape: (2350, 88)
NaN in data: 0

Target distribution:
1    1312
0    1038
% UP: 55.83%

Validation Accuracy: 0.59
...
Model saved to models/model.joblib
```

### Run all training steps at once

**macOS / Linux:**
```bash
python download_finanse_data.py && \
python process_data.py && \
python train_model.py
```

**Windows (Command Prompt):**
```cmd
python download_finanse_data.py && python process_data.py && python train_model.py
```

**Windows (PowerShell):**
```powershell
python download_finanse_data.py; python process_data.py; python train_model.py
```

---

## 5. How to Run Predictions (Backtest)

After training (Step 4 above), run the backtest script to evaluate the strategy on unseen data (dates ≥ `SPLIT_DATE`).

```bash
python test_model.py
```

This script:

1. Loads the trained model from `models/model.joblib`.
2. Loads processed features from `data/processed/AAPL_features.csv`.
3. Generates buy/sell signals in two variants:
   - **Without ADX filter** – uses only the model's predicted probability.
   - **With ADX filter** – additionally requires ADX > `ADX_THRESHOLD` to confirm a trend.
4. Simulates portfolio performance starting from `INITIAL_CAPITAL` USD.
5. Computes risk metrics (Sharpe ratio, max drawdown) and compares against Buy & Hold.
6. Saves charts to `outputs/charts/`.

Expected output (example):
```
============================================================
RESULTS SUMMARY
============================================================
Market Return: 42.35%
Strategy Return (without filter): 38.12%
Strategy Return (with ADX filter): 51.74%

Outperformance (without filter): -4.23%
Outperformance (with ADX filter): +9.39%
============================================================
Charts saved: outputs/charts/backtest.png, outputs/charts/portfolio_value.png
============================================================
```

---

## 6. Outputs

| File | Description |
|---|---|
| `data/raw/AAPL_data.csv` | Raw OHLCV price data |
| `data/processed/AAPL_features.csv` | Processed data with all technical indicators |
| `models/model.joblib` | Trained GradientBoosting model |
| `outputs/charts/feature_importance.png` | Top 20 most important features |
| `outputs/charts/backtest.png` | Strategy returns vs Buy & Hold |
| `outputs/charts/portfolio_value.png` | Portfolio value with ADX indicator overlay |

---

## 7. Troubleshooting

### `python` not found on macOS

Use `python3` instead of `python`, or add a shell alias:

```bash
alias python=python3
```

### Virtual environment activation fails on Windows (PowerShell)

Run once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then activate again:
```powershell
.venv\Scripts\Activate.ps1
```

### `ModuleNotFoundError` after activation

Make sure you installed the package inside the virtual environment:
```bash
pip install -e .
```

### `DataNotFoundError` when training

You must run the pipeline steps in order:
```
download_finanse_data.py  →  process_data.py  →  train_model.py  →  test_model.py
```

### `ModelNotFoundError` when running the backtest

The model file `models/model.joblib` does not exist yet. Run `train_model.py` first.

### Charts are not displayed

Charts are saved to `outputs/charts/` regardless of whether a graphical display is available. Open the `.png` files directly in your file manager.

### Running unit tests

```bash
# Run all tests
pytest tests/

# Verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src
```
