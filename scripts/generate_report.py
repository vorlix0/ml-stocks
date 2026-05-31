"""
Generate an HTML report from the trained model and backtest results.

Reads the saved model and features CSV, runs the backtest computations,
and writes docs/index.html with embedded charts (base64) and a metrics table.

Usage::

    python scripts/generate_report.py
"""
from __future__ import annotations

import base64
import datetime
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so imports work whether this
# script is executed from the repo root or from the scripts/ subfolder.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config.py resolves paths relative to cwd

from config import BACKTEST_CONFIG, DATA_CONFIG, MODEL_CONFIG  # noqa: E402
from src.backtest import RiskMetrics, TradingSimulator, TradingStrategy  # noqa: E402
from src.model import ModelTrainer  # noqa: E402
from src.utils import load_csv_safe, validate_features_data  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_b64(path: Path) -> str:
    """Return a data-URI string for a PNG image."""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode()
    return f"data:image/png;base64,{data}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt_float(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run_backtest() -> dict:
    """Run backtest and return a dict with all metrics and series."""
    model = ModelTrainer.load_model()
    df = load_csv_safe(DATA_CONFIG.features_file)
    validate_features_data(df)

    split_date = MODEL_CONFIG.SPLIT_DATE
    feature_cols = [col for col in df.columns if col not in MODEL_CONFIG.EXCLUDED_COLUMNS]
    df_test = df[df.index >= split_date].copy()
    x_test = df_test[feature_cols]

    y_pred_proba = model.predict_proba(x_test)[:, 1]
    strategy = TradingStrategy(df_test, y_pred_proba)

    signals_no_filter = strategy.generate_signals(use_adx_filter=False)
    signals_adx = strategy.generate_signals(use_adx_filter=True)

    returns_no_filter = strategy.calculate_strategy_returns(signals_no_filter)
    returns_adx = strategy.calculate_strategy_returns(signals_adx)

    cum_market = strategy.get_market_returns()
    cum_no_filter = strategy.calculate_cumulative_returns(returns_no_filter)
    cum_adx = strategy.calculate_cumulative_returns(returns_adx)

    simulator = TradingSimulator(df_test)
    results_market = simulator.simulate_buy_and_hold()
    results_no_filter = simulator.simulate(signals_no_filter)
    results_adx = simulator.simulate(signals_adx)

    metrics_market = RiskMetrics(df_test["Returns"], cum_market)
    metrics_no_filter = RiskMetrics(returns_no_filter, cum_no_filter)
    metrics_adx = RiskMetrics(returns_adx, cum_adx)

    return {
        "results": {
            "Market (Buy & Hold)": results_market,
            "ML Strategy (no filter)": results_no_filter,
            "ML Strategy (ADX filter)": results_adx,
        },
        "risk": {
            "Market (Buy & Hold)": metrics_market.get_all_metrics(),
            "ML Strategy (no filter)": metrics_no_filter.get_all_metrics(),
            "ML Strategy (ADX filter)": metrics_adx.get_all_metrics(),
        },
        "cum": {
            "Market (Buy & Hold)": cum_market,
            "ML Strategy (no filter)": cum_no_filter,
            "ML Strategy (ADX filter)": cum_adx,
        },
        "test_start": str(df_test.index[0].date()),
        "test_end": str(df_test.index[-1].date()),
        "ticker": DATA_CONFIG.TICKER,
    }


def build_html(data: dict) -> str:
    """Build and return the full HTML report string."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sha = _git_sha()

    strategy_names = list(data["results"].keys())

    # --- metrics table rows ---
    rows = ""
    for name in strategy_names:
        r = data["results"][name]
        m = data["risk"][name]
        rows += f"""
        <tr>
          <td>{name}</td>
          <td class="num">{_fmt_pct(m['total_return'])}</td>
          <td class="num">{_fmt_float(m['sharpe_ratio'])}</td>
          <td class="num">{_fmt_pct(m['max_drawdown'])}</td>
          <td class="num">{_fmt_pct(m['volatility'])}</td>
          <td class="num">${r['final_value']:,.0f}</td>
          <td class="num">${r['profit']:,.0f}</td>
        </tr>"""

    # --- charts ---
    charts_html = ""
    chart_files = [
        (BACKTEST_CONFIG.backtest_plot_file, "Cumulative Returns – Backtest"),
        (BACKTEST_CONFIG.portfolio_plot_file, "Portfolio Value with ADX"),
        (BACKTEST_CONFIG.feature_importance_plot_file, "Feature Importance"),
    ]
    for path, title in chart_files:
        if Path(path).exists():
            charts_html += f"""
      <figure>
        <figcaption>{title}</figcaption>
        <img src="{_img_b64(path)}" alt="{title}" />
      </figure>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ML Stocks – Model Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1100px; margin: 0 auto; padding: 2rem 1rem;
      background: #f8f9fa; color: #212529;
    }}
    h1 {{ color: #0d6efd; }}
    .meta {{ color: #6c757d; font-size: .9rem; margin-bottom: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; }}
    th, td {{ padding: .6rem 1rem; border: 1px solid #dee2e6; text-align: left; }}
    th {{ background: #0d6efd; color: #fff; }}
    tr:nth-child(even) {{ background: #e9ecef; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    figure {{ margin: 2rem 0; }}
    figcaption {{ font-weight: 600; margin-bottom: .5rem; }}
    img {{ max-width: 100%; border: 1px solid #dee2e6; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>📈 ML Stocks – Model Report</h1>
  <p class="meta">
    Ticker: <strong>{data['ticker']}</strong> &nbsp;|&nbsp;
    Test period: <strong>{data['test_start']}</strong> → <strong>{data['test_end']}</strong>
    &nbsp;|&nbsp; Generated: <strong>{now}</strong>
    &nbsp;|&nbsp; Commit: <code>{sha}</code>
  </p>

  <h2>Backtest Results</h2>
  <table>
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Total Return</th>
        <th>Sharpe Ratio</th>
        <th>Max Drawdown</th>
        <th>Volatility</th>
        <th>Final Value</th>
        <th>Profit</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
  <p class="meta">Initial capital: $10,000 &nbsp;|&nbsp; Risk-free rate: 0%</p>

  <h2>Charts</h2>
  {charts_html}
</body>
</html>
"""


def main() -> None:
    print("📊 Running backtest …")
    data = run_backtest()

    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "index.html"

    print("🖊  Generating HTML report …")
    html = build_html(data)
    out_file.write_text(html, encoding="utf-8")

    print(f"✅ Report written to {out_file}")


if __name__ == "__main__":
    main()
