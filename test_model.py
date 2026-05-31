"""Backward-compatible script delegating to the unified CLI command."""

from cli import backtest


def main() -> None:
    """Run backtesting via the unified CLI implementation."""
    backtest()


if __name__ == "__main__":
    main()
