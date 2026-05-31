"""Backward-compatible script delegating to the unified CLI command."""

from cli import tune


def main() -> None:
    """Run hyperparameter tuning via the unified CLI implementation."""
    tune()


if __name__ == "__main__":
    main()
