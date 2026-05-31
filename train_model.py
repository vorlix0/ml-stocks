"""Backward-compatible script delegating to the unified CLI command."""

from cli import train


def main() -> None:
    """Run model training via the unified CLI implementation."""
    train()


if __name__ == "__main__":
    main()
