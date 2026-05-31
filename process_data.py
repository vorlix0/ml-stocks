"""Backward-compatible script delegating to the unified CLI command."""

from cli import process


def main() -> None:
    """Run feature processing via the unified CLI implementation."""
    process()


if __name__ == "__main__":
    main()
