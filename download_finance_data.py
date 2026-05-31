"""Backward-compatible script delegating to the unified CLI command."""

from cli import download


def main() -> None:
    """Run data download via the unified CLI implementation."""
    download()


if __name__ == "__main__":
    main()
