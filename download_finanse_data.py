# download_finanse_data.py
"""Script for downloading financial data."""

from src.data import DataDownloader


def main():
    """Main function for downloading data."""
    downloader = DataDownloader()
    downloader.download_and_save()


if __name__ == "__main__":
    main()
