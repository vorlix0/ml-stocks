"""
Module for downloading financial data.
"""
import logging
import re
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_CONFIG
from src.exceptions import DownloadError, EmptyDataError, InvalidDataError

logger = logging.getLogger("forex_ml.data.downloader")

# Allowlist pattern for ticker symbols (e.g. AAPL, BRK.B, ^GSPC, BTC-USD)
_TICKER_PATTERN = re.compile(r'^[A-Z0-9.\-\^]{1,10}$')


class DataDownloader:
    """Class for downloading financial data from yfinance."""

    def __init__(self, ticker: str = None, start_date: str = None, end_date: str = None):
        """
        Initializes the downloader.

        Args:
            ticker: Ticker symbol (default from config)
            start_date: Start date (default from config)
            end_date: End date (default from config)

        Raises:
            InvalidDataError: When ticker symbol has an invalid format
        """
        raw_ticker = ticker or DATA_CONFIG.TICKER
        self._validate_ticker(raw_ticker)
        self.ticker = raw_ticker.upper()
        self.start_date = start_date or DATA_CONFIG.START_DATE
        self.end_date = end_date or DATA_CONFIG.END_DATE

    @staticmethod
    def _validate_ticker(ticker: str) -> None:
        """Validates ticker symbol against an allowlist pattern.

        Args:
            ticker: Ticker symbol to validate

        Raises:
            InvalidDataError: When ticker format is invalid
        """
        if not _TICKER_PATTERN.match(ticker.upper()):
            raise InvalidDataError(
                f"Invalid ticker symbol: '{ticker}'. "
                "Ticker must be 1-10 characters and contain only "
                "uppercase letters, digits, '.', '-', or '^'."
            )

    def download(self) -> pd.DataFrame:
        """
        Downloads OHLCV data for configured ticker.

        Returns:
            DataFrame with OHLCV data

        Raises:
            DownloadError: When download fails
            EmptyDataError: When no data was downloaded
        """
        logger.info(f"Downloading data for: {self.ticker}")
        logger.info(f"Range: {self.start_date} - {self.end_date}")

        try:
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                multi_level_index=False
            )
        except Exception as e:
            raise DownloadError(
                f"Error downloading data for {self.ticker}: {e}"
            ) from e

        if data.empty:
            raise EmptyDataError(
                f"No data downloaded for {self.ticker} "
                f"in range {self.start_date} - {self.end_date}"
            )

        logger.info(f"Downloaded {len(data)} records")
        return data

    def download_and_save(self, output_path: str = None) -> pd.DataFrame:
        """
        Downloads data and saves to CSV file.

        Args:
            output_path: Output file path (default from config)

        Returns:
            DataFrame with OHLCV data

        Raises:
            DownloadError: When download fails
            EmptyDataError: When no data was downloaded
            OSError: When file cannot be saved
        """
        data = self.download()

        output_path = output_path or DATA_CONFIG.data_file

        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path)
        except Exception as e:
            raise OSError(f"Cannot save data to {output_path}: {e}") from e

        logger.info(f"Data saved to: {output_path}")

        return data
