"""
Module for downloading financial data.
"""
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_CONFIG
from src.exceptions import DownloadError, EmptyDataError

logger = logging.getLogger("forex_ml.data.downloader")


class DataDownloader:
    """Class for downloading financial data from yfinance."""
    
    def __init__(self, ticker: str = None, start_date: str = None, end_date: str = None):
        """
        Initializes the downloader.
        
        Args:
            ticker: Ticker symbol (default from config)
            start_date: Start date (default from config)
            end_date: End date (default from config)
        """
        self.ticker = ticker or DATA_CONFIG.TICKER
        self.start_date = start_date or DATA_CONFIG.START_DATE
        self.end_date = end_date or DATA_CONFIG.END_DATE
    
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
            )
        
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
            IOError: When file cannot be saved
        """
        data = self.download()
        
        output_path = output_path or DATA_CONFIG.data_file
        
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path)
        except Exception as e:
            raise IOError(f"Cannot save data to {output_path}: {e}")
        
        logger.info(f"Data saved to: {output_path}")
        
        return data
