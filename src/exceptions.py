"""
Module with custom project exceptions.
"""


class ForexMLError(Exception):
    """Base exception for forex-ml-hf project."""
    pass


class DataNotFoundError(ForexMLError):
    """Exception raised when data file doesn't exist."""
    pass


class EmptyDataError(ForexMLError):
    """Exception raised when data is empty."""
    pass


class InvalidDataError(ForexMLError):
    """Exception raised when data is invalid."""
    pass


class ModelNotFoundError(ForexMLError):
    """Exception raised when model file doesn't exist."""
    pass


class ModelNotTrainedError(ForexMLError):
    """Exception raised when model hasn't been trained."""
    pass


class DownloadError(ForexMLError):
    """Wyjątek rzucany gdy pobieranie danych się nie powiedzie."""
    pass
