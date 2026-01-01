"""
Logger configuration module for forex-ml-hf project.
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "forex_ml",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Logger format
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "forex_ml") -> logging.Logger:
    """
    Gets existing logger or creates new one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, configure it
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


# Domyślny logger dla projektu
logger = setup_logger()
