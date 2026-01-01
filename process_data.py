# process_data.py
"""Script for data processing and feature creation."""

import sys
import pandas as pd

from config import DATA_CONFIG
from src.data import FeatureEngineer
from src.exceptions import ForexMLError, DataNotFoundError
from src.utils import load_csv_safe, validate_ohlcv_data


def main():
    """Main function for data processing."""
    input_file = DATA_CONFIG.data_file
    output_file = DATA_CONFIG.features_file
    
    try:
        print(f"Loading data from: {input_file}")
        data = load_csv_safe(
            input_file,
            required_columns=['Open', 'High', 'Low', 'Close', 'Volume']
        )
        validate_ohlcv_data(data)
        
        print("DF showcase:")
        print(data.head())
        
        # Create features
        feature_engineer = FeatureEngineer(data)
        data_features = feature_engineer.create_all_features()
        
        print(data_features.head())
        print(f"Features: {data_features.shape[1]}")
        
        data_features.to_csv(output_file)
        print(f"Features saved to: {output_file}")
        
        return data_features
    
    except DataNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run first: python download_finanse_data.py")
        sys.exit(1)
    except ForexMLError as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
