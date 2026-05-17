import pandas as pd

from pathlib import Path
from ml_pipeline.feature_pipeline.extract import get_bitcoin_price_data, get_bitcoin_active_addresses
from ml_pipeline.feature_pipeline.transform import transform_data
from ml_pipeline.config.storage_data import FEATURE_PATH, get_storage_options

def save_to_feature_store(df: pd.DataFrame) -> None:
    """
    Saves the transformed DataFrame to the configured Feature Store (Local or Cloud).
    Acts as a Feature Store by appending new data and preventing duplicates.
    """
    options = get_storage_options()
    
    try:
        print(f"Found existing feature store at '{FEATURE_PATH}'. Merging new data...")
        existing_df = pd.read_parquet(FEATURE_PATH, storage_options=options)
        
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
        
        print(f"Merged successfully. Feature store now contains {len(combined_df)} total rows.")
        
    except Exception:
        print(f"No existing feature store found. Creating a new one at '{FEATURE_PATH}'...")
        combined_df = df.sort_values(by="date").reset_index(drop=True)
        print(f"Created new feature store with {len(combined_df)} rows.")

    combined_df.to_parquet(FEATURE_PATH, index=False, storage_options=options)
    print("Data safely written to storage.")


if __name__ == "__main__":    
    try:
        print("\n--- Running Extract ---")
        raw_price_df = get_bitcoin_price_data(days=3650)
        raw_address_df = get_bitcoin_active_addresses(days=3650)
        
        print("\n--- Running Transform ---")
        final_feature_df = transform_data(raw_price_df, raw_address_df)
        
        print("\n--- Running Load ---")
        save_to_feature_store(final_feature_df)
        
        print("\n=== Pipeline Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Pipeline failed: {e}")