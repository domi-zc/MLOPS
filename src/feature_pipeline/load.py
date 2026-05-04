import pandas as pd

from pathlib import Path
from src.feature_pipeline.extract import get_bitcoin_price_data, get_blockchain_metric
from src.feature_pipeline.transform import transform_data

def save_to_feature_store(df: pd.DataFrame, file_path: str = "data/feature_store.parquet") -> None:
    """
    Saves the transformed DataFrame to a local Parquet file.
    Acts as a local Feature Store by appending new data and preventing duplicates.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.exists():
        print(f"Found existing feature store at '{file_path}'. Merging new data...")
        existing_df = pd.read_parquet(path)
        
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
        
        print(f"Merged successfully. Feature store now contains {len(combined_df)} total rows.")
        
    else:
        print(f"No existing feature store found. Creating a new one at '{file_path}'...")
        combined_df = df.sort_values(by="date").reset_index(drop=True)
        print(f"Created new feature store with {len(combined_df)} rows.")

    combined_df.to_parquet(path, index=False)
    print("Data safely written to disk.")


if __name__ == "__main__":    
    try:
        print("\n--- Running Extract ---")
        raw_price_df = get_bitcoin_price_data(days=3650)
        raw_address_df = get_blockchain_metric("n-unique-addresses", days=3650)
        
        print("\n--- Running Transform ---")
        final_feature_df = transform_data(raw_price_df, raw_address_df)
        
        print("\n--- Running Load ---")
        save_to_feature_store(final_feature_df)
        
        print("\n=== Pipeline Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Pipeline failed: {e}")