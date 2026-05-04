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
    
    # Ensure the directory exists (it will create a 'data/' folder if it doesn't exist)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. If we already have a saved Feature Store, we want to merge the new data
    if path.exists():
        print(f"Found existing feature store at '{file_path}'. Merging new data...")
        
        # Load the old data
        existing_df = pd.read_parquet(path)
        
        # Combine old and new data
        combined_df = pd.concat([existing_df, df])
        
        # UPSERT LOGIC: If we pulled data for Jan 1st yesterday, and pulled it again today,
        # we have a duplicate row. We drop the duplicate, keeping the absolute newest version.
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        
        # Sort it perfectly by date so the timeline remains intact
        combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
        
        print(f"Merged successfully. Feature store now contains {len(combined_df)} total rows.")
        
    # 2. If this is our very first time running the pipeline, just save the file
    else:
        print(f"No existing feature store found. Creating a new one at '{file_path}'...")
        combined_df = df.sort_values(by="date").reset_index(drop=True)
        print(f"Created new feature store with {len(combined_df)} rows.")

    # 3. Save the final merged dataset back to the hard drive as a Parquet file
    combined_df.to_parquet(path, index=False)
    print("Data safely written to disk.")


if __name__ == "__main__":
    # Import from your sibling files to test the FULL pipeline end-to-end
    
    print("=== Commencing End-to-End Feature Pipeline Test ===")
    
    try:
        # 1. EXTRACT (Bronze)
        print("\n--- Running Extract ---")
        raw_price_df = get_bitcoin_price_data(days=365)
        raw_address_df = get_blockchain_metric("n-unique-addresses", days=365)
        
        # 2. TRANSFORM (Silver/Gold)
        print("\n--- Running Transform ---")
        final_feature_df = transform_data(raw_price_df, raw_address_df)
        
        # 3. LOAD (Feature Store)
        print("\n--- Running Load ---")
        # We save it into a "data" folder at the root of your project
        save_to_feature_store(final_feature_df)
        
        print("\n=== Pipeline Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Pipeline failed: {e}")