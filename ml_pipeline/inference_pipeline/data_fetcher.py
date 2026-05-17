import pandas as pd

from ml_pipeline.config.storage_data import FEATURE_PATH, get_storage_options
from ml_pipeline.config.model_data import FEATURE_COLS

class LiveDataFetcher:
    def __init__(self):
        self.days_to_fetch = 30

    def get_todays_features(self) -> pd.DataFrame:
        """
        Pulls data using the official extract pipeline, transforms it, 
        and returns ONLY today's row formatted for XGBoost.
        """
        print(f"Fetching latest features directly from Feature Store ({FEATURE_PATH})...")

        options = get_storage_options()

        try:
            df = pd.read_parquet(FEATURE_PATH, storage_options=options)
        except Exception as e:
            raise FileNotFoundError(f"CRITICAL: Could not read Feature Store for inference. Error: {e}")
        
        df = df.sort_values("date").reset_index(drop=True)
        todays_data = df.iloc[[-1]].copy()
        
        todays_features = todays_data[FEATURE_COLS]
        print("Successfully loaded today's pre-computed features!")

        return todays_features