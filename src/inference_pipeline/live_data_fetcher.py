import pandas as pd

from src.feature_pipeline.extract import get_bitcoin_price_data, get_bitcoin_active_addresses
from src.feature_pipeline.transform import transform_data
from src.training_pipeline.metadata import FEATURE_COLS

class LiveDataFetcher:
    def __init__(self):
        self.days_to_fetch = 30

    def get_todays_features(self) -> pd.DataFrame:
        """
        Pulls data using the official extract pipeline, transforms it, 
        and returns ONLY today's row formatted for XGBoost.
        """
        print("Fetching live data from APIs...")
        
        raw_price_df = get_bitcoin_price_data(days=self.days_to_fetch)
        raw_address_df = get_bitcoin_active_addresses(days=self.days_to_fetch)
        
        print("Calculating feature engineering runway...")
        df = transform_data(raw_price_df, raw_address_df, is_live=True)
        
        todays_data = df.iloc[[-1]].copy()
        todays_features = todays_data[FEATURE_COLS]
        
        print(f"Features successfully generated for today!")

        return todays_features