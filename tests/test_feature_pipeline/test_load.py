import pytest
import pandas as pd

from unittest.mock import patch
from ml_pipeline.feature_pipeline.load import save_to_feature_store

MOCK_FEATURE_PATH = "ml_pipeline.feature_pipeline.load.FEATURE_PATH"
MOCK_STORAGE_OPTS = "ml_pipeline.feature_pipeline.load.get_storage_options"


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """
    Provides a tiny dummy feature DataFrame for testing the load process.
    """
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "price_usd": [100.0, 110.0],
        "target": [1, 0]
    })


def test_save_to_feature_store_creates_new(sample_feature_df: pd.DataFrame, tmp_path) -> None:
    """
    Tests that if no feature store exists, a new one is created safely in the temporary path.
    """
    fake_path = str(tmp_path / "fake_feature_store.parquet")
    
    with patch(MOCK_FEATURE_PATH, fake_path), patch(MOCK_STORAGE_OPTS, return_value={}):
        
        save_to_feature_store(sample_feature_df)
        
        saved_df = pd.read_parquet(fake_path)
        assert len(saved_df) == 2


def test_save_to_feature_store_merges_and_deduplicates(sample_feature_df: pd.DataFrame, tmp_path) -> None:
    """
    Tests that overlapping data is merged and duplicates are dropped, keeping the newest data.
    """
    fake_path = str(tmp_path / "fake_feature_store.parquet")
    
    existing_df = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2020-01-01"]),
        "price_usd": [90.0, 95.0], # The 95.0 should be overwritten by the incoming 100.0
        "target": [0, 0]
    })
    existing_df.to_parquet(fake_path)
    
    with patch(MOCK_FEATURE_PATH, fake_path), patch(MOCK_STORAGE_OPTS, return_value={}):
        
        save_to_feature_store(sample_feature_df)
        saved_df = pd.read_parquet(fake_path)
        
        assert len(saved_df) == 3 
        
        updated_price = saved_df.loc[saved_df["date"] == pd.Timestamp("2020-01-01"), "price_usd"].iloc[0]
        assert updated_price == 100.0