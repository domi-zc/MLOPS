import pytest
import pandas as pd

from unittest.mock import patch
from ml_pipeline.inference_pipeline.data_fetcher import LiveDataFetcher

MOCK_READ_PARQUET = "ml_pipeline.inference_pipeline.data_fetcher.pd.read_parquet"
MOCK_STORAGE_OPTS = "ml_pipeline.inference_pipeline.data_fetcher.get_storage_options"
MOCK_FEATURE_COLS = "ml_pipeline.inference_pipeline.data_fetcher.FEATURE_COLS"


def test_get_todays_features_sorts_and_filters_correctly() -> None:
    """
    Tests that the fetcher correctly sorts the data by date, extracts only the 
    most recent single row, and filters out non-feature columns (like target/date).
    """
    mock_df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
        "feat_1": [30.0, 10.0, 20.0],
        "ignored_col": ["a", "b", "c"]
    })
    
    with patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, return_value=mock_df), \
         patch(MOCK_FEATURE_COLS, ["feat_1"]):
         
        fetcher = LiveDataFetcher()
        result_df = fetcher.get_todays_features()
        
        assert len(result_df) == 1
        assert result_df["feat_1"].iloc[0] == 30.0
        assert list(result_df.columns) == ["feat_1"]


def test_get_todays_features_raises_error_on_missing_file() -> None:
    """
    Tests that if the Feature Store is completely missing, the fetcher throws a 
    critical FileNotFoundError to alert the monitoring systems immediately.
    """
    with patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, side_effect=Exception("S3 bucket down")):
         
        fetcher = LiveDataFetcher()
        
        with pytest.raises(FileNotFoundError, match="CRITICAL: Could not read Feature Store"):
            fetcher.get_todays_features()