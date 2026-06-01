import pytest
import pandas as pd

from unittest.mock import patch, MagicMock
from ml_pipeline.feature_pipeline.extract import get_bitcoin_price_data, get_bitcoin_active_addresses


@patch("ml_pipeline.feature_pipeline.extract.yf.Ticker")
def test_get_bitcoin_price_data(mock_ticker: MagicMock) -> None:
    """
    Tests that Yahoo Finance extraction correctly requests and formats price data 
    without actually hitting the live internet API.
    """
    mock_history = MagicMock()
    mock_history.history.return_value = pd.DataFrame({
        "Open": [50000.0, 51000.0]
    }, index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"], name="Date"))
    
    mock_ticker.return_value = mock_history

    result_df = get_bitcoin_price_data(days=2)

    assert "Date" in result_df.columns
    assert "Open" in result_df.columns
    assert len(result_df) == 2


@patch("ml_pipeline.feature_pipeline.extract.CoinMetricsClient")
def test_get_bitcoin_active_addresses(mock_cm_client: MagicMock) -> None:
    """
    Tests that CoinMetrics extraction correctly requests and formats network data
    without hitting the live API.
    """
    mock_client_instance = MagicMock()
    mock_cm_client.return_value = mock_client_instance
    
    mock_metric_data = MagicMock()
    mock_metric_data.to_dataframe.return_value = pd.DataFrame({
        "time": ["2020-01-01", "2020-01-02"],
        "AdrActCnt": [1000, 1100]
    })
    
    mock_client_instance.get_asset_metrics.return_value = mock_metric_data

    result_df = get_bitcoin_active_addresses(days=2)

    assert "AdrActCnt" in result_df.columns
    assert len(result_df) == 2