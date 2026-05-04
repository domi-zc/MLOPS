import pandas as pd
import yfinance as yf

from coinmetrics.api_client import CoinMetricsClient
from datetime import datetime, timedelta, timezone


def get_bitcoin_price_data(days: int = 30) -> pd.DataFrame:
    """
    Fetches daily Bitcoin price data using Yahoo Finance.
    """
    btc = yf.Ticker("BTC-USD")

    hist_df = btc.history(period=f"{days}d")
    hist_df = hist_df.reset_index()
    
    df = hist_df[["Date", "Close"]]
    
    print(f"Bitcoin price data of the last {days} days successfully downloaded.")
    return df


def get_blockchain_metric(metric_name: str = "AdrActCnt", days: int = 30) -> pd.DataFrame:
    """
    Fetches daily on-chain metrics using the official CoinMetricsClient.
    """
    client = CoinMetricsClient()
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%d')
    
    metric_data = client.get_asset_metrics(
        assets='btc',
        metrics=metric_name,
        start_time=start_date
    )
    
    df = metric_data.to_dataframe()
    
    print(f"Metric '{metric_name}' successfully downloaded. Total rows: {len(df)}.")
    return df