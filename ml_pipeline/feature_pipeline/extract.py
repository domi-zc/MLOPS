import pandas as pd
import yfinance as yf

from coinmetrics.api_client import CoinMetricsClient
from datetime import datetime, timedelta, timezone

def get_bitcoin_price_data(start_date: str | None = None, days: int | None = 30) -> pd.DataFrame:
    """
    Fetches daily Bitcoin price data using Yahoo Finance.
    """
    btc = yf.Ticker("BTC-USD")

    if days is not None:
        safe_days = days
    else:
        safe_days = 30

    if start_date:
        hist_df = btc.history(start=start_date)
        range_msg = f"since {start_date}"
    else:
        hist_df = btc.history(period=f"{safe_days}d")
        range_msg = f"of the last {safe_days} days"
        
    hist_df = hist_df.reset_index()

    df = hist_df[["Date", "Open"]]
    
    print(f"Bitcoin price data {range_msg} successfully downloaded.")
    return df

def get_bitcoin_active_addresses(start_date: str | None = None, days: int | None = 30) -> pd.DataFrame:
    """
    Fetches daily on-chain metrics using the official CoinMetricsClient.
    """
    client = CoinMetricsClient()
    
    if not start_date:
        if days is not None:
            safe_days = days
        else:
            safe_days = 30

        start_date = (datetime.now(timezone.utc) - timedelta(days=safe_days)).strftime('%Y-%m-%d')
    
    metric_data = client.get_asset_metrics(
        assets='btc',
        metrics='AdrActCnt',
        start_time=start_date
    )
    
    df = metric_data.to_dataframe()
    
    print(f"Metric 'AdrActCnt' successfully downloaded. Total rows: {len(df)}.")
    return df