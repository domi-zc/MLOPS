import time
import requests
import pandas as pd
import yfinance as yf


def _fetch_with_retry(url: str, headers: dict = None, params: dict = None, max_retries: int = 3) -> dict:
    """
    A robust fetcher that implements Exponential Backoff for 429 Rate Limit errors.
    """
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
            
        elif response.status_code == 429:
            wait_time = int(response.headers.get('Retry-After', 2 ** attempt))
            print(f"Rate limited. Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            
        else:
            response.raise_for_status()
            
    raise Exception(f"Failed to fetch data from {url} after {max_retries} attempts.")


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


def get_blockchain_metric(metric_name: str, days: int = 30) -> pd.DataFrame:
    """
    Fetches on-chain metrics from the Blockchain.com API.
    """
    url = f"https://api.blockchain.info/charts/{metric_name}"
    params = {
        "timespan": f"{days}days",
        "format": "json",
        "sampled": "false" 
    }
    
    data = _fetch_with_retry(url, params=params)
    df = pd.DataFrame(data["values"]) 
    
    print(f"Metric '{metric_name}' of the last {days} days successfully downloaded.")
    return df