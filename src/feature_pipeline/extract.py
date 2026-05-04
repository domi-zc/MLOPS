import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from pycoingecko import CoinGeckoAPI

# Load variables from .env
load_dotenv()

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
    Fetches daily Bitcoin price data using the PyCoinGecko wrapper.
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: COINGECKO_API_KEY is missing from the environment variables.")
    
    cg = CoinGeckoAPI(demo_api_key=api_key)
    
    data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price_usd"])
    
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


if __name__ == "__main__":    
    try:
        price_df = get_bitcoin_price_data(days=5)
        print(price_df.head())
        
        addresses_df = get_blockchain_metric("n-unique-addresses", days=5)
        print(addresses_df.head())
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")