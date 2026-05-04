import numpy as np
import pandas as pd

def _normalize_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw Yahoo Finance Dates to standard UTC dates.
    """
    price_df = price_df.rename(columns={"Close": "price_usd"})

    price_df["date"] = pd.to_datetime(price_df["Date"])
    price_df["date"] = price_df["date"].dt.tz_localize(None).dt.normalize()
    
    return price_df.drop(columns=["Date"])


def _normalize_address_data(addresses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes raw CoinMetrics data to standard UTC dates.
    """
    addresses_df = addresses_df.drop(columns=["asset"], errors="ignore")
    addresses_df = addresses_df.rename(columns={
        "time": "date", 
        "AdrActCnt": "n-unique-addresses"
    })

    addresses_df["date"] = pd.to_datetime(addresses_df["date"]).dt.tz_localize(None).dt.normalize()
    addresses_df["n-unique-addresses"] = pd.to_numeric(addresses_df["n-unique-addresses"])
    
    return addresses_df[["date", "n-unique-addresses"]]


def _merge_and_handle_lag(price_df: pd.DataFrame, addresses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges datasets and forward-fills any intermittent missing on-chain data.
    """
    df = pd.merge(price_df, addresses_df, on="date", how="left")
    df = df.sort_values(by="date").reset_index(drop=True)
    
    df["n-unique-addresses"] = df["n-unique-addresses"].ffill()
    
    return df


def _engineer_quantitative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates financial metrics including returns, deviations, volatility, and momentum.
    """
    # Calculate daily percentage returns (% gain or drop).
    df["return"] = df["price_usd"].pct_change()

    # Calculate Moving Average Deviations.
    df["ma_deviation_7d"] = ( df["price_usd"] / df["price_usd"].rolling(window=7).mean() ) - 1
    df["ma_deviation_14d"] = ( df["price_usd"] / df["price_usd"].rolling(window=14).mean() ) - 1

    # Calculate rolling volatility based on return.
    df["volatility_7d"] = df["return"].rolling(window=7).std()

    # Calculate absolute momentum over 7-day period.
    df["momentum_7d"] = df["price_usd"].pct_change(periods=7)

    # Calculate smoothed network adoption to remove weekend seasonality.
    df["addresses_7d_ma"] = df["n-unique-addresses"].rolling(window=7).mean()
    
    return df


def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the binary classification target predicting tomorrow's price direction.
    """
    # Shift prices backwards to align tomorrow's price with today's features.
    df["future_price"] = df["price_usd"].shift(-1)
    
    # 1 if price is rising, 0 if price is falling.
    df["target"] = (df["future_price"] > df["price_usd"]).astype(int)
    
    # Drop the raw future price to prevent data leakage during model training.
    df = df.drop(columns=["future_price"])
    
    return df


def _clean_and_format_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes invalid data points and enforces the final Feature Store column order.
    """
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna().reset_index(drop=True)

    final_columns = [
        "date", 
        "price_usd", 
        "return",
        "ma_deviation_7d",
        "ma_deviation_14d",
        "volatility_7d",
        "momentum_7d",
        "n-unique-addresses", 
        "addresses_7d_ma", 
        "target"
    ]
    
    return df[final_columns]


def transform_data(price_df: pd.DataFrame, addresses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main orchestrator function for the transformation layer.
    Transforms raw extraction data into clean, ML-ready features.
    """
    print("Starting data transformation pipeline...")
    
    price_df = _normalize_price_data(price_df)
    addresses_df = _normalize_address_data(addresses_df)
    
    df = _merge_and_handle_lag(price_df, addresses_df)
    df = _engineer_quantitative_features(df)
    df = _create_target_variable(df)
    df = _clean_and_format_schema(df)
    
    print(f"Transformation complete. Generated {len(df)} ready-to-train rows.")
    return df