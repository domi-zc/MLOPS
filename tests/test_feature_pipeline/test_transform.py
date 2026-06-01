import pytest
import pandas as pd
import numpy as np

from ml_pipeline.feature_pipeline.transform import transform_data 


@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a fake 40-day dataset matching the Yahoo Finance & Coinmetrics data.
    """
    dates = pd.date_range(start="2020-01-01", periods=40)
    prices = np.linspace(100, 200, 40).tolist()
    
    prices[35] = 150.0 # 2020-02-05
    prices[36] = 145.0 # 2020-02-06
    prices[37] = 160.0 # 2020-02-07
    
    price_df = pd.DataFrame({
        "Date": dates,
        "price_usd": prices
    })
    
    address_df = pd.DataFrame({
        "date": dates,
        "n-unique-addresses": [1000] * 40
    })
    
    return price_df, address_df


def test_transform_data_creates_target_correctly(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests the fundamental machine learning target: 
    1 if tomorrow's price is strictly greater than today's, else 0.
    """
    raw_price, raw_address = dummy_data
    
    result_df = transform_data(raw_price, raw_address)
    
    target_down = result_df.loc[result_df['date'] == pd.Timestamp("2020-02-05"), 'target'].iloc[0]
    target_up = result_df.loc[result_df['date'] == pd.Timestamp("2020-02-06"), 'target'].iloc[0]

    assert target_down == 0
    assert target_up == 1


def test_transform_data_drops_nans(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests that rolling averages correctly force the dropping of early rows.
    """
    raw_price, raw_address = dummy_data
    
    result_df = transform_data(raw_price, raw_address)
    
    assert len(result_df) < len(raw_price), "Pipeline failed to drop NaN rows."
    assert result_df.isna().sum().sum() == 0, "There are still NaNs in the final feature dataset!"


def test_transform_data_calculates_moving_averages(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests that the math behind your feature engineering is actually correct.
    """
    raw_price, raw_address = dummy_data
    
    result_df = transform_data(raw_price, raw_address)
    
    assert "ma_deviation_7d" in result_df.columns, "Missing 7-day MA Deviation column"
    assert "volatility_7d" in result_df.columns, "Missing 7-day Volatility column"
    
    first_valid_row = result_df.iloc[0]
    assert pd.notna(first_valid_row["ma_deviation_7d"])
    assert pd.notna(first_valid_row["volatility_7d"])


def test_transform_data_math_accuracy(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests the exact mathematical output of the engineered features 
    to ensure our custom financial formulas are correct.
    """
    raw_price, raw_address = dummy_data
    
    result_df = transform_data(raw_price, raw_address)
    test_day = result_df[result_df["date"] == "2020-02-06"].iloc[0]
    
    # Return Math: (145.0 - 150.0) / 150.0 = -0.0333
    assert round(test_day["return"], 4) == -0.0333 
    
    # Address Math: (1000 - 1000) / 1000 = 0.0
    assert test_day["address_change_1d"] == 0.0


def test_transform_data_handles_infinity_from_zero_price(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests that if a price flash-crashes to 0.0, the resulting mathematical 
    'Infinity' calculations are caught and removed before breaking the ML model.
    """
    raw_price, raw_address = dummy_data
    raw_price.loc[20, "price_usd"] = 0.0 # Inject a 0.0 price to cause a divide-by-zero (infinity) on the next day's return
    
    result_df = transform_data(raw_price, raw_address)
    
    numeric_columns = result_df.select_dtypes(include=np.number)
    assert not np.isinf(numeric_columns.to_numpy()).any(), "Infinities leaked into the dataset!"


def test_transform_data_handles_missing_address_data(dummy_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """
    Tests that if the API fails to return on-chain data for a specific day, 
    the pipeline successfully forward-fills the missing data instead of dropping the row.
    """
    raw_price, raw_address = dummy_data

    clean_result = transform_data(raw_price.copy(), raw_address.copy())
    expected_row_count = len(clean_result)

    raw_address.loc[25, "n-unique-addresses"] = np.nan # Inject a missing value (NaN) into the middle of the address dataset
    
    dirty_result = transform_data(raw_price, raw_address)
    
    assert len(dirty_result) == expected_row_count, "API Outage caused a dropped row! ffill() failed."