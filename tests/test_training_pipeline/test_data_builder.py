import pytest
import pandas as pd

from unittest.mock import patch
from ml_pipeline.training_pipeline.data_builder import TrainingDataBuilder

MOCK_READ_PARQUET = "ml_pipeline.training_pipeline.data_builder.pd.read_parquet"
MOCK_STORAGE_OPTS = "ml_pipeline.training_pipeline.data_builder.get_storage_options"
MOCK_FEATURE_COLS = "ml_pipeline.training_pipeline.data_builder.FEATURE_COLS"
MOCK_TARGET_COL = "ml_pipeline.training_pipeline.data_builder.TARGET_COL"


@pytest.fixture
def mock_dataset() -> pd.DataFrame:
    """
    Creates a 10-row dataset. Row 5 contains a NaN to test dropping logic.
    """
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10),
        "feat_1": [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat_2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    return df


@patch(MOCK_FEATURE_COLS, ["feat_1", "feat_2"])
@patch(MOCK_TARGET_COL, "target")
@patch(MOCK_STORAGE_OPTS, return_value={})
def test_load_and_split_handles_nans_and_splits_correctly(mock_storage, mock_dataset: pd.DataFrame) -> None:
    """
    Tests that the builder correctly drops NaNs and strictly splits data 
    chronologically based on the test_size percentage.
    """
    with patch(MOCK_READ_PARQUET, return_value=mock_dataset):
        builder = TrainingDataBuilder(test_size=0.2)
        X_train_val, y_train_val, X_test, y_test = builder.load_and_split()
        
        # 10 rows total (1 has "None" and is dropped) -> 9 clean rows
        # 80% train (7 rows) and 20% test (2 rows)
        assert len(X_train_val) == 7
        assert len(X_test) == 2
        
        assert X_train_val["feat_1"].iloc[-1] < X_test["feat_1"].iloc[0]


def test_get_all_data() -> None:
    """
    Tests that the final champion retraining data successfully glues 
    the historical and future datasets back together.
    """
    X_train = pd.DataFrame({"feat_1": [1.0, 2.0]})
    y_train = pd.Series([0, 1])
    
    X_test = pd.DataFrame({"feat_1": [3.0]})
    y_test = pd.Series([0])
    
    builder = TrainingDataBuilder()
    X_all, y_all = builder.get_all_data(X_train, y_train, X_test, y_test)
    
    assert len(X_all) == 3
    assert len(y_all) == 3
    assert X_all["feat_1"].iloc[-1] == 3.0