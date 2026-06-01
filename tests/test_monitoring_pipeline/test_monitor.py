import pytest
import pandas as pd
import numpy as np

from unittest.mock import patch, MagicMock, mock_open
from ml_pipeline.monitoring_pipeline.monitor import grade_recent_performance, lock_in_tomorrows_prediction

MOCK_MODEL_FETCHER = "ml_pipeline.monitoring_pipeline.monitor.ModelFetcher"
MOCK_LIVE_FETCHER = "ml_pipeline.monitoring_pipeline.monitor.LiveDataFetcher"
MOCK_READ_PARQUET = "ml_pipeline.monitoring_pipeline.monitor.pd.read_parquet"
MOCK_PRED_PATH = "ml_pipeline.monitoring_pipeline.monitor.get_prediction_path"
MOCK_STORAGE_OPTS = "ml_pipeline.monitoring_pipeline.monitor.get_storage_options"
MOCK_OPEN = "builtins.open"


def test_grade_recent_performance_no_drift() -> None:
    """
    Tests that when model precision is healthy and has sufficient trade data, 
    no data drift is signaled.
    """
    mock_mf_instance = MagicMock()
    mock_mf_instance.get_champion_model.return_value = ("model", 0.5, {"version": "v2"})
    
    truth_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "target": [1, 1, 1, 1]
    })
    
    guesses_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "prediction": [1, 1, 1, 1]
    })
    
    def smart_read(path, **kwargs):
        if "fake_pred" in str(path):
            return guesses_df
        return truth_df

    with patch(MOCK_MODEL_FETCHER, return_value=mock_mf_instance), \
         patch(MOCK_PRED_PATH, return_value="fake_pred.parquet"), \
         patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, side_effect=smart_read), \
         patch.dict("os.environ", {"GITHUB_OUTPUT": "fake_env.txt"}), \
         patch(MOCK_OPEN, mock_open()) as mocked_file:
         
        grade_recent_performance()
        
        mocked_file().write.assert_called_once_with("drift_detected=false\n")


def test_grade_recent_performance_drift_detected() -> None:
    """
    Tests that when model precision falls below the acceptable threshold, 
    drift is successfully flagged and an alert environment variable is exported.
    """
    mock_mf_instance = MagicMock()
    mock_mf_instance.get_champion_model.return_value = ("model", 0.5, {"version": "v2"})
    
    truth_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "target": [0, 0, 0, 0]
    })
    
    guesses_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "prediction": [1, 1, 1, 1]
    })
    
    def smart_read(path, **kwargs):
        if "fake_pred" in str(path):
            return guesses_df
        return truth_df

    with patch(MOCK_MODEL_FETCHER, return_value=mock_mf_instance), \
         patch(MOCK_PRED_PATH, return_value="fake_pred.parquet"), \
         patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, side_effect=smart_read), \
         patch.dict("os.environ", {"GITHUB_OUTPUT": "fake_env.txt"}), \
         patch(MOCK_OPEN, mock_open()) as mocked_file:
         
        grade_recent_performance()
        
        mocked_file().write.assert_called_once_with("drift_detected=true\n")


def test_grade_recent_performance_insufficient_data() -> None:
    """
    Tests that if the model executed fewer than 3 trades within the window, 
    the function exits gracefully without evaluating drift metrics.
    """
    mock_mf_instance = MagicMock()
    mock_mf_instance.get_champion_model.return_value = ("model", 0.5, {"version": "v2"})
    
    truth_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "target": [1, 1, 1, 1]
    })
    
    guesses_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=4),
        "prediction": [1, 1, 0, 0]
    })
    
    def smart_read(path, **kwargs):
        if "fake_pred" in str(path):
            return guesses_df
        return truth_df

    with patch(MOCK_MODEL_FETCHER, return_value=mock_mf_instance), \
         patch(MOCK_PRED_PATH, return_value="fake_pred.parquet"), \
         patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, side_effect=smart_read), \
         patch.dict("os.environ", {"GITHUB_OUTPUT": "fake_env.txt"}), \
         patch(MOCK_OPEN, mock_open()) as mocked_file:
         
        grade_recent_performance()
        
        mocked_file().write.assert_not_called()


def test_lock_in_tomorrows_prediction_appends_correctly(tmp_path) -> None:
    """
    Tests that tomorrow's live directional prediction is correctly calculated 
    and appended smoothly to the active log.
    """
    mock_lf_instance = MagicMock()
    mock_lf_instance.get_todays_features.return_value = pd.DataFrame([{"feat_1": 0.5}])
    
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    mock_mf_instance = MagicMock()
    mock_mf_instance.get_champion_model.return_value = (mock_model, 0.6, {"version": "v2"})
    
    fake_pred_path = str(tmp_path / "fake_pred.parquet")
    
    existing_log = pd.DataFrame({
        "date": [pd.to_datetime("2026-05-31")],
        "prediction": [0],
        "probability": [0.4]
    })
    existing_log.to_parquet(fake_pred_path, index=False)

    with patch(MOCK_LIVE_FETCHER, return_value=mock_lf_instance), \
         patch(MOCK_MODEL_FETCHER, return_value=mock_mf_instance), \
         patch(MOCK_PRED_PATH, return_value=fake_pred_path), \
         patch(MOCK_STORAGE_OPTS, return_value={}):
         
        lock_in_tomorrows_prediction()
        
        saved_df = pd.read_parquet(fake_pred_path)
        
        assert len(saved_df) == 2
        assert saved_df["prediction"].iloc[-1] == 1
        assert saved_df["probability"].iloc[-1] == 0.7