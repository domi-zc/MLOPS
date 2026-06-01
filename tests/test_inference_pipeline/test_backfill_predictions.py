import pytest
import pandas as pd
import numpy as np

from unittest.mock import patch, MagicMock
from ml_pipeline.inference_pipeline.backfill_predictions import run_backfill_predictions

MOCK_READ_PARQUET = "ml_pipeline.inference_pipeline.backfill_predictions.pd.read_parquet"
MOCK_MODEL_FETCHER = "ml_pipeline.inference_pipeline.backfill_predictions.ModelFetcher"
MOCK_PRED_PATH = "ml_pipeline.inference_pipeline.backfill_predictions.get_prediction_path"
MOCK_STORAGE_OPTS = "ml_pipeline.inference_pipeline.backfill_predictions.get_storage_options"
MOCK_FEATURE_COLS = "ml_pipeline.inference_pipeline.backfill_predictions.FEATURE_COLS"


def test_run_backfill_predictions_success(tmp_path) -> None:
    """
    Tests that a sufficiently large dataset is successfully sliced, evaluated 
    by the champion model, and securely written to the prediction log.
    """
    mock_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=100),
        "feat_1": np.random.rand(100)
    })
    
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 60)
    
    mock_mf_instance = MagicMock()
    mock_mf_instance.get_champion_model.return_value = (mock_model, 0.6, {"version": "v3"})
    
    fake_log_path = str(tmp_path / "fake_log.parquet")
    
    with patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, return_value=mock_df), \
         patch(MOCK_FEATURE_COLS, ["feat_1"]), \
         patch(MOCK_MODEL_FETCHER, return_value=mock_mf_instance), \
         patch(MOCK_PRED_PATH, return_value=fake_log_path):
         
        run_backfill_predictions(days_to_simulate=60)
        
    saved_df = pd.read_parquet(fake_log_path)
    
    assert len(saved_df) == 60
    assert saved_df["prediction"].iloc[0] == 1


def test_run_backfill_predictions_not_enough_data() -> None:
    """
    Tests that the script exits safely without attempting to predict 
    if the feature store contains fewer rows than the requested simulation window.
    """
    mock_df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30) # Only 30 days of data
    })
    
    with patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, return_value=mock_df), \
         patch(MOCK_MODEL_FETCHER) as mock_mf:
         
        run_backfill_predictions(days_to_simulate=60)
        
        # Prove the script aborted before ever calling the model fetcher
        mock_mf.assert_not_called()


def test_run_backfill_predictions_handles_missing_file() -> None:
    """
    Tests that the script catches a missing Feature Store file and exits safely 
    instead of crashing the CI/CD pipeline.
    """
    with patch(MOCK_STORAGE_OPTS, return_value={}), \
         patch(MOCK_READ_PARQUET, side_effect=Exception("File not found")), \
         patch(MOCK_MODEL_FETCHER) as mock_mf:
         
        run_backfill_predictions(days_to_simulate=60)
        
        mock_mf.assert_not_called()