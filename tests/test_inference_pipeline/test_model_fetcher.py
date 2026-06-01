import pytest

from unittest.mock import patch, MagicMock
from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher

MOCK_WANDB_API = "ml_pipeline.inference_pipeline.model_fetcher.wandb.Api"
MOCK_JOBLIB_LOAD = "ml_pipeline.inference_pipeline.model_fetcher.joblib.load"


def test_get_champion_model_returns_correct_artifacts() -> None:
    """
    Tests that the fetcher successfully connects to W&B, parses the metadata dictionary, 
    downloads the artifact, and loads the physical model file.
    """
    mock_api = MagicMock()
    mock_artifact = MagicMock()
    
    mock_artifact.metadata = {
        "optimal_threshold": 0.65,
        "val_precision": 0.85,
        "val_accuracy": 0.70
    }
    mock_artifact.version = "v4"
    mock_artifact.download.return_value = "/fake/model/dir"
    
    mock_api.return_value.artifact.return_value = mock_artifact
    
    with patch(MOCK_WANDB_API, return_value=mock_api.return_value), \
         patch(MOCK_JOBLIB_LOAD, return_value="fake_fitted_model"):
         
        fetcher = ModelFetcher()
        model, threshold, metrics = fetcher.get_champion_model()
        
        assert model == "fake_fitted_model"
        assert threshold == 0.65
        assert metrics["version"] == "v4"
        assert metrics["precision"] == 0.85