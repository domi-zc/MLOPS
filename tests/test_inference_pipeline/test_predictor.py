import pytest
import pandas as pd
import numpy as np

from unittest.mock import MagicMock
from ml_pipeline.inference_pipeline.predictor import BitcoinPredictor


def test_predictor_buy_signal() -> None:
    """
    Tests that if the model's confidence exceeds the dynamic threshold, 
    it strictly outputs a 1 (Buy Signal).
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    
    metrics = {"precision": 0.85} # Threshold lower than confidence
    predictor = BitcoinPredictor(model=mock_model, threshold=0.6, metrics=metrics)
    
    features_df = pd.DataFrame([{"feat_1": 1.0}])
    result = predictor.predict(features_df)
    
    assert result["prediction"] == 1
    assert result["probability"] == 70.0
    assert result["precision"] == 85.0

 
def test_predictor_sell_signal() -> None:
    """
    Tests that if the model's confidence is below the threshold, 
    it strictly outputs a 0 (Do Not Buy Signal).
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    
    metrics = {"precision": 0.85}
    predictor = BitcoinPredictor(model=mock_model, threshold=0.6, metrics=metrics)
    
    features_df = pd.DataFrame([{"feat_1": 1.0}])
    result = predictor.predict(features_df)
    
    assert result["prediction"] == 0