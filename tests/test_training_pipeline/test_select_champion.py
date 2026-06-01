import pytest

from unittest.mock import patch, MagicMock, ANY
from ml_pipeline.training_pipeline.select_champion import pick_and_promote_champion

MOCK_WANDB_API = "ml_pipeline.training_pipeline.select_champion.wandb.Api"
MOCK_OS_MAKEDIRS = "ml_pipeline.training_pipeline.select_champion.os.makedirs"
MOCK_JSON_DUMP = "ml_pipeline.training_pipeline.select_champion.json.dump"
MOCK_OPEN = "builtins.open"


def test_pick_and_promote_champion_finds_best_run() -> None:
    """
    Tests that the script correctly filters out failed runs, ignores runs below 
    the recall threshold, and selects the run with the highest precision.
    """
    mock_api = MagicMock()
    mock_sweep = MagicMock()
    
    run_1 = MagicMock(state="finished", id="run1", config={"lr": 0.1})
    run_1.summary = {"val_recall": 0.8, "val_precision": 0.6}
    
    run_2 = MagicMock(state="failed") 
    
    run_3 = MagicMock(state="finished", id="run3", config={"lr": 0.2})
    run_3.summary = {"val_recall": 0.2, "val_precision": 0.9} 
    
    run_4 = MagicMock(state="finished", id="run4", config={"lr": 0.05})
    run_4.summary = {"val_recall": 0.7, "val_precision": 0.8} 
    
    mock_sweep.runs = [run_1, run_2, run_3, run_4]
    mock_api.return_value.sweep.return_value = mock_sweep
    
    with patch(MOCK_WANDB_API, return_value=mock_api.return_value), \
         patch(MOCK_OS_MAKEDIRS), \
         patch(MOCK_OPEN), \
         patch(MOCK_JSON_DUMP) as mock_json:
         
        result = pick_and_promote_champion("fake_entity", "fake_proj", "sweep123", min_recall=0.5)
        
        assert result is True
        
        mock_json.assert_called_once_with({"lr": 0.05}, ANY, indent=4)


def test_pick_and_promote_champion_fails_gracefully() -> None:
    """
    Tests that the script safely returns False if no models meet the minimum recall threshold.
    """
    mock_api = MagicMock()
    mock_sweep = MagicMock()
    
    run_1 = MagicMock(state="finished")
    run_1.summary = {"val_recall": 0.1, "val_precision": 0.9}
    
    mock_sweep.runs = [run_1]
    mock_api.return_value.sweep.return_value = mock_sweep
    
    with patch(MOCK_WANDB_API, return_value=mock_api.return_value):
        result = pick_and_promote_champion("fake_entity", "fake_proj", "sweep123", min_recall=0.5)
        
        assert result is False