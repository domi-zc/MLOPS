import pytest
from unittest.mock import patch, MagicMock, ANY

from ml_pipeline.training_pipeline.train import main

MOCK_WANDB_INIT = "ml_pipeline.training_pipeline.train.wandb.init"
MOCK_WANDB_LOG = "ml_pipeline.training_pipeline.train.wandb.log"
MOCK_WANDB_FINISH = "ml_pipeline.training_pipeline.train.wandb.finish"
MOCK_WANDB_ARTIFACT = "ml_pipeline.training_pipeline.train.wandb.Artifact"
MOCK_JOBLIB_DUMP = "ml_pipeline.training_pipeline.train.joblib.dump"
MOCK_DATA_BUILDER = "ml_pipeline.training_pipeline.train.TrainingDataBuilder"
MOCK_TRAINER = "ml_pipeline.training_pipeline.train.WalkForwardTrainer"
MOCK_OS_EXISTS = "ml_pipeline.training_pipeline.train.os.path.exists"
MOCK_JSON_LOAD = "ml_pipeline.training_pipeline.train.json.load"
MOCK_OPEN = "builtins.open"


def test_main_trains_and_logs_champion_model() -> None:
    """
    Tests that the script successfully trains the final model, saves it to disk, 
    and tags it as a 'champion' in W&B if the precision is high enough.
    """
    mock_run = MagicMock()
    mock_run.config = {"test_size": 0.15}
    
    mock_builder = MagicMock()
    mock_builder.load_and_split.return_value = ("X_tr", "y_tr", "X_te", "y_te")
    mock_builder.get_all_data.return_value = ("X_all", "y_all")
    
    mock_trainer = MagicMock()
    
    mock_metrics = {
        "val_accuracy": 0.7, 
        "val_precision": 0.8, # precision score to trigger "champion" alias
        "val_recall": 0.6, 
        "val_f1": 0.7, 
        "optimal_threshold": 0.55
    }
    mock_trainer.run_cross_validation.return_value = mock_metrics
    mock_trainer.train_production_model.return_value = "fake_fitted_model"

    with patch(MOCK_OS_EXISTS, return_value=False), \
         patch(MOCK_WANDB_INIT, return_value=mock_run), \
         patch(MOCK_WANDB_LOG), \
         patch(MOCK_DATA_BUILDER, return_value=mock_builder), \
         patch(MOCK_TRAINER, return_value=mock_trainer), \
         patch(MOCK_JOBLIB_DUMP) as mock_joblib, \
         patch(MOCK_WANDB_ARTIFACT) as mock_artifact_class, \
         patch(MOCK_WANDB_FINISH):
         
        main()
        
        mock_joblib.assert_called_once_with("fake_fitted_model", ANY)
        
        mock_artifact_class.assert_called_once()
        created_artifact_metadata = mock_artifact_class.call_args[1]["metadata"]
        assert created_artifact_metadata["val_precision"] == 0.8
        
        mock_run.log_artifact.assert_called_once_with(mock_artifact_class.return_value, aliases=["latest", "champion"])


def test_main_uses_sweep_config_if_available() -> None:
    """
    Tests that if a best_config.json file exists from a previous sweep, 
    the training script loads it instead of using the default fallback.
    """
    fake_sweep_config = {"learning_rate": 0.03, "max_depth": 7}
    
    mock_builder = MagicMock()
    mock_builder.load_and_split.return_value = ("X_tr", "y_tr", "X_te", "y_te")
    mock_builder.get_all_data.return_value = ("X_all", "y_all")
    
    mock_trainer = MagicMock()
    mock_trainer.run_cross_validation.return_value = {
        "val_accuracy": 0.7, "val_precision": 0.8, 
        "val_recall": 0.6, "val_f1": 0.7, "optimal_threshold": 0.55
    }
    
    with patch(MOCK_OS_EXISTS, return_value=True), \
         patch(MOCK_OPEN), \
         patch(MOCK_JSON_LOAD, return_value=fake_sweep_config), \
         patch(MOCK_WANDB_INIT) as mock_init, \
         patch(MOCK_WANDB_LOG), \
         patch(MOCK_DATA_BUILDER, return_value=mock_builder), \
         patch(MOCK_TRAINER, return_value=mock_trainer), \
         patch(MOCK_JOBLIB_DUMP), \
         patch(MOCK_WANDB_ARTIFACT), \
         patch(MOCK_WANDB_FINISH):
         
        main()
        
        mock_init.assert_called_once_with(
            project=ANY,
            job_type="train_production",
            config=fake_sweep_config
        )