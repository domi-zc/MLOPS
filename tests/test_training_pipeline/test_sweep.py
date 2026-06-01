import pytest

from unittest.mock import patch, MagicMock
from ml_pipeline.training_pipeline.sweep import main, evaluate_config

MOCK_WANDB_INIT = "ml_pipeline.training_pipeline.sweep.wandb.init"
MOCK_WANDB_LOG = "ml_pipeline.training_pipeline.sweep.wandb.log"
MOCK_WANDB_SWEEP = "ml_pipeline.training_pipeline.sweep.wandb.sweep"
MOCK_WANDB_AGENT = "ml_pipeline.training_pipeline.sweep.wandb.agent"
MOCK_TRAINING_BUILDER = "ml_pipeline.training_pipeline.sweep.TrainingDataBuilder"
MOCK_TRAINER = "ml_pipeline.training_pipeline.sweep.WalkForwardTrainer"
MOCK_DOTENV = "ml_pipeline.training_pipeline.sweep.dotenv.set_key"
MOCK_OS_GETENV = "ml_pipeline.training_pipeline.sweep.os.getenv"


def test_evaluate_config_executes_training_loop() -> None:
    """
    Tests that the sweep agent successfully loads data, trains the model, 
    and logs the metrics back to Weights & Biases without crashing.
    """
    mock_run = MagicMock()
    mock_run.config = {"test_size": 0.2, "learning_rate": 0.1}
    
    mock_builder_instance = MagicMock()
    mock_builder_instance.load_and_split.return_value = ("X_train", "y_train", "X_test", "y_test")
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.run_cross_validation.return_value = {"val_precision": 0.8, "val_recall": 0.7}
    
    with patch(MOCK_WANDB_INIT, return_value=mock_run), \
         patch(MOCK_TRAINING_BUILDER, return_value=mock_builder_instance), \
         patch(MOCK_TRAINER, return_value=mock_trainer_instance), \
         patch(MOCK_WANDB_LOG) as mock_log:
         
        evaluate_config()
        
        mock_log.assert_called_once_with({"val_precision": 0.8, "val_recall": 0.7})


def test_main_initializes_sweep_and_saves_env(tmp_path) -> None:
    """
    Tests that the main orchestrator generates a sweep ID and saves it securely to the .env file.
    """
    fake_env_path = tmp_path / ".env"
    
    def smart_getenv(key, default=None):
        if key == "WANDB_ENTITY":
            return "fake_entity"
        return None

    with patch("ml_pipeline.training_pipeline.sweep.argparse.ArgumentParser.parse_args", return_value=MagicMock(count=5)), \
         patch(MOCK_OS_GETENV, side_effect=smart_getenv), \
         patch(MOCK_WANDB_SWEEP, return_value="sweep_abc123"), \
         patch("ml_pipeline.training_pipeline.sweep.Path", return_value=fake_env_path), \
         patch(MOCK_DOTENV) as mock_dotenv, \
         patch(MOCK_WANDB_AGENT) as mock_agent:
         
        main()
        
        mock_dotenv.assert_called_once_with(fake_env_path, "SWEEP_ID", "sweep_abc123")
        mock_agent.assert_called_once_with("sweep_abc123", function=evaluate_config, count=5)