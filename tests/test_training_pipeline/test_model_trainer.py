import pytest
import pandas as pd
import numpy as np

from ml_pipeline.training_pipeline.model_trainer import WalkForwardTrainer


@pytest.fixture
def mock_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Generates enough dummy data to survive a 5-fold TimeSeriesSplit.
    """
    np.random.seed(42)
    X = pd.DataFrame({
        "feat_1": np.random.rand(100),
        "feat_2": np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, size=100))
    
    return X, y


def test_build_model_respects_config() -> None:
    """
    Ensures the trainer initializes the XGBoost model with the exact hyperparameters 
    provided by the WandB sweep config.
    """
    config = {
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 50
    }
    trainer = WalkForwardTrainer(config=config)
    model = trainer._build_model()
    
    assert model.get_params()["max_depth"] == 7
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["n_estimators"] == 50


def test_run_cross_validation_returns_metrics(mock_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Proves the Walk-Forward loop executes fully and returns the correctly formatted 
    metrics dictionary required for logging to Weights & Biases.
    """
    X_train_val, y_train_val = mock_training_data
    
    trainer = WalkForwardTrainer(config={"n_estimators": 5, "max_depth": 3})
    metrics = trainer.run_cross_validation(X_train_val, y_train_val)
    
    expected_keys = ["val_accuracy", "val_precision", "val_recall", "val_f1", "optimal_threshold"]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_train_production_model(mock_training_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """
    Ensures the final 100% data retraining function returns a fitted XGBoost instance.
    """
    X_all, y_all = mock_training_data
    
    trainer = WalkForwardTrainer(config={"n_estimators": 5, "max_depth": 3})
    model = trainer.train_production_model(X_all, y_all)
    
    assert hasattr(model, "predict"), "Returned object is not a valid fitted model"