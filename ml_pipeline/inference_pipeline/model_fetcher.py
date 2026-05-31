import joblib
import wandb

from pathlib import Path
from ml_pipeline.config.model_data import (
    WANDB_PROJECT_NAME, 
    MODEL_ARTIFACT_NAME, 
    MODEL_FILENAME
)

class ModelFetcher:
    def __init__(self):
        self.WANDB_PROJECT_NAME = WANDB_PROJECT_NAME
        self.artifact_name = MODEL_ARTIFACT_NAME

    def get_champion_model(self) -> tuple:
        """
        Downloads the champion model from W&B and returns the model, threshold, and metrics.
        """
        print("Connecting to W&B to download Champion model...")
        
        api = wandb.Api()
        artifact = api.artifact(f"{self.WANDB_PROJECT_NAME}/{self.artifact_name}:champion")
        
        metadata = artifact.metadata
        threshold = metadata.get("optimal_threshold", 0.5)
        metrics = {
            "accuracy": metadata.get("val_accuracy", 0.0),
            "precision": metadata.get("val_precision", 0.0),
            "recall": metadata.get("val_recall", 0.0),
            "f1": metadata.get("val_f1", 0.0),
            "version": artifact.version
        }
        
        model_dir = artifact.download()
        model_path = Path(model_dir) / MODEL_FILENAME
        
        model = joblib.load(model_path)
        print(f"Model loaded successfully. Optimal Threshold: {threshold:.2f}")

        return model, threshold, metrics