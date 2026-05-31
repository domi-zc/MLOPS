import joblib
import wandb
import os
import json

from ml_pipeline.training_pipeline.data_builder import TrainingDataBuilder
from ml_pipeline.training_pipeline.model_trainer import WalkForwardTrainer
from ml_pipeline.config.model_data import (
    WANDB_PROJECT_NAME,
    MODEL_ARTIFACT_NAME,
    MODEL_PATH,
    FEATURE_COLS,
    DEFAULT_CONFIG
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    config_path = os.path.join("models", "best_config.json")
    
    if os.path.exists(config_path):
        print(f"Found {config_path}! Using optimized sweep hyperparameters...")
        with open(config_path, "r") as f:
            config_to_use = json.load(f)
    else:
        config_to_use = DEFAULT_CONFIG
        print("No sweep config found. Using DEFAULT_CONFIG...")

    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        job_type="train_production",
        config=config_to_use
    )
    
    training_data_builder = TrainingDataBuilder(test_size=run.config.get("test_size", 0.15))
    X_train_val, y_train_val, X_test, y_test = training_data_builder.load_and_split()
    
    trainer = WalkForwardTrainer(config=dict(run.config))
    val_metrics = trainer.run_cross_validation(X_train_val, y_train_val)
    
    wandb.log(val_metrics)
    print(f"\nAccuracy: {val_metrics['val_accuracy']:.2f} | Precision: {val_metrics['val_precision']:.2f} | Recall: {val_metrics['val_recall']:.2f} | F1: {val_metrics['val_f1']:.2f}")

    print("Executing final production training on 100% of data...")
    X_all, y_all = training_data_builder.get_all_data(X_train_val, y_train_val, X_test, y_test)
    final_model = trainer.train_production_model(X_all, y_all)

    joblib.dump(final_model, MODEL_PATH)
    
    artifact = wandb.Artifact(
        name=MODEL_ARTIFACT_NAME,
        type="model",
        metadata={
            "features": FEATURE_COLS,
            "optimal_threshold": val_metrics["optimal_threshold"],
            "val_accuracy": val_metrics["val_accuracy"],
            "val_precision": val_metrics["val_precision"],
            "val_recall": val_metrics["val_recall"],
            "val_f1": val_metrics["val_f1"],
            "trained_on": "100_percent_all_data"
        }
    )
    artifact.add_file(local_path=str(MODEL_PATH))
    
    aliases = ["latest"]
    if val_metrics["val_precision"] >= 0.6:
        aliases.append("champion")
        
    run.log_artifact(artifact, aliases=aliases)
    wandb.finish()

if __name__ == "__main__":
    main()