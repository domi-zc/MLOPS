import os
import argparse
import joblib
import wandb

from src.training_pipeline.data_builder import TrainingDataBuilder
from src.training_pipeline.model_trainer import WalkForwardTrainer
from src.training_pipeline.metadata import (
    PROJECT_NAME,
    MODEL_ARTIFACT_NAME,
    MODEL_PATH,
    FEATURE_COLS,
    DEFAULT_CONFIG,
    SWEEP_CONFIG
)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def main():
    run = wandb.init(
        project=PROJECT_NAME,
        job_type="train",
        config=DEFAULT_CONFIG
    )
    
    training_data_builder = TrainingDataBuilder(test_size=run.config.get("test_size", 0.15))
    X_train_val, y_train_val, X_test, y_test = training_data_builder.load_and_split()
    
    trainer = WalkForwardTrainer(config=dict(run.config))
    val_metrics = trainer.run_cross_validation(X_train_val, y_train_val)
    
    wandb.log(val_metrics)
    print(f"\nAccuracy: {val_metrics['val_accuracy']:.2f} | Precision: {val_metrics['val_precision']:.2f} | Recall: {val_metrics['val_recall']:.2f} | F1: {val_metrics['val_f1']:.2f} | Optimal Treshold: {val_metrics['optimal_threshold']:.2f}")

    is_sweep = run.sweep_id is not None

    if is_sweep:
        print("Sweep Run Detected. Skipping production training and model registry.")
    else:
        X_all, y_all = training_data_builder.get_all_data(X_train_val, y_train_val, X_test, y_test)
        final_model = trainer.train_production_model(X_all, y_all)

        joblib.dump(final_model, MODEL_PATH)
        
        artifact = wandb.Artifact(
            name=MODEL_ARTIFACT_NAME,
            type="model",
            metadata={
                "features": FEATURE_COLS,
                "optimal_threshold": val_metrics["optimal_threshold"],
                "val_precision": val_metrics["val_precision"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--count", type=int, default=10)
    
    args = parser.parse_args()

    if args.sweep:
        print(f"Initializing Bayesian Sweep for {args.count} runs...")
        sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project=PROJECT_NAME)
        wandb.agent(sweep_id, function=main, count=args.count)
    else:
        print("Executing single production run...")
        main()