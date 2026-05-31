import argparse
import wandb
import os
import dotenv

from pathlib import Path
from ml_pipeline.training_pipeline.data_builder import TrainingDataBuilder
from ml_pipeline.training_pipeline.model_trainer import WalkForwardTrainer
from ml_pipeline.config.model_data import WANDB_PROJECT_NAME, SWEEP_CONFIG

def evaluate_config():
    """
    The target function for the W&B agent to run during the sweep.
    """
    run = wandb.init(project=WANDB_PROJECT_NAME, job_type="sweep")
    
    training_data_builder = TrainingDataBuilder(test_size=run.config.get("test_size", 0.15))
    X_train_val, y_train_val, _, _ = training_data_builder.load_and_split()
    
    trainer = WalkForwardTrainer(config=dict(run.config))
    val_metrics = trainer.run_cross_validation(X_train_val, y_train_val)
    
    wandb.log(val_metrics)
    print(f"Sweep Run Complete | Precision: {val_metrics['val_precision']:.2f} | Recall: {val_metrics['val_recall']:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()

    print(f"Initializing Bayesian Sweep for {args.count} runs...")

    wandb_entity = os.getenv("WANDB_ENTITY")

    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG, 
        project=WANDB_PROJECT_NAME, 
        entity=wandb_entity
    )
    
    github_env_path = os.getenv("GITHUB_ENV")
    
    if github_env_path:
        with open(github_env_path, "a") as f:
            f.write(f"SWEEP_ID={sweep_id}\n")
        print(f"Successfully injected SWEEP_ID={sweep_id} into GitHub Actions environment.")
    else:
        env_path = Path(".env")
        env_path.touch(exist_ok=True)
        dotenv.set_key(env_path, "SWEEP_ID", sweep_id)
        print(f"\nSuccessfully wrote SWEEP_ID={sweep_id} to your .env file!")

    wandb.agent(sweep_id, function=evaluate_config, count=args.count)

if __name__ == "__main__":
    main()