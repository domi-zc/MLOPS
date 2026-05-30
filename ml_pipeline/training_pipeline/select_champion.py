import os
import wandb
import shutil
import json

from ml_pipeline.config.model_data import WANDB_PROJECT_NAME, MIN_RECALL_THRESHOLD

def pick_and_promote_champion(entity_name, project_name, sweep_id, min_recall):
    """
    Searches a W&B sweep for the best model meeting the minimum recall, maximizing Precision.
    Downloads the model and tags it as the 'champion'.
    """
    api = wandb.Api()
    sweep = api.sweep(f"{entity_name}/{project_name}/{sweep_id}")
    
    best_run = None
    highest_precision = 0
    
    print(f"Scanning {len(sweep.runs)} runs in sweep {sweep_id}...")
    
    for run in sweep.runs:
        if run.state != "finished":
            continue

        metrics = run.summary
        recall = metrics.get("val_recall", 0)
        precision = metrics.get("val_precision", 0)
        
        if recall >= min_recall:
            if precision > highest_precision:
                highest_precision = precision
                best_run = run

    if best_run is None:
        print(f"CRITICAL: No runs met the criteria of Recall >= {min_recall}!")
        return False
        
    print(f"\n---CHAMPION FOUND---")
    print(f"Run ID:    {best_run.id}")
    print(f"Recall:    {best_run.summary.get('val_recall'):.4f}")
    print(f"Precision: {highest_precision:.4f}")
    
    print(f"Winning Hyperparameters: {best_run.config}")

    best_config = best_run.config
    
    os.makedirs("models", exist_ok=True)
    config_path = os.path.join("models", "best_config.json")
    
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(f"Successfully saved winning hyperparameters to {config_path}!")
    
    return True

if __name__ == "__main__":
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    SWEEP_ID = os.getenv("SWEEP_ID")
    
    if not WANDB_ENTITY or not SWEEP_ID:
        raise ValueError("CRITICAL: WANDB_ENTITY or SWEEP_ID is missing from the environment!")
    
    pick_and_promote_champion(
        entity_name=WANDB_ENTITY, 
        project_name=WANDB_PROJECT_NAME, 
        sweep_id=SWEEP_ID, 
        min_recall=MIN_RECALL_THRESHOLD
    )