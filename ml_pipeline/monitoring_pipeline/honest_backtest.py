import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ml_pipeline.config.storage_data import FEATURE_PATH, PREDICTION_PATH, get_storage_options
from ml_pipeline.config.model_data import (
    FEATURE_COLS,
    TARGET_COL,
    DEFAULT_CONFIG
)
from ml_pipeline.training_pipeline.model_trainer import WalkForwardTrainer

def run_honest_backtest(days_to_simulate=300):
    print(f"Starting Honest Walk-Forward Backtest for the last {days_to_simulate} days...")
    
    options = get_storage_options()
    
    try:
        df = pd.read_parquet(FEATURE_PATH, storage_options=options)
    except Exception as e:
        print(f"Error: Could not read Feature Store at {FEATURE_PATH}. Details: {e}")
        return
    
    if len(df) <= days_to_simulate:
        print("Not enough data to run simulation.")
        return

    past_df = df.iloc[:-days_to_simulate].copy()
    future_df = df.iloc[-days_to_simulate:].copy()

    log_entries = []

    for index, row in tqdm(future_df.iterrows(), total=days_to_simulate):
        X_train = past_df[FEATURE_COLS]
        y_train = past_df[TARGET_COL]
        
        trainer = WalkForwardTrainer(config=DEFAULT_CONFIG)
        val_metrics = trainer.run_cross_validation(X_train, y_train)
        threshold = val_metrics["optimal_threshold"]
        
        model = trainer._build_model()
        model.fit(X_train, y_train)
        
        X_today = pd.DataFrame([row[FEATURE_COLS]])
        probability = model.predict_proba(X_today)[:, 1][0]
        prediction = 1 if probability >= threshold else 0
        
        log_entries.append({
            "date": row["date"],
            "prediction": prediction,
            "probability": float(probability)
        })
        
        past_df = pd.concat([past_df, pd.DataFrame([row])]).reset_index(drop=True)

    log_df = pd.DataFrame(log_entries)

    print(f"Uploading backtest log to {PREDICTION_PATH}...")
    log_df.to_parquet(PREDICTION_PATH, index=False, storage_options=options)
    
    print(f"\nHonest Backtest Complete! Saved {len(log_df)} predictions.")

if __name__ == "__main__":
    run_honest_backtest()