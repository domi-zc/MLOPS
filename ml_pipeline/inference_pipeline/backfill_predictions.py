import pandas as pd

from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher
from ml_pipeline.config.storage_data import FEATURE_PATH, get_prediction_path, get_storage_options
from ml_pipeline.config.model_data import FEATURE_COLS

def run_backfill_predictions(days_to_simulate=60):
    print(f"Starting Prediction Backfill for the last {days_to_simulate} days...")
    
    options = get_storage_options()
    
    try:
        df = pd.read_parquet(FEATURE_PATH, storage_options=options)
    except Exception as e:
        print(f"Error: Could not read Feature Store at {FEATURE_PATH}. Details: {e}")
        return
    
    if len(df) <= days_to_simulate:
        print("Not enough data to run backfill.")
        return

    future_df = df.iloc[-days_to_simulate:].copy()

    model_fetcher = ModelFetcher()
    model, threshold, metrics = model_fetcher.get_champion_model()
    current_version = metrics.get("version", "v0")

    print(f"Generating baseline predictions using {current_version}...")

    X_batch = future_df[FEATURE_COLS]
    probabilities = model.predict_proba(X_batch)[:, 1]
    
    predictions = (probabilities >= threshold).astype(int) 

    log_df = pd.DataFrame({
        "date": future_df["date"],
        "prediction": predictions,
        "probability": probabilities
    })

    log_path = get_prediction_path(current_version)

    print(f"Uploading backfill log to {log_path}...")
    log_df.to_parquet(log_path, index=False, storage_options=options)
    
    print(f"\nBackfill Complete! Saved {len(log_df)} predictions for {current_version}.")

if __name__ == "__main__":
    run_backfill_predictions()