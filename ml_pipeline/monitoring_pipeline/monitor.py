import pandas as pd
import os

from datetime import datetime, timezone
from sklearn.metrics import precision_score
from ml_pipeline.inference_pipeline.data_fetcher import LiveDataFetcher
from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher
from ml_pipeline.config.storage_data import FEATURE_PATH, get_prediction_path, get_storage_options

def grade_recent_performance():
    """
    Reads the vault, compares past predictions to the actual targets in the Feature Store,
    and calculates the rolling 60-day precision. Incorporates the minimum trade volume rule.
    """
    options = get_storage_options()

    model_fetcher = ModelFetcher()
    _, _, metrics = model_fetcher.get_champion_model()
    current_version = metrics.get("version", "v0")

    prediction_path = get_prediction_path(current_version)
    
    try:
        truth_df = pd.read_parquet(FEATURE_PATH, storage_options=options)
    except Exception:
        print(f"No feature store found at {FEATURE_PATH}.")
        return
    
    try:
        guesses_df = pd.read_parquet(prediction_path, storage_options=options)
    except Exception:
        print(f"No prediction log found for {current_version}. Waiting for more data...")
        return

    merged_df = pd.merge(guesses_df, truth_df[['date', 'target']], on='date', how='inner')

    recent = merged_df.tail(60)
    
    buy_signals = recent[recent['prediction'] == 1]
    total_shots = len(buy_signals)
    
    if total_shots < 3:
        print("The model made less than 3 trades in the last 60 days. Too little data to detect drift!")
        return
        
    precision = precision_score(recent['target'], recent['prediction'], zero_division=0)
    print(f"Rolling 60-Day Precision (based on {total_shots} trades): {precision * 100:.2f}%")

    if precision < 0.45:
        print("\nDRIFT DETECTED!")
        print("The market meta has changed. The model missed its recent shots.")
        
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("drift_detected=true\n")
    else:
        print("No drift detected. Model is hitting its targets. ")
        
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write("drift_detected=false\n")

def lock_in_tomorrows_prediction():
    """
    Fetches today's live data, makes a prediction for tomorrow, 
    and locks it in the Parquet vault so it can't cheat.
    """
    fetcher = LiveDataFetcher()
    todays_features = fetcher.get_todays_features()

    model_fetcher = ModelFetcher()
    model, threshold, metrics = model_fetcher.get_champion_model()

    probability = model.predict_proba(todays_features)[:, 1][0]
    raw_prediction = 1 if probability >= threshold else 0

    today_date = pd.to_datetime(datetime.now(timezone.utc).date())
    new_log = pd.DataFrame([{
        "date": today_date,
        "prediction": raw_prediction,
        "probability": float(probability)
    }])

    options = get_storage_options()

    current_version = metrics.get("version", "v0")

    prediction_path = get_prediction_path(current_version)

    try:
        existing_log = pd.read_parquet(prediction_path, storage_options=options)
        updated_log = pd.concat([existing_log, new_log]).drop_duplicates(subset=['date'], keep='last')
        print(f"Appending prediction to {current_version} cloud log...")
    except Exception:
        print(f"Creating a new prediction log for {current_version} in the cloud...")
        updated_log = new_log

    updated_log.to_parquet(prediction_path, index=False, storage_options=options)
    print(f"Locked in prediction '{raw_prediction}' for date: {today_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    grade_recent_performance()
    lock_in_tomorrows_prediction()