import pandas as pd
import os

from datetime import datetime, timezone
from sklearn.metrics import precision_score
from ml_pipeline.inference_pipeline.data_fetcher import LiveDataFetcher
from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher
from ml_pipeline.config.storage_data import FEATURE_PATH, get_prediction_path, get_storage_options

def grade_the_past_30_days():
    """
    Reads the vault, compares past predictions to the actual targets in the Feature Store,
    and calculates the rolling 30-day precision. Incorporates the minimum trade volume rule.
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
    
    try:
        guesses_df = pd.read_parquet(get_prediction_path(), storage_options=options)
    except Exception:
        print(f"No prediction log found at {get_prediction_path()}.")
        return

    merged_df = pd.merge(guesses_df, truth_df[['date', 'target']], on='date', how='inner')

    if len(merged_df) < 5:
        print(f"Only {len(merged_df)} days of graded history. Waiting for more data...")
        return

    recent_30 = merged_df.tail(30)
    
    buy_signals = recent_30[recent_30['prediction'] == 1]
    total_shots = len(buy_signals)
    
    if total_shots == 0:
        print("No drift detected. The model made 0 trades in the last 30 days.")
        return
        
    precision = precision_score(recent_30['target'], recent_30['prediction'], zero_division=0)
    print(f"Rolling 30-Day Precision (based on {total_shots} trades): {precision * 100:.2f}%")

    if precision < 0.55:
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
    model, threshold, _ = model_fetcher.get_champion_model()

    probability = model.predict_proba(todays_features)[:, 1][0]
    raw_prediction = 1 if probability >= threshold else 0

    today_date = pd.to_datetime(datetime.now(timezone.utc).date())
    new_log = pd.DataFrame([{
        "date": today_date,
        "prediction": raw_prediction,
        "probability": float(probability)
    }])

    options = get_storage_options()

    model_fetcher = ModelFetcher()
    _, _, metrics = model_fetcher.get_champion_model()
    current_version = metrics.get("version", "v0")

    prediction_path = get_prediction_path(current_version)

    try:
        existing_log = pd.read_parquet(prediction_path, storage_options=options)
        updated_log = pd.concat([existing_log, new_log]).drop_duplicates(subset=['date'], keep='last')
        print("Appending prediction to existing cloud log...")
    except Exception:
        print("Creating a new prediction log in the cloud...")
        updated_log = new_log

    updated_log.to_parquet(prediction_path, index=False, storage_options=options)
    print(f"Locked in prediction '{raw_prediction}' for date: {today_date.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    grade_the_past_30_days()
    lock_in_tomorrows_prediction()