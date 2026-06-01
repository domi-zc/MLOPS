import os

from pathlib import Path
from dotenv import load_dotenv
from ml_pipeline.config.model_data import MODEL_ARTIFACT_NAME

# Load environment variables
load_dotenv()

STORAGE_MODE = os.getenv("STORAGE_MODE", "cloud").lower() # "local" or "cloud"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "bitcoin-feature-store")

FEATURE_DIR = "feature_store"
PREDICTION_DIR = "prediction_logs"

FEATURE_FILE = f"features.parquet"
PREDICTION_FILE = f"predictions.parquet"

def get_prediction_path(version: str = "v0") -> str:
    """
    Generates a foldername based on the model version.
    """
    modelname_version = f"{MODEL_ARTIFACT_NAME}_{version}"
    
    if STORAGE_MODE == "cloud":
        return f"gs://{GCS_BUCKET_NAME}/{PREDICTION_DIR}/{modelname_version}/{PREDICTION_FILE}"
    else:
        local_dir = f"data/{PREDICTION_DIR}/{modelname_version}"
        os.makedirs(local_dir, exist_ok=True)
        
        return f"{local_dir}/{PREDICTION_FILE}"

if STORAGE_MODE == "cloud":
    FEATURE_PATH = f"gs://{GCS_BUCKET_NAME}/{FEATURE_DIR}/{FEATURE_FILE}"
else:
    FEATURE_PATH = f"data/{FEATURE_DIR}/{FEATURE_FILE}"
    os.makedirs(f"data/{FEATURE_DIR}", exist_ok=True)
    os.makedirs(f"data/{PREDICTION_DIR}", exist_ok=True)
    
def get_storage_options():
    mode = os.getenv("STORAGE_MODE", "local")
    
    if mode == "cloud":
        return {} 
    
    return {}