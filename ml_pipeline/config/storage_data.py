import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

STORAGE_MODE = os.getenv("STORAGE_MODE", "cloud").lower() # "local" or "cloud"
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "bitcoin-feature-store")

if STORAGE_MODE == "cloud":
    FEATURE_PATH = f"s3://{BUCKET_NAME}/feature_store.parquet"
    LOG_PATH = f"s3://{BUCKET_NAME}/prediction_log.parquet"
else:
    FEATURE_PATH = "data/feature_store.parquet"
    LOG_PATH = "data/prediction_log.parquet"
    
def get_storage_options():
    if STORAGE_MODE == "local":
        return None  
        
    return {
        "key": os.getenv("GCS_ACCESS_KEY"),
        "secret": os.getenv("GCS_SECRET_KEY"),
        "client_kwargs": {
            "endpoint_url": "https://storage.googleapis.com",
            "region_name": "eu-central-2"
        },
        "config_kwargs": {
            "s3": {"addressing_style": "path"},
            "request_checksum_calculation": "when_required",
            "response_checksum_validation": "when_required"
        }
    }