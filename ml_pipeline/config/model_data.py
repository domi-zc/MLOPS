from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

WANDB_PROJECT_NAME = "bitcoin-on-chain-prediction"
MODEL_ARTIFACT_NAME = "xgboost-sniper-bot"

MODEL_FILENAME = "xgboost_model.joblib"
MODEL_PATH = Path("models") / MODEL_FILENAME

FEATURE_COLS = [
    "price_usd",
    "return",
    "ma_deviation_7d",
    "ma_deviation_14d",
    "volatility_7d",
    "momentum_7d",
    "address_change_1d",
    "address_momentum_7d",
    "address_volatility_7d"
]
TARGET_COL = "target"

MIN_RECALL_THRESHOLD = 0.27

DEFAULT_CONFIG = {
    "model_type": "xgboost",
    "max_depth": 5, # How deep the trees can grow
    "learning_rate": 0.0141, # How aggressively each tree fixes mistakes
    "n_estimators": 82, # Total number of trees
    "subsample": 0.5802, # Percentage of historical days used per tree (prevents memorizing the chart / overfitting)
    "colsample_bytree": 0.9693, # Percentage of features used per tree
    "test_size": 0.15
}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {
        "name": "val_precision",
        "goal": "maximize"
    },
    "parameters": {
        "max_depth": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 8
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.01,
            "max": 0.2
        },
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 50,
            "max": 500
        },
        "subsample": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1
        },
        "colsample_bytree": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1
        }
    }
}