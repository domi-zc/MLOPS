import pandas as pd

from src.training_pipeline.metadata import FEATURE_PATH, FEATURE_COLS, TARGET_COL

class TrainingDataBuilder:
    def __init__(self, test_size: float = 0.15):
        self.test_size = test_size

    def load_and_split(self) -> tuple:
        """
        Loads the Parquet file and splits it chronologically.
        Returns the historical chunk (for Walk-Forward) and the future chunk (Test).
        """
        print("Loading data from Feature Store...")
        
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(f"Could not find {FEATURE_PATH}.")
            
        df = pd.read_parquet(FEATURE_PATH)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]) # Remove any row containing a singular NaN. Prevents XGBoost from crashing
        
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]
        
        split_idx = int(len(df) * (1 - self.test_size))
        
        X_train_val = X.iloc[:split_idx].reset_index(drop=True)
        y_train_val = y.iloc[:split_idx].reset_index(drop=True)
        
        X_test = X.iloc[split_idx:].reset_index(drop=True)
        y_test = y.iloc[split_idx:].reset_index(drop=True)
        
        print(f"Data Splitting Complete:")
        print(f"-> Train/Val (Historical): {len(X_train_val)} days")
        print(f"-> Test (Future): {len(X_test)} days")
        
        return X_train_val, y_train_val, X_test, y_test

    def get_all_data(self, X_train_val, y_train_val, X_test, y_test) -> tuple:
        """
        Combines all historical and future data together.
        Used at the very end to train the final Champion model.
        """
        X_all = pd.concat([X_train_val, X_test], ignore_index=True)
        y_all = pd.concat([y_train_val, y_test], ignore_index=True)

        return X_all, y_all