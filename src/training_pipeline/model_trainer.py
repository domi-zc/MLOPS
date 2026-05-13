import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class WalkForwardTrainer:
    def __init__(self, config: dict):
        self.config = config

    def _evaluate_predictions(self, y_true, y_prediction):
        return {
            "accuracy": accuracy_score(y_true, y_prediction),
            "precision": precision_score(y_true, y_prediction, zero_division=0),
            "recall": recall_score(y_true, y_prediction, zero_division=0),
            "f1": f1_score(y_true, y_prediction, zero_division=0)
        }

    def _build_model(self) -> XGBClassifier:
        return XGBClassifier(
            max_depth=self.config.get("max_depth", 5),
            learning_rate=self.config.get("learning_rate", 0.1),
            n_estimators=self.config.get("n_estimators", 100),
            subsample=self.config.get("subsample", 0.8),
            colsample_bytree=self.config.get("colsample_bytree", 0.8),
            random_state=13,
            eval_metric="logloss"
        )

    def run_cross_validation(self, X_train_val: pd.DataFrame, y_train_val: pd.Series) -> dict:
        """
        Executes Walk-Forward Validation and dynamic threshold tuning.
        """
        print("Starting Walk-Forward Cross Validation...")
        ts_cross_validation = TimeSeriesSplit(n_splits=5)
        
        fold_metrics = [] # Save accuracy, precision, recall and f1 of every run
        best_thresholds = [] # Best threshold of every run

        X_arr = X_train_val.values
        y_arr = y_train_val.values

        for fold, (train_index, val_index) in enumerate(ts_cross_validation.split(X_arr)):
            X_fold_train, X_fold_val = X_arr[train_index], X_arr[val_index] # 
            y_fold_train, y_fold_val = y_arr[train_index], y_arr[val_index] # 

            model = self._build_model()
            model.fit(X_fold_train, y_fold_train)
            
            y_probability = model.predict_proba(X_fold_val)[:, 1] # Only grab second column of output

            best_fold_precision = 0
            best_fold_threshold = 0.5
            
            for threshold in np.arange(0.5, 0.75, 0.01):
                y_pred_tuned = (y_probability >= threshold).astype(int) # If y_probability is >= threshold it becomes 1 (Buy), otherwise 0 (Hold)

                precision = precision_score(np.array(y_fold_val), y_pred_tuned, zero_division=0)
                recall = recall_score(np.array(y_fold_val), y_pred_tuned, zero_division=0)
                
                if precision > best_fold_precision and recall > 0.1:
                    best_fold_precision = precision
                    best_fold_threshold = threshold
                    
            best_thresholds.append(best_fold_threshold)
            y_pred_best = (y_probability >= best_fold_threshold).astype(int)

            best_metrics = self._evaluate_predictions(y_fold_val, y_pred_best)
            fold_metrics.append(best_metrics)

            print(f"Fold {fold+1} | Accuracy: {best_metrics['accuracy']:.2f} | Precision: {best_metrics['precision']:.2f} | Recall: {best_metrics['recall']:.2f} | F1: {best_metrics['f1']:.2f} | Threshold: {best_fold_threshold:.2f}")

        metrics_df = pd.DataFrame(fold_metrics)

        return {
            "val_accuracy": metrics_df["accuracy"].mean(),
            "val_precision": metrics_df["precision"].mean(),
            "val_recall": metrics_df["recall"].mean(),
            "val_f1": metrics_df["f1"].mean(),
            "optimal_threshold": float(np.mean(best_thresholds))
        }

    def train_production_model(self, X_all: pd.DataFrame, y_all: pd.Series) -> XGBClassifier:
        """
        Trains the final model on 100% of the available data.
        """
        print("\nRetraining Champion Architecture on 100% of data...")
        
        model = self._build_model()
        model.fit(X_all, y_all)

        return model