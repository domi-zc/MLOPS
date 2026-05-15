import pandas as pd

class BitcoinPredictor:
    def __init__(self, model, threshold: float, metrics: dict):
        self.model = model
        self.threshold = threshold
        self.metrics = metrics

    def predict(self, todays_features: pd.DataFrame) -> dict:
        """
        Runs inference on today's features using the provided model and threshold.
        """
        print("Running inference...")
        
        probability = self.model.predict_proba(todays_features)[:, 1][0]
        prediction = 1 if probability >= self.threshold else 0
        
        result = {
            "prediction": "BUY" if prediction == 1 else "DON'T BUY",
            "probability": float(round(probability * 100, 2)),
            "threshold_used": float(round(self.threshold * 100, 2)),
            "model_precision": float(round(self.metrics.get("precision", 0) * 100, 2))
        }
        
        return result