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
            "prediction": prediction,
            "probability": float(round(probability * 100, 2)),
            "threshold": float(round(self.threshold * 100, 2)),
            "precision": float(round(self.metrics.get("precision", 0) * 100, 2))
        }
        
        return result

if __name__ == "__main__":
    from ml_pipeline.inference_pipeline.data_fetcher import LiveDataFetcher
    from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher
    
    data_fetcher = LiveDataFetcher()
    features_df = data_fetcher.get_todays_features()
    
    model_fetcher = ModelFetcher()
    model, threshold, metrics = model_fetcher.get_champion_model()
    
    predictor = BitcoinPredictor(model, threshold, metrics)
    decision = predictor.predict(features_df)
    
    print("\n--- Final Trading Decision ---")
    for key, value in decision.items():
        print(f"{key.replace('_', ' ').title()}: {value}")