import logging

from fastapi import APIRouter, HTTPException, Request
from ml_pipeline.inference_pipeline.data_fetcher import LiveDataFetcher
from ml_pipeline.inference_pipeline.predictor import BitcoinPredictor

router = APIRouter(tags=["Inference"])
logger = logging.getLogger(__name__)

@router.get("/predict")
async def get_prediction(request: Request):

    # Check if model is in memory
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is currently offline.")

    try:
        logger.info("Fetching live data...")
        data_fetcher = LiveDataFetcher()
        live_features = data_fetcher.get_todays_features()
        
        logger.info("Running inference...")
        predictor = BitcoinPredictor(
            model=request.app.state.model, 
            threshold=request.app.state.threshold, 
            metrics=request.app.state.metrics
        )
        
        return predictor.predict(live_features)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))