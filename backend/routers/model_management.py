import os
import logging

from fastapi import APIRouter, Request, HTTPException
from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher

router = APIRouter(tags=["Model Management"])
logger = logging.getLogger(__name__)


@router.post("/refresh-model")
async def refresh_model(request: Request):
    """
    Forces the API to drop its current model and download the latest Champion.
    """
    logger.info("Downloading latest champion model...")
    try:
        fetcher = ModelFetcher()
        model, threshold, metrics = fetcher.get_champion_model()
        
        request.app.state.model = model
        request.app.state.threshold = threshold
        request.app.state.metrics = metrics
        
        logger.info("Swap successful. New Champion model is live.")
        
        return {
            "status": "success", 
            "message": "Champion model updated in memory.",
            "new_threshold": threshold
        }
    except Exception as e:
        logger.error(f"Failed to hot-swap model: {e}")
        raise HTTPException(status_code=500, detail=str(e))