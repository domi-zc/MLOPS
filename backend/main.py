from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from ml_pipeline.inference_pipeline.model_fetcher import ModelFetcher
from backend.routers import predict, stats, health, model_management

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Downloading and loading the champion XGBoost model...")
    try:
        fetcher = ModelFetcher()
        model, threshold, metrics = fetcher.get_champion_model()
        
        app.state.model = model
        app.state.threshold = threshold
        app.state.metrics = metrics
        
        logger.info("Model loaded successfully into memory.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    if hasattr(app.state, "model"):
        del app.state.model

app = FastAPI(title="Bitcoin ML Sniper API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/backend-api")
app.include_router(stats.router, prefix="/backend-api")
app.include_router(health.router, prefix="/backend-api")
app.include_router(model_management.router, prefix="/backend-api")