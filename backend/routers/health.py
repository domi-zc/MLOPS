from fastapi import APIRouter, Request

router = APIRouter(tags=["System"])

@router.get("/health")
async def health_check(request: Request):
    model_loaded = hasattr(request.app.state, "model")
    return {"status": "online", "model_loaded": model_loaded}