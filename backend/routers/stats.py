from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["Statistics"])

@router.get("/stats")
async def get_stats(request: Request):
    if not hasattr(request.app.state, "metrics"):
        raise HTTPException(status_code=503, detail="Model metrics are currently offline.")

    return {
        "threshold": request.app.state.threshold,
        "metrics": request.app.state.metrics
    }