from fastapi import APIRouter, Request

from app.api.dependencies import SettingsDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["salud"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: SettingsDep) -> HealthResponse:
    """
    Indica si el servicio está operativo y si el modelo fue cargado.
    """
    loaded = request.app.state.reviewer is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=settings.app_version,
        model_loaded=loaded,
        detail=request.app.state.reviewer_error,
    )
