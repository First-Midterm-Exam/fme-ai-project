"""Estado del servicio."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.core.config import Settings, get_settings
from app.schemas.revision import HealthResponse

router = APIRouter(tags=["salud"])


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    loaded = getattr(request.app.state, "revisor", None) is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=settings.app_version,
        model_loaded=loaded,
        detail=getattr(request.app.state, "revisor_error", None),
    )
