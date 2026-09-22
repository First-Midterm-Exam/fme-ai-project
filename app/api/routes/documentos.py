"""Endpoints de revision de documentos."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.api.dependencies import get_revisor
from app.core.config import Settings, get_settings
from app.core.security import verify_api_key
from app.schemas.revision import RevisionFormatoResponse
from app.services.revisor_formato import RevisorFormato

router = APIRouter(
    prefix="/documentos",
    tags=["documentos"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/revision-formato",
    response_model=RevisionFormatoResponse,
    summary="Revisa el formato de un documento a partir de su imagen",
)
def revision_formato(
    imagen: Annotated[UploadFile, File(description="Imagen del documento")],
    revisor: Annotated[RevisorFormato, Depends(get_revisor)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RevisionFormatoResponse:
    # Funcion sincrona: FastAPI la ejecuta en un hilo aparte y la
    # inferencia no bloquea el event loop.
    if imagen.content_type and not imagen.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser una imagen",
        )
    content = imagen.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"La imagen supera {settings.max_upload_mb} MB",
        )
    return RevisionFormatoResponse.from_result(revisor.review(content))
