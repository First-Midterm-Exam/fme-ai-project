from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.api.dependencies import FormatReviewerDep, SettingsDep
from app.api.security import verify_api_key
from app.schemas.review import RevisionFormatoResponse

IMAGE_CONTENT_PREFIX = "image/"

router = APIRouter(
    prefix="/documentos",
    tags=["documentos"],
    dependencies=[Depends(verify_api_key)],
)


def _read_image(imagen: UploadFile, max_bytes: int, max_mb: int) -> bytes:
    content_type = imagen.content_type
    if content_type and not content_type.startswith(IMAGE_CONTENT_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo debe ser una imagen",
        )
    content = imagen.file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"La imagen supera {max_mb} MB",
        )
    return content


@router.post(
    "/revision-formato",
    response_model=RevisionFormatoResponse,
    summary="Revisa el formato de un documento a partir de su imagen",
)
def revision_formato(
    imagen: Annotated[UploadFile, File(description="Imagen del documento")],
    reviewer: FormatReviewerDep,
    settings: SettingsDep,
) -> RevisionFormatoResponse:
    """
    Aplica OCR a la imagen, identifica encabezados, campos y
    respuestas con LayoutLMv3 y evalúa las reglas de formato.
    """
    content = _read_image(
        imagen, settings.max_upload_bytes, settings.max_upload_mb
    )
    return RevisionFormatoResponse.from_result(reviewer.review(content))
