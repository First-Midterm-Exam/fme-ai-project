"""Construccion e inyeccion de los servicios usados por las rutas."""

import logging

from fastapi import FastAPI, Request

from app.core.config import Settings
from app.domain.exceptions import ModelUnavailableError
from app.services.entity_extractor import LayoutLMv3EntityExtractor
from app.services.ocr import TesseractOcr
from app.services.revisor_formato import RevisorFormato

logger = logging.getLogger(__name__)


def build_revisor(settings: Settings) -> RevisorFormato:
    return RevisorFormato(
        ocr=TesseractOcr(
            settings.ocr_languages,
            settings.tesseract_cmd,
            settings.tessdata_dir,
        ),
        extractor=LayoutLMv3EntityExtractor(
            settings.model_dir, settings.model_device
        ),
        min_passing_score=settings.min_passing_score,
    )


def init_revisor(app: FastAPI, settings: Settings) -> None:
    """Carga el modelo al iniciar; si falla, el servicio arranca en modo
    degradado y el endpoint responde 503 con el motivo."""
    app.state.revisor = None
    app.state.revisor_error = None
    try:
        app.state.revisor = build_revisor(settings)
    except Exception as error:  # noqa: BLE001 - se reporta en /health
        logger.exception("No se pudo cargar el modelo")
        app.state.revisor_error = str(error)


def get_revisor(request: Request) -> RevisorFormato:
    revisor = getattr(request.app.state, "revisor", None)
    if revisor is None:
        reason = getattr(request.app.state, "revisor_error", None)
        raise ModelUnavailableError(reason or "Modelo no cargado")
    return revisor
