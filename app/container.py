import logging

from fastapi import FastAPI

from app.core.config import Settings
from app.infrastructure.layoutlmv3_extractor import LayoutLMv3EntityExtractor
from app.infrastructure.tesseract_ocr import TesseractOcr
from app.services.format_reviewer import FormatReviewer

logger = logging.getLogger(__name__)


def build_format_reviewer(settings: Settings) -> FormatReviewer:
    return FormatReviewer(
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


def init_app_state(app: FastAPI, settings: Settings) -> None:
    app.state.settings = settings
    app.state.reviewer = None
    app.state.reviewer_error = None
    try:
        app.state.reviewer = build_format_reviewer(settings)
    except Exception as error:
        logger.exception("No se pudo inicializar el revisor de formato")
        app.state.reviewer_error = str(error)
