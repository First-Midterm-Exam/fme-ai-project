"""Extraccion de palabras y cajas mediante OCR."""

import os
from pathlib import Path
from typing import Protocol

import pytesseract
from PIL import Image

from app.domain.exceptions import OcrUnavailableError
from app.domain.models import Word


class OcrEngine(Protocol):
    def extract(self, image: Image.Image) -> list[Word]:
        """Devuelve las palabras en orden de lectura."""


def normalize_box(
    left: int, top: int, right: int, bottom: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Escala una caja en pixeles al rango 0-1000 usado por LayoutLMv3."""

    def scale(value: int, size: int) -> int:
        return int(max(0, min(1000, 1000 * value / size)))

    return (
        scale(left, width),
        scale(top, height),
        scale(right, width),
        scale(bottom, height),
    )


class TesseractOcr:
    """Motor OCR basado en Tesseract (mismo que usa LayoutLMv3)."""

    def __init__(
        self,
        languages: str,
        tesseract_cmd: str | None = None,
        tessdata_dir: Path | None = None,
    ):
        self._languages = languages
        # Carpeta de idiomas propia (no requiere permisos de administrador).
        # Se usa la variable de entorno porque ``--tessdata-dir`` pierde las
        # comillas en Windows y falla con rutas que contienen espacios.
        if tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, image: Image.Image) -> list[Word]:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._languages,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as error:
            raise OcrUnavailableError(
                "Tesseract no esta instalado o no esta en el PATH"
            ) from error
        except pytesseract.TesseractError as error:
            raise OcrUnavailableError(
                f"Tesseract fallo: {error.message}"
            ) from error

        width, height = image.size
        words = []
        for index, text in enumerate(data["text"]):
            text = text.strip()
            if not text or float(data["conf"][index]) < 0:
                continue
            left, top = data["left"][index], data["top"][index]
            box = normalize_box(
                left,
                top,
                left + data["width"][index],
                top + data["height"][index],
                width,
                height,
            )
            words.append(Word(text=text, box=box))
        return words
