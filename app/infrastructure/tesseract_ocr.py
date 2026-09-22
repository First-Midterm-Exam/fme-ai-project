import os
from pathlib import Path

import pytesseract
from PIL import Image

from app.domain.exceptions import OcrUnavailableError
from app.domain.models import Word

BOX_SCALE = 1000


def normalize_box(
    left: int, top: int, right: int, bottom: int, width: int, height: int
) -> tuple[int, int, int, int]:
    def scale(value: int, size: int) -> int:
        return int(max(0, min(BOX_SCALE, BOX_SCALE * value / size)))

    return (
        scale(left, width),
        scale(top, height),
        scale(right, width),
        scale(bottom, height),
    )


class TesseractOcr:
    def __init__(
        self,
        languages: str,
        tesseract_cmd: str | None = None,
        tessdata_dir: Path | None = None,
    ):
        self._languages = languages
        if tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve())
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, image: Image.Image) -> list[Word]:
        data = self._read_data(image)
        width, height = image.size
        words = []
        for index, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
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

    def _read_data(self, image: Image.Image) -> dict[str, list]:
        try:
            return pytesseract.image_to_data(
                image,
                lang=self._languages,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as error:
            raise OcrUnavailableError(
                "Tesseract no está instalado o no está en el PATH"
            ) from error
        except pytesseract.TesseractError as error:
            raise OcrUnavailableError(
                f"Tesseract falló: {error.message}"
            ) from error
