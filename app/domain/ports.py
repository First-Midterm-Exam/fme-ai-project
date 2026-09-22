from typing import Protocol

from PIL import Image

from app.domain.models import Entity, Word


class OcrEngine(Protocol):
    def extract(self, image: Image.Image) -> list[Word]:
        ...


class EntityExtractor(Protocol):
    def extract(self, image: Image.Image, words: list[Word]) -> list[Entity]:
        ...
