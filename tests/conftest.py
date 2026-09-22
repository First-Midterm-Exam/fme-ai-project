import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_revisor
from app.core.config import Settings, get_settings
from app.domain.models import Entity, EntityType, Word
from app.main import create_app
from app.services.revisor_formato import RevisorFormato


class FakeOcr:
    def __init__(self, words):
        self.words = words

    def extract(self, image):
        return list(self.words)


class FakeExtractor:
    def __init__(self, entities):
        self.entities = entities

    def extract(self, image, words):
        return list(self.entities)


def make_words(*texts):
    return [Word(text, (0, 0, 10, 10)) for text in texts]


COMPLETE_ENTITIES = [
    Entity(EntityType.HEADER, "Plan de Proyecto", 0.95),
    Entity(EntityType.QUESTION, "Responsable:", 0.9),
    Entity(EntityType.ANSWER, "Ana", 0.9),
]


@pytest.fixture
def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def complete_revisor():
    return RevisorFormato(
        ocr=FakeOcr(make_words("Plan", "de", "Proyecto", "Responsable:")),
        extractor=FakeExtractor(COMPLETE_ENTITIES),
        min_passing_score=60,
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(model_dir=tmp_path / "sin-modelo", api_key=None)


@pytest.fixture
def client_factory(settings):
    def build(revisor=None, **overrides):
        current = settings.model_copy(update=overrides)
        app = create_app(current)
        app.dependency_overrides[get_settings] = lambda: current
        if revisor is not None:
            app.dependency_overrides[get_revisor] = lambda: revisor
        return TestClient(app)

    return build
