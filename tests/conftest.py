import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_format_reviewer
from app.core.config import Settings
from app.main import create_app
from app.services.format_reviewer import FormatReviewer
from tests.fakes import COMPLETE_ENTITIES, FakeExtractor, FakeOcr, make_words


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def complete_reviewer() -> FormatReviewer:
    return FormatReviewer(
        ocr=FakeOcr(make_words("Plan", "de", "Proyecto", "Responsable:")),
        extractor=FakeExtractor(COMPLETE_ENTITIES),
        min_passing_score=60,
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(model_dir=tmp_path / "sin-modelo", api_key=None)


@pytest.fixture
def client_factory(settings):
    def build(reviewer=None, **overrides) -> TestClient:
        app = create_app(settings.model_copy(update=overrides))
        if reviewer is not None:
            app.dependency_overrides[get_format_reviewer] = lambda: reviewer
        return TestClient(app)

    return build
