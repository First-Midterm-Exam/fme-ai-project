import pytest

from app.domain.exceptions import InvalidImageError
from app.domain.models import Entity, EntityType, Severity
from app.services.document_types import UNKNOWN_TYPE
from app.services.format_reviewer import FormatReviewer
from tests.fakes import FakeExtractor, FakeOcr, make_words


def test_complete_document_complies(complete_reviewer, png_bytes):
    result = complete_reviewer.review(png_bytes)

    assert result.complies is True
    assert result.score == 100
    assert result.detected_type == "Plan de proyecto"
    assert result.findings == []


def test_empty_document_fails_with_high_finding(png_bytes):
    reviewer = FormatReviewer(FakeOcr([]), FakeExtractor([]), 60)

    result = reviewer.review(png_bytes)

    assert result.complies is False
    assert result.detected_type == UNKNOWN_TYPE
    assert Severity.HIGH in {f.severity for f in result.findings}


def test_unanswered_fields_are_reported(png_bytes):
    entities = [
        Entity(EntityType.HEADER, "Acta de reunion", 0.9),
        Entity(EntityType.QUESTION, "Asistentes:", 0.9),
        Entity(EntityType.QUESTION, "Acuerdos:", 0.9),
    ]
    reviewer = FormatReviewer(
        FakeOcr(make_words("Acta", "de", "reunion")),
        FakeExtractor(entities),
        60,
    )

    result = reviewer.review(png_bytes)

    assert result.complies is True
    assert result.score == 90
    assert [f.element for f in result.findings] == ["respuestas"]


def test_invalid_image_raises(complete_reviewer):
    with pytest.raises(InvalidImageError):
        complete_reviewer.review(b"no es una imagen")
