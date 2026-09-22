import pytest

from app.domain.exceptions import InvalidImageError
from app.domain.models import Entity, EntityType, Severity
from app.services.document_types import UNKNOWN_TYPE
from app.services.entity_extractor import group_entities
from app.services.revisor_formato import RevisorFormato
from tests.conftest import FakeExtractor, FakeOcr, make_words


def test_group_entities_merges_bio_sequences():
    words = make_words("Fecha", "de", "entrega:", "12/05", "nota")
    predictions = [
        ("B-QUESTION", 0.9),
        ("I-QUESTION", 0.8),
        ("I-QUESTION", 0.7),
        ("B-ANSWER", 0.6),
        ("O", 0.99),
    ]

    entities = group_entities(words, predictions)

    assert [(e.type, e.text) for e in entities] == [
        (EntityType.QUESTION, "Fecha de entrega:"),
        (EntityType.ANSWER, "12/05"),
    ]
    assert entities[0].confidence == pytest.approx(0.8)


def test_group_entities_starts_new_entity_on_type_change():
    words = make_words("A", "B")
    entities = group_entities(
        words, [("I-HEADER", 0.9), ("I-ANSWER", 0.9)]
    )
    assert [e.type for e in entities] == [
        EntityType.HEADER,
        EntityType.ANSWER,
    ]


def test_complete_document_complies(complete_revisor, png_bytes):
    result = complete_revisor.review(png_bytes)

    assert result.complies is True
    assert result.score == 100
    assert result.detected_type == "Plan de proyecto"
    assert result.findings == []


def test_empty_document_fails_with_high_finding(png_bytes):
    revisor = RevisorFormato(FakeOcr([]), FakeExtractor([]), 60)

    result = revisor.review(png_bytes)

    assert result.complies is False
    assert result.detected_type == UNKNOWN_TYPE
    assert Severity.HIGH in {f.severity for f in result.findings}


def test_unanswered_fields_are_reported(png_bytes):
    entities = [
        Entity(EntityType.HEADER, "Acta de reunion", 0.9),
        Entity(EntityType.QUESTION, "Asistentes:", 0.9),
        Entity(EntityType.QUESTION, "Acuerdos:", 0.9),
    ]
    revisor = RevisorFormato(
        FakeOcr(make_words("Acta", "de", "reunion")),
        FakeExtractor(entities),
        60,
    )

    result = revisor.review(png_bytes)

    assert result.complies is True
    assert result.score == 90
    assert [f.element for f in result.findings] == ["respuestas"]


def test_invalid_image_raises(complete_revisor):
    with pytest.raises(InvalidImageError):
        complete_revisor.review(b"no es una imagen")
