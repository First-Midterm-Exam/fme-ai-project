import pytest

from app.domain.bio import decode_entities
from app.domain.models import EntityType
from tests.fakes import make_words


def test_decode_entities_merges_bio_sequences():
    words = make_words("Fecha", "de", "entrega:", "12/05", "nota")
    predictions = [
        ("B-QUESTION", 0.9),
        ("I-QUESTION", 0.8),
        ("I-QUESTION", 0.7),
        ("B-ANSWER", 0.6),
        ("O", 0.99),
    ]

    entities = decode_entities(words, predictions)

    assert [(e.type, e.text) for e in entities] == [
        (EntityType.QUESTION, "Fecha de entrega:"),
        (EntityType.ANSWER, "12/05"),
    ]
    assert entities[0].confidence == pytest.approx(0.8)


def test_decode_entities_starts_new_entity_on_type_change():
    entities = decode_entities(
        make_words("A", "B"), [("I-HEADER", 0.9), ("I-ANSWER", 0.9)]
    )

    assert [e.type for e in entities] == [
        EntityType.HEADER,
        EntityType.ANSWER,
    ]


def test_decode_entities_ignores_unknown_labels():
    entities = decode_entities(make_words("A"), [("B-OTHER", 0.9)])

    assert entities == []
