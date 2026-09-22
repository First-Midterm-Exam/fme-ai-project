from app.domain.models import Entity, EntityType, Word

LabelPrediction = tuple[str, float]


def _parse_label(label: str) -> tuple[str, EntityType | None]:
    prefix, _, tag = label.partition("-")
    entity_type = EntityType(tag) if tag in EntityType else None
    return prefix, entity_type


def _build_entity(
    entity_type: EntityType, texts: list[str], scores: list[float]
) -> Entity:
    return Entity(
        type=entity_type,
        text=" ".join(texts),
        confidence=round(sum(scores) / len(scores), 4),
    )


def decode_entities(
    words: list[Word], predictions: list[LabelPrediction]
) -> list[Entity]:
    entities: list[Entity] = []
    current_type: EntityType | None = None
    texts: list[str] = []
    scores: list[float] = []

    for word, (label, score) in zip(words, predictions):
        prefix, entity_type = _parse_label(label)
        continues = prefix == "I" and entity_type is current_type
        if entity_type is None or not continues:
            if current_type is not None:
                entities.append(_build_entity(current_type, texts, scores))
            current_type, texts, scores = entity_type, [], []
        if entity_type is not None:
            texts.append(word.text)
            scores.append(score)

    if current_type is not None:
        entities.append(_build_entity(current_type, texts, scores))
    return entities
