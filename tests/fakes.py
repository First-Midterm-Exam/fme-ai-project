from app.domain.models import Entity, EntityType, Word


class FakeOcr:
    def __init__(self, words: list[Word]):
        self._words = words

    def extract(self, image) -> list[Word]:
        return list(self._words)


class FakeExtractor:
    def __init__(self, entities: list[Entity]):
        self._entities = entities

    def extract(self, image, words: list[Word]) -> list[Entity]:
        return list(self._entities)


def make_words(*texts: str) -> list[Word]:
    return [Word(text, (0, 0, 10, 10)) for text in texts]


COMPLETE_ENTITIES = [
    Entity(EntityType.HEADER, "Plan de Proyecto", 0.95),
    Entity(EntityType.QUESTION, "Responsable:", 0.9),
    Entity(EntityType.ANSWER, "Ana", 0.9),
]
