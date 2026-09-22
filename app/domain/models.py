from dataclasses import dataclass, field
from enum import StrEnum


class EntityType(StrEnum):
    HEADER = "HEADER"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"


class Severity(StrEnum):
    HIGH = "alta"
    MEDIUM = "media"
    LOW = "baja"


@dataclass(frozen=True)
class Word:
    text: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Entity:
    type: EntityType
    text: str
    confidence: float


@dataclass(frozen=True)
class DocumentAnalysis:
    words: tuple[Word, ...]
    entities: tuple[Entity, ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    def entities_of(self, entity_type: EntityType) -> list[Entity]:
        return [e for e in self.entities if e.type is entity_type]

    def count(self, entity_type: EntityType) -> int:
        return len(self.entities_of(entity_type))

    @property
    def mean_confidence(self) -> float:
        if not self.entities:
            return 0.0
        total = sum(entity.confidence for entity in self.entities)
        return total / len(self.entities)


@dataclass(frozen=True)
class Finding:
    severity: Severity
    element: str
    message: str


@dataclass(frozen=True)
class ReviewResult:
    complies: bool
    score: int
    detected_type: str
    summary: str
    findings: list[Finding] = field(default_factory=list)
