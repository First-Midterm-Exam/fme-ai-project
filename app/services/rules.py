from collections.abc import Callable
from dataclasses import dataclass

from app.domain.models import (
    DocumentAnalysis,
    EntityType,
    Finding,
    Severity,
)
from app.services.document_types import UNKNOWN_TYPE

LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class RuleContext:
    analysis: DocumentAnalysis
    detected_type: str


Rule = Callable[[RuleContext], Finding | None]


def require_text(context: RuleContext) -> Finding | None:
    if context.analysis.words:
        return None
    return Finding(
        Severity.HIGH,
        "texto",
        "No se detectó texto legible en la imagen.",
    )


def require_header(context: RuleContext) -> Finding | None:
    if context.analysis.count(EntityType.HEADER):
        return None
    return Finding(
        Severity.MEDIUM,
        "encabezado",
        "No se identificó un encabezado o título del documento.",
    )


def require_fields(context: RuleContext) -> Finding | None:
    if context.analysis.count(EntityType.QUESTION):
        return None
    return Finding(
        Severity.MEDIUM,
        "campos",
        "No se identificaron campos o secciones rotuladas.",
    )


def check_answered_fields(context: RuleContext) -> Finding | None:
    questions = context.analysis.count(EntityType.QUESTION)
    answers = context.analysis.count(EntityType.ANSWER)
    if answers >= questions:
        return None
    return Finding(
        Severity.LOW,
        "respuestas",
        f"Se detectaron {questions} campos y solo {answers} respuestas; "
        "algunos campos podrían estar vacíos.",
    )


def require_known_type(context: RuleContext) -> Finding | None:
    if context.detected_type != UNKNOWN_TYPE:
        return None
    return Finding(
        Severity.LOW,
        "tipo_documento",
        "No se reconoció el tipo de producto de trabajo CMMI.",
    )


def check_confidence(context: RuleContext) -> Finding | None:
    analysis = context.analysis
    if not analysis.entities:
        return None
    if analysis.mean_confidence >= LOW_CONFIDENCE_THRESHOLD:
        return None
    return Finding(
        Severity.LOW,
        "calidad",
        "La confianza del análisis es baja; verifique la nitidez "
        "de la imagen.",
    )


DEFAULT_RULES: tuple[Rule, ...] = (
    require_text,
    require_header,
    require_fields,
    check_answered_fields,
    require_known_type,
    check_confidence,
)
