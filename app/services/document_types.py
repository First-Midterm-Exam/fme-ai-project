"""Catalogo de productos de trabajo CMMI y deteccion del tipo."""

import unicodedata
from dataclasses import dataclass

from app.domain.models import DocumentAnalysis, EntityType

UNKNOWN_TYPE = "No identificado"
HEADER_WEIGHT = 2


@dataclass(frozen=True)
class DocumentType:
    name: str
    keywords: tuple[str, ...]


# Para soportar un nuevo tipo basta con agregarlo a esta lista.
CMMI_DOCUMENT_TYPES = (
    DocumentType(
        "Plan de proyecto",
        ("plan de proyecto", "project plan", "cronograma"),
    ),
    DocumentType(
        "Especificación de requisitos",
        ("especificacion de requisitos", "requisito", "requirement"),
    ),
    DocumentType(
        "Plan de pruebas",
        ("plan de pruebas", "caso de prueba", "test plan", "test case"),
    ),
    DocumentType(
        "Acta de reunión",
        ("acta", "minuta", "reunion", "meeting minutes"),
    ),
    DocumentType(
        "Gestión de configuración",
        ("gestion de configuracion", "linea base", "configuration"),
    ),
    DocumentType(
        "Registro de riesgos",
        ("riesgo", "risk"),
    ),
    DocumentType(
        "Solicitud de cambio",
        ("solicitud de cambio", "change request"),
    ),
    DocumentType(
        "Aseguramiento de calidad",
        ("aseguramiento de calidad", "quality assurance", "auditoria"),
    ),
    DocumentType(
        "Documento de diseño",
        ("documento de diseno", "arquitectura", "design"),
    ),
    DocumentType(
        "Informe de seguimiento",
        ("seguimiento", "informe de estado", "status report"),
    ),
)


def normalize_text(text: str) -> str:
    """Minusculas y sin tildes para comparar palabras clave."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def detect_document_type(
    analysis: DocumentAnalysis,
    catalog: tuple[DocumentType, ...] = CMMI_DOCUMENT_TYPES,
) -> str:
    """Elige el tipo con mas coincidencias; los encabezados pesan doble."""
    full_text = normalize_text(analysis.text)
    headers = normalize_text(" ".join(
        entity.text for entity in analysis.entities_of(EntityType.HEADER)
    ))

    best_name, best_score = UNKNOWN_TYPE, 0
    for document_type in catalog:
        score = sum(
            full_text.count(keyword) + HEADER_WEIGHT * headers.count(keyword)
            for keyword in document_type.keywords
        )
        if score > best_score:
            best_name, best_score = document_type.name, score
    return best_name
