"""Caso de uso: revisar el formato de un documento a partir de su imagen."""

import io

from PIL import Image, UnidentifiedImageError

from app.domain.exceptions import InvalidImageError
from app.domain.models import (
    DocumentAnalysis,
    EntityType,
    Finding,
    ReviewResult,
    Severity,
)
from app.services.document_types import detect_document_type
from app.services.entity_extractor import EntityExtractor
from app.services.ocr import OcrEngine
from app.services.rules import DEFAULT_RULES, Rule, RuleContext

PENALTIES = {
    Severity.HIGH: 40,
    Severity.MEDIUM: 20,
    Severity.LOW: 10,
}


class RevisorFormato:
    """Orquesta OCR, modelo y reglas; no conoce detalles de HTTP."""

    def __init__(
        self,
        ocr: OcrEngine,
        extractor: EntityExtractor,
        min_passing_score: int,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
    ):
        self._ocr = ocr
        self._extractor = extractor
        self._min_passing_score = min_passing_score
        self._rules = rules

    def review(self, content: bytes) -> ReviewResult:
        image = self._open_image(content)
        words = self._ocr.extract(image)
        entities = self._extractor.extract(image, words)
        analysis = DocumentAnalysis(tuple(words), tuple(entities))

        detected_type = detect_document_type(analysis)
        context = RuleContext(analysis, detected_type)
        findings = [
            finding
            for finding in (rule(context) for rule in self._rules)
            if finding is not None
        ]
        score = self._score(findings)
        complies = score >= self._min_passing_score and not any(
            finding.severity is Severity.HIGH for finding in findings
        )
        return ReviewResult(
            complies=complies,
            score=score,
            detected_type=detected_type,
            summary=self._summary(analysis, detected_type, findings),
            findings=findings,
        )

    @staticmethod
    def _open_image(content: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError(
                "El archivo no es una imagen valida"
            ) from error
        return image.convert("RGB")

    @staticmethod
    def _score(findings: list[Finding]) -> int:
        penalty = sum(PENALTIES[finding.severity] for finding in findings)
        return max(0, 100 - penalty)

    @staticmethod
    def _summary(
        analysis: DocumentAnalysis,
        detected_type: str,
        findings: list[Finding],
    ) -> str:
        observations = (
            f"{len(findings)} observación(es)." if findings
            else "Sin observaciones."
        )
        return (
            f"Tipo de documento: {detected_type}. "
            f"Se identificaron {analysis.count(EntityType.HEADER)} "
            f"encabezado(s), {analysis.count(EntityType.QUESTION)} "
            f"campo(s) y {analysis.count(EntityType.ANSWER)} "
            f"respuesta(s). {observations}"
        )
