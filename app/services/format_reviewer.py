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
from app.domain.ports import EntityExtractor, OcrEngine
from app.services.document_types import detect_document_type
from app.services.rules import DEFAULT_RULES, Rule, RuleContext

MAX_SCORE = 100
PENALTIES = {
    Severity.HIGH: 40,
    Severity.MEDIUM: 20,
    Severity.LOW: 10,
}


class FormatReviewer:
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
        analysis = self._analyze(self._open_image(content))
        detected_type = detect_document_type(analysis)
        findings = self._evaluate_rules(RuleContext(analysis, detected_type))
        score = self._score(findings)
        return ReviewResult(
            complies=self._complies(score, findings),
            score=score,
            detected_type=detected_type,
            summary=self._summary(analysis, detected_type, findings),
            findings=findings,
        )

    def _analyze(self, image: Image.Image) -> DocumentAnalysis:
        words = self._ocr.extract(image)
        entities = self._extractor.extract(image, words)
        return DocumentAnalysis(tuple(words), tuple(entities))

    def _evaluate_rules(self, context: RuleContext) -> list[Finding]:
        findings = (rule(context) for rule in self._rules)
        return [finding for finding in findings if finding is not None]

    def _complies(self, score: int, findings: list[Finding]) -> bool:
        has_high = any(f.severity is Severity.HIGH for f in findings)
        return score >= self._min_passing_score and not has_high

    @staticmethod
    def _open_image(content: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageError(
                "El archivo no es una imagen válida"
            ) from error
        return image.convert("RGB")

    @staticmethod
    def _score(findings: list[Finding]) -> int:
        penalty = sum(PENALTIES[finding.severity] for finding in findings)
        return max(0, MAX_SCORE - penalty)

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
