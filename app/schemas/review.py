from typing import Literal, Self

from pydantic import BaseModel, Field

from app.domain.models import Finding, ReviewResult


class HallazgoSchema(BaseModel):
    severidad: Literal["alta", "media", "baja"]
    elemento: str
    mensaje: str

    @classmethod
    def from_finding(cls, finding: Finding) -> Self:
        return cls(
            severidad=finding.severity.value,
            elemento=finding.element,
            mensaje=finding.message,
        )


class RevisionFormatoResponse(BaseModel):
    cumple: bool
    puntaje: int = Field(ge=0, le=100)
    tipo_detectado: str
    resumen: str
    hallazgos: list[HallazgoSchema]

    @classmethod
    def from_result(cls, result: ReviewResult) -> Self:
        return cls(
            cumple=result.complies,
            puntaje=result.score,
            tipo_detectado=result.detected_type,
            resumen=result.summary,
            hallazgos=[
                HallazgoSchema.from_finding(finding)
                for finding in result.findings
            ],
        )
