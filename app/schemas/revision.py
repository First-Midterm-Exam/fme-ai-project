"""Contrato JSON consumido por Laravel (RevisionFormatoController)."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.models import ReviewResult


class HallazgoSchema(BaseModel):
    severidad: Literal["alta", "media", "baja"]
    elemento: str
    mensaje: str


class RevisionFormatoResponse(BaseModel):
    cumple: bool
    puntaje: int = Field(ge=0, le=100)
    tipo_detectado: str
    resumen: str
    hallazgos: list[HallazgoSchema]

    @classmethod
    def from_result(cls, result: ReviewResult) -> "RevisionFormatoResponse":
        return cls(
            cumple=result.complies,
            puntaje=result.score,
            tipo_detectado=result.detected_type,
            resumen=result.summary,
            hallazgos=[
                HallazgoSchema(
                    severidad=finding.severity.value,
                    elemento=finding.element,
                    mensaje=finding.message,
                )
                for finding in result.findings
            ],
        )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    detail: str | None = None
