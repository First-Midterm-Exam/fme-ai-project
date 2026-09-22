"""Punto de entrada de la aplicacion FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import init_revisor
from app.api.routes import documentos, health
from app.core.config import Settings, get_settings
from app.domain.exceptions import (
    InvalidImageError,
    ModelUnavailableError,
    OcrUnavailableError,
    RevisionError,
)

ERROR_STATUS = {
    InvalidImageError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    OcrUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_revisor(app, settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    @app.exception_handler(RevisionError)
    async def handle_revision_error(
        request: Request, error: RevisionError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=ERROR_STATUS.get(
                type(error), status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            content={"detail": str(error)},
        )

    @app.get("/", include_in_schema=False)
    def root() -> str:
        return (
            "FastAPI service is running correctly - "
            f"Version {settings.app_version}"
        )

    app.include_router(health.router)
    app.include_router(documentos.router, prefix=settings.api_prefix)
    return app


app = create_app()
