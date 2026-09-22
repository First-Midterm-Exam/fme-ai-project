from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import documents, health
from app.container import init_app_state
from app.core.config import Settings
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_app_state(app, settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    @app.get("/", include_in_schema=False)
    def root() -> str:
        return (
            "FastAPI service is running correctly - "
            f"Version {settings.app_version}"
        )

    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_prefix)
    return app


app = create_app()
