from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    InvalidImageError,
    ModelUnavailableError,
    OcrUnavailableError,
    ReviewError,
)

ERROR_STATUS = {
    InvalidImageError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    OcrUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


async def handle_review_error(
    request: Request, error: ReviewError
) -> JSONResponse:
    return JSONResponse(
        status_code=ERROR_STATUS.get(
            type(error), status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content={"detail": str(error)},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ReviewError, handle_review_error)
