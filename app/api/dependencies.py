from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings
from app.domain.exceptions import ModelUnavailableError
from app.services.format_reviewer import FormatReviewer


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_format_reviewer(request: Request) -> FormatReviewer:
    reviewer = request.app.state.reviewer
    if reviewer is None:
        reason = request.app.state.reviewer_error or "Modelo no cargado"
        raise ModelUnavailableError(reason)
    return reviewer


SettingsDep = Annotated[Settings, Depends(get_settings)]
FormatReviewerDep = Annotated[FormatReviewer, Depends(get_format_reviewer)]
