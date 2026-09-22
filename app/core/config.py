"""Configuracion de la aplicacion leida desde variables de entorno."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parametros del servicio; cada campo se sobreescribe desde ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "FME AI Service"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    api_key: SecretStr | None = None

    model_dir: Path = Path("models/fme-layoutlmv3-funsd")
    model_device: str = "cpu"

    ocr_languages: str = "spa+eng"
    tesseract_cmd: str | None = None
    tessdata_dir: Path | None = None

    max_upload_mb: int = 10
    min_passing_score: int = 60

    @field_validator(
        "api_key", "tesseract_cmd", "tessdata_dir", mode="before"
    )
    @classmethod
    def empty_as_none(cls, value):
        """Una variable vacia en ``.env`` equivale a no configurarla."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
