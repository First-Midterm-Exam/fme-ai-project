from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
    log_level: str = "INFO"

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
    def empty_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
