import logging
import shutil
import zipfile
from pathlib import Path

from app.domain.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

MODEL_CONFIG_FILE = "config.json"
SOURCE_MARKER = ".source_archive"
IGNORED_ARCHIVE_DIRS = ("__MACOSX",)


def prepare_model_dir(model_dir: Path) -> Path:
    archive = model_dir.with_suffix(".zip")
    if archive.is_file() and _archive_changed(archive, model_dir):
        _install_archive(archive, model_dir)

    if (model_dir / MODEL_CONFIG_FILE).is_file():
        return model_dir
    raise ModelUnavailableError(
        f"No se encontró el modelo en '{model_dir}' ni '{archive}'"
    )


def _archive_signature(archive: Path) -> str:
    stat = archive.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _archive_changed(archive: Path, model_dir: Path) -> bool:
    marker = model_dir / SOURCE_MARKER
    if not marker.is_file():
        return True
    return marker.read_text(encoding="utf-8") != _archive_signature(archive)


def _install_archive(archive: Path, model_dir: Path) -> None:
    logger.info("Instalando modelo desde %s", archive)
    staging = model_dir.with_name(f"{model_dir.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        root = _extract_to_staging(archive, staging)
        shutil.rmtree(model_dir, ignore_errors=True)
        shutil.move(root, model_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    (model_dir / SOURCE_MARKER).write_text(
        _archive_signature(archive), encoding="utf-8"
    )


def _extract_to_staging(archive: Path, staging: Path) -> Path:
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    except zipfile.BadZipFile as error:
        raise ModelUnavailableError(
            f"'{archive}' no es un archivo zip válido"
        ) from error

    root = _find_model_root(staging)
    if root is None:
        raise ModelUnavailableError(
            f"'{archive}' no contiene {MODEL_CONFIG_FILE}"
        )
    return root


def _find_model_root(directory: Path) -> Path | None:
    for config in sorted(directory.rglob(MODEL_CONFIG_FILE)):
        if not any(part in IGNORED_ARCHIVE_DIRS for part in config.parts):
            return config.parent
    return None
