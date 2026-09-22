import zipfile

import pytest

from app.domain.exceptions import ModelUnavailableError
from app.infrastructure.model_store import SOURCE_MARKER, prepare_model_dir


def write_zip(path, files):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)


def test_uses_existing_folder_without_zip(tmp_path):
    model_dir = tmp_path / "modelo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    assert prepare_model_dir(model_dir) == model_dir


def test_new_zip_replaces_existing_model(tmp_path):
    model_dir = tmp_path / "modelo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("viejo")
    (model_dir / "obsoleto.bin").write_text("x")
    write_zip(tmp_path / "modelo.zip", {"config.json": "nuevo"})

    prepare_model_dir(model_dir)

    assert (model_dir / "config.json").read_text() == "nuevo"
    assert not (model_dir / "obsoleto.bin").exists()
    assert (model_dir / SOURCE_MARKER).is_file()


def test_same_zip_is_not_extracted_twice(tmp_path):
    model_dir = tmp_path / "modelo"
    write_zip(tmp_path / "modelo.zip", {"config.json": "v1"})
    prepare_model_dir(model_dir)
    (model_dir / "config.json").write_text("sin tocar")

    prepare_model_dir(model_dir)

    assert (model_dir / "config.json").read_text() == "sin tocar"


def test_zip_with_nested_folder_is_supported(tmp_path):
    model_dir = tmp_path / "modelo"
    write_zip(tmp_path / "modelo.zip", {"export/config.json": "{}"})

    prepare_model_dir(model_dir)

    assert (model_dir / "config.json").is_file()


def test_invalid_zip_keeps_current_model(tmp_path):
    model_dir = tmp_path / "modelo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("actual")
    (tmp_path / "modelo.zip").write_text("no es zip")

    with pytest.raises(ModelUnavailableError):
        prepare_model_dir(model_dir)

    assert (model_dir / "config.json").read_text() == "actual"
    assert not (tmp_path / "modelo.tmp").exists()


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        prepare_model_dir(tmp_path / "modelo")
