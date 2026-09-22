"""Deteccion de entidades con el modelo LayoutLMv3 exportado desde Colab."""

import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Tokenizer,
)

from app.domain.exceptions import ModelUnavailableError
from app.domain.models import Entity, EntityType, Word

logger = logging.getLogger(__name__)

METADATA_FILE = "fme_metadata.json"
SOURCE_MARKER = ".source_archive"
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128

WordPrediction = tuple[str, float]


class EntityExtractor(Protocol):
    def extract(self, image: Image.Image, words: list[Word]) -> list[Entity]:
        """Agrupa las palabras en entidades del formulario."""


def resolve_model_dir(model_dir: Path) -> Path:
    """Devuelve la carpeta del modelo lista para cargar.

    Si existe ``<model_dir>.zip`` (el archivo exportado por Colab) y es
    distinto del que se extrajo la ultima vez, reemplaza la carpeta con
    su contenido. Asi, actualizar el modelo consiste solo en copiar el
    nuevo ``.zip`` en ``models/`` y reiniciar el servicio.
    """
    archive = model_dir.with_suffix(".zip")
    if archive.is_file() and _archive_changed(archive, model_dir):
        _extract_archive(archive, model_dir)

    if (model_dir / "config.json").is_file():
        return model_dir
    raise ModelUnavailableError(
        f"No se encontro el modelo en '{model_dir}' ni '{archive}'"
    )


def _archive_signature(archive: Path) -> str:
    stat = archive.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _archive_changed(archive: Path, model_dir: Path) -> bool:
    marker = model_dir / SOURCE_MARKER
    if not marker.is_file():
        return True
    return marker.read_text(encoding="utf-8") != _archive_signature(archive)


def _extract_archive(archive: Path, model_dir: Path) -> None:
    """Extrae en una carpeta temporal y solo entonces reemplaza el modelo,
    para que un ``.zip`` invalido no deje al servicio sin modelo."""
    logger.info("Instalando modelo desde %s", archive)
    staging = model_dir.with_name(f"{model_dir.name}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    except zipfile.BadZipFile as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise ModelUnavailableError(
            f"'{archive}' no es un archivo zip valido"
        ) from error

    root = _find_model_root(staging)
    if root is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise ModelUnavailableError(f"'{archive}' no contiene config.json")

    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.move(root, model_dir)
    shutil.rmtree(staging, ignore_errors=True)
    (model_dir / SOURCE_MARKER).write_text(
        _archive_signature(archive), encoding="utf-8"
    )


def _find_model_root(directory: Path) -> Path | None:
    """Admite zips con los archivos en la raiz o dentro de una carpeta."""
    for config in sorted(directory.rglob("config.json")):
        if "__MACOSX" not in config.parts:
            return config.parent
    return None


def group_entities(
    words: list[Word], predictions: list[WordPrediction]
) -> list[Entity]:
    """Agrupa palabras consecutivas con etiquetas BIO en entidades."""
    entities = []
    current_type, texts, scores = None, [], []

    def close():
        if current_type is not None:
            entities.append(Entity(
                type=current_type,
                text=" ".join(texts),
                confidence=round(sum(scores) / len(scores), 4),
            ))

    for word, (label, score) in zip(words, predictions):
        prefix, _, tag = label.partition("-")
        entity_type = EntityType(tag) if tag in EntityType else None
        continues = prefix == "I" and entity_type is current_type
        if entity_type is None or not continues:
            close()
            current_type, texts, scores = entity_type, [], []
        if entity_type is not None:
            texts.append(word.text)
            scores.append(score)
    close()
    return entities


class LayoutLMv3EntityExtractor:
    """Ejecuta el modelo por ventanas para soportar documentos largos."""

    def __init__(self, model_dir: Path, device: str = "cpu"):
        model_dir = resolve_model_dir(model_dir)
        metadata = self._read_metadata(model_dir)
        self._max_length = metadata.get("max_length", DEFAULT_MAX_LENGTH)
        self._stride = metadata.get("stride", DEFAULT_STRIDE)
        self._device = torch.device(device)
        # Clases explicitas: algunos modelos publicados declaran un
        # tokenizador RoBERTa que no procesa cajas; el vocabulario es el
        # mismo, por lo que se fuerza el tokenizador de LayoutLMv3.
        # ``local_files_only`` garantiza que nunca se use internet.
        self._tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            model_dir, local_files_only=True
        )
        self._image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            model_dir, apply_ocr=False, local_files_only=True
        )
        self._model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_dir, local_files_only=True
        ).to(self._device)
        self._model.eval()
        self._id2label = self._model.config.id2label
        logger.info("Modelo cargado desde %s", model_dir)

    @staticmethod
    def _read_metadata(model_dir: Path) -> dict:
        path = model_dir / METADATA_FILE
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def extract(self, image: Image.Image, words: list[Word]) -> list[Entity]:
        if not words:
            return []
        return group_entities(words, self._predict_words(image, words))

    def _predict_words(
        self, image: Image.Image, words: list[Word]
    ) -> list[WordPrediction]:
        encoding = self._tokenizer(
            [word.text for word in words],
            boxes=[list(word.box) for word in words],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        windows = encoding["input_ids"].shape[0]
        pixel_values = self._image_processor(
            image.convert("RGB"), return_tensors="pt"
        )["pixel_values"].expand(windows, -1, -1, -1)

        with torch.inference_mode():
            logits = self._model(
                input_ids=encoding["input_ids"].to(self._device),
                attention_mask=encoding["attention_mask"].to(self._device),
                bbox=encoding["bbox"].to(self._device),
                pixel_values=pixel_values.to(self._device),
            ).logits
        scores, label_ids = logits.softmax(dim=-1).max(dim=-1)

        # Cada palabra se evalua en su primer sub-token; si aparece en
        # varias ventanas se conserva la prediccion mas confiable.
        best: dict[int, WordPrediction] = {}
        for window in range(windows):
            previous = None
            for position, word_id in enumerate(encoding.word_ids(window)):
                if word_id is None or word_id == previous:
                    continue
                previous = word_id
                score = scores[window, position].item()
                if word_id not in best or score > best[word_id][1]:
                    label_id = label_ids[window, position].item()
                    best[word_id] = (self._id2label[label_id], score)
        return [best.get(index, ("O", 0.0)) for index in range(len(words))]
