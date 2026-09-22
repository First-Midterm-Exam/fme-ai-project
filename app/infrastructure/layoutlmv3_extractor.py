import json
import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    BatchEncoding,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Tokenizer,
)

from app.domain.bio import LabelPrediction, decode_entities
from app.domain.models import Entity, Word
from app.infrastructure.model_store import prepare_model_dir

logger = logging.getLogger(__name__)

METADATA_FILE = "fme_metadata.json"
DEFAULT_MAX_LENGTH = 512
DEFAULT_STRIDE = 128
OUTSIDE_LABEL = "O"


class LayoutLMv3EntityExtractor:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        model_dir = prepare_model_dir(model_dir)
        metadata = self._read_metadata(model_dir)
        self._max_length = metadata.get("max_length", DEFAULT_MAX_LENGTH)
        self._stride = metadata.get("stride", DEFAULT_STRIDE)
        self._device = torch.device(device)
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
        return decode_entities(words, self._predict_words(image, words))

    def _predict_words(
        self, image: Image.Image, words: list[Word]
    ) -> list[LabelPrediction]:
        encoding = self._encode_words(words)
        scores, label_ids = self._infer(image, encoding)

        best: dict[int, LabelPrediction] = {}
        for window in range(scores.shape[0]):
            for position, word_id in self._first_subtokens(encoding, window):
                score = scores[window, position].item()
                if word_id not in best or score > best[word_id][1]:
                    label = self._id2label[label_ids[window, position].item()]
                    best[word_id] = (label, score)
        return [
            best.get(index, (OUTSIDE_LABEL, 0.0))
            for index in range(len(words))
        ]

    def _encode_words(self, words: list[Word]) -> BatchEncoding:
        return self._tokenizer(
            [word.text for word in words],
            boxes=[list(word.box) for word in words],
            truncation=True,
            padding="max_length",
            max_length=self._max_length,
            stride=self._stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )

    def _infer(
        self, image: Image.Image, encoding: BatchEncoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return logits.softmax(dim=-1).max(dim=-1)

    @staticmethod
    def _first_subtokens(
        encoding: BatchEncoding, window: int
    ) -> list[tuple[int, int]]:
        positions = []
        previous = None
        for position, word_id in enumerate(encoding.word_ids(window)):
            if word_id is not None and word_id != previous:
                positions.append((position, word_id))
            previous = word_id
        return positions
