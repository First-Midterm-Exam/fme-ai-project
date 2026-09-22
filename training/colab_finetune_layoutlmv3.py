"""Fine-tuning de LayoutLMv3 sobre FUNSD para Google Colab.

Entrena ``microsoft/layoutlmv3-base`` en clasificacion de tokens
(etiquetado de entidades de formularios: HEADER, QUESTION, ANSWER)
usando el dataset FUNSD y exporta el modelo junto con su procesador
en un archivo ``.zip`` listo para ser consumido por el servicio
FastAPI de este proyecto.

Uso en Colab (ver ``training/README.md``):
    1. Entorno de ejecucion con GPU (T4 es suficiente).
    2. Celda 1: instalar las dependencias indicadas en el README.
    3. Celda 2: pegar este archivo completo y ejecutarlo.
"""

import json
import random
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoProcessor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Config:
    """Parametros del experimento; editar aqui antes de ejecutar."""

    base_model: str = "microsoft/layoutlmv3-base"
    dataset_urls: tuple = (
        # Enlace directo publicado en github.com/crcresearch/FUNSD
        "https://www.crc.nd.edu/~pmoreira/funsd.zip",
        # Fuente original, usada como respaldo
        "https://guillaumejaume.github.io/FUNSD/dataset.zip",
    )
    work_dir: Path = Path("/content/work")
    export_name: str = "fme-layoutlmv3-funsd"
    max_length: int = 512
    stride: int = 128
    validation_docs: int = 15
    epochs: int = 20
    learning_rate: float = 1e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    copy_to_drive: bool = False
    drive_dir: Path = Path("/content/drive/MyDrive/fme-ai-models")

    @property
    def data_dir(self):
        return self.work_dir / "data"

    @property
    def output_dir(self):
        return self.work_dir / "checkpoints"

    @property
    def export_dir(self):
        return self.work_dir / self.export_name


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def download_dataset(config):
    """Descarga y descomprime FUNSD; devuelve la raiz del dataset."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    existing = _find_dataset_root(config.data_dir)
    if existing is not None:
        print(f"Dataset ya disponible en {existing}")
        return existing

    archive = config.data_dir / "funsd.zip"
    last_error = None
    for url in config.dataset_urls:
        try:
            print(f"Descargando FUNSD desde {url} ...")
            urllib.request.urlretrieve(url, archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(config.data_dir)
            break
        except (OSError, zipfile.BadZipFile) as error:
            last_error = error
            print(f"  Fallo la descarga: {error}")
    else:
        raise RuntimeError("No se pudo descargar FUNSD") from last_error

    root = _find_dataset_root(config.data_dir)
    if root is None:
        raise RuntimeError("Estructura de FUNSD no reconocida")
    return root


def _find_dataset_root(base):
    """Busca la carpeta que contiene training_data/ y testing_data/."""
    for candidate in base.rglob("training_data"):
        if "__MACOSX" in candidate.parts:
            continue
        root = candidate.parent
        if (root / "testing_data" / "annotations").is_dir():
            return root
    return None


def normalize_box(box, width, height):
    """Escala una caja [x0, y0, x1, y1] al rango 0-1000 de LayoutLMv3."""
    x0, y0, x1, y1 = box
    scaled = [
        1000 * x0 / width,
        1000 * y0 / height,
        1000 * x1 / width,
        1000 * y1 / height,
    ]
    return [int(max(0, min(1000, value))) for value in scaled]


def load_split(split_dir):
    """Lee un split de FUNSD como lista de documentos.

    Cada documento es un dict con ``id``, ``image_path``, ``words``,
    ``boxes`` (normalizadas) y ``labels`` (esquema BIO).
    """
    documents = []
    annotations = sorted((split_dir / "annotations").glob("*.json"))
    for annotation_path in annotations:
        image_path = split_dir / "images" / f"{annotation_path.stem}.png"
        with Image.open(image_path) as image:
            width, height = image.size
        with annotation_path.open(encoding="utf-8") as handle:
            form = json.load(handle)["form"]

        words, boxes, labels = [], [], []
        for entity in form:
            entity_words = [w for w in entity["words"] if w["text"].strip()]
            tag = entity["label"].upper()
            for position, word in enumerate(entity_words):
                words.append(word["text"])
                boxes.append(normalize_box(word["box"], width, height))
                if tag == "OTHER":
                    labels.append("O")
                elif position == 0:
                    labels.append(f"B-{tag}")
                else:
                    labels.append(f"I-{tag}")

        documents.append({
            "id": annotation_path.stem,
            "image_path": str(image_path),
            "words": words,
            "boxes": boxes,
            "labels": [LABEL2ID[label] for label in labels],
        })
    return documents


class FunsdWindowDataset(torch.utils.data.Dataset):
    """Documentos codificados en ventanas de ``max_length`` tokens.

    Los documentos que exceden el limite se dividen en ventanas
    solapadas (``stride``) para no perder palabras por truncamiento.
    """

    def __init__(self, documents, processor, config):
        self.features = []
        for document in documents:
            self.features.extend(
                self._encode(document, processor, config)
            )

    @staticmethod
    def _encode(document, processor, config):
        with Image.open(document["image_path"]) as image:
            pixel_values = processor.image_processor(
                image.convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        encoding = processor.tokenizer(
            document["words"],
            boxes=document["boxes"],
            word_labels=document["labels"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            stride=config.stride,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        windows = []
        for index in range(encoding["input_ids"].shape[0]):
            windows.append({
                "input_ids": encoding["input_ids"][index],
                "attention_mask": encoding["attention_mask"][index],
                "bbox": encoding["bbox"][index],
                "labels": encoding["labels"][index],
                "pixel_values": pixel_values,
            })
        return windows

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def split_validation(documents, size, seed):
    """Separa ``size`` documentos de entrenamiento para validacion."""
    shuffled = documents[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[size:], shuffled[:size]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------


def to_label_sequences(predictions, label_ids):
    """Convierte logits/ids en secuencias de etiquetas sin padding."""
    predicted_ids = np.argmax(predictions, axis=-1)
    true_sequences, predicted_sequences = [], []
    for predicted_row, label_row in zip(predicted_ids, label_ids):
        mask = label_row != IGNORE_INDEX
        true_sequences.append([ID2LABEL[i] for i in label_row[mask]])
        predicted_sequences.append(
            [ID2LABEL[i] for i in predicted_row[mask]]
        )
    return true_sequences, predicted_sequences


def compute_metrics(eval_prediction):
    predictions, label_ids = eval_prediction
    true_seq, predicted_seq = to_label_sequences(predictions, label_ids)
    return {
        "precision": precision_score(true_seq, predicted_seq),
        "recall": recall_score(true_seq, predicted_seq),
        "f1": f1_score(true_seq, predicted_seq),
    }


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def build_trainer(config, processor, train_dataset, eval_dataset):
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=config.seed,
    )
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )


# ---------------------------------------------------------------------------
# Exportacion
# ---------------------------------------------------------------------------


def export_model(config, trainer, processor, test_metrics):
    """Guarda modelo, procesador y metadatos; devuelve la ruta del zip."""
    export_dir = config.export_dir
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    trainer.save_model(str(export_dir))
    processor.save_pretrained(str(export_dir))

    metadata = {
        "name": config.export_name,
        "base_model": config.base_model,
        "task": "token-classification",
        "dataset": "FUNSD",
        "labels": LABELS,
        "max_length": config.max_length,
        "stride": config.stride,
        "apply_ocr": False,
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "config": {
            key: str(value) for key, value in asdict(config).items()
        },
    }
    metadata_path = export_dir / "fme_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    archive = shutil.make_archive(
        str(export_dir), "zip", root_dir=export_dir
    )
    print(f"Modelo exportado en {archive}")

    if config.copy_to_drive:
        _copy_to_drive(config, Path(archive))
    return Path(archive)


def _copy_to_drive(config, archive):
    from google.colab import drive

    drive.mount("/content/drive")
    config.drive_dir.mkdir(parents=True, exist_ok=True)
    target = config.drive_dir / archive.name
    shutil.copy2(archive, target)
    print(f"Copia guardada en Google Drive: {target}")


def offer_download(archive):
    """Inicia la descarga del zip si se ejecuta dentro de Colab."""
    if "google.colab" not in sys.modules:
        return
    from google.colab import files

    files.download(str(archive))


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main(config=None):
    config = config or Config()
    set_seed(config.seed)
    print(f"GPU disponible: {torch.cuda.is_available()}")

    dataset_root = download_dataset(config)
    train_docs = load_split(dataset_root / "training_data")
    test_docs = load_split(dataset_root / "testing_data")
    train_docs, validation_docs = split_validation(
        train_docs, config.validation_docs, config.seed
    )
    print(
        f"Documentos -> train: {len(train_docs)}, "
        f"validacion: {len(validation_docs)}, test: {len(test_docs)}"
    )

    processor = AutoProcessor.from_pretrained(
        config.base_model, apply_ocr=False
    )
    train_dataset = FunsdWindowDataset(train_docs, processor, config)
    validation_dataset = FunsdWindowDataset(
        validation_docs, processor, config
    )
    test_dataset = FunsdWindowDataset(test_docs, processor, config)

    trainer = build_trainer(
        config, processor, train_dataset, validation_dataset
    )
    trainer.train()

    test_output = trainer.predict(test_dataset, metric_key_prefix="test")
    true_seq, predicted_seq = to_label_sequences(
        test_output.predictions, test_output.label_ids
    )
    print(classification_report(true_seq, predicted_seq, digits=4))
    print(f"Metricas en test: {test_output.metrics}")

    archive = export_model(config, trainer, processor, test_output.metrics)
    offer_download(archive)
    return archive


if __name__ == "__main__":
    main()
