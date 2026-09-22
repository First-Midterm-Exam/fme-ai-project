"""Errores de dominio; la capa API los traduce a respuestas HTTP."""


class RevisionError(Exception):
    """Base de los errores controlados del servicio."""


class InvalidImageError(RevisionError):
    """El archivo recibido no es una imagen valida."""


class OcrUnavailableError(RevisionError):
    """El motor OCR no esta instalado o no pudo ejecutarse."""


class ModelUnavailableError(RevisionError):
    """El modelo exportado no esta disponible o no pudo cargarse."""
