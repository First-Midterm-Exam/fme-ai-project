class ReviewError(Exception):
    pass


class InvalidImageError(ReviewError):
    pass


class OcrUnavailableError(ReviewError):
    pass


class ModelUnavailableError(ReviewError):
    pass
