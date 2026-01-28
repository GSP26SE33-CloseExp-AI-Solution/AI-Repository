from typing import Any, Dict, Optional


class AIServiceError(Exception):
    """Base exception for AI Service."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ImageProcessingError(AIServiceError):
    """Error during image processing."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="IMAGE_PROCESSING_ERROR",
            status_code=400,
            details=details,
        )


class ModelNotLoadedError(AIServiceError):
    """Error when model is not loaded."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Model '{model_name}' is not loaded",
            error_code="MODEL_NOT_LOADED",
            status_code=503,
            details={"model_name": model_name},
        )


class OCRExtractionError(AIServiceError):
    """Error during OCR extraction."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="OCR_EXTRACTION_ERROR",
            status_code=422,
            details=details,
        )


class InvalidImageError(AIServiceError):
    """Error for invalid image input."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_IMAGE",
            status_code=400,
            details=details,
        )


class PricingCalculationError(AIServiceError):
    """Error during pricing calculation."""

    def __init__(
        self, message: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="PRICING_CALCULATION_ERROR",
            status_code=422,
            details=details,
        )
