from __future__ import annotations

from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelStore:
    """
    Centralized model storage and lazy loading.
    
    Models are loaded once and cached for reuse.
    """

    def __init__(self) -> None:
        self._yolo_model: Optional[Any] = None
        self._ocr_engine: Optional[Any] = None
        self._pricing_model: Optional[Any] = None
        self._yolo_model_name: Optional[str] = None

    def load_yolo(self, model_name: Optional[str] = None) -> Any:
        """
        Load YOLO model.
        
        Args:
            model_name: Model file name (e.g., 'yolo11n.pt')
            
        Returns:
            Loaded YOLO model
        """
        model_name = model_name or settings.yolo_model_path

        # Return cached model if same name
        if self._yolo_model is not None and self._yolo_model_name == model_name:
            return self._yolo_model

        try:
            from ultralytics import YOLO  # type: ignore

            logger.info(f"Loading YOLO model: {model_name}")
            self._yolo_model = YOLO(model_name)
            self._yolo_model_name = model_name
            logger.info(f"YOLO model loaded successfully: {model_name}")
            return self._yolo_model

        except ImportError as e:
            logger.error("Ultralytics is not installed")
            raise RuntimeError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Failed to load YOLO model '{model_name}': {e}") from e

    def load_ocr(self) -> Any:
        """Load OCR engine (PaddleOCR or EasyOCR)."""
        if self._ocr_engine is not None:
            return self._ocr_engine

        langs = settings.ocr_supported_languages

        # Try PaddleOCR first
        try:
            from paddleocr import PaddleOCR  # type: ignore

            logger.info("Loading PaddleOCR engine")
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="vi",
                show_log=False,
            )
            logger.info("PaddleOCR loaded successfully")
            return self._ocr_engine
        except ImportError:
            pass

        # Fallback to EasyOCR
        try:
            import easyocr  # type: ignore

            reader_kwargs: dict[str, Any] = {"gpu": False}
            if settings.ocr_model_path:
                reader_kwargs["model_storage_directory"] = settings.ocr_model_path
                logger.info(
                    "Loading EasyOCR engine (langs=%s, model_dir=%s)",
                    langs,
                    settings.ocr_model_path,
                )
            else:
                logger.info("Loading EasyOCR engine (langs=%s)", langs)

            self._ocr_engine = easyocr.Reader(langs, **reader_kwargs)
            logger.info("EasyOCR loaded successfully")
            return self._ocr_engine
        except ImportError:
            pass
        except Exception as e:
            logger.error("Failed to load EasyOCR: %s", e)
            raise

        logger.warning("No OCR engine available")
        return None

    @property
    def yolo_loaded(self) -> bool:
        return self._yolo_model is not None

    @property
    def ocr_loaded(self) -> bool:
        return self._ocr_engine is not None

    def load_pricing(self) -> Any:
        """
        Load pricing model.
        
        Currently uses rule-based logic, but can be extended for ML models.
        """
        if self._pricing_model is not None:
            return self._pricing_model

        # Placeholder for ML-based pricing model
        self._pricing_model = "rule-based-v1"
        return self._pricing_model

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        return {
            "yolo": {
                "loaded": self._yolo_model is not None,
                "name": self._yolo_model_name,
            },
            "ocr": {
                "loaded": self._ocr_engine is not None,
                "type": type(self._ocr_engine).__name__ if self._ocr_engine else None,
            },
            "pricing": {
                "loaded": self._pricing_model is not None,
                "type": self._pricing_model,
            },
        }


# Singleton instance
model_store = ModelStore()
