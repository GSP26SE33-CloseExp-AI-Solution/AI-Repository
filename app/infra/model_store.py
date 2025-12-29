from typing import Any, Optional


class ModelStore:
    def __init__(self) -> None:
        self._ocr_model: Optional[Any] = None
        self._pricing_model: Optional[Any] = None

    def load_ocr(self) -> Any:
        if self._ocr_model is None:
            self._ocr_model = "ocr-model-placeholder"
        return self._ocr_model

    def load_pricing(self) -> Any:
        if self._pricing_model is None:
            self._pricing_model = "pricing-model-placeholder"
        return self._pricing_model


model_store = ModelStore()
