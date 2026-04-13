"""
YOLO-based ROI selection for OCR.

Uses the packaged YOLO model (e.g. COCO-pretrained) to find the dominant object,
crops to that region with padding, and falls back to the full image when
detection is missing or unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from app.core.logging import get_logger
from app.infra.model_store import model_store

logger = get_logger(__name__)


@dataclass
class YoloRoiResult:
    """Outcome of YOLO ROI selection."""

    image: Any  # PIL.Image
    applied: bool
    reason: str


def _box_area(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad_x: float,
    pad_y: float,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, w, h
    return x1, y1, x2, y2


def select_ocr_crop(
    image: Any,
    *,
    min_confidence: float = 0.25,
    padding_ratio: float = 0.05,
    min_crop_area_ratio: float = 0.08,
    max_crop_area_ratio: float = 0.98,
) -> YoloRoiResult:
    """
    Return a cropped PIL image focused on the largest high-confidence detection,
    or the original image if cropping is not helpful.

    Args:
        image: PIL.Image (RGB)
        min_confidence: Minimum YOLO confidence to consider a detection
        padding_ratio: Fraction of min(w,h) to expand the box
        min_crop_area_ratio: If best crop is smaller than this fraction of the image, skip crop
        max_crop_area_ratio: If crop covers more than this fraction, skip (already full frame)
    """
    try:
        from PIL import Image  # type: ignore

        if not isinstance(image, Image.Image):
            return YoloRoiResult(image=image, applied=False, reason="not_a_pil_image")

        w, h = image.size
        if w < 32 or h < 32:
            return YoloRoiResult(image=image, applied=False, reason="image_too_small")

        model = model_store.load_yolo()
        import numpy as np

        img_np = np.array(image)
        results = model.predict(img_np, verbose=False)
        if not results:
            return YoloRoiResult(image=image, applied=False, reason="no_results")

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return YoloRoiResult(image=image, applied=False, reason="no_detections")

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        best: Optional[Tuple[float, Tuple[float, float, float, float]]] = None
        for i in range(len(xyxy)):
            c = float(conf[i])
            if c < min_confidence:
                continue
            x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
            area = _box_area((x1, y1, x2, y2))
            if best is None or area > best[0]:
                best = (area, (x1, y1, x2, y2))

        if best is None:
            return YoloRoiResult(image=image, applied=False, reason="no_detections_above_threshold")

        _, (x1, y1, x2, y2) = best
        full_area = float(w * h)
        pad = padding_ratio * min(w, h)
        cx1, cy1, cx2, cy2 = _clamp_box(x1, y1, x2, y2, pad, pad, w, h)
        crop_area = float((cx2 - cx1) * (cy2 - cy1))
        if full_area <= 0:
            return YoloRoiResult(image=image, applied=False, reason="invalid_image_area")

        ratio = crop_area / full_area
        if ratio < min_crop_area_ratio:
            return YoloRoiResult(
                image=image,
                applied=False,
                reason=f"crop_too_small(ratio={ratio:.2f})",
            )
        if ratio > max_crop_area_ratio:
            return YoloRoiResult(
                image=image,
                applied=False,
                reason=f"crop_covers_full_frame(ratio={ratio:.2f})",
            )

        cropped = image.crop((cx1, cy1, cx2, cy2))
        logger.info(
            "YOLO OCR crop applied: box=(%s,%s,%s,%s) conf_threshold=%s",
            cx1,
            cy1,
            cx2,
            cy2,
            min_confidence,
        )
        return YoloRoiResult(image=cropped, applied=True, reason="ok")

    except ImportError as e:
        logger.warning("YOLO ROI skipped (import): %s", e)
        return YoloRoiResult(image=image, applied=False, reason=f"import_error:{e}")
    except Exception as e:
        logger.warning("YOLO ROI skipped: %s", e)
        return YoloRoiResult(image=image, applied=False, reason=f"error:{e}")
