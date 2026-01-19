from __future__ import annotations

import base64
import os
import time
from functools import lru_cache
from io import BytesIO
from typing import Any, List, Tuple, Union

from app.models.vision import Detection, ProductType, QualityAssessment, VisionAnalyzeRequest, VisionAnalyzeResponse


def _strip_data_url_prefix(image_b64: str) -> str:
    if "," in image_b64 and image_b64.strip().lower().startswith("data:"):
        return image_b64.split(",", 1)[1]
    return image_b64


def _load_image_bytes_from_url(image_url: str) -> bytes:
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests is required to fetch image_url") from exc

    resp = requests.get(image_url, timeout=20)
    resp.raise_for_status()
    return resp.content


def _load_image_bytes_from_b64(image_b64: str) -> bytes:
    return base64.b64decode(_strip_data_url_prefix(image_b64), validate=False)


def _decode_image_to_pil(image_bytes: bytes) -> "Any":
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to decode images") from exc

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _decode_image_to_bgr_np(image_bytes: bytes) -> "Any":
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python and numpy are required for quality scoring") from exc

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    return img


@lru_cache(maxsize=1)
def _get_yolo(model_name: str) -> "Any":
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    return YOLO(model_name)


def _pick_yolo_model(requested: str | None) -> str:
    if requested and requested.strip():
        return requested.strip()

    # Allow override via env var for deployments
    env_model = os.getenv("AI_VISION_YOLO_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    # Try a "newest" default first, then fallback.
    # Weight names depend on Ultralytics release line; we handle failures later.
    return "yolo11n.pt"


def _get_yolo_with_fallback(primary: str) -> tuple[Any, str]:
    candidates = [primary]
    if primary != "yolo11n.pt":
        candidates.append("yolo11n.pt")

    last_exc: Exception | None = None
    for name in candidates:
        try:
            return _get_yolo(name), name
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            continue
    raise RuntimeError(f"Unable to load YOLO model: {primary}") from last_exc


def _run_yolo(model_name: str, image_input: Union[str, "Any"]) -> Tuple[List[Detection], float, str]:
    model, resolved_model_name = _get_yolo_with_fallback(model_name)
    start = time.perf_counter()
    results = model(image_input, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if not results:
        return [], elapsed_ms, resolved_model_name

    r0 = results[0]
    names = getattr(r0, "names", None) or getattr(model, "names", {})

    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return [], elapsed_ms, resolved_model_name

    xyxy = getattr(boxes, "xyxy", None)
    conf = getattr(boxes, "conf", None)
    cls = getattr(boxes, "cls", None)
    if xyxy is None or conf is None or cls is None:
        return [], elapsed_ms, resolved_model_name

    xyxy_list = xyxy.cpu().tolist() if hasattr(xyxy, "cpu") else xyxy.tolist()
    conf_list = conf.cpu().tolist() if hasattr(conf, "cpu") else conf.tolist()
    cls_list = cls.cpu().tolist() if hasattr(cls, "cpu") else cls.tolist()

    detections: List[Detection] = []
    for idx, (box, score, class_id) in enumerate(
        zip(xyxy_list, conf_list, cls_list, strict=False)
    ):
        class_idx = int(class_id)
        class_name = names.get(class_idx, str(class_idx)) if isinstance(names, dict) else str(class_idx)
        detections.append(
            Detection(
                index=idx,
                class_name=class_name,
                confidence=float(score),
                xyxy=[float(v) for v in box],
            )
        )

    return detections, elapsed_ms, resolved_model_name


def _color_for_class(name: str) -> tuple[int, int, int]:
    # Stable pseudo-random color per class name.
    h = abs(hash(name)) % 360
    # HSV -> RGB (simple)
    c = 255
    x = int(c * (1 - abs(((h / 60.0) % 2) - 1)))
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int(r), int(g), int(b)


def _annotate_image(
    image_pil: Any,
    detections: List[Detection],
    quality: QualityAssessment,
    header_text: str,
) -> bytes:
    try:
        from PIL import ImageDraw, ImageFont  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to annotate images") from exc

    img = image_pil.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    header = f"{header_text} | quality={quality.label} ({quality.score:.2f})"
    header_color = (0, 255, 255)
    draw.rectangle([6, 6, 6 + 8 * len(header) + 10, 30], fill=header_color + (160,))
    draw.text((10, 10), header, fill=(0, 0, 0), font=font)

    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = _color_for_class(det.class_name)
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

        if det.quality is not None:
            label = (
                f"{det.class_name} {det.confidence:.2f}"
                f" | {det.product_type}"
                f" | {det.quality.label} {det.quality.score:.2f}"
            )
        else:
            label = f"{det.class_name} {det.confidence:.2f} | {det.product_type}"
        # estimate label box size
        text_w = max(1, 8 * len(label))
        text_h = 14
        tx1 = float(x1)
        ty1 = float(max(0.0, y1 - text_h - 2))
        tx2 = tx1 + text_w + 6
        ty2 = ty1 + text_h + 4
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color + (220,))
        draw.text((tx1 + 3, ty1 + 2), label, fill=(255, 255, 255), font=font)

    out = BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _crop_bgr(image_bgr: "Any", xyxy: List[float]) -> "Any":
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return image_bgr
    return image_bgr[y1:y2, x1:x2]


def _infer_product_type(class_name: str) -> ProductType:
    name = class_name.strip().lower()
    # Ultralytics COCO classes sometimes include food items; map them to our supermarket buckets.
    fruit = {"apple", "banana", "orange"}
    vegetable = {"broccoli", "carrot", "hot dog", "sandwich"}  # demo-friendly fallback
    fish = {"fish"}
    meat = {"steak", "meat", "chicken"}

    if name in fruit:
        return "fruit"
    if name in vegetable:
        return "vegetable"
    if name in fish:
        return "fish"
    if name in meat:
        return "meat"
    return "unknown"


def _encode_pil_png_b64(image_pil: Any) -> str:
    out = BytesIO()
    image_pil.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def analyze_product_image_png(payload: VisionAnalyzeRequest, yolo_model: str = "yolo11n.pt") -> bytes:
    """Return annotated PNG bytes for quick client display."""
    if not payload.image_url and not payload.image_b64:
        raise ValueError("image_url or image_b64 is required")

    if payload.image_url:
        image_bytes = _load_image_bytes_from_url(str(payload.image_url))
    else:
        image_bytes = _load_image_bytes_from_b64(payload.image_b64 or "")

    image_pil = _decode_image_to_pil(image_bytes)
    requested_model = _pick_yolo_model(payload.model) if hasattr(payload, "model") else None
    chosen_model = requested_model or yolo_model
    detections, _inference_ms, _resolved_model = _run_yolo(chosen_model, image_pil)
    detections = [d for d in detections if d.confidence >= payload.min_confidence]

    image_bgr = _decode_image_to_bgr_np(image_bytes)
    quality = _assess_quality(image_bgr)

    enriched: List[Detection] = []
    for idx, det in enumerate(detections):
        det.index = idx
        det.product_type = _infer_product_type(det.class_name)
        roi = _crop_bgr(image_bgr, det.xyxy)
        det.quality = _assess_quality(roi)
        enriched.append(det)

    header_text = "multi-object"
    return _annotate_image(image_pil, enriched, quality, header_text)


def _assess_quality(image_bgr: "Any") -> QualityAssessment:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Blur metric: variance of Laplacian
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness/contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Simple score combining metrics (demo heuristic)
    # Normalize blur_var into [0,1] with a soft cap
    blur_score = min(1.0, max(0.0, blur_var / 300.0))

    # Brightness: prefer mid-range ~[60..190] on 0..255
    if brightness < 40:
        brightness_score = 0.2
    elif brightness < 70:
        brightness_score = 0.6
    elif brightness <= 200:
        brightness_score = 1.0
    else:
        brightness_score = 0.6

    # Contrast: very low contrast makes details hard
    contrast_score = min(1.0, max(0.0, contrast / 60.0))

    score = 0.5 * blur_score + 0.3 * brightness_score + 0.2 * contrast_score

    reasons: List[str] = []
    if blur_var < 80:
        reasons.append("image is blurry")
    if brightness < 50:
        reasons.append("image is too dark")
    elif brightness > 220:
        reasons.append("image is overexposed")
    if contrast < 20:
        reasons.append("low contrast")

    if score >= 0.75:
        label: str = "good"
    elif score >= 0.5:
        label = "ok"
    else:
        label = "poor"

    return QualityAssessment(
        label=label,  # type: ignore[arg-type]
        score=round(float(score), 3),
        metrics={
            "blur_var": round(blur_var, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
        },
        reasons=reasons,
    )


def analyze_product_image(payload: VisionAnalyzeRequest, yolo_model: str = "yolo11n.pt") -> VisionAnalyzeResponse:
    if not payload.image_url and not payload.image_b64:
        raise ValueError("image_url or image_b64 is required")

    if payload.image_url:
        image_bytes = _load_image_bytes_from_url(str(payload.image_url))
    else:
        image_bytes = _load_image_bytes_from_b64(payload.image_b64 or "")

    # YOLO can accept a URL string or a PIL image.
    # We use PIL to avoid double-download when we already fetched bytes.
    image_pil = _decode_image_to_pil(image_bytes)

    requested_model = _pick_yolo_model(payload.model) if hasattr(payload, "model") else None
    chosen_model = requested_model or yolo_model
    detections, inference_ms, resolved_model = _run_yolo(chosen_model, image_pil)

    detections = [d for d in detections if d.confidence >= payload.min_confidence]

    image_bgr = _decode_image_to_bgr_np(image_bytes)
    quality = _assess_quality(image_bgr)

    enriched_detections: List[Detection] = []
    for idx, det in enumerate(detections):
        det.index = idx
        det.product_type = _infer_product_type(det.class_name)
        roi = _crop_bgr(image_bgr, det.xyxy)
        det.quality = _assess_quality(roi)

        if getattr(payload, "return_crops", True) and idx < getattr(payload, "max_crops", 10):
            # Crop from original PIL (RGB) using the same bbox.
            x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            crop = image_pil.crop((x1, y1, x2, y2))
            det.crop_image_content_type = "image/png"
            det.crop_image_b64 = _encode_pil_png_b64(crop)

        enriched_detections.append(det)

    annotated_b64: str | None = None
    annotated_ct: str | None = None
    if getattr(payload, "return_annotated_image", True):
        annotated_png = _annotate_image(image_pil, enriched_detections, quality, "multi-object")
        annotated_b64 = base64.b64encode(annotated_png).decode("ascii")
        annotated_ct = "image/png"

    return VisionAnalyzeResponse(
        model=resolved_model,
        inference_ms=round(inference_ms, 2),
        detections=enriched_detections,
        quality=quality,
        annotated_image_content_type=annotated_ct,
        annotated_image_b64=annotated_b64,
    )
