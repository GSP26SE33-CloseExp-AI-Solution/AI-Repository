from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.exceptions import ImageProcessingError, ModelNotLoadedError
from app.core.logging import get_logger
from app.infra.model_store import model_store
from app.models.common import BoundingBox
from app.models.vision import (
    Detection,
    ProductType,
    QualityAssessment,
    QualityLabel,
    VisionAnalyzeRequest,
    VisionAnalyzeResponse,
)

logger = get_logger(__name__)


# Mapping from YOLO class names to product types
CLASS_TO_PRODUCT_TYPE = {
    "apple": ProductType.FRUIT,
    "orange": ProductType.FRUIT,
    "banana": ProductType.FRUIT,
    "broccoli": ProductType.VEGETABLE,
    "carrot": ProductType.VEGETABLE,
    "bottle": ProductType.BEVERAGE,
    "wine glass": ProductType.BEVERAGE,
    "cup": ProductType.BEVERAGE,
    "sandwich": ProductType.BAKERY,
    "cake": ProductType.BAKERY,
    "donut": ProductType.BAKERY,
    "pizza": ProductType.BAKERY,
    "hot dog": ProductType.MEAT,
}


def _load_image(request: VisionAnalyzeRequest) -> Tuple[Any, bytes]:
    """Load image from URL or base64."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImageProcessingError("Pillow is required for image processing") from e

    if request.image_url:
        try:
            import requests

            resp = requests.get(str(request.image_url), timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            raise ImageProcessingError(f"Failed to fetch image: {e}") from e
    elif request.image_b64:
        try:
            b64_data = request.image_b64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
        except Exception as e:
            raise ImageProcessingError(f"Invalid base64 image: {e}") from e
    else:
        raise ImageProcessingError("No image provided")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image, image_bytes
    except Exception as e:
        raise ImageProcessingError(f"Failed to decode image: {e}") from e


def _assess_image_quality(image: Any) -> QualityAssessment:
    """Assess image quality based on various metrics."""
    try:
        import numpy as np
        from PIL import ImageStat

        # Calculate brightness
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3 / 255

        # Calculate contrast (standard deviation)
        contrast = sum(stat.stddev) / 3 / 255

        # Estimate blur (using Laplacian variance)
        try:
            import cv2

            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian_var / 500)
        except ImportError:
            blur_score = 0.5  # Default if cv2 not available

        # Calculate overall quality score
        quality_score = (brightness * 0.3 + contrast * 0.3 + blur_score * 0.4)

        # Determine quality label
        reasons = []
        if quality_score >= 0.6:
            label = QualityLabel.GOOD
        elif quality_score >= 0.4:
            label = QualityLabel.OK
        else:
            label = QualityLabel.POOR

        if brightness < 0.3:
            reasons.append("Image is too dark")
        elif brightness > 0.8:
            reasons.append("Image is overexposed")

        if contrast < 0.2:
            reasons.append("Low contrast")

        if blur_score < 0.3:
            reasons.append("Image appears blurry")

        return QualityAssessment(
            label=label,
            score=quality_score,
            metrics={
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "sharpness": round(blur_score, 3),
            },
            reasons=reasons,
        )

    except Exception as e:
        logger.warning(f"Quality assessment failed: {e}")
        return QualityAssessment(
            label=QualityLabel.OK,
            score=0.5,
            metrics={"brightness": 0.5, "contrast": 0.5, "sharpness": 0.5},
            reasons=["Quality assessment unavailable"],
        )


def _get_product_type(class_name: str) -> ProductType:
    """Map YOLO class name to product type."""
    return CLASS_TO_PRODUCT_TYPE.get(class_name.lower(), ProductType.UNKNOWN)


def _crop_detection(
    image: Any,
    bbox: BoundingBox,
) -> Tuple[Optional[str], Optional[str]]:
    """Crop detection from image and return as base64."""
    try:
        cropped = image.crop((
            int(bbox.x1),
            int(bbox.y1),
            int(bbox.x2),
            int(bbox.y2),
        ))
        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return b64, "image/png"
    except Exception as e:
        logger.warning(f"Failed to crop detection: {e}")
        return None, None


def analyze_product_image(request: VisionAnalyzeRequest) -> VisionAnalyzeResponse:
    """
    Analyze product image for object detection.
    
    Args:
        request: Vision analysis request
        
    Returns:
        Detection results with metadata
    """
    import time

    start_time = time.perf_counter()

    # Load image
    image, image_bytes = _load_image(request)
    width, height = image.size

    # Load model
    model_name = request.model or settings.yolo_model_path
    try:
        model = model_store.load_yolo(model_name)
    except Exception as e:
        raise ModelNotLoadedError(model_name) from e

    # Run inference
    import numpy as np

    image_np = np.array(image)
    results = model(image_np, conf=request.min_confidence, verbose=False)

    # Process detections
    detections: List[Detection] = []
    class_counts: Counter = Counter()
    product_type_counts: Counter = Counter()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            if len(detections) >= request.max_detections:
                break

            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )

            product_type = _get_product_type(class_name)
            class_counts[class_name] += 1
            product_type_counts[product_type.value] += 1

            # Optional: crop detection
            crop_b64, crop_type = None, None
            if request.return_crops:
                crop_b64, crop_type = _crop_detection(image, bbox)

            detection = Detection(
                index=len(detections),
                class_name=class_name,
                confidence=conf,
                bounding_box=bbox,
                product_type=product_type,
                crop_image_b64=crop_b64,
                crop_image_content_type=crop_type,
            )
            detections.append(detection)

    # Assess image quality
    image_quality = None
    if request.assess_quality:
        image_quality = _assess_image_quality(image)

    # Generate annotated image
    annotated_b64, annotated_type = None, None
    if request.return_annotated_image and results:
        try:
            annotated = results[0].plot()
            from PIL import Image as PILImage

            annotated_pil = PILImage.fromarray(annotated)
            buffer = BytesIO()
            annotated_pil.save(buffer, format="PNG")
            annotated_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            annotated_type = "image/png"
        except Exception as e:
            logger.warning(f"Failed to generate annotated image: {e}")

    inference_time = (time.perf_counter() - start_time) * 1000

    return VisionAnalyzeResponse(
        detections=detections,
        detection_count=len(detections),
        image_quality=image_quality,
        class_summary=dict(class_counts),
        product_type_summary=dict(product_type_counts),
        annotated_image_b64=annotated_b64,
        annotated_image_content_type=annotated_type,
        model=model_name,
        inference_time_ms=round(inference_time, 2),
        image_dimensions={"width": width, "height": height},
    )


def analyze_product_image_png(request: VisionAnalyzeRequest) -> bytes:
    """
    Analyze image and return annotated PNG.
    
    Args:
        request: Vision analysis request
        
    Returns:
        PNG image bytes
    """
    import numpy as np

    # Load image
    image, _ = _load_image(request)

    # Load model
    model_name = request.model or settings.yolo_model_path
    model = model_store.load_yolo(model_name)

    # Run inference
    image_np = np.array(image)
    results = model(image_np, conf=request.min_confidence, verbose=False)

    # Generate annotated image
    if results:
        annotated = results[0].plot()
        from PIL import Image as PILImage

        annotated_pil = PILImage.fromarray(annotated)
        buffer = BytesIO()
        annotated_pil.save(buffer, format="PNG")
        return buffer.getvalue()

    # Return original if no detections
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
