# AI Service cho Price Suggestion & OCR

Dịch vụ AI (Python/FastAPI) được .NET API gọi sang để trích xuất thông tin sản phẩm (OCR) và gợi ý giá bán cho hàng cận hạn. README này chỉ nêu cấu trúc, quy tắc và bề mặt API; phần setup đã được khai báo sẵn trong repo.

## 1) Mục tiêu & Phạm vi
- OCR: nhận ảnh/URL → trả về HSD, NSX, tên sản phẩm, thương hiệu, barcode/SKU.
- Price Suggestion: nhận metadata (loại, hạn còn lại, giá gốc, vùng/brand/demand) → gợi ý giá + độ tin cậy.
- Health/Readiness: phục vụ giám sát từ .NET API.

## 2) Cấu trúc thư mục
```
app/
  api/         # FastAPI routers (health, ocr, pricing)
  models/      # Pydantic schemas
  services/    # OCR & pricing logic (stubs)
  core/        # config, logging
  infra/       # model loading placeholders
scripts/       # tiện ích huấn luyện/batch
tests/         # chỗ đặt unit/service tests
requirements.txt
```

## 3) Quy tắc phát triển
- Ngôn ngữ: code/comment ASCII; docstring ngắn gọn, typed đầy đủ.
- Schema: pydantic v2, `extra="forbid"` cho input models.
- Logging: dùng `app.core.logging.setup_logging` cho app entry.
- Config: lấy từ biến môi trường prefix `AI_` (xem `app/core/config.py`).
- API versioning: tiền tố `/v1/...`; giữ backward compatibility khi thay đổi.
- Kiểm thử: thêm unit tests cho services/routers khi bổ sung logic mới.

## 4) Bề mặt API hiện có (stub)
- `GET /health` và `GET /ready`
- `POST /v1/ocr/extract`
- `POST /v1/pricing/suggest`
- `POST /v1/vision/analyze` (demo multi-object: nhận diện + đánh giá từng object)
- `POST /v1/vision/analyze.png` (trả thẳng ảnh PNG đã annotate)

### `POST /v1/vision/analyze` (Demo)
Mục tiêu: ảnh có nhiều sản phẩm khác nhau → AI phát hiện nhiều object và đánh giá từng object riêng lẻ.
Output của mỗi object gồm bbox (`xyxy`), nhãn YOLO (`class_name`), suy luận loại hàng cận hạn (`product_type`) và `quality` của vùng crop theo bbox.

Request (JSON):
- `image_url` hoặc `image_b64`
- `min_confidence` (mặc định 0.25)
- `return_crops` (mặc định true): trả crop ảnh (PNG base64) cho từng object
- `max_crops` (mặc định 10)

Ví dụ (curl):
```bash
curl -X POST http://127.0.0.1:8000/v1/vision/analyze \
  -H "Content-Type: application/json" \
  -d "{\"image_url\":\"https://ultralytics.com/images/bus.jpg\"}"
```

Response: gồm `detections` (YOLO) và `quality` (label/score/metrics/reasons).
Ngoài ra, mặc định response có `annotated_image_b64` (PNG) để client hiển thị trực tiếp ảnh đã vẽ box + text.

Hiển thị nhanh ở web:
- Tạo URL: `data:image/png;base64,<annotated_image_b64>`

Nếu muốn lấy thẳng ảnh (không base64), dùng endpoint PNG:
```bash
curl -X POST http://127.0.0.1:8000/v1/vision/analyze.png \
  -H "Content-Type: application/json" \
  -d "{\"image_url\":\"https://ultralytics.com/images/bus.jpg\"}" \
  --output annotated.png
```

## 5) Tích hợp .NET
- Gọi HTTP nội bộ tới service AI; cấu hình base URL qua `AI_SERVICE_BASE_URL` phía .NET.
- Bọc retry + timeout, giới hạn kích thước ảnh; có thể gửi URL thay vì file lớn.

## 6) Ghi chú triển khai
- Model nặng: tải một lần ở startup (xem `infra/model_store.py`).
- Khi huấn luyện nội bộ: version hóa model và lưu dưới `infra/model_store` hoặc storage ngoài.

## 7) Cài đặt nhanh (Windows)
```bash
pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn app.api.main:app --reload
```

Mặc định Vision dùng weights `yolo11n.pt` (có thể override bằng request field `model` hoặc env var `AI_VISION_YOLO_MODEL`).

Nếu bạn muốn dùng Ultralytics/YOLO "mới nhất từ GitHub" (bleeding-edge):
```bash
pip install -U git+https://github.com/ultralytics/ultralytics.git
```

Nếu cài `ultralytics` báo lỗi Torch/CUDA, bạn có thể dùng CPU-only (tuỳ máy) hoặc giữ nguyên và chỉ gọi các endpoint OCR/Pricing stub.