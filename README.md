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

## 5) Tích hợp .NET
- Gọi HTTP nội bộ tới service AI; cấu hình base URL qua `AI_SERVICE_BASE_URL` phía .NET.
- Bọc retry + timeout, giới hạn kích thước ảnh; có thể gửi URL thay vì file lớn.

## 6) Ghi chú triển khai
- Model nặng: tải một lần ở startup (xem `infra/model_store.py`).
- Khi huấn luyện nội bộ: version hóa model và lưu dưới `infra/model_store` hoặc storage ngoài.