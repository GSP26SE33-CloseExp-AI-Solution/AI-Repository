# CloseExp AI Service

AI Service cho hệ thống quản lý sản phẩm sắp hết hạn (CloseExp).

## Tính năng chính

### 1. OCR Service (`/v1/ocr`)
- Trích xuất ngày hết hạn (HSD) và ngày sản xuất (NSX) từ hình ảnh
- Nhận dạng barcode/QR code với hỗ trợ **mã vạch Việt Nam (GS1 893)**
- Tra cứu thông tin công ty từ mã vạch (Vinamilk, TH True Milk, Masan, Acecook, ...)
- Trích xuất thành phần, hướng dẫn bảo quản, thông tin dinh dưỡng từ bao bì
- Nhận dạng xuất xứ và chứng nhận chất lượng (HACCP, ISO, VietGAP, ...)
- Hỗ trợ tiếng Việt và tiếng Anh
- Trả về độ tin cậy cho mỗi trường

### 2. Pricing Service (`/v1/pricing`)
- Đề xuất giá bán cho sản phẩm sắp hết hạn
- Tính toán dựa trên:
  - Số ngày còn lại đến hạn sử dụng
  - Loại sản phẩm (dairy, meat, seafood, bakery, etc.)
  - Chỉ số nhu cầu thị trường
  - Chiến lược định giá (aggressive, balanced, conservative)
- Trả về giá đề xuất, % giảm giá, độ tin cậy và giải thích

### 3. Vision Service (`/v1/vision`)
- Phát hiện sản phẩm trong hình ảnh sử dụng YOLO
- Đánh giá chất lượng hình ảnh
- Phân loại sản phẩm theo danh mục
- Trả về hình ảnh đã đánh dấu (annotated)

### 4. Fresh Produce Service (`/v1/fresh-produce`) - **MỚI**
- Nhận dạng sản phẩm tươi sống: rau củ, trái cây, thịt, hải sản
- Cung cấp tên tiếng Việt và tiếng Anh
- Thông tin thời hạn sử dụng điển hình
- Hướng dẫn bảo quản phù hợp
- Các chỉ báo nhận biết độ tươi của sản phẩm

## Cấu trúc Project

```
app/
├── api/                    # API endpoints
│   ├── main.py            # FastAPI application
│   ├── health.py          # Health check endpoints
│   ├── ocr.py             # OCR endpoints
│   ├── pricing.py         # Pricing endpoints
│   ├── vision.py          # Vision endpoints
│   ├── fresh_produce.py   # Fresh produce endpoints (NEW)
│   └── deps.py            # Dependencies
├── core/                   # Core configurations
│   ├── config.py          # Settings
│   ├── logging.py         # Logging setup
│   ├── exceptions.py      # Custom exceptions
│   └── security.py        # API key auth
├── models/                 # Pydantic schemas
│   ├── common.py          # Shared models
│   ├── ocr.py             # OCR models
│   ├── pricing.py         # Pricing models
│   └── vision.py          # Vision models
├── services/               # Business logic
│   ├── ocr.py             # OCR service
│   ├── pricing.py         # Pricing service
│   ├── vision.py          # Vision service
│   └── vietnamese_product.py  # Vietnamese product recognition (NEW)
├── infra/                  # Infrastructure
│   └── model_store.py     # Model management
└── utils/                  # Utilities
    ├── image.py           # Image processing
    └── validators.py      # Input validation
```

## Cài đặt

### Yêu cầu
- Python 3.11+
- CUDA (optional, cho GPU acceleration)

### 1. Clone và tạo virtual environment

```bash
git clone <repository-url>
cd AI-Repository
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu hình environment

```bash
copy .env.example .env
# Chỉnh sửa .env theo nhu cầu
```

### 4. Chạy server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Truy cập API docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker

### Local development

```bash
cp .env.example .env
# pip install -r requirements-dev.txt   # includes optional llama-cpp-python
docker compose up --build
```

### Production image (Gemini LLM, YOLO baked in, API key required)

```bash
docker compose -f docker-compose.prod.yml up --build
```

Set in `.env` or platform secrets: `AI_API_KEY` (>=32 chars), `AI_GEMINI_API_KEY`.

### Deploy on Render

1. Connect repo → **New Web Service** → Runtime: **Docker**
2. Use root `Dockerfile` and `render.yaml`
3. Set secrets: `AI_API_KEY`, `AI_GEMINI_API_KEY`
4. Point BE `AIService__BaseUrl` to Render AI URL; `AIService__ApiKey` = same `AI_API_KEY`

### Requirements split

| File | Use |
|------|-----|
| `requirements-prod.txt` | Docker / production |
| `requirements-dev.txt` | Local dev + CI (+ llama-cpp optional) |
| `requirements.txt` | Alias → dev |

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### OCR Extract

```bash
curl -X POST http://localhost:8000/v1/ocr/extract \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/product.jpg",
    "extract_dates": true,
    "extract_barcode": true
  }'
```

### Pricing Suggest

```bash
curl -X POST http://localhost:8000/v1/pricing/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "product_type": "dairy",
    "days_to_expire": 3,
    "base_price": 50000,
    "strategy": "balanced"
  }'
```

### Vision Analyze

```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/products.jpg",
    "min_confidence": 0.25,
    "return_annotated_image": true
  }'
```

## Testing

```bash
# Chạy tất cả tests
pytest

# Chạy với coverage
pytest --cov=app --cov-report=html

# Chạy tests cụ thể
pytest tests/services/test_pricing.py -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_APP_NAME` | CloseExp AI Service | Tên ứng dụng |
| `AI_APP_VERSION` | 1.0.0 | Version |
| `AI_ENVIRONMENT` | development | Environment (development/staging/production) |
| `AI_DEBUG` | false | Debug mode |
| `AI_PORT` | 8000 | Server port |
| `AI_API_KEY` | - | **Required in production** — shared secret with BE (`AIService:ApiKey`) |
| `AI_LLM_PROVIDER` | auto | `gemini` in production; `auto` tries GGUF then Gemini locally |
| `AI_ALLOWED_ORIGINS` | *(empty prod)* | Leave empty in production (no browser access) |
| `AI_YOLO_MODEL_PATH` | yolo11n.pt | Path đến YOLO model |
| `AI_LOG_LEVEL` | INFO | Log level |
| `AI_LOG_FORMAT` | json | Log format (json/console) |

## Tích hợp với Backend .NET

### Response Format

Tất cả API đều trả về JSON với format chuẩn:

```json
{
  "field1": "value1",
  "field2": "value2",
  "confidence": 0.85,
  "processing_time_ms": 123.45
}
```

### Error Response

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {}
  }
}
```

### API Key Authentication

Nếu `AI_API_KEY` được cấu hình, thêm header:

```
X-API-Key: your-api-key
```

### Gọi từ .NET

```csharp
// Cấu hình HttpClient
services.AddHttpClient("AIService", client =>
{
    client.BaseAddress = new Uri(Configuration["AI_SERVICE_BASE_URL"]);
    client.DefaultRequestHeaders.Add("X-API-Key", Configuration["AI_API_KEY"]);
    client.Timeout = TimeSpan.FromSeconds(30);
});

// Gọi Pricing API
var response = await httpClient.PostAsJsonAsync("/v1/pricing/suggest", new
{
    product_type = "dairy",
    days_to_expire = 3,
    base_price = 50000
});
```

## Quy tắc phát triển

- **Code style**: Sử dụng type hints đầy đủ, docstrings ngắn gọn
- **Schema**: Pydantic v2 với `extra="forbid"` cho input validation
- **Logging**: Structured logging với JSON format cho production
- **Config**: Environment variables với prefix `AI_`
- **API versioning**: Prefix `/v1/...`, giữ backward compatibility
- **Testing**: Unit tests cho tất cả services và routers

## License

MIT License - xem file [LICENSE](LICENSE)