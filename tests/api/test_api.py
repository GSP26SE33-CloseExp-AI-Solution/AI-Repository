import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health(self, client: TestClient) -> None:
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_ready(self, client: TestClient) -> None:
        """Test readiness check."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ready", "not_ready"]
        assert "checks" in data
        assert "timestamp" in data

    def test_info(self, client: TestClient) -> None:
        """Test service info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "endpoints" in data


class TestPricingEndpoints:
    """Test pricing API endpoints."""

    def test_suggest_price_valid(self, client: TestClient) -> None:
        """Test valid pricing request."""
        response = client.post(
            "/v1/pricing/suggest",
            json={
                "product_type": "dairy",
                "days_to_expire": 3,
                "base_price": 50000,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "suggested_price" in data
        assert "discount_percent" in data
        assert "confidence" in data
        assert "urgency_level" in data

    def test_suggest_price_with_all_fields(self, client: TestClient) -> None:
        """Test pricing request with all optional fields."""
        response = client.post(
            "/v1/pricing/suggest",
            json={
                "product_type": "meat",
                "days_to_expire": 2,
                "base_price": 100000,
                "region": "HCMC",
                "brand": "Test Brand",
                "demand_index": 1.2,
                "strategy": "aggressive",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urgency_level"] == "high"

    def test_suggest_price_invalid_category(self, client: TestClient) -> None:
        """Test pricing request with invalid category."""
        response = client.post(
            "/v1/pricing/suggest",
            json={
                "product_type": "invalid_category",
                "days_to_expire": 3,
                "base_price": 50000,
            },
        )
        assert response.status_code == 422

    def test_suggest_price_invalid_price(self, client: TestClient) -> None:
        """Test pricing request with invalid price."""
        response = client.post(
            "/v1/pricing/suggest",
            json={
                "product_type": "dairy",
                "days_to_expire": 3,
                "base_price": -100,
            },
        )
        assert response.status_code == 422

    def test_suggest_price_missing_required(self, client: TestClient) -> None:
        """Test pricing request with missing required fields."""
        response = client.post(
            "/v1/pricing/suggest",
            json={
                "product_type": "dairy",
            },
        )
        assert response.status_code == 422


class TestOCREndpoints:
    """Test OCR API endpoints."""

    def test_extract_missing_image(self, client: TestClient) -> None:
        """Test OCR request without image."""
        response = client.post(
            "/v1/ocr/extract",
            json={},
        )
        assert response.status_code == 400

    def test_extract_with_invalid_url(self, client: TestClient) -> None:
        """Test OCR request with invalid URL."""
        response = client.post(
            "/v1/ocr/extract",
            json={
                "image_url": "not-a-valid-url",
            },
        )
        # Should either fail validation or fail to fetch
        assert response.status_code in [400, 422]
