"""Tests for the FastAPI prediction API."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_503_without_model(self, client: TestClient) -> None:
        payload = {
            "store_latitude": 31.23,
            "store_longitude": 121.47,
            "delivery_latitude": 31.24,
            "delivery_longitude": 121.48,
            "created_at": "2026-03-15T12:30:00",
            "total_items": 3,
            "subtotal": 45.0,
            "num_distinct_items": 2,
            "total_onshift_riders": 15,
            "total_busy_riders": 10,
            "total_outstanding_orders": 25,
        }
        response = client.post("/predict", json=payload)
        # Model not loaded in test environment
        assert response.status_code == 503

    def test_predict_validates_input(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestBatchEndpoint:
    def test_batch_validates_empty_list(self, client: TestClient) -> None:
        response = client.post("/predict/batch", json={"orders": []})
        # Empty list should succeed with empty predictions
        assert response.status_code == 200
        assert response.json()["predictions"] == []
