"""Pydantic schemas for API request/response validation."""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for a single ETA prediction."""

    store_latitude: float = Field(..., ge=-90, le=90)
    store_longitude: float = Field(..., ge=-180, le=180)
    delivery_latitude: float = Field(..., ge=-90, le=90)
    delivery_longitude: float = Field(..., ge=-180, le=180)
    total_items: int = Field(..., gt=0)
    subtotal: float = Field(..., ge=0)
    num_distinct_items: int = Field(..., gt=0)
    created_at: datetime
    total_onshift_riders: int = Field(default=10, ge=0)
    total_busy_riders: int = Field(default=5, ge=0)
    total_outstanding_orders: int = Field(default=10, ge=0)

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "store_latitude": 31.2304,
                "store_longitude": 121.4737,
                "delivery_latitude": 31.2396,
                "delivery_longitude": 121.4997,
                "total_items": 3,
                "subtotal": 45.0,
                "num_distinct_items": 2,
                "created_at": "2026-04-20T12:30:00",
                "total_onshift_riders": 15,
                "total_busy_riders": 10,
                "total_outstanding_orders": 25,
            }
        ]
    }}


class PredictionResponse(BaseModel):
    """Output schema for ETA prediction."""

    estimated_delivery_minutes: float
    confidence_interval: dict[str, float]
    model_version: str


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    n_features: int
    feature_names: list[str]
