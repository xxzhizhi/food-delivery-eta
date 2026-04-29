"""FastAPI application for ETA prediction service."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src import __version__
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.features.pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Food Delivery ETA Prediction API",
    description="Predict estimated delivery time for food orders.",
    version=__version__,
)

# Global model and pipeline state
_model = None
_pipeline = None
_model_version = "not_loaded"

MODEL_PATH = Path("models/saved/lgbm_model.pkl")
PIPELINE_PATH = Path("models/saved/feature_pipeline.pkl")


def _load_artifacts() -> None:
    """Load model and feature pipeline from disk."""
    global _model, _pipeline, _model_version

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        _model_version = "lgbm_v1.0"
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("No model found at %s", MODEL_PATH)

    if PIPELINE_PATH.exists():
        _pipeline = joblib.load(PIPELINE_PATH)
        logger.info("Feature pipeline loaded from %s", PIPELINE_PATH)
    else:
        _pipeline = FeaturePipeline()
        logger.warning("Using default feature pipeline (not fitted)")


@app.on_event("startup")
async def startup() -> None:
    _load_artifacts()


def _request_to_dataframe(req: PredictionRequest) -> pd.DataFrame:
    """Convert a single request to a DataFrame row."""
    return pd.DataFrame([req.model_dump()])


def _predict_single(req: PredictionRequest) -> PredictionResponse:
    """Generate prediction for a single order."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _request_to_dataframe(req)
    X = _pipeline.transform_predict(df)
    eta = float(_model.predict(X)[0])

    # Simple confidence interval (±20% as placeholder; replace with quantile regression)
    margin = eta * 0.15
    return PredictionResponse(
        estimated_delivery_minutes=round(eta, 1),
        confidence_interval={
            "lower": round(max(eta - margin, 5.0), 1),
            "upper": round(eta + margin, 1),
        },
        model_version=_model_version,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict ETA for a single delivery order."""
    return _predict_single(request)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict ETA for a batch of delivery orders."""
    predictions = [_predict_single(order) for order in request.orders]
    return BatchPredictionResponse(predictions=predictions)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version=__version__,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return model metadata."""
    if _pipeline is None or not _pipeline.feature_names_:
        raise HTTPException(status_code=503, detail="Pipeline not fitted")
    return ModelInfoResponse(
        model_name="LightGBM",
        model_version=_model_version,
        n_features=len(_pipeline.feature_names_),
        feature_names=_pipeline.feature_names_,
    )
