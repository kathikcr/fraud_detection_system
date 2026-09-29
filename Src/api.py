"""Incremental HTTP API for the fraud detection platform."""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, model_validator

from Src.artifacts import load_model_artifact
from Src.inference import FraudInference, InferenceError

LOGGER = logging.getLogger(__name__)
StrictNumber = StrictFloat | StrictInt


class HealthResponse(BaseModel):
    """Liveness status for the running API process."""

    status: Literal["ok"]


class PredictionRequest(BaseModel):
    """Strict raw Dataset 2 transaction schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    Time: StrictNumber
    V1: StrictNumber
    V2: StrictNumber
    V3: StrictNumber
    V4: StrictNumber
    V5: StrictNumber
    V6: StrictNumber
    V7: StrictNumber
    V8: StrictNumber
    V9: StrictNumber
    V10: StrictNumber
    V11: StrictNumber
    V12: StrictNumber
    V13: StrictNumber
    V14: StrictNumber
    V15: StrictNumber
    V16: StrictNumber
    V17: StrictNumber
    V18: StrictNumber
    V19: StrictNumber
    V20: StrictNumber
    V21: StrictNumber
    V22: StrictNumber
    V23: StrictNumber
    V24: StrictNumber
    V25: StrictNumber
    V26: StrictNumber
    V27: StrictNumber
    V28: StrictNumber
    Amount: StrictNumber

    @model_validator(mode="after")
    def validate_numeric_constraints(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            try:
                finite = math.isfinite(value)
            except (OverflowError, TypeError):
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")
        if self.Time < 0:
            raise ValueError("Time cannot be negative")
        if self.Amount < 0:
            raise ValueError("Amount cannot be negative")
        return self


class PredictionResponse(BaseModel):
    """Public prediction response; model score semantics stay explicit."""

    model_name: str
    score: float
    score_kind: Literal["probability", "anomaly"]
    score_basis: str
    risk_level: Literal["low", "medium", "high"]
    model_alert: bool
    decision_method: str
    inference_ms: float


def create_app(
    *,
    inference_service: FraudInference | None = None,
    artifact_dir: str | Path | None = None,
) -> FastAPI:
    """Construct API routes; load a configured artifact once on first prediction.

    `FRAUD_MODEL_ARTIFACT_DIR` is the runtime configuration key. The model
    path is never accepted from an HTTP request and no local absolute path is
    embedded in this module.
    """
    if inference_service is not None and artifact_dir is not None:
        raise ValueError("Provide either inference_service or artifact_dir, not both")
    if inference_service is not None and not isinstance(inference_service, FraudInference):
        raise TypeError("inference_service must be a FraudInference instance")

    application = FastAPI(
        title="Fraud Detection Analytics API",
        version="0.2.0",
        description="Incremental API for the fraud detection platform.",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    application.state.inference_service = inference_service
    application.state.artifact_dir = str(artifact_dir) if artifact_dir is not None else None
    application.state.inference_lock = threading.Lock()

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"loc": list(error.get("loc", ())), "msg": error.get("msg", "Invalid request"),
             "type": error.get("type", "value_error")}
            for error in exc.errors()
        ]
        LOGGER.warning(
            "api_request_rejected",
            extra={"event_type": "api_request_rejected", "path": request.url.path,
                   "method": request.method, "error_count": len(errors)},
        )
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Report that the HTTP process can serve requests."""
        LOGGER.info("api_health_check", extra={"event_type": "api_health_check", "status": "ok"})
        return HealthResponse(status="ok")

    @application.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(
        payload: PredictionRequest,
        request: Request,
    ) -> PredictionResponse:
        """Return one transaction's risk score and model alert decision."""
        service = _get_inference_service(request)
        try:
            result = service.predict_transaction(payload.model_dump())
        except InferenceError as exc:
            LOGGER.exception(
                "api_prediction_failed",
                extra={"event_type": "api_prediction_failed", "model": "configured",
                       "error_type": type(exc).__name__},
            )
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        return PredictionResponse(**asdict(result))

    return application


def _get_inference_service(request: Request) -> FraudInference:
    """Resolve and cache the configured model service outside request data."""
    application = request.app
    service = application.state.inference_service
    if service is not None:
        return service
    with application.state.inference_lock:
        service = application.state.inference_service
        if service is not None:
            return service
        configured_dir = application.state.artifact_dir or os.getenv("FRAUD_MODEL_ARTIFACT_DIR")
        if not configured_dir:
            LOGGER.error("api_prediction_unavailable", extra={
                "event_type": "api_prediction_unavailable", "reason": "artifact_not_configured"
            })
            raise HTTPException(status_code=503, detail="Prediction service is unavailable")
        try:
            artifact = load_model_artifact(configured_dir)
            service = FraudInference(artifact)
        except Exception as exc:
            LOGGER.error("api_model_load_failed", extra={
                "event_type": "api_model_load_failed", "error_type": type(exc).__name__
            })
            raise HTTPException(status_code=503, detail="Prediction service is unavailable") from exc
        application.state.inference_service = service
        return service


app = create_app()
