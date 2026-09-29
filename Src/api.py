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
from fastapi.responses import HTMLResponse, JSONResponse
from Src.dashboard import dashboard_shell
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr, model_validator

from Src.artifacts import LoadedArtifact, load_model_artifact
from Src.inference import FraudInference, InferenceError
from Src.investigation import InvestigationError, TransactionInvestigator

LOGGER = logging.getLogger(__name__)
StrictNumber = StrictFloat | StrictInt
MAX_BACKGROUND_CSV_BYTES = 1_000_000
MAX_BACKGROUND_ROWS = 100


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


class InvestigationRequest(BaseModel):
    """Case reference kept separate from the Dataset 2 feature object."""

    model_config = ConfigDict(extra="forbid", strict=True)

    transaction_id: StrictStr | StrictInt
    transaction: PredictionRequest

    @model_validator(mode="after")
    def validate_transaction_id(self):
        value = self.transaction_id
        if isinstance(value, str) and (not value.strip() or len(value.strip()) > 128):
            raise ValueError("transaction_id must contain 1 to 128 characters")
        return self


class FeatureContributionResponse(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    model_name: str
    score_kind: Literal["probability", "anomaly"]
    score_basis: str
    score: float
    baseline_score: float
    output_scale: str
    attributions: list[FeatureContributionResponse]
    max_evals: int


class InvestigationResponse(BaseModel):
    transaction_id: str
    time: float
    amount: float
    prediction: PredictionResponse
    explanation: ExplanationResponse
    top_contributors: list[FeatureContributionResponse]
    investigation_ms: float


def create_app(
    *,
    inference_service: FraudInference | None = None,
    investigator_service: TransactionInvestigator | None = None,
    artifact_dir: str | Path | None = None,
    background_path: str | Path | None = None,
) -> FastAPI:
    """Construct API routes; load configured resources on first scoring request.

    `FRAUD_MODEL_ARTIFACT_DIR` is the runtime configuration key. The model
    path is never accepted from an HTTP request and no local absolute path is
    embedded in this module.
    """
    if inference_service is not None and artifact_dir is not None:
        raise ValueError("Provide either inference_service or artifact_dir, not both")
    if inference_service is not None and background_path is not None:
        raise ValueError("Provide either inference_service or background_path, not both")
    if inference_service is not None and investigator_service is not None:
        raise ValueError("Provide at most one injected model service")
    if inference_service is not None and not isinstance(inference_service, FraudInference):
        raise TypeError("inference_service must be a FraudInference instance")
    if investigator_service is not None and artifact_dir is not None:
        raise ValueError("Provide either investigator_service or artifact_dir, not both")
    if investigator_service is not None and not isinstance(investigator_service, TransactionInvestigator):
        raise TypeError("investigator_service must be a TransactionInvestigator instance")
    if background_path is not None and investigator_service is not None:
        raise ValueError("Provide either investigator_service or background_path, not both")

    application = FastAPI(
        title="Fraud Detection Analytics API",
        version="0.2.0",
        description="Incremental API for the fraud detection platform.",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    application.state.inference_service = inference_service
    application.state.investigator_service = investigator_service
    application.state.loaded_artifact = None
    application.state.artifact_dir = str(artifact_dir) if artifact_dir is not None else None
    application.state.background_path = str(background_path) if background_path is not None else None
    application.state.inference_lock = threading.RLock()

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard():
        """Serve the directly accessible local dashboard shell."""
        return dashboard_shell()

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

    @application.post("/investigate", response_model=InvestigationResponse, tags=["investigation"])
    def investigate(payload: InvestigationRequest, request: Request) -> InvestigationResponse:
        """Return one case's prediction and local feature attributions."""
        service = _get_investigator_service(request)
        try:
            result = service.investigate(payload.transaction_id, payload.transaction.model_dump())
        except InvestigationError as exc:
            LOGGER.exception(
                "api_investigation_failed",
                extra={"event_type": "api_investigation_failed", "model": "configured",
                       "error_type": type(exc).__name__},
            )
            raise HTTPException(status_code=500, detail="Investigation failed") from exc
        return InvestigationResponse(**asdict(result))

    return application


def _get_inference_service(request: Request) -> FraudInference:
    """Resolve and cache the configured model service outside request data."""
    application = request.app
    with application.state.inference_lock:
        service = application.state.inference_service
        if service is not None:
            return service
        try:
            artifact = _get_loaded_artifact_locked(application)
            service = FraudInference(artifact)
        except Exception as exc:
            LOGGER.error("api_model_load_failed", extra={
                "event_type": "api_model_load_failed", "error_type": type(exc).__name__
            })
            raise HTTPException(status_code=503, detail="Prediction service is unavailable") from exc
        application.state.inference_service = service
        return service


def _get_investigator_service(request: Request) -> TransactionInvestigator:
    """Resolve and cache the artifact plus training-only SHAP background."""
    application = request.app
    with application.state.inference_lock:
        service = application.state.investigator_service
        if service is not None:
            return service
        try:
            artifact = _get_loaded_artifact_locked(application)
            configured_background = (
                application.state.background_path or os.getenv("FRAUD_SHAP_BACKGROUND_PATH")
            )
            if not configured_background:
                raise FileNotFoundError("training background is not configured")
            background = _read_training_background(configured_background)
            service = TransactionInvestigator(artifact, background)
        except Exception as exc:
            LOGGER.error("api_investigator_unavailable", extra={
                "event_type": "api_investigator_unavailable", "error_type": type(exc).__name__
            })
            raise HTTPException(status_code=503, detail="Investigation service is unavailable") from exc
        application.state.investigator_service = service
        return service


def _get_loaded_artifact_locked(application) -> LoadedArtifact:
    artifact = application.state.loaded_artifact
    if artifact is not None:
        return artifact
    configured_dir = application.state.artifact_dir or os.getenv("FRAUD_MODEL_ARTIFACT_DIR")
    if not configured_dir:
        LOGGER.error("api_model_not_configured", extra={
            "event_type": "api_model_not_configured", "reason": "artifact_not_configured"
        })
        raise FileNotFoundError("model artifact is not configured")
    artifact = load_model_artifact(configured_dir)
    application.state.loaded_artifact = artifact
    return artifact


def _read_training_background(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("SHAP background file cannot be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size > MAX_BACKGROUND_CSV_BYTES:
        raise ValueError("SHAP background file is missing or exceeds the size limit")
    background = pd.read_csv(source)
    if background.empty or len(background) > MAX_BACKGROUND_ROWS:
        raise ValueError(f"SHAP background must contain 1 to {MAX_BACKGROUND_ROWS} rows")
    values = background.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("SHAP background values must be finite")
    return background


app = create_app()
