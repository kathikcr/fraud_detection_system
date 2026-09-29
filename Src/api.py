"""HTTP API shell; endpoints are added one feature at a time."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Liveness status for the running API process."""

    status: Literal["ok"]


def create_app() -> FastAPI:
    """Construct the API application without loading models or datasets."""
    application = FastAPI(
        title="Fraud Detection Analytics API",
        version="0.1.0",
        description="Incremental API for the fraud detection platform.",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Report that the HTTP process can serve requests."""
        LOGGER.info("api_health_check", extra={"event_type": "api_health_check", "status": "ok"})
        return HealthResponse(status="ok")

    return application


app = create_app()
