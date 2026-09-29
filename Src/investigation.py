"""One-transaction investigation view joining inference and SHAP evidence."""

from __future__ import annotations

import logging
import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Src.artifacts import LoadedArtifact
from Src.explainability import (
    DEFAULT_MAX_EVALS,
    DEFAULT_BACKGROUND_ROWS,
    ExplainabilityError,
    FeatureContribution,
    FraudExplainer,
    LocalExplanation,
)
from Src.inference import FraudInference, InferenceError, TransactionPrediction
from Src.preprocessing import DATASET2_FEATURES

LOGGER = logging.getLogger(__name__)


class InvestigationError(ValueError):
    """Raised when an investigation cannot be assembled safely."""


@dataclass(frozen=True)
class TransactionInvestigation:
    """Prediction and local explanatory evidence for a caller-identified case."""

    transaction_id: str
    time: float
    amount: float
    prediction: TransactionPrediction
    explanation: LocalExplanation
    top_contributors: tuple[FeatureContribution, ...]
    investigation_ms: float


class TransactionInvestigator:
    """Combine the existing inference and explanation layers for review."""

    def __init__(
        self,
        artifact: LoadedArtifact,
        training_background: pd.DataFrame,
        *,
        seed: int = 42,
        background_rows: int = DEFAULT_BACKGROUND_ROWS,
        max_evals: int = DEFAULT_MAX_EVALS,
        top_contributors: int = 5,
    ):
        if type(max_evals) is not int:
            raise InvestigationError("max_evals must be an integer")
        if type(top_contributors) is not int or not 1 <= top_contributors <= len(DATASET2_FEATURES):
            raise InvestigationError(f"top_contributors must be from 1 to {len(DATASET2_FEATURES)}")
        self._inference = FraudInference(artifact)
        self._explainer = FraudExplainer(
            artifact, training_background, seed=seed, background_rows=background_rows
        )
        self._artifact = artifact
        self._max_evals = max_evals
        self._top_contributors = top_contributors

    def investigate(
        self,
        transaction_id: str | int,
        transaction: Mapping[str, object],
    ) -> TransactionInvestigation:
        """Score and explain a transaction; identifier is metadata only."""
        case_id = _validate_transaction_id(transaction_id)
        started = time.perf_counter()
        try:
            prediction = self._inference.predict_transaction(transaction)
            raw = pd.DataFrame(
                [{feature: transaction[feature] for feature in DATASET2_FEATURES}],
                columns=DATASET2_FEATURES,
            )
            transformed_values = np.asarray(self._artifact.preprocessor.transform(raw), dtype=float)
            names = tuple(self._artifact.risk_scorer.feature_names)
            if transformed_values.shape != (1, len(names)) or not np.isfinite(transformed_values).all():
                raise InvestigationError("Artifact preprocessing returned invalid model features")
            transformed = pd.DataFrame(transformed_values, columns=names)
            explanation = self._explainer.explain_transactions(
                transformed, max_evals=self._max_evals
            )[0]
            if explanation.score_kind != prediction.score_kind or not np.isclose(
                explanation.score, prediction.score, rtol=1e-6, atol=1e-4
            ):
                raise InvestigationError("Explanation does not match the transaction prediction")
            top = tuple(sorted(
                explanation.attributions,
                key=lambda item: abs(item.shap_value),
                reverse=True,
            )[:self._top_contributors])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result = TransactionInvestigation(
                transaction_id=case_id,
                time=float(transaction["Time"]),
                amount=float(transaction["Amount"]),
                prediction=prediction,
                explanation=explanation,
                top_contributors=top,
                investigation_ms=elapsed_ms,
            )
            LOGGER.info(
                "transaction_investigation_complete",
                extra={"event_type": "transaction_investigation_complete",
                       "model": prediction.model_name, "score_kind": prediction.score_kind,
                       "model_alert": prediction.model_alert, "contributor_count": len(top),
                       "investigation_ms": round(elapsed_ms, 3)},
            )
            return result
        except (InferenceError, ExplainabilityError) as exc:
            LOGGER.warning(
                "transaction_investigation_rejected",
                extra={"event_type": "transaction_investigation_rejected",
                       "model": self._artifact.manifest.get("model_name", "unknown"),
                       "error_type": type(exc).__name__},
            )
            raise InvestigationError(f"Could not investigate transaction: {exc}") from exc


def _validate_transaction_id(value: str | int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, numbers.Integral)):
        raise InvestigationError("transaction_id must be a non-empty string or integer")
    result = str(value).strip()
    if not result or len(result) > 128:
        raise InvestigationError("transaction_id must contain 1 to 128 characters")
    return result
