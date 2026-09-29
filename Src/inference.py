"""Validated raw Dataset 2 transaction inference over loaded model artifacts."""

from __future__ import annotations

import logging
import math
import numbers
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Src.artifacts import LoadedArtifact
from Src.preprocessing import DATASET2_FEATURES
from Src.risk_scoring import RiskScoringError

LOGGER = logging.getLogger(__name__)


class InferenceError(ValueError):
    """Raised when an input transaction or loaded model cannot be scored."""


@dataclass(frozen=True)
class TransactionPrediction:
    """Model alert and typed risk score for one transaction."""

    model_name: str
    score: float
    score_kind: str
    score_basis: str
    risk_level: str
    model_alert: bool
    decision_method: str
    inference_ms: float


class FraudInference:
    """Transform and score raw Dataset 2 transaction mappings.

    The caller supplies an already loaded, project-validated artifact. This
    layer never accepts a filesystem path or loads a model per request.
    """

    def __init__(self, artifact: LoadedArtifact):
        if not isinstance(artifact, LoadedArtifact):
            raise InferenceError("artifact must be a LoadedArtifact returned by the project artifact loader")
        if artifact.manifest.get("target_column") != "Class":
            raise InferenceError("Inference supports Dataset 2 artifacts targeting Class only")
        if tuple(artifact.manifest.get("feature_names", ())) != tuple(artifact.risk_scorer.feature_names):
            raise InferenceError("Artifact risk scorer features do not match its manifest")
        if artifact.manifest.get("model_type") != type(artifact.estimator).__name__:
            raise InferenceError("Artifact estimator type does not match its manifest")
        self._artifact = artifact

    def predict_transaction(self, transaction: Mapping[str, object]) -> TransactionPrediction:
        """Validate, transform, and score one raw Dataset 2 transaction."""
        return self.predict_transactions([transaction])[0]

    def predict_transactions(
        self, transactions: Sequence[Mapping[str, object]]
    ) -> tuple[TransactionPrediction, ...]:
        """Validate and score a non-empty batch atomically."""
        started = time.perf_counter()
        try:
            rows = _validate_transactions(transactions)
            raw = pd.DataFrame(rows, columns=DATASET2_FEATURES)
            transformed_values = self._artifact.preprocessor.transform(raw)
            transformed_values = np.asarray(transformed_values, dtype=float)
            if transformed_values.shape != (len(rows), len(self._artifact.risk_scorer.feature_names)):
                raise InferenceError("Preprocessor returned an unexpected feature shape")
            if not np.isfinite(transformed_values).all():
                raise InferenceError("Preprocessing produced non-finite model features")
            transformed = pd.DataFrame(
                transformed_values, columns=self._artifact.risk_scorer.feature_names
            )
            scores = self._artifact.risk_scorer.score_transactions(transformed)
            decision_method, alerts = self._model_alerts(transformed, scores)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results = tuple(
                TransactionPrediction(
                    model_name=score.model_name,
                    score=score.score,
                    score_kind=score.score_kind,
                    score_basis=score.score_basis,
                    risk_level=score.risk_level,
                    model_alert=bool(alert),
                    decision_method=decision_method,
                    inference_ms=elapsed_ms,
                )
                for score, alert in zip(scores, alerts, strict=True)
            )
            LOGGER.info(
                "transaction_inference_complete",
                extra={"event_type": "transaction_inference_complete",
                       "model": self._artifact.manifest["model_name"],
                       "score_kind": self._artifact.manifest["score_kind"],
                       "row_count": len(results), "inference_ms": round(elapsed_ms, 3)},
            )
            return results
        except (InferenceError, RiskScoringError) as exc:
            LOGGER.warning(
                "transaction_inference_rejected",
                extra={"event_type": "transaction_inference_rejected",
                       "model": self._artifact.manifest.get("model_name", "unknown"),
                       "error_type": type(exc).__name__},
            )
            if isinstance(exc, InferenceError):
                raise
            raise InferenceError(f"Transaction could not be scored: {exc}") from exc
        except Exception as exc:
            LOGGER.exception(
                "transaction_inference_failed",
                extra={"event_type": "transaction_inference_failed",
                       "model": self._artifact.manifest.get("model_name", "unknown"),
                       "error_type": type(exc).__name__},
            )
            raise InferenceError("Transaction inference failed") from exc

    def _model_alerts(self, transformed: pd.DataFrame, scores) -> tuple[str, np.ndarray]:
        manifest = self._artifact.manifest
        if manifest["score_kind"] == "probability":
            threshold = manifest["decision_threshold"]
            estimator = self._artifact.estimator
            classes = list(getattr(estimator, "classes_", ()))
            if 1 not in classes or not hasattr(estimator, "predict_proba"):
                raise InferenceError("Supervised artifact must expose positive-class probabilities")
            probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
            if (probabilities.ndim != 2 or probabilities.shape != (len(transformed), len(classes))
                    or not np.isfinite(probabilities).all()):
                raise InferenceError("Estimator returned malformed probabilities for its decision rule")
            alerts = probabilities[:, classes.index(1)] >= threshold
            return f"positive-class probability >= {threshold:g}", alerts
        predictions = np.asarray(self._artifact.estimator.predict(transformed))
        if predictions.shape != (len(transformed),) or not np.isin(predictions, (-1, 1)).all():
            raise InferenceError("Isolation Forest returned malformed native decisions")
        return "Isolation Forest native predict == -1", predictions == -1


def _validate_transactions(transactions: Sequence[Mapping[str, object]]) -> list[dict[str, float]]:
    if isinstance(transactions, (str, bytes, Mapping)) or not isinstance(transactions, Sequence):
        raise InferenceError("transactions must be a sequence of transaction objects")
    if not transactions:
        raise InferenceError("transactions must contain at least one transaction")
    validated = []
    expected = set(DATASET2_FEATURES)
    for index, transaction in enumerate(transactions):
        if not isinstance(transaction, Mapping):
            raise InferenceError(f"Transaction at index {index} must be an object")
        if any(not isinstance(key, str) for key in transaction):
            raise InferenceError(f"Transaction at index {index} contains a non-string feature name")
        provided = set(transaction)
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing:
            raise InferenceError(f"Transaction at index {index} is missing required features: {missing}")
        if extra:
            raise InferenceError(f"Transaction at index {index} contains unsupported fields: {extra}")
        row = {}
        for feature in DATASET2_FEATURES:
            value = transaction[feature]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise InferenceError(f"Transaction at index {index} feature {feature} must be a numeric value")
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise InferenceError(f"Transaction at index {index} feature {feature} must be finite") from exc
            if not math.isfinite(numeric):
                raise InferenceError(f"Transaction at index {index} feature {feature} must be finite")
            if feature in ("Time", "Amount") and numeric < 0:
                raise InferenceError(f"Transaction at index {index} feature {feature} cannot be negative")
            row[feature] = numeric
        validated.append(row)
    return validated
