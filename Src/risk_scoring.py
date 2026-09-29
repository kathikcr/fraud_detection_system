"""Convert fitted model outputs to explicit, typed 0–100 risk scores."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
RiskScoreKind = Literal["probability", "anomaly"]


class RiskScoringError(ValueError):
    """Raised when an estimator, risk scale, or transaction input is invalid."""


@dataclass(frozen=True)
class RiskBandPolicy:
    """Configurable display-only bands over the resulting 0–100 score."""

    low_upper: float = 30.0
    medium_upper: float = 70.0

    def __post_init__(self):
        values = (self.low_upper, self.medium_upper)
        if not all(np.isfinite(value) for value in values) or not 0 < self.low_upper < self.medium_upper < 100:
            raise RiskScoringError("Risk bands must satisfy 0 < low_upper < medium_upper < 100")


@dataclass(frozen=True)
class RiskScore:
    model_name: str
    score: float
    risk_level: Literal["low", "medium", "high"]
    score_kind: RiskScoreKind
    score_basis: str


@dataclass(frozen=True)
class RiskScorer:
    """Inference adapter for one fitted model and its explicit score scale."""

    model_name: str
    estimator: object
    score_kind: RiskScoreKind
    feature_names: tuple[str, ...]
    policy: RiskBandPolicy
    sorted_reference_anomaly_scores: tuple[float, ...] = ()

    def score_transaction(self, features: pd.DataFrame) -> RiskScore:
        """Score one already-preprocessed transaction row."""
        if not isinstance(features, pd.DataFrame) or len(features) != 1:
            raise RiskScoringError("score_transaction requires a one-row preprocessed pandas DataFrame")
        return self.score_transactions(features)[0]

    def score_transactions(self, features: pd.DataFrame) -> tuple[RiskScore, ...]:
        """Score a non-empty batch of already-preprocessed feature rows."""
        _validate_feature_frame(features, self.feature_names)
        if self.score_kind == "probability":
            classes = list(getattr(self.estimator, "classes_", ()))
            if 1 not in classes or not hasattr(self.estimator, "predict_proba"):
                raise RiskScoringError("Probability scoring requires a fitted estimator with positive class 1")
            probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape != (len(features), len(classes)):
                raise RiskScoringError("Estimator returned malformed probability output")
            raw_scores = probabilities[:, classes.index(1)]
            if not np.isfinite(raw_scores).all() or ((raw_scores < 0) | (raw_scores > 1)).any():
                raise RiskScoringError("Estimator probabilities must be finite values between 0 and 1")
            risk_values = raw_scores * 100.0
            basis = "uncalibrated model probability × 100"
        elif self.score_kind == "anomaly":
            if not self.sorted_reference_anomaly_scores:
                raise RiskScoringError("Anomaly scoring requires training-reference anomaly scores")
            if not hasattr(self.estimator, "score_samples"):
                raise RiskScoringError("Anomaly scoring requires a fitted estimator with score_samples")
            anomaly_scores = -np.asarray(self.estimator.score_samples(features), dtype=float)
            if not np.isfinite(anomaly_scores).all():
                raise RiskScoringError("Estimator returned non-finite anomaly scores")
            reference = np.asarray(self.sorted_reference_anomaly_scores)
            risk_values = np.searchsorted(reference, anomaly_scores, side="right") / len(reference) * 100.0
            basis = "percentile of training-reference anomaly scores"
        else:
            raise RiskScoringError(f"Unsupported score kind: {self.score_kind}")

        results = tuple(
            RiskScore(
                model_name=self.model_name,
                score=float(value),
                risk_level=_risk_level(float(value), self.policy),
                score_kind=self.score_kind,
                score_basis=basis,
            )
            for value in risk_values
        )
        LOGGER.info(
            "risk_scores_generated",
            extra={"event_type": "risk_scores_generated", "model": self.model_name,
                   "score_kind": self.score_kind, "row_count": len(results)},
        )
        return results


def build_risk_scorer(
    model_name: str,
    estimator,
    *,
    score_kind: RiskScoreKind,
    feature_names: tuple[str, ...],
    reference_features: pd.DataFrame | None = None,
    policy: RiskBandPolicy | None = None,
) -> RiskScorer:
    """Bind one already-fitted model to a 0–100 score scale.

    For anomaly models, pass only training features as `reference_features`.
    The resulting empirical percentile uses no fraud labels.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise RiskScoringError("model_name must be a non-empty string")
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise RiskScoringError("feature_names must be non-empty and unique")
    if not hasattr(estimator, "n_features_in_"):
        raise RiskScoringError("Risk scoring requires an already-fitted estimator")
    if int(estimator.n_features_in_) != len(feature_names):
        raise RiskScoringError("Estimator feature count does not match feature_names")
    fitted_names = getattr(estimator, "feature_names_in_", None)
    if fitted_names is not None and tuple(fitted_names) != tuple(feature_names):
        raise RiskScoringError("Estimator feature names do not match feature_names")
    if score_kind not in ("probability", "anomaly"):
        raise RiskScoringError(f"Unsupported score_kind: {score_kind}")

    sorted_reference: tuple[float, ...] = ()
    if score_kind == "probability":
        if not hasattr(estimator, "predict_proba") or 1 not in list(getattr(estimator, "classes_", ())):
            raise RiskScoringError("Probability scoring requires a fitted estimator with positive class 1")
        if reference_features is not None:
            raise RiskScoringError("reference_features are only used for anomaly score normalization")
    else:
        if reference_features is None:
            raise RiskScoringError("Anomaly scoring requires training-only reference_features")
        _validate_feature_frame(reference_features, tuple(feature_names))
        if not hasattr(estimator, "score_samples"):
            raise RiskScoringError("Anomaly scoring requires a fitted estimator with score_samples")
        values = -np.asarray(estimator.score_samples(reference_features), dtype=float)
        if not np.isfinite(values).all():
            raise RiskScoringError("Reference anomaly scores must be finite")
        sorted_reference = tuple(float(value) for value in np.sort(values))

    return RiskScorer(
        model_name=model_name.strip(), estimator=estimator, score_kind=score_kind,
        feature_names=tuple(feature_names), policy=policy or RiskBandPolicy(),
        sorted_reference_anomaly_scores=sorted_reference,
    )


def _validate_feature_frame(features: pd.DataFrame, expected_names: tuple[str, ...]) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise RiskScoringError("Feature input must be a non-empty pandas DataFrame")
    if tuple(features.columns) != expected_names:
        raise RiskScoringError("Feature columns and order do not match the fitted model")
    try:
        values = features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise RiskScoringError("Feature values must be numeric") from exc
    if not np.isfinite(values).all():
        raise RiskScoringError("Feature values must be finite; preprocess/impute them before scoring")


def _risk_level(score: float, policy: RiskBandPolicy) -> Literal["low", "medium", "high"]:
    if score < policy.low_upper:
        return "low"
    if score < policy.medium_upper:
        return "medium"
    return "high"
