"""Local SHAP explanations for saved Dataset 2 model scores."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from Src.artifacts import LoadedArtifact
from Src.risk_scoring import RiskScoringError

LOGGER = logging.getLogger(__name__)
DEFAULT_BACKGROUND_ROWS = 100
MAX_BACKGROUND_ROWS = 100
DEFAULT_MAX_EVALS = 500
MAX_EVALS = 10_000
MAX_EXPLANATION_ROWS = 32


class ExplainabilityError(ValueError):
    """Raised when SHAP explanation inputs or output are invalid."""


@dataclass(frozen=True)
class FeatureContribution:
    """One feature value and its signed local SHAP contribution."""

    feature_name: str
    feature_value: float
    shap_value: float


@dataclass(frozen=True)
class LocalExplanation:
    """SHAP decomposition of one model's existing 0–100 risk score."""

    model_name: str
    score_kind: str
    score_basis: str
    score: float
    baseline_score: float
    output_scale: str
    attributions: tuple[FeatureContribution, ...]
    max_evals: int


class FraudExplainer:
    """Explain risk scores for preprocessed Dataset 2 feature rows.

    An independent masker uses a bounded, deterministic sample of training
    features as its background. All supported estimators share the SHAP
    permutation explainer, so values explain the same 0–100 output returned by
    the risk scorer. For Isolation Forest, that output is a training-reference
    anomaly percentile, not a fraud probability.
    """

    def __init__(
        self,
        artifact: LoadedArtifact,
        background_features: pd.DataFrame,
        *,
        seed: int = 42,
        background_rows: int = DEFAULT_BACKGROUND_ROWS,
    ):
        if not isinstance(artifact, LoadedArtifact):
            raise ExplainabilityError("artifact must be a LoadedArtifact returned by the project artifact loader")
        if artifact.manifest.get("target_column") != "Class":
            raise ExplainabilityError("Explainability supports Dataset 2 artifacts targeting Class only")
        if artifact.manifest.get("model_type") != type(artifact.estimator).__name__:
            raise ExplainabilityError("Artifact estimator type does not match its manifest")
        if tuple(artifact.manifest.get("feature_names", ())) != tuple(artifact.risk_scorer.feature_names):
            raise ExplainabilityError("Artifact risk scorer features do not match its manifest")
        _validate_frame(background_features, artifact.risk_scorer.feature_names, "background_features")
        if type(seed) is not int:
            raise ExplainabilityError("seed must be an integer")
        if type(background_rows) is not int or not 1 <= background_rows <= MAX_BACKGROUND_ROWS:
            raise ExplainabilityError(f"background_rows must be between 1 and {MAX_BACKGROUND_ROWS}")

        self._artifact = artifact
        self.feature_names = tuple(artifact.risk_scorer.feature_names)
        self.background_features = _bounded_background(background_features, background_rows, seed)
        self._model = shap.Explainer(
            self._score_function,
            masker=shap.maskers.Independent(self.background_features.to_numpy(dtype=float)),
            algorithm="permutation",
            feature_names=list(self.feature_names),
            seed=seed,
        )

    def explain_transactions(
        self,
        features: pd.DataFrame,
        *,
        max_evals: int = DEFAULT_MAX_EVALS,
    ) -> tuple[LocalExplanation, ...]:
        """Explain up to 32 preprocessed transactions, retaining score units."""
        _validate_frame(features, self.feature_names, "features")
        if len(features) > MAX_EXPLANATION_ROWS:
            raise ExplainabilityError(f"At most {MAX_EXPLANATION_ROWS} transactions can be explained at once")
        min_evals = 2 * len(self.feature_names) + 1
        if type(max_evals) is not int or not min_evals <= max_evals <= MAX_EVALS:
            raise ExplainabilityError(f"max_evals must be an integer from {min_evals} to {MAX_EVALS}")

        started = time.perf_counter()
        try:
            values = features.loc[:, self.feature_names]
            explanation = self._model(values, max_evals=max_evals, silent=True)
            shap_values = np.asarray(explanation.values, dtype=float)
            base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            if shap_values.shape != (len(values), len(self.feature_names)):
                raise ExplainabilityError("SHAP returned an unexpected attribution shape")
            if base_values.size == 1 and len(values) > 1:
                base_values = np.repeat(base_values, len(values))
            if base_values.size != len(values) or not np.isfinite(shap_values).all() or not np.isfinite(base_values).all():
                raise ExplainabilityError("SHAP returned non-finite or incomplete attribution values")

            scores = self._score_function(values.to_numpy(dtype=float))
            results = []
            for index, row_values in enumerate(shap_values):
                score = float(scores[index])
                reconstructed = float(base_values[index] + row_values.sum())
                if not np.isclose(reconstructed, score, rtol=1e-5, atol=1e-4):
                    raise ExplainabilityError("SHAP values do not reconstruct the model score")
                results.append(LocalExplanation(
                    model_name=self._artifact.manifest["model_name"],
                    score_kind=self._artifact.manifest["score_kind"],
                    score_basis=self._artifact.manifest["score_basis"],
                    score=score,
                    baseline_score=float(base_values[index]),
                    output_scale="risk score on the 0–100 scale",
                    attributions=tuple(
                        FeatureContribution(str(name), float(feature_value), float(shap_value))
                        for name, feature_value, shap_value in zip(
                            self.feature_names, values.iloc[index].to_numpy(dtype=float), row_values, strict=True
                        )
                    ),
                    max_evals=max_evals,
                ))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "risk_score_explanation_complete",
                extra={"event_type": "risk_score_explanation_complete",
                       "model": self._artifact.manifest["model_name"],
                       "score_kind": self._artifact.manifest["score_kind"],
                       "row_count": len(results), "max_evals": max_evals,
                       "inference_ms": round(elapsed_ms, 3)},
            )
            return tuple(results)
        except ExplainabilityError:
            raise
        except (RiskScoringError, ValueError, TypeError, RuntimeError) as exc:
            LOGGER.warning(
                "risk_score_explanation_failed",
                extra={"event_type": "risk_score_explanation_failed",
                       "model": self._artifact.manifest.get("model_name", "unknown"),
                       "error_type": type(exc).__name__},
            )
            raise ExplainabilityError(f"Could not explain model scores: {exc}") from exc

    def _score_function(self, matrix) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names) or not np.isfinite(values).all():
            raise ExplainabilityError("SHAP generated an invalid model input")
        frame = pd.DataFrame(values, columns=self.feature_names)
        scores = self._artifact.risk_scorer.score_transactions(frame)
        return np.asarray([result.score for result in scores], dtype=float)


def _validate_frame(features: pd.DataFrame, expected_names: tuple[str, ...], label: str) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ExplainabilityError(f"{label} must be a non-empty pandas DataFrame")
    if tuple(features.columns) != tuple(expected_names):
        raise ExplainabilityError(f"{label} columns and order must match the fitted model")
    try:
        values = features.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExplainabilityError(f"{label} values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ExplainabilityError(f"{label} values must be finite")


def _bounded_background(features: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(features) <= limit:
        return features.copy()
    return features.sample(n=limit, replace=False, random_state=seed).copy()
