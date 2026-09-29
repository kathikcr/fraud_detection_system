"""Constrained save/load for fitted Dataset 2 model bundles.

Bundles use skops (not pickle/joblib), a fixed manifest, checksums, explicit
estimator/type allowlists, and exact feature/preprocessing metadata. Load only
artifacts created by this project in a trusted local directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
import skops
from skops import io as skops_io
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from Src.risk_scoring import RiskBandPolicy, RiskScorer

LOGGER = logging.getLogger(__name__)
MANIFEST_NAME = "manifest.json"
BUNDLE_NAME = "bundle.skops"
REFERENCE_NAME = "anomaly_reference.npy"
FORMAT_VERSION = 1
MAX_MANIFEST_BYTES = 2_000_000
MAX_ARTIFACT_FILE_BYTES = 2_000_000_000
MAX_TRAIN_ROWS = 100_000_000

SUPPORTED_MODELS = {
    "LogisticRegression": (LogisticRegression, "probability"),
    "RandomForestClassifier": (RandomForestClassifier, "probability"),
    "XGBClassifier": (XGBClassifier, "probability"),
    "IsolationForest": (IsolationForest, "anomaly"),
}
ALLOWED_UNTRUSTED_TYPES = {
    "numpy.dtype",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
}


class ArtifactError(ValueError):
    """Raised for unsupported, corrupt, incompatible, or unsafe artifacts."""


@dataclass(frozen=True)
class LoadedArtifact:
    estimator: object
    preprocessor: ColumnTransformer
    risk_scorer: RiskScorer
    manifest: Mapping[str, Any]


def save_model_artifact(
    artifact_dir: str | Path,
    *,
    model_name: str,
    estimator,
    preprocessor: ColumnTransformer,
    risk_scorer: RiskScorer,
    dataset_id: str,
    split_strategy: str,
    train_rows: int,
    decision_threshold: float | None = None,
) -> Path:
    """Write an immutable model/preprocessor/scorer bundle and JSON manifest."""
    model_type, score_kind = _validate_bundle_inputs(
        model_name, estimator, preprocessor, risk_scorer,
        dataset_id, split_strategy, train_rows, decision_threshold,
    )
    if score_kind == "probability":
        decision_threshold = float(decision_threshold)
    destination = Path(artifact_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(f"Artifact destination already exists; artifacts are immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        bundle_path = staging / BUNDLE_NAME
        reference_path = staging / REFERENCE_NAME
        manifest_path = staging / MANIFEST_NAME
        skops_io.dump({"estimator": estimator, "preprocessor": preprocessor}, bundle_path)
        bundle_hash = _sha256_file(bundle_path)
        files = {BUNDLE_NAME: bundle_hash}

        if score_kind == "anomaly":
            with reference_path.open("wb") as stream:
                np.save(stream, np.asarray(risk_scorer.sorted_reference_anomaly_scores, dtype=np.float64), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[REFERENCE_NAME] = _sha256_file(reference_path)

        manifest = _create_manifest(
            model_name=model_name.strip(), model_type=model_type, estimator=estimator,
            preprocessor=preprocessor, risk_scorer=risk_scorer,
            dataset_id=dataset_id.strip(), split_strategy=split_strategy.strip(),
            train_rows=train_rows, decision_threshold=decision_threshold, files=files,
        )
        manifest_tmp = staging / f".{MANIFEST_NAME}.tmp"
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_tmp, manifest_path)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    LOGGER.info(
        "model_artifact_saved",
        extra={"event_type": "model_artifact_saved", "model": model_name.strip(),
               "score_kind": score_kind, "artifact_dir": str(destination), "train_rows": train_rows},
    )
    return destination


def load_model_artifact(artifact_dir: str | Path) -> LoadedArtifact:
    """Load a project-format bundle after path, checksum, type, and version checks."""
    root = Path(artifact_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ArtifactError("artifact_dir must be a directory")
    manifest_path = _safe_child(root, MANIFEST_NAME)
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ArtifactError("Artifact manifest is missing or exceeds the size limit")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ArtifactError) as exc:
        raise ArtifactError(f"Could not read artifact manifest: {exc}") from exc
    _validate_manifest(manifest)
    _validate_runtime_versions(manifest["versions"], manifest["model_type"])

    bundle_path = _safe_child(root, BUNDLE_NAME)
    _verify_file(bundle_path, manifest["files"][BUNDLE_NAME])
    reference: tuple[float, ...] = ()
    if manifest["score_kind"] == "anomaly":
        reference_path = _safe_child(root, REFERENCE_NAME)
        _verify_file(reference_path, manifest["files"][REFERENCE_NAME])
        try:
            reference_values = np.load(reference_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"Could not read anomaly reference array: {exc}") from exc
        if reference_values.ndim != 1 or reference_values.size == 0 or not np.isfinite(reference_values).all():
            raise ArtifactError("Anomaly reference array must be a non-empty finite vector")
        if reference_values.dtype != np.dtype("float64") or reference_values.size != manifest["train_rows"]:
            raise ArtifactError("Anomaly reference array dtype or row count does not match the manifest")
        if np.any(np.diff(reference_values) < 0):
            raise ArtifactError("Anomaly reference values must be sorted")
        reference = tuple(float(value) for value in reference_values)

    try:
        untrusted = set(skops_io.get_untrusted_types(file=bundle_path))
    except Exception as exc:
        raise ArtifactError(f"Could not inspect serialized object types: {exc}") from exc
    allowed = _allowed_types_for_model(manifest["model_type"])
    if not untrusted.issubset(allowed):
        unexpected = sorted(untrusted - allowed)
        raise ArtifactError(f"Bundle contains unapproved serialized types: {unexpected}")
    try:
        bundle = skops_io.load(file=bundle_path, trusted=sorted(untrusted))
    except Exception as exc:
        raise ArtifactError(f"Could not safely deserialize model bundle: {exc}") from exc
    if not isinstance(bundle, dict) or set(bundle) != {"estimator", "preprocessor"}:
        raise ArtifactError("Serialized bundle has an unexpected object structure")
    estimator = bundle["estimator"]
    preprocessor = bundle["preprocessor"]
    expected_class, expected_score_kind = SUPPORTED_MODELS[manifest["model_type"]]
    if type(estimator) is not expected_class or type(preprocessor) is not ColumnTransformer:
        raise ArtifactError("Serialized estimator/preprocessor type does not match the approved artifact metadata")
    _validate_loaded_objects(estimator, preprocessor, manifest)
    policy_data = manifest["risk_policy"]
    policy = RiskBandPolicy(policy_data["low_upper"], policy_data["medium_upper"])
    risk_scorer = RiskScorer(
        model_name=manifest["model_name"], estimator=estimator, score_kind=expected_score_kind,
        feature_names=tuple(manifest["feature_names"]), policy=policy,
        sorted_reference_anomaly_scores=reference,
    )
    LOGGER.info(
        "model_artifact_loaded",
        extra={"event_type": "model_artifact_loaded", "model": manifest["model_name"],
               "score_kind": manifest["score_kind"], "artifact_dir": str(root)},
    )
    return LoadedArtifact(estimator, preprocessor, risk_scorer, manifest)


def _validate_bundle_inputs(model_name, estimator, preprocessor, risk_scorer, dataset_id, split_strategy, train_rows, threshold):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ArtifactError("model_name must be a non-empty string")
    model_type = type(estimator).__name__
    if model_type not in SUPPORTED_MODELS or type(estimator) is not SUPPORTED_MODELS[model_type][0]:
        raise ArtifactError(f"Unsupported estimator type: {model_type}")
    expected_score_kind = SUPPORTED_MODELS[model_type][1]
    if not hasattr(estimator, "n_features_in_"):
        raise ArtifactError("Estimator is not fitted")
    if type(preprocessor) is not ColumnTransformer or not hasattr(preprocessor, "transformers_"):
        raise ArtifactError("A fitted sklearn ColumnTransformer is required")
    feature_names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    if not feature_names or len(feature_names) != int(estimator.n_features_in_):
        raise ArtifactError("Fitted estimator feature count does not match the preprocessor output")
    fitted_names = getattr(estimator, "feature_names_in_", None)
    if fitted_names is not None and tuple(fitted_names) != feature_names:
        raise ArtifactError("Fitted estimator feature order does not match preprocessor output")
    if (
        risk_scorer.estimator is not estimator
        or risk_scorer.score_kind != expected_score_kind
        or tuple(risk_scorer.feature_names) != feature_names
    ):
        raise ArtifactError("Risk scorer must reference this estimator, score kind, and exact feature order")
    if not isinstance(dataset_id, str) or not dataset_id.strip() or not isinstance(split_strategy, str) or not split_strategy.strip():
        raise ArtifactError("dataset_id and split_strategy must be non-empty strings")
    if type(train_rows) is not int or train_rows < 1 or train_rows > MAX_TRAIN_ROWS:
        raise ArtifactError("train_rows must be a positive integer")
    if expected_score_kind == "probability":
        if threshold is None:
            raise ArtifactError("Supervised artifacts require a numeric decision_threshold")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("Supervised artifacts require a numeric decision_threshold") from exc
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactError("Supervised artifacts require a finite decision_threshold between 0 and 1")
    elif threshold is not None:
        raise ArtifactError("Isolation Forest artifacts use native anomaly decisions and must not have a probability threshold")
    if risk_scorer.model_name != model_name.strip() or not isinstance(risk_scorer.policy, RiskBandPolicy):
        raise ArtifactError("Risk scorer model name and policy must match the artifact")
    if expected_score_kind == "anomaly":
        values = risk_scorer.sorted_reference_anomaly_scores
        if len(values) != train_rows or not np.isfinite(values).all() or tuple(sorted(values)) != tuple(values):
            raise ArtifactError("Isolation Forest artifact requires finite sorted training-reference anomaly scores")
    return model_type, expected_score_kind


def _create_manifest(*, model_name, model_type, estimator, preprocessor, risk_scorer, dataset_id, split_strategy, train_rows, decision_threshold, files):
    return {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "model_type": model_type,
        "target_column": "Class",
        "dataset_id": dataset_id,
        "score_kind": risk_scorer.score_kind,
        "score_basis": (
            "uncalibrated model probability × 100"
            if risk_scorer.score_kind == "probability"
            else "percentile of training-reference anomaly scores"
        ),
        "decision_threshold": decision_threshold,
        "feature_names": list(risk_scorer.feature_names),
        "split_strategy": split_strategy,
        "train_rows": train_rows,
        "model_config": _json_safe(estimator.get_params(deep=False)),
        "preprocessing_config": _preprocessing_summary(preprocessor),
        "risk_policy": {
            "low_upper": risk_scorer.policy.low_upper,
            "medium_upper": risk_scorer.policy.medium_upper,
        },
        "versions": _runtime_versions(),
        "files": files,
    }


def _validate_manifest(manifest):
    required = {
        "format_version", "model_name", "model_type", "target_column", "dataset_id", "score_kind",
        "score_basis", "decision_threshold", "feature_names", "split_strategy", "train_rows",
        "model_config", "preprocessing_config", "risk_policy", "versions", "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ArtifactError("Artifact manifest has missing or unexpected fields")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ArtifactError(f"Unsupported artifact format version: {manifest['format_version']}")
    if manifest["target_column"] != "Class":
        raise ArtifactError("Only Dataset 2 Class artifacts are supported")
    if not isinstance(manifest["model_type"], str) or manifest["model_type"] not in SUPPORTED_MODELS:
        raise ArtifactError(f"Unsupported artifact model type: {manifest['model_type']}")
    if manifest["score_kind"] != SUPPORTED_MODELS[manifest["model_type"]][1]:
        raise ArtifactError("Manifest score_kind does not match the approved model type")
    expected_basis = ("uncalibrated model probability × 100" if manifest["score_kind"] == "probability"
                      else "percentile of training-reference anomaly scores")
    if manifest["score_basis"] != expected_basis:
        raise ArtifactError("Manifest score_basis does not match the approved score kind")
    for field in ("model_name", "dataset_id", "split_strategy"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise ArtifactError(f"Manifest field {field} must be a non-empty string")
    names = manifest["feature_names"]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ArtifactError("Manifest feature_names must be a unique non-empty string list")
    if not isinstance(manifest["train_rows"], int) or manifest["train_rows"] < 1:
        raise ArtifactError("Manifest train_rows must be positive")
    if manifest["train_rows"] > MAX_TRAIN_ROWS:
        raise ArtifactError("Manifest train_rows exceeds the safety limit")
    dataset_path = PureWindowsPath(manifest["dataset_id"])
    if Path(manifest["dataset_id"]).is_absolute() or dataset_path.is_absolute() or ":" in manifest["dataset_id"]:
        raise ArtifactError("Manifest dataset_id must be a logical identifier, not a local path")
    if not isinstance(manifest["risk_policy"], dict) or set(manifest["risk_policy"]) != {"low_upper", "medium_upper"}:
        raise ArtifactError("Manifest risk_policy has an invalid structure")
    RiskBandPolicy(manifest["risk_policy"]["low_upper"], manifest["risk_policy"]["medium_upper"])
    if not isinstance(manifest["files"], dict) or BUNDLE_NAME not in manifest["files"]:
        raise ArtifactError("Manifest must include the approved bundle file")
    expected_files = {BUNDLE_NAME} | ({REFERENCE_NAME} if manifest["score_kind"] == "anomaly" else set())
    if set(manifest["files"]) != expected_files:
        raise ArtifactError("Manifest references unsupported artifact files")
    for filename, digest in manifest["files"].items():
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ArtifactError(f"Invalid SHA-256 checksum for {filename}")
    threshold = manifest["decision_threshold"]
    if manifest["score_kind"] == "probability":
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ArtifactError("Probability artifact threshold is invalid")
    elif threshold is not None:
        raise ArtifactError("Anomaly artifact cannot define a probability decision threshold")
    for field in ("model_config", "preprocessing_config", "versions"):
        if not isinstance(manifest[field], dict):
            raise ArtifactError(f"Manifest field {field} must be an object")
    expected_versions = {"python", "python_major_minor", "numpy", "pandas", "scikit_learn", "xgboost", "skops"}
    if set(manifest["versions"]) != expected_versions or any(not isinstance(v, str) or not v for v in manifest["versions"].values()):
        raise ArtifactError("Manifest versions has an invalid structure")
    if not manifest["model_config"] or manifest["preprocessing_config"].get("type") != "ColumnTransformer":
        raise ArtifactError("Manifest model or preprocessing configuration is invalid")


def _validate_runtime_versions(saved, model_type):
    current = _runtime_versions()
    fields = ["python_major_minor", "numpy", "pandas", "scikit_learn", "skops"]
    if model_type == "XGBClassifier":
        fields.append("xgboost")
    for field in fields:
        if saved.get(field) != current.get(field):
            raise ArtifactError(
                f"Runtime version mismatch for {field}: artifact={saved.get(field)!r}, current={current.get(field)!r}"
            )


def _validate_loaded_objects(estimator, preprocessor, manifest):
    names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    if names != tuple(manifest["feature_names"]):
        raise ArtifactError("Loaded preprocessor feature order does not match the manifest")
    if int(estimator.n_features_in_) != len(names):
        raise ArtifactError("Loaded estimator feature count does not match the preprocessor")
    estimator_names = getattr(estimator, "feature_names_in_", None)
    if estimator_names is not None and tuple(estimator_names) != names:
        raise ArtifactError("Loaded estimator feature order does not match the preprocessor")
    if _json_safe(estimator.get_params(deep=False)) != manifest["model_config"]:
        raise ArtifactError("Loaded estimator configuration does not match the manifest")
    if _preprocessing_summary(preprocessor) != manifest["preprocessing_config"]:
        raise ArtifactError("Loaded preprocessing configuration does not match the manifest")


def _preprocessing_summary(preprocessor):
    summary = []
    for name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            transformer_data = {"type": "drop"}
        elif transformer == "passthrough":
            transformer_data = {"type": "passthrough"}
        else:
            steps = []
            if hasattr(transformer, "steps"):
                for step_name, step in transformer.steps:
                    steps.append({"name": step_name, "type": type(step).__name__, "params": _json_safe(step.get_params(deep=False))})
                transformer_data = {"type": type(transformer).__name__, "steps": steps}
            else:
                transformer_data = {"type": type(transformer).__name__, "params": _json_safe(transformer.get_params(deep=False))}
        summary.append({"name": name, "columns": [str(column) for column in columns], "transformer": transformer_data})
    return {
        "type": type(preprocessor).__name__,
        "remainder": _json_safe(preprocessor.remainder),
        "verbose_feature_names_out": _json_safe(preprocessor.verbose_feature_names_out),
        "fit_scope": "training partition only",
        "transformers": summary,
        "output_feature_names": [str(name) for name in preprocessor.get_feature_names_out()],
    }


def _runtime_versions():
    import xgboost

    return {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "skops": skops.__version__,
    }


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _allowed_types_for_model(model_type):
    expected, _ = SUPPORTED_MODELS[model_type]
    allowed = set(ALLOWED_UNTRUSTED_TYPES)
    if expected is not XGBClassifier:
        allowed -= {"xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"}
    return allowed


def _safe_child(root: Path, filename: str) -> Path:
    candidate = root / filename
    if candidate.is_symlink():
        raise ArtifactError(f"Artifact files may not be symlinks: {filename}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ArtifactError(f"Artifact file is missing or inaccessible: {filename}") from exc
    if resolved.parent != root:
        raise ArtifactError(f"Artifact path escapes artifact directory: {filename}")
    return resolved


def _verify_file(path: Path, expected_digest: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"Artifact file is missing: {path.name}")
    if path.stat().st_size > MAX_ARTIFACT_FILE_BYTES:
        raise ArtifactError(f"Artifact file exceeds size limit: {path.name}")
    actual = _sha256_file(path)
    if actual != expected_digest:
        raise ArtifactError(f"Artifact checksum mismatch: {path.name}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value):
    raise ArtifactError(f"Invalid JSON constant in manifest: {value}")
