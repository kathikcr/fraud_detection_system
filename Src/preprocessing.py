"""Leakage-safe, dataset-specific preprocessing and data partitioning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)
DATASET1_TARGET = "FraudIndicator"
DATASET2_TARGET = "Class"
DATASET2_FEATURES = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
DATASET1_EXCLUDED = (
    "TransactionID", "CustomerID", "MerchantID", "MerchantName", "Location",
    "FraudIndicator", "SuspiciousFlag", "AnomalyScore",
)
DATASET1_NUMERIC = ("Amount", "TransactionAmount", "TimestampHour", "TimestampDayOfWeek", "TimestampMonth")
DATASET1_CATEGORICAL = ("Category",)


class PreprocessingError(ValueError):
    """Raised when labels, features, or split constraints are invalid."""


@dataclass(frozen=True)
class PreparedSplits:
    """Fitted train-only transformation and independent prepared partitions."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: tuple[str, ...]
    target_column: str
    split_strategy: str
    split_boundaries: Mapping[str, object]
    class_counts: Mapping[str, Mapping[int, int]]
    excluded_columns: tuple[str, ...]


def prepare_dataset1(
    frame: pd.DataFrame,
    *,
    random_state: int = 42,
) -> PreparedSplits:
    """Create seeded stratified splits for synthetic Dataset 1.

    The target, identifiers, merchant descriptors, suspicious flag, and
    anomaly score are excluded. Only the transaction amounts, timestamp-derived
    calendar fields, and transaction category are model-input candidates.
    """
    _require_columns(frame, (DATASET1_TARGET, "Amount", "TransactionAmount", "Timestamp", "Category"), "Dataset 1")
    target = _validate_target(frame[DATASET1_TARGET], DATASET1_TARGET)
    features = pd.DataFrame(index=frame.index)
    for column in ("Amount", "TransactionAmount"):
        features[column] = _numeric_feature(frame[column], column)
    timestamps = pd.to_datetime(frame["Timestamp"], errors="coerce", format="ISO8601")
    if timestamps.isna().any():
        raise PreprocessingError("Dataset 1 Timestamp contains missing or unparseable values")
    features["TimestampHour"] = timestamps.dt.hour.astype(float)
    features["TimestampDayOfWeek"] = timestamps.dt.dayofweek.astype(float)
    features["TimestampMonth"] = timestamps.dt.month.astype(float)
    features["Category"] = frame["Category"].astype(object).where(frame["Category"].notna(), np.nan)

    indices = np.arange(len(features))
    try:
        train_idx, holdout_idx = train_test_split(
            indices, test_size=0.30, random_state=random_state, stratify=target.to_numpy()
        )
        validation_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.50, random_state=random_state, stratify=target.iloc[holdout_idx].to_numpy()
        )
    except ValueError as exc:
        raise PreprocessingError(f"Dataset 1 needs enough examples of each class for stratified 70/15/15 splits: {exc}") from exc

    return _fit_and_transform(
        features, target, train_idx, validation_idx, test_idx,
        numeric_columns=DATASET1_NUMERIC,
        categorical_columns=DATASET1_CATEGORICAL,
        target_column=DATASET1_TARGET,
        strategy=f"stratified random 70/15/15 (seed={random_state}); synthetic dataset only",
        boundaries={"random_state": random_state},
        excluded_columns=DATASET1_EXCLUDED,
    )


def prepare_dataset2(frame: pd.DataFrame) -> PreparedSplits:
    """Create chronological 70/15/15 splits, keeping equal Time values together.

    Dataset 2 is time ordered in the observed source file. Forward evaluation
    better reflects predicting later transactions from earlier observations;
    class prevalence is allowed to vary across partitions and is reported.
    """
    _require_columns(frame, (*DATASET2_FEATURES, DATASET2_TARGET), "Dataset 2")
    if frame.empty:
        raise PreprocessingError("Dataset 2 is empty")
    target = _validate_target(frame[DATASET2_TARGET], DATASET2_TARGET)
    features = pd.DataFrame(index=frame.index)
    for column in DATASET2_FEATURES:
        features[column] = _numeric_feature(frame[column], column)
    time_values = features["Time"]
    if time_values.isna().any():
        raise PreprocessingError("Dataset 2 Time contains missing values and cannot define chronological splits")
    if not time_values.is_monotonic_increasing:
        raise PreprocessingError("Dataset 2 Time must be non-decreasing for chronological splitting; input was not reordered")

    train_end, validation_end = _time_group_boundaries(time_values)
    train_idx = np.arange(0, train_end)
    validation_idx = np.arange(train_end, validation_end)
    test_idx = np.arange(validation_end, len(features))
    for split_name, split_idx in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
        if len(np.unique(target.iloc[split_idx])) != 2:
            raise PreprocessingError(f"Dataset 2 chronological {split_name} partition must contain both target classes")

    return _fit_and_transform(
        features, target, train_idx, validation_idx, test_idx,
        numeric_columns=DATASET2_FEATURES,
        categorical_columns=(),
        target_column=DATASET2_TARGET,
        strategy="chronological 70/15/15 by row count; equal Time values kept together; no shuffling",
        boundaries={
            "train_last_time": float(time_values.iloc[train_end - 1]),
            "validation_first_time": float(time_values.iloc[train_end]),
            "validation_last_time": float(time_values.iloc[validation_end - 1]),
            "test_first_time": float(time_values.iloc[validation_end]),
        },
        excluded_columns=(DATASET2_TARGET,),
    )


def _time_group_boundaries(time_values: pd.Series) -> tuple[int, int]:
    """Choose whole-timestamp boundaries nearest to 70% and 85% row quantiles."""
    group_ends = np.flatnonzero(time_values.ne(time_values.shift(-1)).to_numpy()) + 1
    n_rows = len(time_values)
    first = _nearest_valid_boundary(group_ends, round(n_rows * 0.70), lower=1, upper=n_rows - 2)
    second = _nearest_valid_boundary(group_ends, round(n_rows * 0.85), lower=first + 1, upper=n_rows - 1)
    if first <= 0 or second <= first or second >= n_rows:
        raise PreprocessingError("Dataset 2 does not have enough distinct Time groups for three chronological partitions")
    return first, second


def _nearest_valid_boundary(candidates: np.ndarray, target: int, *, lower: int, upper: int) -> int:
    valid = candidates[(candidates >= lower) & (candidates <= upper)]
    if valid.size == 0:
        raise PreprocessingError("Dataset 2 does not have enough distinct Time groups for three chronological partitions")
    return int(valid[np.argmin(np.abs(valid - target))])


def _fit_and_transform(
    features: pd.DataFrame,
    target: pd.Series,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    numeric_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    target_column: str,
    strategy: str,
    boundaries: Mapping[str, object],
    excluded_columns: tuple[str, ...],
) -> PreparedSplits:
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), list(numeric_columns)))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_columns)))
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.fit(features.iloc[train_idx])
    names = tuple(str(name) for name in preprocessor.get_feature_names_out())

    def transform(indices: np.ndarray) -> pd.DataFrame:
        values = preprocessor.transform(features.iloc[indices])
        return pd.DataFrame(values, columns=names, index=features.index[indices])

    partitions = {
        "train": (features.iloc[train_idx], target.iloc[train_idx]),
        "validation": (features.iloc[validation_idx], target.iloc[validation_idx]),
        "test": (features.iloc[test_idx], target.iloc[test_idx]),
    }
    counts = {
        name: {int(label): int(count) for label, count in y.value_counts().sort_index().items()}
        for name, (_, y) in partitions.items()
    }
    LOGGER.info(
        "preprocessing_complete",
        extra={"event_type": "preprocessing_complete", "target": target_column,
               "split_strategy": strategy, "class_counts": counts,
               "feature_count": len(names), "train_only_fit": True},
    )
    return PreparedSplits(
        X_train=transform(train_idx), X_validation=transform(validation_idx), X_test=transform(test_idx),
        y_train=partitions["train"][1].copy(),
        y_validation=partitions["validation"][1].copy(),
        y_test=partitions["test"][1].copy(),
        preprocessor=preprocessor, feature_names=names, target_column=target_column,
        split_strategy=strategy, split_boundaries=dict(boundaries), class_counts=counts,
        excluded_columns=excluded_columns,
    )


def _validate_target(target: pd.Series, name: str) -> pd.Series:
    if target.isna().any() or not target.isin({0, 1}).all():
        raise PreprocessingError(f"Target {name} must contain only non-null 0/1 values")
    if target.nunique() != 2:
        raise PreprocessingError(f"Target {name} must contain both classes 0 and 1")
    return target.astype("int8")


def _numeric_feature(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce")
    bad = values.notna() & converted.isna()
    if bad.any():
        raise PreprocessingError(f"Feature {name} contains non-numeric values")
    finite = converted.dropna().to_numpy(dtype=float)
    if finite.size and not np.isfinite(finite).all():
        raise PreprocessingError(f"Feature {name} contains infinite values")
    return converted.astype(float)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise PreprocessingError(f"{name} preprocessing requires a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise PreprocessingError(f"{name} is missing required preprocessing columns: {missing}")
