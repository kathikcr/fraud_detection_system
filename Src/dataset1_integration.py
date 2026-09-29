"""Validate Dataset 1 relationships and assemble a privacy-conscious table."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from Src.ingestion import LoadedCsv, load_dataset1

LOGGER = logging.getLogger(__name__)
TRANSACTION_KEY = "TransactionID"
TARGET_COLUMN = "FraudIndicator"
PII_COLUMNS_EXCLUDED = ("Name", "Address", "Age", "AccountBalance", "LastLogin")


class Dataset1IntegrationError(ValueError):
    """Raised when Dataset 1 tables cannot be joined without ambiguity or row loss."""


@dataclass(frozen=True)
class RelationshipAudit:
    source_table: str
    target_table: str
    key: str
    cardinality: str
    matched_reference_keys: int
    unmatched_reference_keys: int
    unreferenced_target_keys: int


@dataclass(frozen=True)
class Dataset1IntegrationReport:
    input_transaction_rows: int
    output_rows: int
    transaction_key: str
    target_column: str
    target_distribution: Mapping[object, int]
    relationships: tuple[RelationshipAudit, ...]
    excluded_personal_columns: tuple[str, ...]


@dataclass(frozen=True)
class Dataset1IntegrationResult:
    frame: pd.DataFrame
    report: Dataset1IntegrationReport


def build_dataset1_transaction_table(
    tables: Mapping[str, pd.DataFrame | LoadedCsv],
) -> Dataset1IntegrationResult:
    """Validate cardinality/coverage then make a transaction-grain table.

    Personal customer profile and account-activity fields are key-checked but
    intentionally omitted. Dataset 1 fraud/anomaly flags remain analytics labels,
    not approved model predictors.
    """
    required = {
        "transaction_records", "transaction_metadata", "fraud_indicators",
        "suspicious_activity", "customer_data", "account_activity",
        "merchant_data", "transaction_category_labels", "amount_data", "anomaly_scores",
    }
    missing_tables = sorted(required - set(tables))
    if missing_tables:
        raise Dataset1IntegrationError(f"Missing Dataset 1 tables: {missing_tables}")
    frames = {name: _frame(value, name) for name, value in tables.items()}

    transactions = frames["transaction_records"]
    _require_columns(transactions, "transaction_records", (TRANSACTION_KEY, "CustomerID", "Amount"))
    _validate_key(transactions, TRANSACTION_KEY, "transaction_records", unique=True)
    input_rows = len(transactions)
    if input_rows == 0:
        raise Dataset1IntegrationError("transaction_records is empty")

    audits: list[RelationshipAudit] = []
    transaction_ids = transactions[[TRANSACTION_KEY]]

    metadata = frames["transaction_metadata"]
    _require_columns(metadata, "transaction_metadata", (TRANSACTION_KEY, "Timestamp", "MerchantID"))
    audits.append(_audit_relationship(
        "transaction_records", "transaction_metadata", transaction_ids, metadata,
        TRANSACTION_KEY, "one-to-one", exact=True,
    ))

    for name in ("fraud_indicators", "transaction_category_labels", "amount_data", "anomaly_scores"):
        related = frames[name]
        if name == "fraud_indicators":
            _require_columns(related, name, (TRANSACTION_KEY, TARGET_COLUMN))
        elif name == "transaction_category_labels":
            _require_columns(related, name, (TRANSACTION_KEY, "Category"))
        elif name == "amount_data":
            _require_columns(related, name, (TRANSACTION_KEY, "TransactionAmount"))
        else:
            _require_columns(related, name, (TRANSACTION_KEY, "AnomalyScore"))
        audits.append(_audit_relationship(
            "transaction_records", name, transaction_ids, related,
            TRANSACTION_KEY, "one-to-one", exact=True,
        ))

    for name in ("customer_data", "account_activity", "suspicious_activity"):
        related = frames[name]
        _require_columns(related, name, ("CustomerID",))
        if name == "suspicious_activity":
            _require_columns(related, name, ("CustomerID", "SuspiciousFlag"))
        audits.append(_audit_relationship(
            "transaction_records", name, transactions[["CustomerID"]], related,
            "CustomerID", "many-to-one", exact=False,
        ))

    _require_columns(metadata, "transaction_metadata", ("MerchantID",))
    merchant_data = frames["merchant_data"]
    _require_columns(merchant_data, "merchant_data", ("MerchantID", "MerchantName", "Location"))
    audits.append(_audit_relationship(
        "transaction_metadata", "merchant_data", metadata[["MerchantID"]], merchant_data,
        "MerchantID", "many-to-one", exact=False,
    ))

    target = frames["fraud_indicators"]
    target_values = target[TARGET_COLUMN]
    if target_values.isna().any() or not target_values.isin({0, 1}).all():
        raise Dataset1IntegrationError("FraudIndicator must be non-null and contain only 0/1")
    expected_target_by_id = target.set_index(TRANSACTION_KEY)[TARGET_COLUMN]

    result = transactions[[TRANSACTION_KEY, "Amount", "CustomerID"]].copy()
    result = _merge_checked(result, metadata[[TRANSACTION_KEY, "Timestamp", "MerchantID"]], TRANSACTION_KEY, "transaction_metadata", input_rows)
    result = _merge_checked(result, target[[TRANSACTION_KEY, TARGET_COLUMN]], TRANSACTION_KEY, "fraud_indicators", input_rows)
    result = _merge_checked(result, frames["transaction_category_labels"][[TRANSACTION_KEY, "Category"]], TRANSACTION_KEY, "transaction_category_labels", input_rows)
    result = _merge_checked(result, frames["amount_data"][[TRANSACTION_KEY, "TransactionAmount"]], TRANSACTION_KEY, "amount_data", input_rows)
    result = _merge_checked(result, frames["anomaly_scores"][[TRANSACTION_KEY, "AnomalyScore"]], TRANSACTION_KEY, "anomaly_scores", input_rows)
    result = _merge_checked(result, frames["suspicious_activity"][["CustomerID", "SuspiciousFlag"]], "CustomerID", "suspicious_activity", input_rows)
    result = _merge_checked(result, merchant_data[["MerchantID", "MerchantName", "Location"]], "MerchantID", "merchant_data", input_rows)

    if not result[TARGET_COLUMN].reset_index(drop=True).equals(
        result[TRANSACTION_KEY].map(expected_target_by_id).reset_index(drop=True)
    ):
        raise Dataset1IntegrationError("FraudIndicator changed during join; target preservation check failed")
    if result[TRANSACTION_KEY].duplicated().any():
        raise Dataset1IntegrationError("TransactionID is no longer unique after joins")

    distribution = {
        value.item() if hasattr(value, "item") else value: int(count)
        for value, count in result[TARGET_COLUMN].value_counts(sort=False).items()
    }
    report = Dataset1IntegrationReport(
        input_transaction_rows=input_rows,
        output_rows=len(result),
        transaction_key=TRANSACTION_KEY,
        target_column=TARGET_COLUMN,
        target_distribution=distribution,
        relationships=tuple(audits),
        excluded_personal_columns=PII_COLUMNS_EXCLUDED,
    )
    LOGGER.info(
        "dataset1_join_complete",
        extra={
            "event_type": "dataset1_join_complete",
            "input_transaction_rows": report.input_transaction_rows,
            "output_rows": report.output_rows,
            "relationship_count": len(report.relationships),
            "target_distribution": dict(report.target_distribution),
        },
    )
    return Dataset1IntegrationResult(frame=result, report=report)


def load_dataset1_transaction_table(data_root=None) -> Dataset1IntegrationResult:
    """Load Dataset 1's CSVs and return a verified transaction-level table."""
    return build_dataset1_transaction_table(load_dataset1(data_root))


def _frame(value: pd.DataFrame | LoadedCsv, name: str) -> pd.DataFrame:
    frame = value.frame if isinstance(value, LoadedCsv) else value
    if not isinstance(frame, pd.DataFrame):
        raise Dataset1IntegrationError(f"Table {name} must be a pandas DataFrame or LoadedCsv")
    return frame


def _require_columns(frame: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise Dataset1IntegrationError(f"Table {name} is missing required columns: {missing}")


def _validate_key(frame: pd.DataFrame, key: str, name: str, *, unique: bool) -> None:
    _require_columns(frame, name, (key,))
    null_count = int(frame[key].isna().sum())
    if null_count:
        raise Dataset1IntegrationError(f"Table {name} has {null_count} null values in key {key}")
    if unique and frame[key].duplicated().any():
        duplicate_count = int(frame[key].duplicated().sum())
        raise Dataset1IntegrationError(f"Table {name} key {key} has {duplicate_count} duplicate values")


def _audit_relationship(
    source_name: str,
    target_name: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    key: str,
    cardinality: str,
    *,
    exact: bool,
) -> RelationshipAudit:
    _validate_key(source, key, source_name, unique=(cardinality == "one-to-one"))
    _validate_key(target, key, target_name, unique=True)
    source_keys = set(source[key].tolist())
    target_keys = set(target[key].tolist())
    missing = source_keys - target_keys
    extra = target_keys - source_keys
    if missing:
        raise Dataset1IntegrationError(
            f"Foreign-key integrity failed: {source_name}.{key} has {len(missing)} "
            f"unmatched value(s) in {target_name}.{key}; examples={_examples(missing)}"
        )
    if exact and extra:
        raise Dataset1IntegrationError(
            f"Unexpected orphan key(s): {target_name}.{key} contains {len(extra)} "
            f"value(s) absent from {source_name}.{key}; examples={_examples(extra)}"
        )
    return RelationshipAudit(
        source_table=source_name,
        target_table=target_name,
        key=key,
        cardinality=cardinality,
        matched_reference_keys=len(source_keys & target_keys),
        unmatched_reference_keys=len(missing),
        unreferenced_target_keys=len(extra),
    )


def _merge_checked(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    table_name: str,
    expected_rows: int,
) -> pd.DataFrame:
    try:
        merged = left.merge(right, on=key, how="left", validate="many_to_one", sort=False)
    except pd.errors.MergeError as exc:
        raise Dataset1IntegrationError(f"Join to {table_name} violates many-to-one key cardinality: {exc}") from exc
    if len(merged) != expected_rows or merged[TRANSACTION_KEY].duplicated().any():
        raise Dataset1IntegrationError(
            f"Join to {table_name} changed transaction grain: expected {expected_rows} unique transactions, "
            f"got {len(merged)} rows"
        )
    return merged


def _examples(values: set[object], limit: int = 5) -> list[object]:
    return sorted(values, key=str)[:limit]
