import numpy as np
import pytest

from Src.inference import InferenceError, _validate_transactions
from Src.preprocessing import DATASET2_FEATURES


def valid_transaction():
    return {name: 0.0 for name in DATASET2_FEATURES}


def test_raw_transaction_validation_preserves_canonical_feature_order():
    transaction = valid_transaction()
    transaction["Amount"] = 0
    rows = _validate_transactions([transaction])

    assert list(rows[0]) == list(DATASET2_FEATURES)
    assert rows[0]["Amount"] == 0.0


@pytest.mark.parametrize("bad_value", ["1.2", True, None, np.nan, np.inf, -np.inf])
def test_raw_transaction_rejects_wrong_type_missing_or_non_finite_values(bad_value):
    transaction = valid_transaction()
    transaction["V1"] = bad_value
    with pytest.raises(InferenceError):
        _validate_transactions([transaction])


@pytest.mark.parametrize("feature", ["Time", "Amount"])
def test_raw_transaction_rejects_negative_time_and_amount(feature):
    transaction = valid_transaction()
    transaction[feature] = -0.01
    with pytest.raises(InferenceError, match="cannot be negative"):
        _validate_transactions([transaction])


def test_raw_transaction_rejects_missing_extra_target_and_empty_batch():
    transaction = valid_transaction()
    missing = transaction.copy()
    del missing["V28"]
    extra = transaction | {"Class": 0}

    with pytest.raises(InferenceError, match="missing required"):
        _validate_transactions([missing])
    with pytest.raises(InferenceError, match="unsupported fields"):
        _validate_transactions([extra])
    with pytest.raises(InferenceError, match="at least one"):
        _validate_transactions([])
    with pytest.raises(InferenceError, match="object"):
        _validate_transactions([None])


def test_raw_transaction_rejects_numeric_values_that_overflow_float64():
    transaction = valid_transaction()
    transaction["Amount"] = 10**1000
    with pytest.raises(InferenceError, match="finite"):
        _validate_transactions([transaction])
