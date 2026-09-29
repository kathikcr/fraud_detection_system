import pytest

from Src.investigation import InvestigationError, _validate_transaction_id


@pytest.mark.parametrize("value, expected", [(42, "42"), ("  case-42  ", "case-42")])
def test_transaction_id_is_normalized_as_case_metadata(value, expected):
    assert _validate_transaction_id(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", True, 3.5, object(), "x" * 129])
def test_invalid_transaction_ids_are_rejected(value):
    with pytest.raises(InvestigationError):
        _validate_transaction_id(value)
