from Src.dataset1_integration import load_dataset1_transaction_table


def test_dataset1_join_keeps_transaction_row_count_and_target(dataset1_csv_root):
    result = load_dataset1_transaction_table(dataset1_csv_root)

    assert result.report.input_transaction_rows == 3
    assert result.report.output_rows == 3
    assert result.frame["TransactionID"].is_unique
    assert result.frame["FraudIndicator"].value_counts().to_dict() == {0: 2, 1: 1}
    assert all(check.unmatched_reference_keys == 0 for check in result.report.relationships)
    assert not {"Name", "Address", "Age", "AccountBalance", "LastLogin"} & set(result.frame.columns)
