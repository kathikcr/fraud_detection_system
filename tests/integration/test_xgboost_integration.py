import numpy as np
import pandas as pd

from Src.preprocessing import prepare_dataset2
from Src.xgboost_baseline import run_xgboost_baseline, write_xgboost_report


def test_xgboost_baseline_integrates_with_preprocessing_and_writes_report(tmp_path):
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)

    result = run_xgboost_baseline(prepare_dataset2(frame))
    report = write_xgboost_report(result, tmp_path / "xgboost.md")
    content = report.read_text(encoding="utf-8")

    assert report.is_file()
    assert "scale_pos_weight" in content
    assert "Validation and test partitions were not used during fitting" in content
    assert "PR-AUC" in content
    assert f"{result.test.pr_auc:.6f}" in content
