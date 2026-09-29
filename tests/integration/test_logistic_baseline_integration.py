import numpy as np
import pandas as pd

from Src.logistic_baseline import run_logistic_regression_baseline, write_baseline_report
from Src.preprocessing import prepare_dataset2


def test_logistic_baseline_runs_on_preprocessed_dataset_and_writes_measured_report(tmp_path):
    count = 800
    row = np.arange(count)
    frame = pd.DataFrame({"Time": row // 4, "Amount": (row % 31).astype(float)})
    for index in range(1, 29):
        frame[f"V{index}"] = np.sin(row / (index + 1))
    frame["Class"] = (row % 20 == 0).astype(int)

    result = run_logistic_regression_baseline(prepare_dataset2(frame))
    report = write_baseline_report(result, tmp_path / "logistic-baseline.md")
    content = report.read_text(encoding="utf-8")

    assert report.is_file()
    assert "Precision" in content and "Recall" in content and "PR-AUC" in content
    assert "not tuned" in content
    assert f"{result.test.pr_auc:.6f}" in content
    assert "V1–V28" in content
