import numpy as np
import pandas as pd

from Src.preprocessing import prepare_dataset2
from Src.random_forest_baseline import run_random_forest_baseline, write_random_forest_report


def test_random_forest_baseline_runs_with_preprocessor_and_writes_report(tmp_path):
    row = np.arange(400)
    frame = pd.DataFrame({"Time": row // 4, "Amount": row % 31})
    for index in range(1, 29):
        frame[f"V{index}"] = np.cos(row / (index + 2))
    frame["Class"] = (row % 20 == 0).astype(int)

    result = run_random_forest_baseline(prepare_dataset2(frame))
    report = write_random_forest_report(result, tmp_path / "random-forest.md")
    text = report.read_text(encoding="utf-8")

    assert report.is_file()
    assert "balanced_subsample" in text
    assert "PR-AUC" in text and "| FP |" in text
    assert f"{result.test.pr_auc:.6f}" in text
