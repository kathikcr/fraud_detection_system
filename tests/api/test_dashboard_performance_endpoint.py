from pathlib import Path

from fastapi.testclient import TestClient

from Src import dashboard
from Src.api import create_app


REPORT = """# Dataset 2 Model Evaluation

- Target: `Class` (fraud class 1)
- Split: chronological 70/15/15 by row count; equal Time values kept together; no shuffling

## Validation results

| Model | Score type | Decision rule | Prevalence | Alert rate | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | probability | probability >= 0.5 (fixed) | 0.1311% | 2.2962% | 0.053007 | 0.928571 | 0.100289 | 0.982762 | 0.839355 | 41,737 | 929 | 4 | 52 |

## Test results

| Model | Score type | Decision rule | Prevalence | Alert rate | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | probability | probability >= 0.5 (fixed) | 0.1217% | 1.7649% | 0.057029 | 0.826923 | 0.106700 | 0.977236 | 0.706915 | 41,958 | 711 | 9 | 43 |
| Isolation Forest | anomaly | Isolation Forest native predict cutoff (contamination=auto) | 0.1217% | 3.7569% | 0.023053 | 0.711538 | 0.044659 | 0.933870 | 0.046077 | 41,101 | 1,568 | 15 | 37 |

## Interpretation limits
"""


def test_performance_endpoint_reads_metrics_and_keeps_score_kinds_distinct(tmp_path, monkeypatch):
    report_path = tmp_path / "model_comparison.md"
    report_path.write_text(REPORT, encoding="utf-8")
    monkeypatch.setattr(dashboard, "EVALUATION_REPORT", report_path)
    dashboard.clear_performance_cache()

    with TestClient(create_app()) as client:
        response = client.get("/dashboard/performance?partition=test")
        validation = client.get("/dashboard/performance?partition=validation")

    assert response.status_code == 200
    result = response.json()
    assert result["partition"] == "test"
    assert result["target"].startswith("Class")
    assert len(result["models"]) == 2
    assert result["models"][0]["prevalence"] == 0.001217
    assert result["models"][0]["metrics"]["pr_auc"] == 0.706915
    assert result["models"][0]["confusion_matrix"] == {"tn": 41958, "fp": 711, "fn": 9, "tp": 43}
    assert result["models"][1]["score_kind"] == "anomaly"
    assert result["models"][1]["rows"] == 42721
    assert "not fraud probabilities" in result["limitations"][1]
    assert validation.status_code == 200
    assert round(validation.json()["models"][0]["prevalence"], 6) == 0.001311


def test_performance_endpoint_rejects_invalid_partition_and_missing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "EVALUATION_REPORT", tmp_path / "missing.md")
    dashboard.clear_performance_cache()

    with TestClient(create_app()) as client:
        invalid = client.get("/dashboard/performance?partition=development")
        missing = client.get("/dashboard/performance?partition=validation")

    assert invalid.status_code == 422
    assert missing.status_code == 503


def test_performance_curve_endpoint_serves_only_known_png_artifacts():
    with TestClient(create_app()) as client:
        curve = client.get("/dashboard/performance/roc.png")
        unknown = client.get("/dashboard/performance/other.png")

    assert curve.status_code == 200
    assert curve.headers["content-type"] == "image/png"
    assert curve.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert unknown.status_code == 422
