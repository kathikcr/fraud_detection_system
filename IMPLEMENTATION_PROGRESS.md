# Implementation Progress

- **Completed:** Data pipeline, Dataset 1 integration, EDA, leakage-safe preprocessing, four Dataset 2 baselines, unified evaluation, typed risk scoring, safe artifact save/load, transaction inference, SHAP explainability, and transaction-level investigation.
- **Current state:** `Src.investigation.TransactionInvestigator` combines validated raw input scoring and local SHAP explanation for one transaction. A caller-supplied case reference stays outside model inputs and logs. It verifies prediction/explanation score parity and returns source Time/Amount, model decision, and top contributors.
- **Tests/checks:** 112 tests PASS, including supervised and anomaly investigation workflows, invalid case references/payloads, SHAP consistency, and all previous regression suites.
- **Scientific/security constraints:** No actual fraud label is inferred as ground truth. Probability and anomaly semantics remain distinct. Explanations are local model explanations, not causal claims. The case reference is metadata only and the investigation is not persisted.
- **Docs:** `INVESTIGATION.md`; implementation `Src/investigation.py`; unit/integration tests in `tests/unit/test_investigation.py` and `tests/integration/test_investigation_integration.py`.
- **Next:** Begin API design/implementation one endpoint at a time; no HTTP endpoint is included in this checkpoint.
