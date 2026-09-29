# Model artifact persistence

`Src.artifacts` stores a fitted Dataset 2 estimator, its fitted `ColumnTransformer`, and the matching risk-scoring configuration as one immutable artifact directory. The estimator/preprocessor bundle is serialized with `skops`; model loading does not use pickle or joblib. Artifact directories contain:

- `bundle.skops`: the estimator and preprocessor.
- `manifest.json`: format/model identity, Dataset 2 target, split and training-row metadata, ordered output feature names, model/preprocessing configuration, risk-score definition, decision threshold, runtime versions, and SHA-256 digests.
- `anomaly_reference.npy`: training-only sorted anomaly scores for Isolation Forest bundles. The array is loaded with NumPy pickle support disabled.

Example after fitting a model on the train partition:

```python
from Src.artifacts import load_model_artifact, save_model_artifact
from Src.risk_scoring import build_risk_scorer

scorer = build_risk_scorer(
    "Logistic Regression", estimator,
    score_kind="probability",
    feature_names=splits.feature_names,
)
path = save_model_artifact(
    "artifacts/logistic-regression-v1",
    model_name="Logistic Regression",
    estimator=estimator,
    preprocessor=splits.preprocessor,
    risk_scorer=scorer,
    dataset_id="kaggle-credit-card-fraud",
    split_strategy="forward-chronological",
    train_rows=len(splits.X_train),
    decision_threshold=0.5,
)

loaded = load_model_artifact(path)
transformed = loaded.preprocessor.transform(raw_transaction_frame)
```

For Isolation Forest, build the scorer with `score_kind="anomaly"` and pass only `splits.X_train` as `reference_features`; omit `decision_threshold`. The loader checks the model family, exact feature order, checksum, risk-score basis, runtime compatibility, and loaded model/preprocessing configuration before returning a `LoadedArtifact`. Transform raw transactions with the loaded preprocessor, then pass a DataFrame with the stored feature names to `loaded.risk_scorer`.

Destinations must not already exist. Write a new versioned directory for each saved model revision. The manifest deliberately records a logical `dataset_id`, never a local data path, and the repository does not publish fitted model binaries or source CSVs.

Checksums detect accidental or unsophisticated modification; they are not cryptographic signatures. Load artifacts only from a trusted source. Runtime version differences are rejected rather than silently attempting an incompatible load. The serialized-type allowlist is restricted to the four implemented model families: Logistic Regression, Random Forest, XGBoost, and Isolation Forest.
