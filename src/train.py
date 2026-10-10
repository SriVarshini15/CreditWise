from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEMOGRAPHIC_COLUMNS, MLFLOW_TRACKING_URI, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, RAW_DATA_PATH
from src.data_loader import detect_id_column, detect_target_column
from src.evaluate import compute_metrics, precision_at_k, recall_at_k
from src.preprocessing import build_preprocessor, prepare_feature_frame
from src.queue_generation import generate_review_queue


def load_and_prepare_training_data(data_path: str | Path = RAW_DATA_PATH):
    df = pd.read_csv(data_path)
    target_col = detect_target_column(df)
    id_col = detect_id_column(df)
    X, y, feature_cols = prepare_feature_frame(df, target_col, id_col, excluded_columns=DEMOGRAPHIC_COLUMNS)
    return df, X, y, feature_cols, id_col, target_col


def _split_dataset(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


import time

def fit_model(pipeline, X_train, y_train, X_val, y_val):
    start = time.perf_counter()
    print("Starting preprocessing and model fitting...", flush=True)

    pipeline.fit(X_train, y_train)

    print(
        f"Model fitting completed in {time.perf_counter() - start:.1f} seconds.",
        flush=True,
    )

    start = time.perf_counter()
    val_scores = pipeline.predict_proba(X_val)[:, 1]

    print(
        f"Validation prediction completed in {time.perf_counter() - start:.1f} seconds.",
        flush=True,
    )

    return pipeline, val_scores


def train_and_evaluate():
    df, X, y, feature_cols, id_col, target_col = load_and_prepare_training_data()
    X_train, X_val, X_test, y_train, y_val, y_test = _split_dataset(X, y)
    numeric_candidates = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_candidates = [c for c in X.columns if c not in numeric_candidates and c not in ["ID", "id", "customer_id"]]
    model_specs = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=10, random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    }
    results = {}
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in model_specs.items():
        preprocessor = build_preprocessor(X_train, categorical_columns=cat_candidates, pay_columns=[c for c in X_train.columns if c.startswith("PAY_")])
        pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        with mlflow.start_run(run_name=name) as run:
            pipeline, val_scores = fit_model(pipeline, X_train, y_train, X_val, y_val)
            val_metrics = compute_metrics(y_val, val_scores)
            test_scores = pipeline.predict_proba(X_test)[:, 1]
            test_metrics = compute_metrics(y_test, test_scores)
            test_metrics["precision_at_500"] = precision_at_k(y_test, test_scores, 500)
            test_metrics["recall_at_500"] = recall_at_k(y_test, test_scores, 500)
            val_metrics["precision_at_500"] = precision_at_k(y_val, val_scores, 500)
            val_metrics["recall_at_500"] = recall_at_k(y_val, val_scores, 500)
            artifact_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(pipeline, artifact_path)
            mlflow.log_params({
                "model_type": name,
                "random_state": 42,
                "feature_count": len(feature_cols),
                "target_column": target_col,
                "id_column": id_col,
            })
            mlflow.log_metrics({**{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)}, **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, float)}})
            mlflow.log_artifact(str(artifact_path), artifact_path=f"artifacts/{name}")
            results[name] = {"val": val_metrics, "test": test_metrics, "run_id": run.info.run_id}
    selected = max(results, key=lambda k: results[k]["val"]["roc_auc"])
    selected_pipeline = joblib.load(MODELS_DIR / f"{selected}.joblib")
    reference_values = X_train.median()
    for feature in (column for column in X_train.columns if column.startswith("PAY_")):
        modes = X_train[feature].mode(dropna=True)
        if not modes.empty:
            reference_values[feature] = modes.iloc[0]
    queue = generate_review_queue(
        df,
        X,
        selected_pipeline,
        reference_values,
        id_col,
        model_version=results[selected]["run_id"],
    )
    queue.to_csv(OUTPUTS_DIR / "top_500_review_queue.csv", index=False)
    with open(REPORTS_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "selected_model": selected,
            "results": results,
            "feature_columns": feature_cols,
            "target_column": target_col,
            "id_column": id_col,
            "queue_path": str(OUTPUTS_DIR / "top_500_review_queue.csv"),
        }, fh, indent=2, default=str)
    return results, selected


if __name__ == "__main__":
    train_and_evaluate()
