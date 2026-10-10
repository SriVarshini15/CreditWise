from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR
from src.queue_generation import generate_review_queue
from src.train import _split_dataset, load_and_prepare_training_data


def main() -> Path:
    summary_path = REPORTS_DIR / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Training summary is missing; train a model before scoring the queue.")

    with summary_path.open(encoding="utf-8") as file:
        summary = json.load(file)

    selected_model = summary["selected_model"]
    model_path = MODELS_DIR / f"{selected_model}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Selected model artifact is missing: {model_path}")

    df, X, y, feature_columns, id_col, target_col = load_and_prepare_training_data()
    if feature_columns != summary["feature_columns"]:
        raise ValueError("Current dataset features do not match the saved training summary.")
    if target_col != summary["target_column"] or id_col != summary["id_column"]:
        raise ValueError("Current dataset ID/target columns do not match the saved training summary.")

    X_train, _, _, _, _, _ = _split_dataset(X, y)
    reference_values = X_train.median()
    for feature in (column for column in X_train.columns if column.startswith("PAY_")):
        modes = X_train[feature].mode(dropna=True)
        if not modes.empty:
            reference_values[feature] = modes.iloc[0]

    queue = generate_review_queue(
        df,
        X,
        joblib.load(model_path),
        reference_values,
        id_col,
        model_version=f"{selected_model}:{hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]}",
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "top_500_review_queue.csv"
    queue.to_csv(output_path, index=False)
    print(f"Saved {len(queue)} ranked accounts with local risk explanations to {output_path}")
    return output_path


if __name__ == "__main__":
    main()
