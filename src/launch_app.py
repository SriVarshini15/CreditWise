from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.config import MODELS_DIR, OUTPUTS_DIR, RAW_DATA_PATH, REPORTS_DIR

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPORTS_DIR / "training_summary.json"
QUEUE_PATH = OUTPUTS_DIR / "top_500_review_queue.csv"


def _run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def ensure_artifacts() -> None:
    if not RAW_DATA_PATH.is_file():
        print("Raw dataset is missing; downloading the UCI dataset.", flush=True)
        _run_module("src.download_data")

    if not SUMMARY_PATH.is_file():
        model_missing = True
    else:
        with SUMMARY_PATH.open(encoding="utf-8") as file:
            summary = json.load(file)
        model_path = MODELS_DIR / f"{summary['selected_model']}.joblib"
        model_missing = not model_path.is_file()

    if model_missing:
        print("Trained model artifacts are missing; training candidate models.", flush=True)
        _run_module("src.train")
    elif not QUEUE_PATH.is_file():
        print("Review queue is missing; regenerating it from the saved model.", flush=True)
        _run_module("src.score_queue")

    required_artifacts = (RAW_DATA_PATH, SUMMARY_PATH, QUEUE_PATH)
    missing_artifacts = [str(path) for path in required_artifacts if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            f"Startup did not create required application artifacts: {', '.join(missing_artifacts)}"
        )


def main() -> None:
    ensure_artifacts()
    port = str(int(os.environ.get("PORT", "3000")))
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
