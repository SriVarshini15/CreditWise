from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_PATH


def fetch_uciml_dataset() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=350)
        raw_df = dataset.data.original.copy()
        rename_map = {}
        for record in dataset.variables.to_dict("records"):
            original_name = record["name"]
            label = record.get("description")
            if original_name == "Y":
                rename_map[original_name] = "default payment next month"
            elif label and pd.notna(label):
                rename_map[original_name] = str(label)
            else:
                rename_map[original_name] = str(original_name)
        df = raw_df.rename(columns=rename_map)
        return df
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError("Unable to download the UCI dataset automatically") from exc


def save_raw_dataset(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    destination = Path(path) if path is not None else RAW_DATA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination


def main() -> None:
    try:
        df = fetch_uciml_dataset()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    path = save_raw_dataset(df)
    print(f"Saved UCI credit card default data to {path}")


if __name__ == "__main__":
    main()
