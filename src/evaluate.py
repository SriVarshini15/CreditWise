from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: pd.Series | np.ndarray, y_prob: pd.Series | np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def precision_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int = 500) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(scores) == 0:
        return 0.0
    ranked = np.argsort(scores)[::-1][: min(k, len(scores))]
    if len(ranked) == 0:
        return 0.0
    return float(y_true[ranked].mean())


def recall_at_k(y_true: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int = 500) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]
    actual_positives = int(y_true.sum())
    if actual_positives == 0:
        return 0.0
    return float(y_true[top_idx].sum() / actual_positives)
