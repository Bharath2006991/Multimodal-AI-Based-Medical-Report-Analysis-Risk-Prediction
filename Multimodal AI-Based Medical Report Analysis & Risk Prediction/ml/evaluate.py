"""
Model Evaluation and Benchmarking Metrics Module
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix, Brier Score, and Latency.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, classification_report
)
from sklearn.preprocessing import label_binarize


def compute_multi_class_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes mean multi-class Brier score (lower is better calibrated)."""
    n_classes = y_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def evaluate_model_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    training_time_sec: float = 0.0,
    cv_mean: float = 0.0,
    cv_std: float = 0.0,
    class_names: List[str] = ["Low Risk", "Moderate Risk", "High Risk"]
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier across clinical ML performance metrics.
    """
    # Latency measurement
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_total_time = (time.perf_counter() - start_infer) * 1000.0  # ms
    latency_per_sample = inference_total_time / max(len(X_test), 1)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        # Softmax
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
    else:
        # Fallback dummy probabilities from predictions
        y_prob = np.zeros((len(y_pred), len(class_names)))
        for i, p in enumerate(y_pred):
            y_prob[i, int(p)] = 1.0

    # 1. Standard Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # 2. Area Under Curves
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    try:
        roc_auc_ovr = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        roc_auc_ovr = 0.5

    try:
        pr_auc_macro = float(average_precision_score(y_test_bin, y_prob, average="macro"))
    except Exception:
        pr_auc_macro = 0.5

    # 3. Calibration / Brier Score
    brier = compute_multi_class_brier_score(y_test, y_prob)

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()

    # Per-class metrics
    per_class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "roc_auc_ovr": round(roc_auc_ovr, 4),
        "pr_auc_macro": round(pr_auc_macro, 4),
        "cv_mean": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "brier_score": round(brier, 4),
        "training_time_sec": round(training_time_sec, 3),
        "prediction_latency_ms": round(latency_per_sample, 4),
        "confusion_matrix": cm_list,
        "class_report": per_class_report
    }

    return metrics
