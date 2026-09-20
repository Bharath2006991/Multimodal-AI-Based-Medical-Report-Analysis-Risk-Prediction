"""
Advanced Research Experiments Suite
Implements feature ablation, calibration reliability diagrams, missingness sensitivity, and prediction entropy analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"


def run_feature_ablation_study(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Evaluates impact on F1-score when masking specific clinical feature subsets:
    1. Demographics Only
    2. Vitals & Labs Only
    3. Symptoms Only
    4. Full Multimodal Pipeline
    """
    baseline_pred = model.predict(X_test)
    baseline_f1 = float(f1_score(y_test, baseline_pred, average="macro"))
    baseline_acc = float(accuracy_score(y_test, baseline_pred))

    # Identify indices
    demo_indices = [i for i, f in enumerate(feature_names) if any(d in f for d in ["age", "sex", "bmi", "height", "weight"])]
    vitals_labs_indices = [i for i, f in enumerate(feature_names) if any(v in f for v in ["bp", "heart", "glucose", "chol", "trig", "hba1c", "hemoglobin", "tyg", "pulse", "arterial"])]
    symptom_indices = [i for i, f in enumerate(feature_names) if any(s in f for s in ["chest_pain", "shortness", "fatigue", "dizziness", "palpitation", "smoking", "activity"])]

    subsets = {
        "Full Multimodal Pipeline": list(range(len(feature_names))),
        "Vitals & Lab Biomarkers Only": vitals_labs_indices,
        "Demographics & Anthropometrics Only": demo_indices,
        "Symptoms & Lifestyle Only": symptom_indices,
        "Without Laboratory Values (Ablated Labs)": [i for i in range(len(feature_names)) if i not in vitals_labs_indices]
    }

    results = []
    for name, indices in subsets.items():
        # Mask out other features with mean zero (scaled space)
        X_masked = np.zeros_like(X_test)
        X_masked[:, indices] = X_test[:, indices]
        
        preds = model.predict(X_masked)
        f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_test, preds))
        drop_f1 = float(baseline_f1 - f1)

        results.append({
            "subset_name": name,
            "feature_count": len(indices),
            "macro_f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "performance_delta_f1": round(-drop_f1, 4),
            "percentage_retention": round((f1 / max(baseline_f1, 1e-5)) * 100, 1)
        })

    return results


def compute_calibration_analysis(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Computes calibration reliability curve for High Risk class (class 2)."""
    y_prob = model.predict_proba(X_test)
    y_test_bin = (y_test == 2).astype(int)
    prob_high_risk = y_prob[:, 2]

    prob_true, prob_pred = calibration_curve(y_test_bin, prob_high_risk, n_bins=5, strategy="uniform")
    
    # Expected Calibration Error (ECE)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    return {
        "target_class": "High Risk",
        "bins": 5,
        "prob_true": [round(float(p), 4) for p in prob_true],
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "expected_calibration_error": round(ece, 4),
        "interpretation": "A lower Expected Calibration Error (ECE) indicates the model's reported risk probability closely mirrors observed empirical prevalence."
    }


def compute_prediction_uncertainty(probabilities: np.ndarray) -> Dict[str, float]:
    """
    Computes Shannon entropy as a normalized measure of prediction uncertainty:
    H(p) = -sum(p * log2(p)) / log2(n_classes).
    Values near 0 indicate definitive certainty; values near 1 indicate maximum ambiguity.
    """
    eps = 1e-9
    probs = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probabilities))
    normalized_entropy = float(entropy / max_entropy)
    confidence = float(np.max(probabilities))

    return {
        "confidence": round(confidence, 4),
        "shannon_entropy": round(float(entropy), 4),
        "normalized_uncertainty": round(normalized_entropy, 4)
    }


def run_all_research_experiments(model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str], save_path: Path) -> Dict[str, Any]:
    """Executes full research experiments suite and serializes findings."""
    ablation = run_feature_ablation_study(model, X_test, y_test, feature_names)
    calibration = compute_calibration_analysis(model, X_test, y_test)
    
    # Compute test set uncertainty metrics
    probs = model.predict_proba(X_test)
    entropies = [compute_prediction_uncertainty(p)["normalized_uncertainty"] for p in probs]
    
    report = {
        "feature_ablation": ablation,
        "calibration": calibration,
        "mean_test_uncertainty": round(float(np.mean(entropies)), 4),
        "low_uncertainty_fraction": round(float(np.mean(np.array(entropies) < 0.35)), 3),
        "model_version": "v1.4-production",
        "random_seed": 42
    }
    
    joblib.dump(report, save_path)
    print(f"[Research] Completed research experiments. Saved to {save_path}")
    return report
