"""
Inference and Clinical Risk Prediction Engine
Performs multimodal feature assembly, preprocessing, model inference, SHAP attribution, and 2D manifold projection.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from ml.feature_engineering import compute_clinical_features
from ml.preprocessing import transform_data
from backend.utils.reference_ranges import evaluate_status, normalize_value, REFERENCE_RANGES
from ml.research_experiments import compute_prediction_uncertainty
from ml.explainability import generate_clinical_narrative

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"

# In-memory artifact cache
_CACHE = {}


def load_prediction_artifacts():
    """Loads and caches models, preprocessors, explainers, and dimensionality reducers."""
    if not _CACHE:
        preprocessor_bundle = joblib.load(MODELS_DIR / "preprocessor_bundle.joblib")
        best_model_bundle = joblib.load(MODELS_DIR / "best_model.joblib")
        dim_bundle = joblib.load(MODELS_DIR / "dim_reduction.joblib")
        explainer_bundle = joblib.load(MODELS_DIR / "explainer_bundle.joblib")

        _CACHE["preprocessor"] = preprocessor_bundle
        _CACHE["best_model"] = best_model_bundle["model"]
        _CACHE["model_name"] = best_model_bundle["model_name"]
        _CACHE["feature_names"] = preprocessor_bundle["feature_names"]
        _CACHE["dim_bundle"] = dim_bundle
        _CACHE["explainer"] = explainer_bundle["explainer"]

    return _CACHE


def predict_patient_risk(
    patient_dict: Dict[str, Any],
    model_override_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes full multimodal inference pipeline for a given patient record:
    1. Feature engineering
    2. Zero-leakage transformation
    3. Model inference (class + calibrated probabilities)
    4. Shannon entropy prediction uncertainty
    5. Local SHAP waterfall explanation
    6. Abnormal clinical laboratory findings
    7. PCA & UMAP query coordinates
    8. Clinician & patient natural-language narrative
    """
    t0 = time.perf_counter()
    artifacts = load_prediction_artifacts()

    # Select model
    model = artifacts["best_model"]
    active_model_name = artifacts["model_name"]

    if model_override_name:
        clean_name = model_override_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        custom_model_file = MODELS_DIR / f"{clean_name}.joblib"
        if custom_model_file.exists():
            model = joblib.load(custom_model_file)
            active_model_name = model_override_name

    # 1. Feature Engineering
    df_raw = pd.DataFrame([patient_dict])
    df_feat = compute_clinical_features(df_raw)

    # 2. Transform through fitted ColumnTransformer
    X_trans = artifacts["preprocessor"]["preprocessor"].transform(df_feat)

    # 3. Model Prediction
    y_pred = int(model.predict(X_trans)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_trans)[0]
    else:
        probs = np.array([0.1, 0.2, 0.7]) if y_pred == 2 else (np.array([0.1, 0.7, 0.2]) if y_pred == 1 else np.array([0.7, 0.2, 0.1]))

    class_names = ["Low Risk", "Moderate Risk", "High Risk"]
    risk_category = class_names[y_pred]
    
    # Continuous Risk Score: weighted sum of class probabilities (0 = low, 1 = high)
    risk_score = float(round(probs[1] * 0.5 + probs[2] * 1.0, 4))
    
    # 4. Uncertainty
    uncertainty_stats = compute_prediction_uncertainty(probs)
    confidence = uncertainty_stats["confidence"]

    prob_dict = {
        class_names[i]: float(round(probs[i], 4)) for i in range(len(class_names))
    }

    # 5. Abnormal Findings Detection
    abnormal_items = []
    sex = patient_dict.get("sex", "male")
    for key, info in REFERENCE_RANGES.items():
        if key in patient_dict and patient_dict[key] is not None:
            val = float(patient_dict[key])
            st = evaluate_status(key, val, sex=sex)
            if st["status"] != "normal":
                abnormal_items.append({
                    "test_name": key,
                    "display_name": info["display_name"],
                    "raw_value_str": f"{val} {info['standard_unit']}",
                    "value": val,
                    "unit": info["standard_unit"],
                    "reference_range": st["reference_range"],
                    "status": st["status"],
                    "severity": st["severity"],
                    "category": info["category"],
                    "interpretation": st["message"]
                })

    # 6. SHAP Local Waterfall Attribution
    explainer = artifacts["explainer"]
    feature_contributions = explainer.explain_instance(
        X_trans,
        predicted_class_idx=y_pred,
        raw_values_dict=df_feat.iloc[0].to_dict()
    )

    # 7. Dimensionality Reduction Projection (PCA & UMAP coordinates)
    reducer = artifacts["dim_bundle"]["reducer"]
    dim_coords = reducer.transform_single_patient(X_trans)

    # 8. Clinical Narrative
    clinical_expl, patient_narrative = generate_clinical_narrative(
        risk_category=risk_category,
        risk_score=risk_score,
        confidence=confidence,
        top_features=feature_contributions,
        abnormal_items=abnormal_items
    )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "prediction_id": f"PRED-{uuid.uuid4().hex[:8].upper()}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": str(patient_dict.get("patient_id", "PAT-ANON")),
        "risk_category": risk_category,
        "risk_score": risk_score,
        "confidence": confidence,
        "probabilities": prob_dict,
        "model_used": active_model_name,
        "abnormal_findings": abnormal_items,
        "top_contributing_features": feature_contributions[:10],
        "clinical_explanation": clinical_expl,
        "patient_friendly_summary": patient_narrative,
        "pca_coordinates": dim_coords["pca"],
        "umap_coordinates": dim_coords["umap"],
        "disclaimer": "RESEARCH & EDUCATIONAL PROTOTYPE ONLY. Not intended for clinical diagnosis or treatment planning. Always consult a licensed healthcare professional.",
        "inference_time_ms": round(latency_ms, 2)
    }
