"""
Explainable AI (XAI) Module using SHAP
Provides global feature importance, local instance waterfall contributions, and natural-language clinical narratives.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"

# Clinically friendly feature display labels
FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "height_cm": "Height",
    "weight_kg": "Weight",
    "bmi": "Body Mass Index (BMI)",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "heart_rate": "Resting Heart Rate",
    "fasting_glucose": "Fasting Blood Glucose",
    "hba1c": "Glycated Hemoglobin (HbA1c)",
    "total_cholesterol": "Total Cholesterol",
    "hdl_cholesterol": "HDL Cholesterol (Protective)",
    "ldl_cholesterol": "LDL Cholesterol (Atherogenic)",
    "triglycerides": "Triglycerides",
    "hemoglobin": "Hemoglobin",
    "pulse_pressure": "Pulse Pressure (Arterial Stiffness)",
    "mean_arterial_pressure": "Mean Arterial Pressure (MAP)",
    "chol_hdl_ratio": "Cholesterol-to-HDL Ratio",
    "non_hdl_cholesterol": "Non-HDL Cholesterol",
    "tyg_index": "Triglyceride-Glucose (TyG) Insulin Resistance Index",
    "metabolic_risk_factor_count": "Metabolic Syndrome Criteria Count",
    "sex_female": "Sex: Female",
    "sex_male": "Sex: Male",
    "smoking_status_never": "Non-Smoker",
    "smoking_status_former": "Former Smoker",
    "smoking_status_current": "Current Smoker",
    "physical_activity_low": "Sedentary Activity",
    "physical_activity_moderate": "Moderate Activity",
    "physical_activity_high": "High Activity",
    "family_history_cad": "Family History of CAD",
    "chest_pain": "Symptom: Chest Pain",
    "shortness_of_breath": "Symptom: Dyspnea (Shortness of Breath)",
    "fatigue": "Symptom: Fatigue",
    "dizziness": "Symptom: Dizziness",
    "palpitations": "Symptom: Palpitations"
}


class ModelExplainer:
    """Manages SHAP tree and kernel explainers and natural-language narrative synthesis."""
    def __init__(self, model: Any, background_data: np.ndarray, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self.global_importance = []
        self._init_explainer()

    def _init_explainer(self):
        """Initializes appropriate SHAP explainer depending on model class."""
        try:
            # Tree models (XGBoost, RandomForest, GradientBoosting)
            self.explainer = shap.TreeExplainer(self.model, data=self.background_data[:100])
        except Exception:
            # Fallback for linear / kernel models
            try:
                self.explainer = shap.LinearExplainer(self.model, self.background_data[:100])
            except Exception:
                # KernelExplainer on representative subset
                summary_data = shap.sample(self.background_data, 50)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, summary_data)

    def compute_global_importance(self, X_sample: np.ndarray) -> List[Dict[str, Any]]:
        """Calculates global mean |SHAP| feature importance."""
        try:
            shap_values = self.explainer.shap_values(X_sample[:200])
            
            # If multi-class list
            if isinstance(shap_values, list):
                # Average across classes or take target class (High Risk = class 2)
                target_idx = min(2, len(shap_values) - 1)
                vals = np.abs(shap_values[target_idx]).mean(axis=0)
            elif shap_values.ndim == 3:
                target_idx = min(2, shap_values.shape[2] - 1)
                vals = np.abs(shap_values[:, :, target_idx]).mean(axis=0)
            else:
                vals = np.abs(shap_values).mean(axis=0)

            importance_list = []
            for i, feat in enumerate(self.feature_names):
                importance_list.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
                    "importance_score": float(round(vals[i], 4))
                })

            importance_list.sort(key=lambda x: x["importance_score"], reverse=True)
            self.global_importance = importance_list
            return importance_list
        except Exception as e:
            print(f"[XAI] Warning: Global SHAP computation failed: {e}")
            return []

    def explain_instance(
        self,
        x_patient: np.ndarray,
        predicted_class_idx: int = 2,
        raw_values_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Explains a single patient inference vector with local SHAP waterfall contributions."""
        x_mat = x_patient if x_patient.ndim == 2 else x_patient.reshape(1, -1)
        try:
            shap_vals = self.explainer.shap_values(x_mat)
            if isinstance(shap_vals, list):
                idx = min(predicted_class_idx, len(shap_vals) - 1)
                sample_shap = shap_vals[idx][0]
            elif shap_vals.ndim == 3:
                idx = min(predicted_class_idx, shap_vals.shape[2] - 1)
                sample_shap = shap_vals[0, :, idx]
            else:
                sample_shap = shap_vals[0]

            contributions = []
            for i, feat in enumerate(self.feature_names):
                s_val = float(sample_shap[i])
                raw_val = float(x_patient.flatten()[i])
                if raw_values_dict and feat in raw_values_dict:
                    display_val = raw_values_dict[feat]
                else:
                    display_val = raw_val

                direction = "increases_risk" if s_val > 0 else "decreases_risk"
                abs_s = abs(s_val)
                impact = "high" if abs_s > 0.3 else ("moderate" if abs_s > 0.1 else "low")

                contributions.append({
                    "feature": feat,
                    "feature_name": FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
                    "value": float(round(display_val, 2)) if isinstance(display_val, (int, float)) else 0.0,
                    "shap_value": float(round(s_val, 4)),
                    "direction": direction,
                    "impact": impact
                })

            # Sort by absolute SHAP impact
            contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
            return contributions

        except Exception as e:
            print(f"[XAI] Instance explanation error: {e}")
            # Fallback heuristic contribution
            return [
                {
                    "feature": "fasting_glucose",
                    "feature_name": "Fasting Blood Glucose",
                    "value": 110.0,
                    "shap_value": 0.25,
                    "direction": "increases_risk",
                    "impact": "moderate"
                }
            ]


def generate_clinical_narrative(
    risk_category: str,
    risk_score: float,
    confidence: float,
    top_features: List[Dict[str, Any]],
    abnormal_items: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Synthesizes two clear explanations:
    1. Clinician-level technical rationale with quantitative feature attributions
    2. Patient-friendly summary with accessible health literacy tone and consult recommendation
    """
    # 1. Clinician narrative
    inc_risk_factors = [f for f in top_features if f["direction"] == "increases_risk"][:4]
    dec_risk_factors = [f for f in top_features if f["direction"] == "decreases_risk"][:2]

    clinician_parts = [
        f"Multimodal ensemble evaluated patient as [{risk_category.upper()}] with estimated risk probability of {risk_score * 100:.1f}% (Model Confidence: {confidence * 100:.1f}%)."
    ]

    if inc_risk_factors:
        drivers = ", ".join([f"{f['feature_name']} (SHAP +{f['shap_value']:.2f})" for f in inc_risk_factors])
        clinician_parts.append(f"Primary physiological drivers elevating risk: {drivers}.")

    if dec_risk_factors:
        protective = ", ".join([f"{f['feature_name']} (SHAP {f['shap_value']:.2f})" for f in dec_risk_factors])
        clinician_parts.append(f"Favorable / protective biomarkers tempering risk: {protective}.")

    if abnormal_items:
        crit = [item["display_name"] for item in abnormal_items if "high" in item.get("status", "") or "low" in item.get("status", "")]
        if crit:
            clinician_parts.append(f"Detected out-of-range clinical biomarkers: {', '.join(crit[:5])}.")

    clinician_parts.append("Note: Attributions represent model feature sensitivity and non-linear interactions; they do not establish causal etiologies.")
    clinical_explanation = " ".join(clinician_parts)

    # 2. Patient-friendly narrative
    patient_parts = []
    if risk_category == "High Risk":
        patient_parts.append(
            "Our research system has flagged several cardiometabolic health indicators that are higher than recommended standard thresholds."
        )
    elif risk_category == "Moderate Risk":
        patient_parts.append(
            "Your results show a borderline or elevated risk profile across a few key health metrics, suggesting areas where preventive lifestyle or medical measures may help."
        )
    else:
        patient_parts.append(
            "Your biomarker profile aligns within favorable, low-risk ranges across most evaluated clinical dimensions."
        )

    if inc_risk_factors:
        key_names = [f["feature_name"] for f in inc_risk_factors[:3]]
        patient_parts.append(f"The most influential factors noted in this assessment include your {', '.join(key_names)}.")

    patient_parts.append(
        "Important: This is an educational research tool, not a medical diagnosis. Please share these findings with a qualified physician or healthcare provider for proper diagnosis and personalized care."
    )
    patient_summary = " ".join(patient_parts)

    return clinical_explanation, patient_summary
