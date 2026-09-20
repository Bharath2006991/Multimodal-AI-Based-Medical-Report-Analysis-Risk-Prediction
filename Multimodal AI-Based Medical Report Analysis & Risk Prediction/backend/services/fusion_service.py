"""
Multimodal Early Fusion Service
Combines structured clinical inputs, PDF lab extractions, and OCR image biomarkers into a unified feature representation.
"""

from typing import Dict, Any, List, Optional
from backend.schemas.schemas import PatientFullProfile, ExtractedLabItem


def fuse_multimodal_patient_data(
    patient_profile: PatientFullProfile,
    extracted_items: Optional[List[ExtractedLabItem]] = None,
    detected_symptoms: Optional[List[str]] = None,
    negated_symptoms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Early Fusion Architecture:
    Merges Demographics + Vitals/Labs + Extracted Report Measurements + Clinical Symptoms
    into a standardized single-patient dictionary.
    
    Generates provenance tracking showing the modality source of each biomarker.
    """
    # 1. Base dictionary from structured input
    demo = patient_profile.demographics
    vitals = patient_profile.vitals_labs
    syms = patient_profile.symptoms

    fused: Dict[str, Any] = {
        "patient_id": demo.patient_id,
        "age": float(demo.age),
        "sex": demo.sex.lower(),
        "height_cm": float(demo.height_cm or 175.0),
        "weight_kg": float(demo.weight_kg or 80.0),
        "bmi": float(demo.bmi or round(demo.weight_kg / ((demo.height_cm / 100.0) ** 2), 1)),
        "smoking_status": demo.smoking_status.lower(),
        "physical_activity": demo.physical_activity.lower(),
        "family_history_cad": int(demo.family_history_cad),
        # Vitals & Labs baseline
        "systolic_bp": float(vitals.systolic_bp),
        "diastolic_bp": float(vitals.diastolic_bp),
        "heart_rate": float(vitals.heart_rate),
        "fasting_glucose": float(vitals.fasting_glucose),
        "hba1c": float(vitals.hba1c or 5.6),
        "total_cholesterol": float(vitals.total_cholesterol),
        "hdl_cholesterol": float(vitals.hdl_cholesterol),
        "ldl_cholesterol": float(vitals.ldl_cholesterol),
        "triglycerides": float(vitals.triglycerides),
        "hemoglobin": float(vitals.hemoglobin or 14.5),
        # Symptoms
        "chest_pain": int(syms.chest_pain),
        "shortness_of_breath": int(syms.shortness_of_breath),
        "fatigue": int(syms.fatigue),
        "dizziness": int(syms.dizziness),
        "palpitations": int(syms.palpitations)
    }

    modality_provenance: Dict[str, str] = {k: "structured_form" for k in fused.keys()}

    # 2. Overlay Extracted Lab Measurements (from PDF or OCR Image)
    if extracted_items:
        for item in extracted_items:
            test_key = item.test_name.lower().replace(" ", "_").replace("-", "_")
            if test_key in fused:
                fused[test_key] = float(item.value)
                modality_provenance[test_key] = "extracted_report_document"

    # 3. Integrate NLP Detected / Negated Symptoms
    if detected_symptoms:
        for s in detected_symptoms:
            s_key = s.lower().replace(" ", "_")
            if s_key in fused:
                fused[s_key] = 1
                modality_provenance[s_key] = "nlp_clinical_text"

    if negated_symptoms:
        for s in negated_symptoms:
            s_key = s.lower().replace(" ", "_")
            if s_key in fused:
                fused[s_key] = 0
                modality_provenance[s_key] = "nlp_negation_verified"

    fused["_modality_provenance"] = modality_provenance
    return fused
