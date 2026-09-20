"""
Unit Tests for Clinical NLP Information Extraction Service
Verifies regex patterns, entity matching, unit normalization, reference ranges, and negation handling.
"""

import pytest
from backend.services.nlp_extractor import (
    extract_medical_entities_from_text,
    detect_symptoms_with_negation,
    extract_blood_pressure_pair,
    clean_clinical_text
)
from backend.utils.reference_ranges import evaluate_status, normalize_value


def test_clean_clinical_text():
    raw = "Patient report \r\nwith multiple   spaces and\n\n\n\nnewlines."
    cleaned = clean_clinical_text(raw)
    assert "\r" not in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_extract_blood_pressure_pair():
    text1 = "Patient vitals: BP: 138/88 mmHg, resting pulse 74."
    pair = extract_blood_pressure_pair(text1)
    assert pair is not None
    assert pair == (138.0, 88.0)

    text2 = "Recorded 120 / 80 during triage."
    pair2 = extract_blood_pressure_pair(text2)
    assert pair2 == (120.0, 80.0)


def test_detect_symptoms_with_negation():
    # Affirmed symptom
    text_affirmed = "Patient reports persistent fatigue and occasional dyspnea."
    det, neg = detect_symptoms_with_negation(text_affirmed)
    assert "fatigue" in det
    assert "shortness_of_breath" in det
    assert len(neg) == 0

    # Negated symptom
    text_negated = "Patient denies chest pain. No history of palpitations. Reports fatigue."
    det2, neg2 = detect_symptoms_with_negation(text_negated)
    assert "chest_pain" in neg2
    assert "palpitations" in neg2
    assert "chest_pain" not in det2
    assert "fatigue" in det2


def test_extract_medical_entities():
    report_text = """
    METABOLIC PROFILE:
    Fasting Blood Sugar: 128 mg/dL
    HbA1c: 6.8%
    Total Cholesterol: 245 mg/dL
    HDL: 38 mg/dL
    Triglycerides: 210 mg/dL
    BP: 142/90 mmHg
    Denies chest pain.
    """
    items, detected, negated = extract_medical_entities_from_text(report_text, patient_sex="male")
    
    extracted_names = {item["test_name"] for item in items}
    assert "fasting_glucose" in extracted_names
    assert "hba1c" in extracted_names
    assert "total_cholesterol" in extracted_names
    assert "hdl_cholesterol" in extracted_names
    assert "triglycerides" in extracted_names
    assert "systolic_bp" in extracted_names
    assert "diastolic_bp" in extracted_names

    # Check status flags
    glucose_item = next(i for i in items if i["test_name"] == "fasting_glucose")
    assert glucose_item["value"] == 128.0
    assert glucose_item["status"] in ["high", "critical_high"]

    assert "chest_pain" in negated


def test_unit_normalization():
    # Test mmol/L to mg/dL for glucose: 7.0 mmol/L * 18.0182 ~ 126.13 mg/dL
    norm_val, norm_unit = normalize_value("fasting_glucose", 7.0, "mmol/L")
    assert norm_unit == "mg/dL"
    assert 125.0 <= norm_val <= 127.0
