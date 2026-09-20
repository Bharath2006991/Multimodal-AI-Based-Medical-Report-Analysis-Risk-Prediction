"""
Clinical NLP Information Extraction Service
Extracts medical laboratory test entities, numeric values, units, reference ranges, and symptoms with negation awareness.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from backend.utils.reference_ranges import REFERENCE_RANGES, evaluate_status, normalize_value

# Medical Entity Synonyms Map
TEST_SYNONYMS: Dict[str, List[str]] = {
    "fasting_glucose": [
        r"fasting\s+blood\s+glucose", r"fasting\s+blood\s+sugar", r"fasting\s+glucose",
        r"fasting\s+sugar", r"\bfbs\b", r"\bfbg\b", r"blood\s+glucose(?:\s+fasting)?",
        r"serum\s+glucose"
    ],
    "hba1c": [
        r"glycated\s+hemoglobin", r"glycosylated\s+hemoglobin", r"\bhba1c\b",
        r"\ba1c\b", r"hemoglobin\s+a1c"
    ],
    "total_cholesterol": [
        r"total\s+cholesterol", r"cholesterol[,\s]+total", r"serum\s+cholesterol",
        r"\bcholesterol\b(?!\s+(?:hdl|ldl))", r"\btc\b"
    ],
    "hdl_cholesterol": [
        r"hdl\s+cholesterol", r"high[\s-]density\s+lipoprotein", r"\bhdl-c\b",
        r"\bhdl\b", r"good\s+cholesterol"
    ],
    "ldl_cholesterol": [
        r"ldl\s+cholesterol", r"low[\s-]density\s+lipoprotein", r"\bldl-c\b",
        r"\bldl\b", r"bad\s+cholesterol"
    ],
    "triglycerides": [
        r"triglycerides?", r"serum\s+triglycerides?", r"\btg\b", r"trigs"
    ],
    "hemoglobin": [
        r"\bhemoglobin\b", r"\bhgb\b", r"\bhb\b(?!\s*a1c)", r"total\s+hemoglobin"
    ],
    "systolic_bp": [
        r"systolic(?:\s+bp|\s+blood\s+pressure)?", r"\bsbp\b"
    ],
    "diastolic_bp": [
        r"diastolic(?:\s+bp|\s+blood\s+pressure)?", r"\bdbp\b"
    ],
    "heart_rate": [
        r"heart\s+rate", r"resting\s+heart\s+rate", r"\bpulse\s+rate\b",
        r"\bpulse\b(?!\s+pressure)", r"\bhr\b"
    ],
    "bmi": [
        r"body\s+mass\s+index", r"\bbmi\b"
    ]
}

# Clinical Symptoms Patterns
SYMPTOM_PATTERNS: Dict[str, List[str]] = {
    "chest_pain": [r"chest\s+pain", r"angina", r"thoracic\s+discomfort", r"chest\s+tightness"],
    "shortness_of_breath": [r"shortness\s+of\s+breath", r"\bdyspnea\b", r"breathlessness", r"difficulty\s+breathing"],
    "fatigue": [r"fatigue", r"exhaustion", r"lethargy", r"tiredness", r"malaise"],
    "dizziness": [r"dizziness", r"lightheadedness", r"vertigo", r"presyncope"],
    "palpitations": [r"palpitations", r"fluttering\s+heart", r"racing\s+heart", r"irregular\s+heartbeat"]
}

# Negation Trigger Phrases
NEGATION_TRIGGERS = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\bnegative\s+for\b", r"\bno\s+history\s+of\b", r"\bno\s+evidence\s+of\b",
    r"\brules?\s+out\b", r"\babsent\b"
]


def clean_clinical_text(text: str) -> str:
    """Cleans document text, normalizes whitespace and special characters."""
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_symptoms_with_negation(text: str) -> Tuple[List[str], List[str]]:
    """
    Scans text for clinical symptoms, checking preceding words in the sentence for negation triggers.
    Returns (detected_symptoms, negated_symptoms).
    """
    detected = set()
    negated = set()

    sentences = re.split(r"[.\n;]", text)
    neg_regex = re.compile(r"|".join(NEGATION_TRIGGERS), re.IGNORECASE)

    for sentence in sentences:
        s_clean = sentence.strip()
        if not s_clean:
            continue

        for symptom_name, patterns in SYMPTOM_PATTERNS.items():
            for pat in patterns:
                match = re.search(pat, s_clean, re.IGNORECASE)
                if match:
                    # Check text before match for negation
                    preceding_text = s_clean[:match.start()]
                    # Look at last 6 words
                    recent_words = " ".join(preceding_text.split()[-6:])
                    if neg_regex.search(recent_words):
                        negated.add(symptom_name)
                    else:
                        detected.add(symptom_name)

    # If negated, remove from detected
    detected = detected - negated
    return list(detected), list(negated)


def extract_blood_pressure_pair(text: str) -> Optional[Tuple[float, float]]:
    """Detects standard BP compound expressions such as '128/84 mmHg' or 'BP: 130 / 85'."""
    pattern = r"(?:(?:bp|blood\s+pressure)[:\s]*)?(\d{2,3})\s*[\/]\s*(\d{2,3})(?:\s*mm\s*hg)?"
    matches = re.finditer(pattern, text, re.IGNORECASE)
    for m in matches:
        sbp = float(m.group(1))
        dbp = float(m.group(2))
        if 70 <= sbp <= 250 and 40 <= dbp <= 140:
            return sbp, dbp
    return None


def extract_medical_entities_from_text(
    text: str,
    patient_sex: Optional[str] = "male"
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Parses full report text, extracts structured lab measurements, verifies ranges,
    and returns detected/negated symptoms.
    """
    cleaned_text = clean_clinical_text(text)
    extracted_items = []
    extracted_keys = set()

    # 1. Check for combined BP expression first
    bp_pair = extract_blood_pressure_pair(cleaned_text)
    if bp_pair:
        sbp, dbp = bp_pair
        sbp_status = evaluate_status("systolic_bp", sbp, patient_sex)
        extracted_items.append({
            "test_name": "systolic_bp",
            "display_name": REFERENCE_RANGES["systolic_bp"]["display_name"],
            "raw_value_str": f"{sbp} mmHg",
            "value": sbp,
            "unit": "mmHg",
            "reference_range": sbp_status["reference_range"],
            "status": sbp_status["status"],
            "severity": sbp_status["severity"],
            "category": "Vitals",
            "interpretation": sbp_status["message"]
        })
        extracted_keys.add("systolic_bp")

        dbp_status = evaluate_status("diastolic_bp", dbp, patient_sex)
        extracted_items.append({
            "test_name": "diastolic_bp",
            "display_name": REFERENCE_RANGES["diastolic_bp"]["display_name"],
            "raw_value_str": f"{dbp} mmHg",
            "value": dbp,
            "unit": "mmHg",
            "reference_range": dbp_status["reference_range"],
            "status": dbp_status["status"],
            "severity": dbp_status["severity"],
            "category": "Vitals",
            "interpretation": dbp_status["message"]
        })
        extracted_keys.add("diastolic_bp")

    # 2. Iterate through reference test dictionary
    lines = cleaned_text.split("\n")
    for test_key, synonyms in TEST_SYNONYMS.items():
        if test_key in extracted_keys:
            continue

        ref_info = REFERENCE_RANGES.get(test_key, {})
        for synonym_pat in synonyms:
            if test_key in extracted_keys:
                break

            # Match lines containing synonym and numeric value
            # Pattern: [Synonym] [:|=|\t|-]? [Value] [Unit]?
            regex_line = re.compile(
                rf"(?:{synonym_pat})[:\s=\t-]*([<>]?\s*\d+(?:\.\d+)?)\s*([a-zA-Z/%/]+)?",
                re.IGNORECASE
            )

            for line in lines:
                m = regex_line.search(line)
                if m:
                    val_str = m.group(1).replace("<", "").replace(">", "").strip()
                    unit_str = m.group(2) if m.group(2) else ref_info.get("standard_unit", "")
                    try:
                        raw_num = float(val_str)
                        # Sanity range check to avoid matching years (e.g., 2026) or telephone numbers
                        norm_val, norm_unit = normalize_value(test_key, raw_num, unit_str)
                        
                        # Validate value is biologically plausible
                        crit_high = ref_info.get("critical_high", 1000.0)
                        if raw_num > crit_high * 1.8:
                            continue  # Likely an ID number or year

                        st = evaluate_status(test_key, norm_val, patient_sex)
                        extracted_items.append({
                            "test_name": test_key,
                            "display_name": ref_info.get("display_name", test_key),
                            "raw_value_str": f"{norm_val} {norm_unit}",
                            "value": norm_val,
                            "unit": norm_unit,
                            "reference_range": st["reference_range"],
                            "status": st["status"],
                            "severity": st["severity"],
                            "category": ref_info.get("category", "General"),
                            "interpretation": st["message"]
                        })
                        extracted_keys.add(test_key)
                        break
                    except ValueError:
                        continue

    # 3. Detect symptoms with negation
    detected_symptoms, negated_symptoms = detect_symptoms_with_negation(cleaned_text)

    return extracted_items, detected_symptoms, negated_symptoms
