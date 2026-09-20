"""
Clinical Reference Ranges and Biomarker Knowledge Base
Compliant with standard clinical lab reference intervals (AHA, ADA, CDC, ATP III).
"""

from typing import Dict, Any, Optional, Tuple

REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "fasting_glucose": {
        "display_name": "Fasting Blood Glucose",
        "standard_unit": "mg/dL",
        "alt_units": {"mmol/L": 18.0182},  # Multiply mmol/L by 18.0182 to get mg/dL
        "normal_range": (70.0, 99.0),
        "borderline_range": (100.0, 125.0),
        "high_threshold": 126.0,
        "low_threshold": 70.0,
        "critical_high": 300.0,
        "critical_low": 50.0,
        "category": "Metabolic",
        "description": "Measures blood sugar level after fasting for at least 8 hours."
    },
    "hba1c": {
        "display_name": "Hemoglobin A1c (Glycated Hb)",
        "standard_unit": "%",
        "alt_units": {},
        "normal_range": (4.0, 5.6),
        "borderline_range": (5.7, 6.4),
        "high_threshold": 6.5,
        "low_threshold": 4.0,
        "critical_high": 10.0,
        "critical_low": 3.5,
        "category": "Metabolic",
        "description": "Reflects average blood glucose over the preceding 2-3 months."
    },
    "total_cholesterol": {
        "display_name": "Total Cholesterol",
        "standard_unit": "mg/dL",
        "alt_units": {"mmol/L": 38.67},
        "normal_range": (120.0, 199.0),
        "borderline_range": (200.0, 239.0),
        "high_threshold": 240.0,
        "low_threshold": 100.0,
        "critical_high": 350.0,
        "critical_low": 80.0,
        "category": "Lipid Panel",
        "description": "Total serum cholesterol concentration."
    },
    "hdl_cholesterol": {
        "display_name": "HDL Cholesterol (Good)",
        "standard_unit": "mg/dL",
        "alt_units": {"mmol/L": 38.67},
        "normal_range": (40.0, 60.0),  # Men >= 40, Women >= 50
        "borderline_range": (40.0, 49.0),
        "high_threshold": 90.0,
        "low_threshold": 40.0,
        "critical_high": 120.0,
        "critical_low": 25.0,
        "category": "Lipid Panel",
        "description": "High-density lipoprotein; higher levels offer cardiovascular protection."
    },
    "ldl_cholesterol": {
        "display_name": "LDL Cholesterol (Bad)",
        "standard_unit": "mg/dL",
        "alt_units": {"mmol/L": 38.67},
        "normal_range": (50.0, 99.0),
        "borderline_range": (100.0, 159.0),
        "high_threshold": 160.0,
        "low_threshold": 40.0,
        "critical_high": 250.0,
        "critical_low": 30.0,
        "category": "Lipid Panel",
        "description": "Low-density lipoprotein; primary atherogenic lipoprotein."
    },
    "triglycerides": {
        "display_name": "Triglycerides",
        "standard_unit": "mg/dL",
        "alt_units": {"mmol/L": 88.57},
        "normal_range": (50.0, 149.0),
        "borderline_range": (150.0, 199.0),
        "high_threshold": 200.0,
        "low_threshold": 40.0,
        "critical_high": 500.0,
        "critical_low": 30.0,
        "category": "Lipid Panel",
        "description": "Main form of fat stored by the body; elevated levels indicate cardiometabolic strain."
    },
    "hemoglobin": {
        "display_name": "Hemoglobin",
        "standard_unit": "g/dL",
        "alt_units": {"g/L": 0.1},
        "normal_range": (13.5, 17.5),  # General adult range
        "borderline_range": (12.0, 13.4),
        "high_threshold": 18.0,
        "low_threshold": 12.0,
        "critical_high": 20.0,
        "critical_low": 7.0,
        "category": "Complete Blood Count",
        "description": "Iron-containing oxygen-transport protein in red blood cells."
    },
    "systolic_bp": {
        "display_name": "Systolic Blood Pressure",
        "standard_unit": "mmHg",
        "alt_units": {},
        "normal_range": (90.0, 119.0),
        "borderline_range": (120.0, 129.0),
        "high_threshold": 130.0,
        "low_threshold": 90.0,
        "critical_high": 180.0,
        "critical_low": 70.0,
        "category": "Vitals",
        "description": "Peak arterial pressure during cardiac contraction."
    },
    "diastolic_bp": {
        "display_name": "Diastolic Blood Pressure",
        "standard_unit": "mmHg",
        "alt_units": {},
        "normal_range": (60.0, 79.0),
        "borderline_range": (80.0, 89.0),
        "high_threshold": 90.0,
        "low_threshold": 60.0,
        "critical_high": 120.0,
        "critical_low": 45.0,
        "category": "Vitals",
        "description": "Minimum arterial pressure during cardiac relaxation."
    },
    "heart_rate": {
        "display_name": "Heart Rate",
        "standard_unit": "bpm",
        "alt_units": {"/min": 1.0, "beats/min": 1.0},
        "normal_range": (60.0, 100.0),
        "borderline_range": (100.0, 110.0),
        "high_threshold": 100.0,
        "low_threshold": 60.0,
        "critical_high": 150.0,
        "critical_low": 40.0,
        "category": "Vitals",
        "description": "Resting cardiac beats per minute."
    },
    "bmi": {
        "display_name": "Body Mass Index (BMI)",
        "standard_unit": "kg/m2",
        "alt_units": {},
        "normal_range": (18.5, 24.9),
        "borderline_range": (25.0, 29.9),
        "high_threshold": 30.0,
        "low_threshold": 18.5,
        "critical_high": 45.0,
        "critical_low": 15.0,
        "category": "Anthropometrics",
        "description": "Weight-to-height ratio used to screen for weight categories."
    }
}


def normalize_value(test_name: str, value: float, unit: Optional[str] = None) -> Tuple[float, str]:
    """Convert input value from alternative units to standard clinical unit if needed."""
    key = test_name.lower().replace(" ", "_").replace("-", "_")
    info = REFERENCE_RANGES.get(key)
    if not info:
        return value, unit or ""

    standard_unit = info["standard_unit"]
    if not unit or unit.lower().strip() == standard_unit.lower():
        return round(value, 2), standard_unit

    clean_unit = unit.strip()
    for alt_u, multiplier in info.get("alt_units", {}).items():
        if alt_u.lower() in clean_unit.lower():
            converted = value * multiplier
            return round(converted, 2), standard_unit

    return round(value, 2), unit or standard_unit


def evaluate_status(test_name: str, value: float, sex: Optional[str] = "male") -> Dict[str, Any]:
    """
    Evaluate whether a laboratory measurement is normal, low, high, borderline, or critical.
    Returns status, severity, reference range text, and interpretation.
    """
    key = test_name.lower().replace(" ", "_").replace("-", "_")
    info = REFERENCE_RANGES.get(key)

    if not info:
        return {
            "status": "unknown",
            "severity": "info",
            "reference_range": "N/A",
            "unit": "",
            "message": f"No baseline reference interval configured for {test_name}"
        }

    unit = info["standard_unit"]
    norm_low, norm_high = info["normal_range"]

    # Handle sex-specific adjustments
    if key == "hdl_cholesterol" and sex and sex.lower().startswith("f"):
        norm_low = 50.0
    elif key == "hemoglobin" and sex and sex.lower().startswith("f"):
        norm_low, norm_high = (12.0, 15.5)

    ref_str = f"{norm_low} - {norm_high} {unit}"
    crit_high = info.get("critical_high", float("inf"))
    crit_low = info.get("critical_low", 0.0)

    if value >= crit_high:
        return {
            "status": "critical_high",
            "severity": "critical",
            "reference_range": ref_str,
            "unit": unit,
            "message": f"Critically high {info['display_name']} ({value} {unit})"
        }
    elif value <= crit_low:
        return {
            "status": "critical_low",
            "severity": "critical",
            "reference_range": ref_str,
            "unit": unit,
            "message": f"Critically low {info['display_name']} ({value} {unit})"
        }
    elif value > norm_high:
        borderline_high = info.get("high_threshold", norm_high)
        if value < borderline_high:
            return {
                "status": "borderline_high",
                "severity": "warning",
                "reference_range": ref_str,
                "unit": unit,
                "message": f"Borderline elevated {info['display_name']}"
            }
        return {
            "status": "high",
            "severity": "danger",
            "reference_range": ref_str,
            "unit": unit,
            "message": f"Elevated {info['display_name']}"
        }
    elif value < norm_low:
        return {
            "status": "low",
            "severity": "warning",
            "reference_range": ref_str,
            "unit": unit,
            "message": f"Below normal {info['display_name']}"
        }
    else:
        return {
            "status": "normal",
            "severity": "success",
            "reference_range": ref_str,
            "unit": unit,
            "message": f"Optimal {info['display_name']}"
        }
