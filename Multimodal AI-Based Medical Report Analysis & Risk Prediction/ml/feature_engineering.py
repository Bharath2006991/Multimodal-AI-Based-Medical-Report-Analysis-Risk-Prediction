"""
Clinical Feature Engineering Pipeline
Calculates clinically validated cardiovascular and metabolic biomarker indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union


def compute_clinical_features(data: Union[pd.DataFrame, Dict[str, Any]]) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Computes standard clinical physiological ratios and indices:
    - Mean Arterial Pressure (MAP): DBP + (SBP - DBP) / 3
    - Pulse Pressure: SBP - DBP
    - Cholesterol-to-HDL Ratio: TC / HDL
    - Non-HDL Cholesterol: TC - HDL
    - Triglyceride-Glucose (TyG) Index: ln( (Triglycerides * Fasting Glucose) / 2 )
    - Metabolic Syndrome Score (ATP III surrogate count)
    """
    is_dict = isinstance(data, dict)
    df = pd.DataFrame([data]) if is_dict else data.copy()

    # 1. Hemodynamics
    sbp = df["systolic_bp"]
    dbp = df["diastolic_bp"]
    df["pulse_pressure"] = sbp - dbp
    df["mean_arterial_pressure"] = (dbp + (sbp - dbp) / 3.0).round(1)

    # 2. Lipid & Metabolic Ratios
    tc = df["total_cholesterol"]
    hdl = df["hdl_cholesterol"]
    tg = df["triglycerides"]
    glu = df["fasting_glucose"]

    # Avoid zero-division
    safe_hdl = np.where(hdl <= 0, 1.0, hdl)
    df["chol_hdl_ratio"] = (tc / safe_hdl).round(2)
    df["non_hdl_cholesterol"] = (tc - hdl).round(1)

    # TyG Index: surrogate marker for insulin resistance
    safe_tg_glu = np.clip((tg * glu) / 2.0, 1.0, None)
    df["tyg_index"] = np.log(safe_tg_glu).round(2)

    # 3. Clinical Metabolic Syndrome Risk Factor Count (ATP III criteria approximation)
    # Factor 1: Elevated BP (SBP >= 130 or DBP >= 85)
    f_bp = ((sbp >= 130) | (dbp >= 85)).astype(int)
    # Factor 2: Elevated Triglycerides (>= 150)
    f_tg = (tg >= 150).astype(int)
    # Factor 3: Reduced HDL (<40 in men or <50 in women)
    is_male = df["sex"].str.lower().isin(["male", "m"])
    f_hdl = ((is_male & (hdl < 40)) | (~is_male & (hdl < 50))).astype(int)
    # Factor 4: Elevated Fasting Glucose (>= 100)
    f_glu = (glu >= 100).astype(int)
    # Factor 5: Elevated BMI (>= 30 as waist circumference surrogate)
    f_bmi = (df["bmi"] >= 30.0).astype(int)

    df["metabolic_risk_factor_count"] = f_bp + f_tg + f_hdl + f_glu + f_bmi

    if is_dict:
        return df.iloc[0].to_dict()
    return df
