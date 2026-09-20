"""
Synthetic Clinical Dataset Generator and Data Loader
Generates realistic, scientifically grounded clinical dataset based on standard epidemiological patterns (NHANES, Framingham, ATP III guidelines).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def generate_synthetic_clinical_data(n_samples: int = 3200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic clinical cohort with physiological correlations and clinical risk categories.
    Follows biological covariance between age, obesity, metabolic biomarkers, and cardiovascular strain.
    """
    np.random.seed(seed)

    # 1. Demographics
    age = np.clip(np.random.normal(53, 14, n_samples).round(), 22, 85)
    sex = np.random.choice(["male", "female"], size=n_samples, p=[0.49, 0.51])
    
    # Sex boolean for calculation
    is_male = (sex == "male").astype(float)

    # 2. Anthropometrics
    # Base height
    height_cm = np.where(is_male == 1, 
                         np.random.normal(176, 7, n_samples), 
                         np.random.normal(163, 6, n_samples)).round(1)
    
    # BMI influenced slightly by age
    base_bmi = np.random.gamma(shape=16, scale=1.7, size=n_samples) # Mean ~ 27.2
    bmi = np.clip(base_bmi + (age - 45) * 0.05 + np.random.normal(0, 1.5, n_samples), 17.5, 46.0).round(1)
    weight_kg = (bmi * ((height_cm / 100.0) ** 2)).round(1)

    # 3. Lifestyle
    smoking_choices = ["never", "former", "current"]
    p_male = [0.55, 0.25, 0.20]
    p_female = [0.68, 0.20, 0.12]
    smoking_status = [
        np.random.choice(smoking_choices, p=p_male if is_m == 1 else p_female)
        for is_m in is_male
    ]
    smoking_current = np.array([1.0 if s == "current" else (0.4 if s == "former" else 0.0) for s in smoking_status])

    physical_activity = np.random.choice(["low", "moderate", "high"], size=n_samples, p=[0.35, 0.45, 0.20])
    activity_low = (physical_activity == "low").astype(float)

    family_history_cad = np.random.binomial(1, 0.28, size=n_samples)

    # 4. Blood Pressure & Hemodynamics
    # SBP increases with age, BMI, and smoking
    base_sbp = 105.0 + 0.45 * (age - 30) + 0.9 * (bmi - 23) + 6.0 * smoking_current + np.random.normal(0, 10, n_samples)
    systolic_bp = np.clip(base_sbp.round(), 90, 210)

    # DBP correlated with SBP
    base_dbp = 68.0 + 0.35 * (systolic_bp - 100) + 0.3 * (bmi - 23) + np.random.normal(0, 6, n_samples)
    diastolic_bp = np.clip(base_dbp.round(), 55, 125)

    # Resting Heart Rate
    base_hr = 70.0 + 0.3 * (bmi - 24) + 5.0 * smoking_current + 4.0 * activity_low + np.random.normal(0, 9, n_samples)
    heart_rate = np.clip(base_hr.round(), 48, 128)

    # 5. Metabolic Lab Values
    # Fasting glucose (correlated with BMI, age, activity)
    base_glucose = 85.0 + 0.25 * (age - 30) + 1.2 * (bmi - 22) + 8.0 * activity_low + np.random.normal(0, 14, n_samples)
    # Add diabetic subgroup cluster
    diabetic_cluster = np.random.binomial(1, 0.12, size=n_samples)
    base_glucose += diabetic_cluster * np.random.exponential(45, size=n_samples)
    fasting_glucose = np.clip(base_glucose.round(1), 65, 310)

    # HbA1c tied strongly to fasting glucose
    base_hba1c = 4.2 + (fasting_glucose - 70) * 0.024 + np.random.normal(0, 0.3, n_samples)
    hba1c = np.clip(base_hba1c.round(1), 4.1, 13.5)

    # 6. Lipid Panel
    # Triglycerides (heavily elevated with high BMI, high glucose, low activity)
    base_trig = 95.0 + 2.8 * (bmi - 22) + 0.4 * (fasting_glucose - 90) + 18.0 * smoking_current + np.random.normal(0, 35, n_samples)
    triglycerides = np.clip(base_trig.round(), 45, 580)

    # HDL (protective, lower in males, high BMI, smoking; higher with activity)
    base_hdl = 52.0 - 0.4 * (bmi - 22) - 4.0 * is_male - 5.0 * smoking_current + 6.0 * (physical_activity == "high") + np.random.normal(0, 8, n_samples)
    hdl_cholesterol = np.clip(base_hdl.round(), 22, 98)

    # LDL (bad cholesterol, influenced by genetics, age, diet)
    base_ldl = 100.0 + 0.5 * (age - 30) + 0.8 * (bmi - 22) + 10.0 * family_history_cad + np.random.normal(0, 24, n_samples)
    ldl_cholesterol = np.clip(base_ldl.round(), 48, 260)

    # Total Cholesterol (Friedewald estimation: TC = HDL + LDL + Trig/5 + error)
    total_cholesterol = np.clip((hdl_cholesterol + ldl_cholesterol + (triglycerides / 5.0) + np.random.normal(0, 6, n_samples)).round(), 105, 380)

    # Hemoglobin
    base_hb = np.where(is_male == 1, 
                       np.random.normal(15.2, 1.2, n_samples), 
                       np.random.normal(13.4, 1.1, n_samples))
    hemoglobin = np.clip(base_hb.round(1), 8.5, 19.5)

    # 7. Clinical Symptoms (more frequent with severe vitals or age)
    p_chest_pain = np.clip(0.04 + 0.08 * (systolic_bp > 145) + 0.06 * (ldl_cholesterol > 160) + 0.08 * family_history_cad, 0.02, 0.40)
    chest_pain = (np.random.rand(n_samples) < p_chest_pain).astype(int)

    p_sob = np.clip(0.05 + 0.09 * (bmi > 32) + 0.06 * (heart_rate > 90) + 0.08 * smoking_current, 0.03, 0.45)
    shortness_of_breath = (np.random.rand(n_samples) < p_sob).astype(int)

    p_fatigue = np.clip(0.12 + 0.12 * (fasting_glucose > 130) + 0.10 * (hemoglobin < 12.0) + 0.08 * activity_low, 0.08, 0.65)
    fatigue = (np.random.rand(n_samples) < p_fatigue).astype(int)

    p_dizziness = np.clip(0.04 + 0.08 * (systolic_bp > 155) + 0.05 * (systolic_bp < 100), 0.02, 0.35)
    dizziness = (np.random.rand(n_samples) < p_dizziness).astype(int)

    p_palpitations = np.clip(0.05 + 0.10 * (heart_rate > 95) + 0.05 * smoking_current, 0.03, 0.35)
    palpitations = (np.random.rand(n_samples) < p_palpitations).astype(int)

    # 8. Ground Truth Risk Scoring (ATP III & Framingham Risk Criteria)
    # Build continuous multi-factor risk index
    risk_score_continuous = (
        0.025 * (age - 40) +
        0.045 * (bmi - 24) +
        0.030 * (systolic_bp - 120) +
        0.020 * (diastolic_bp - 80) +
        0.035 * (fasting_glucose - 95) +
        0.35 * (hba1c - 5.4) +
        0.015 * (ldl_cholesterol - 100) +
        0.008 * (triglycerides - 130) -
        0.030 * (hdl_cholesterol - 45) +
        1.2 * smoking_current +
        0.8 * activity_low +
        1.0 * family_history_cad +
        1.5 * chest_pain +
        0.9 * shortness_of_breath +
        0.5 * fatigue +
        0.4 * dizziness
    )
    # Add minor stochastic variance
    risk_score_continuous += np.random.normal(0, 0.8, n_samples)

    # Discretize into 3 clinically meaningful tiers
    # Percentiles ~ 38% Low, 38% Moderate, 24% High
    p38 = np.percentile(risk_score_continuous, 38)
    p76 = np.percentile(risk_score_continuous, 76)

    risk_category = np.zeros(n_samples, dtype=int)
    risk_category[risk_score_continuous >= p38] = 1
    risk_category[risk_score_continuous >= p76] = 2

    # Map labels
    risk_label = np.where(risk_category == 0, "Low Risk",
                 np.where(risk_category == 1, "Moderate Risk", "High Risk"))

    # Construct DataFrame
    df = pd.DataFrame({
        "patient_id": [f"PAT-{1000 + i}" for i in range(n_samples)],
        "age": age.astype(int),
        "sex": sex,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "physical_activity": physical_activity,
        "family_history_cad": family_history_cad,
        "systolic_bp": systolic_bp.astype(int),
        "diastolic_bp": diastolic_bp.astype(int),
        "heart_rate": heart_rate.astype(int),
        "fasting_glucose": fasting_glucose,
        "hba1c": hba1c,
        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "hemoglobin": hemoglobin,
        "chest_pain": chest_pain,
        "shortness_of_breath": shortness_of_breath,
        "fatigue": fatigue,
        "dizziness": dizziness,
        "palpitations": palpitations,
        "risk_score_continuous": risk_score_continuous.round(3),
        "risk_category": risk_category,
        "risk_label": risk_label
    })

    return df


def prepare_and_save_data(n_samples: int = 3200, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates synthetic dataset, applies stratified train/val/test split, and saves to CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_clinical_data(n_samples=n_samples, seed=seed)
    raw_path = RAW_DIR / "clinical_dataset.csv"
    df.to_csv(raw_path, index=False)
    print(f"[DataLoader] Saved raw dataset with {len(df)} records to {raw_path}")

    # Stratified split: 70% Train, 15% Validation, 15% Test
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["risk_category"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df["risk_category"])

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"[DataLoader] Split saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"[DataLoader] Risk distribution in train: {dict(train_df['risk_label'].value_counts())}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_and_save_data()
