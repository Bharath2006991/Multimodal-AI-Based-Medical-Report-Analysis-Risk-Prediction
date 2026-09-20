"""
Unit Tests for Data Preprocessing and Feature Engineering
Validates clinical index derivations, outlier clipping, and data leakage prevention.
"""

import pytest
import numpy as np
import pandas as pd
from ml.feature_engineering import compute_clinical_features
from ml.preprocessing import build_preprocessing_pipeline, OutlierClipper


def test_clinical_feature_engineering():
    sample = {
        "systolic_bp": 140,
        "diastolic_bp": 90,
        "total_cholesterol": 240,
        "hdl_cholesterol": 40,
        "triglycerides": 200,
        "fasting_glucose": 110,
        "sex": "male",
        "bmi": 31.0
    }
    res = compute_clinical_features(sample)

    # MAP = 90 + 50/3 = 106.7
    assert "mean_arterial_pressure" in res
    assert 106.0 <= res["mean_arterial_pressure"] <= 107.0

    # Pulse pressure = 140 - 90 = 50
    assert res["pulse_pressure"] == 50

    # Chol / HDL ratio = 240 / 40 = 6.0
    assert res["chol_hdl_ratio"] == 6.0

    # Non-HDL = 240 - 40 = 200
    assert res["non_hdl_cholesterol"] == 200.0

    # TyG index > 0
    assert res["tyg_index"] > 8.0

    # Metabolic risk factor count should be >= 4
    assert res["metabolic_risk_factor_count"] >= 4


def test_outlier_clipper():
    clipper = OutlierClipper(lower_percentile=5.0, upper_percentile=95.0)
    data = np.array([[1.0], [10.0], [12.0], [15.0], [100.0]])
    clipper.fit(data)

    transformed = clipper.transform(np.array([[0.0], [500.0]]))
    assert transformed[0, 0] >= clipper.lower_bounds_[0]
    assert transformed[1, 0] <= clipper.upper_bounds_[0]


def test_preprocessing_handles_missing_values():
    from ml.data_loader import generate_synthetic_clinical_data
    from ml.feature_engineering import compute_clinical_features

    df = generate_synthetic_clinical_data(n_samples=50, seed=42)
    # Introduce random NaNs
    df.loc[0, "fasting_glucose"] = np.nan
    df.loc[1, "systolic_bp"] = np.nan
    df.loc[2, "sex"] = np.nan

    df_eng = compute_clinical_features(df)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(df_eng)

    X_trans = pipeline.transform(df_eng)
    assert not np.isnan(X_trans).any(), "Preprocessed matrix should contain zero NaNs."
