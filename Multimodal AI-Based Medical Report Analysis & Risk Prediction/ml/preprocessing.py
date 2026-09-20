"""
Preprocessing Pipeline Module
Prevents data leakage using Scikit-Learn ColumnTransformer and Pipeline.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Optional
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

from ml.feature_engineering import compute_clinical_features

# Feature specifications
NUMERICAL_FEATURES = [
    "age", "height_cm", "weight_kg", "bmi",
    "systolic_bp", "diastolic_bp", "heart_rate",
    "fasting_glucose", "hba1c",
    "total_cholesterol", "hdl_cholesterol", "ldl_cholesterol", "triglycerides",
    "hemoglobin",
    # Engineered features:
    "pulse_pressure", "mean_arterial_pressure",
    "chol_hdl_ratio", "non_hdl_cholesterol",
    "tyg_index", "metabolic_risk_factor_count"
]

CATEGORICAL_FEATURES = [
    "sex", "smoking_status", "physical_activity"
]

BINARY_SYMPTOM_FEATURES = [
    "family_history_cad", "chest_pain", "shortness_of_breath",
    "fatigue", "dizziness", "palpitations"
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_SYMPTOM_FEATURES


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips extreme numerical values to specified percentiles calculated strictly on training set."""
    def __init__(self, lower_percentile: float = 0.5, upper_percentile: float = 99.5):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_mat = np.asarray(X)
        self.lower_bounds_ = np.percentile(X_mat, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X_mat, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        X_mat = np.asarray(X)
        return np.clip(X_mat, self.lower_bounds_, self.upper_bounds_)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer with separate imputers and scalers."""
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(0.5, 99.5)),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("bin", bin_pipeline, BINARY_SYMPTOM_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """Extracts explicit feature names after ColumnTransformer one-hot expansion."""
    feature_names = []
    
    # 1. Numerical features
    feature_names.extend(NUMERICAL_FEATURES)
    
    # 2. Categorical features (one-hot encoded)
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names.extend(encoded_cat_names)
    
    # 3. Binary symptom features
    feature_names.extend(BINARY_SYMPTOM_FEATURES)
    
    return feature_names


def fit_and_save_preprocessor(train_df: pd.DataFrame, save_path: Path) -> Tuple[ColumnTransformer, List[str]]:
    """Fits preprocessor on train_df (after computing clinical features) and serializes to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Apply feature engineering
    df_engineered = compute_clinical_features(train_df)
    
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(df_engineered)
    
    feature_names = get_transformed_feature_names(preprocessor)
    
    # Save bundle
    bundle = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_SYMPTOM_FEATURES
    }
    joblib.dump(bundle, save_path)
    print(f"[Preprocessing] Preprocessor bundle saved to {save_path} ({len(feature_names)} features)")
    
    return preprocessor, feature_names


def transform_data(preprocessor_bundle: dict, data: pd.DataFrame) -> np.ndarray:
    """Transforms raw or engineered input DataFrame using a fitted preprocessor bundle."""
    # Ensure clinical features are computed
    df_feat = compute_clinical_features(data)
    preprocessor = preprocessor_bundle["preprocessor"]
    return preprocessor.transform(df_feat)
