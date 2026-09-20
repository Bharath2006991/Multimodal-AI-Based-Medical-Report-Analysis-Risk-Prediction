"""
Model Benchmarking, Metrics, Explainability, and EDA API Endpoints
Serves benchmark comparison tables, PCA/UMAP coordinates, global SHAP feature importance, and EDA statistics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
import joblib
import pandas as pd
import numpy as np

from backend.config.settings import BASE_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from backend.utils.reference_ranges import REFERENCE_RANGES
from backend.schemas.schemas import ModelBenchmarkResponse, DimReductionResponse

router = APIRouter(prefix="/api", tags=["Models, Metrics & EDA"])

REPORTS_DIR = BASE_DIR / "models" / "reports"


@router.get("/model/metrics", response_model=ModelBenchmarkResponse)
async def get_model_benchmarks():
    """Returns comparative benchmarking metrics across all 7 machine learning models."""
    report_file = REPORTS_DIR / "benchmark_report.json"
    if not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark report not yet generated. Please execute ml/train.py first."
        )
    try:
        with open(report_file, "r") as f:
            data = json.load(f)
        return ModelBenchmarkResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/model/features")
async def get_feature_metadata():
    """Returns list of all model input features, display names, clinical categories, and global SHAP ranking."""
    explainer_file = MODELS_DIR / "explainer_bundle.joblib"
    global_importance = []
    if explainer_file.exists():
        bundle = joblib.load(explainer_file)
        global_importance = bundle.get("global_importance", [])

    return {
        "features_count": len(global_importance),
        "global_shap_importance": global_importance,
        "reference_ranges": REFERENCE_RANGES
    }


@router.get("/dim-reduction")
async def get_dimensionality_reduction_data(method: str = "PCA"):
    """
    Returns precomputed low-dimensional coordinates for dataset visualization.
    Method: 'PCA' (linear variance) or 'UMAP' (non-linear manifold clusters).
    """
    dim_file = MODELS_DIR / "dim_reduction.joblib"
    if not dim_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dimensionality reduction artifact not found.")

    bundle = joblib.load(dim_file)
    method_upper = method.upper()

    if method_upper == "UMAP":
        return {
            "method": "UMAP",
            "points": bundle["umap_points"],
            "interpretation": bundle["umap_interpretation"]
        }
    else:
        return {
            "method": "PCA",
            "explained_variance_ratio": bundle["explained_variance_ratio"],
            "cumulative_variance": bundle["cumulative_variance"],
            "points": bundle["pca_points"],
            "interpretation": bundle["pca_interpretation"]
        }


@router.get("/eda")
async def get_exploratory_data_analysis():
    """Returns dataset summary statistics, risk distribution, and correlation matrix for the frontend EDA view."""
    raw_path = RAW_DATA_DIR / "clinical_dataset.csv"
    if not raw_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw clinical dataset not found.")

    df = pd.read_csv(raw_path)

    # Risk distribution with native python int
    risk_dist = {str(k): int(v) for k, v in df["risk_label"].value_counts().items()}

    # Numeric features for correlation
    corr_cols = [
        "age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate",
        "fasting_glucose", "hba1c", "total_cholesterol", "hdl_cholesterol",
        "ldl_cholesterol", "triglycerides", "risk_score_continuous"
    ]
    corr_matrix = json.loads(df[corr_cols].corr().round(3).to_json())

    # Numerical statistics
    summary_stats = json.loads(df[corr_cols].describe().round(2).to_json())

    return {
        "total_records": len(df),
        "risk_distribution": risk_dist,
        "correlation_features": corr_cols,
        "correlation_matrix": corr_matrix,
        "summary_statistics": summary_stats,
        "outlier_summary": {
            "systolic_bp_over_160": int((df["systolic_bp"] > 160).sum()),
            "glucose_over_140": int((df["fasting_glucose"] > 140).sum()),
            "bmi_over_35": int((df["bmi"] > 35).sum())
        }
    }


@router.get("/research/experiments")
async def get_research_experiments():
    """Returns feature ablation results, calibration reliability curve, and uncertainty analysis."""
    research_file = MODELS_DIR / "research_experiments.joblib"
    if not research_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research experiments not found.")
    
    data = joblib.load(research_file)
    return data
