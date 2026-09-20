"""
Pydantic v2 Schemas for Multimodal Medical AI System
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ==========================================
# Authentication Schemas
# ==========================================
class LoginRequest(BaseModel):
    email: str = Field(..., example="doctor@medai.org")
    password: str = Field(..., example="demo1234")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_name: str
    user_email: str
    role: str


# ==========================================
# Clinical Data Schemas
# ==========================================
class PatientDemographics(BaseModel):
    patient_id: Optional[str] = Field(default="PAT-SAMPLE-001")
    age: float = Field(..., ge=18, le=120, description="Age in years", example=56)
    sex: str = Field(..., description="biological sex: male or female", example="male")
    height_cm: Optional[float] = Field(default=175.0, ge=100, le=250, example=175.0)
    weight_kg: Optional[float] = Field(default=82.0, ge=30, le=300, example=82.0)
    bmi: Optional[float] = Field(default=26.8, ge=12, le=60, example=26.8)
    smoking_status: str = Field(default="never", example="former")  # never, former, current
    physical_activity: str = Field(default="moderate", example="low")  # low, moderate, high
    family_history_cad: bool = Field(default=False, example=True)


class VitalsAndLabs(BaseModel):
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic BP in mmHg", example=138.0)
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic BP in mmHg", example=88.0)
    heart_rate: float = Field(..., ge=40, le=200, description="Resting heart rate in bpm", example=76.0)
    fasting_glucose: float = Field(..., ge=40, le=400, description="Fasting Blood Glucose in mg/dL", example=112.0)
    hba1c: Optional[float] = Field(default=5.9, ge=3.5, le=15.0, description="HbA1c in %", example=5.9)
    total_cholesterol: float = Field(..., ge=80, le=450, description="Total Cholesterol in mg/dL", example=218.0)
    hdl_cholesterol: float = Field(..., ge=15, le=140, description="HDL Cholesterol in mg/dL", example=42.0)
    ldl_cholesterol: float = Field(..., ge=30, le=300, description="LDL Cholesterol in mg/dL", example=136.0)
    triglycerides: float = Field(..., ge=30, le=700, description="Triglycerides in mg/dL", example=185.0)
    hemoglobin: Optional[float] = Field(default=14.5, ge=6.0, le=22.0, description="Hemoglobin in g/dL", example=14.5)


class ClinicalSymptoms(BaseModel):
    chest_pain: bool = Field(default=False, example=False)
    shortness_of_breath: bool = Field(default=False, example=False)
    fatigue: bool = Field(default=False, example=True)
    dizziness: bool = Field(default=False, example=False)
    palpitations: bool = Field(default=False, example=False)
    notes: Optional[str] = Field(default=None, example="Patient reports mild exertional fatigue.")


class PatientFullProfile(BaseModel):
    demographics: PatientDemographics
    vitals_labs: VitalsAndLabs
    symptoms: ClinicalSymptoms


# ==========================================
# Extraction Schemas (NLP & OCR)
# ==========================================
class ExtractedLabItem(BaseModel):
    test_name: str
    display_name: str
    raw_value_str: Optional[str] = None
    value: float
    unit: str
    reference_range: str
    status: str  # normal, low, high, borderline_high, critical_high, critical_low
    severity: str  # success, warning, danger, critical
    category: str
    interpretation: str


class DocumentExtractionResponse(BaseModel):
    filename: str
    file_type: str  # pdf, image
    extracted_text: str
    extracted_measurements: List[ExtractedLabItem]
    detected_symptoms: List[str]
    negated_symptoms: List[str]
    confidence_score: float
    processing_time_ms: float
    warnings: List[str] = []


class DirectTextExtractRequest(BaseModel):
    raw_text: str = Field(..., example="Patient Fasting Blood Sugar: 128 mg/dL, HbA1c: 6.8%, Total Cholesterol: 245 mg/dL. Denies chest pain. Reports persistent fatigue.")


# ==========================================
# Prediction & Multimodal Fusion Schemas
# ==========================================
class MultimodalPredictionRequest(BaseModel):
    patient_data: PatientFullProfile
    extracted_items: Optional[List[ExtractedLabItem]] = None
    model_name: Optional[str] = "XGBoost"  # XGBoost, Random Forest, Logistic Regression, etc.
    use_calibrated: bool = True


class FeatureContribution(BaseModel):
    feature: str
    feature_name: str
    value: float
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"
    impact: str  # "high", "moderate", "low"


class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: str
    patient_id: str
    risk_category: str  # "Low Risk", "Moderate Risk", "High Risk"
    risk_score: float  # 0.0 to 1.0 (calibrated risk probability)
    confidence: float  # 0.0 to 1.0
    probabilities: Dict[str, float]  # e.g. {"Low Risk": 0.12, "Moderate Risk": 0.65, "High Risk": 0.23}
    model_used: str
    abnormal_findings: List[ExtractedLabItem]
    top_contributing_features: List[FeatureContribution]
    clinical_explanation: str
    patient_friendly_summary: str
    pca_coordinates: Optional[List[float]] = None
    umap_coordinates: Optional[List[float]] = None
    disclaimer: str
    inference_time_ms: float


# ==========================================
# Model Benchmarking & Performance Schemas
# ==========================================
class ModelBenchmarkMetric(BaseModel):
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: float
    pr_auc_macro: float
    cv_mean: float
    cv_std: float
    brier_score: float
    training_time_sec: float
    prediction_latency_ms: float
    confusion_matrix: List[List[int]]


class ModelBenchmarkResponse(BaseModel):
    benchmark_timestamp: str
    dataset_samples: int
    models: List[ModelBenchmarkMetric]
    best_model_name: str
    best_metric: str


# ==========================================
# Dimensionality Reduction Schemas
# ==========================================
class CoordinatePoint(BaseModel):
    sample_id: int
    x: float
    y: float
    z: Optional[float] = None
    risk_category: str
    confidence: float


class DimReductionResponse(BaseModel):
    method: str  # "PCA" or "UMAP"
    explained_variance_ratio: Optional[List[float]] = None
    cumulative_variance: Optional[float] = None
    points: List[CoordinatePoint]
    query_point: Optional[Dict[str, float]] = None
    interpretation: str


# ==========================================
# History & Sample Schemas
# ==========================================
class HistoryRecord(BaseModel):
    id: str
    timestamp: str
    patient_id: str
    age: float
    sex: str
    risk_category: str
    risk_score: float
    model_used: str
    abnormal_count: int
    summary: str
