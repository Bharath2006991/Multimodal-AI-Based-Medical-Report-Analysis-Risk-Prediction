"""
Integration Tests for FastAPI Backend Endpoints
Tests health, auth, document extraction, prediction, benchmarks, and dimensionality reduction.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_auth_demo_token():
    response = client.post("/api/auth/demo-token")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_direct_text_extraction():
    payload = {
        "raw_text": "Fasting Blood Sugar: 132 mg/dL. Total Cholesterol: 245 mg/dL. Denies chest pain."
    }
    response = client.post("/api/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["extracted_measurements"]) >= 2
    assert "chest_pain" in data["negated_symptoms"]


def test_sample_data_endpoint():
    response = client.get("/api/sample-data")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    assert len(data["profiles"]) >= 3


def test_sample_pdf_download():
    response = client.get("/api/sample-data/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 0


def test_history_endpoints():
    # 1. Get history
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_full_prediction_workflow():
    payload = {
        "patient_data": {
            "demographics": {
                "patient_id": "TEST-PAT-001",
                "age": 55,
                "sex": "male",
                "height_cm": 176,
                "weight_kg": 85,
                "bmi": 27.4,
                "smoking_status": "former",
                "physical_activity": "moderate",
                "family_history_cad": True
            },
            "vitals_labs": {
                "systolic_bp": 138,
                "diastolic_bp": 88,
                "heart_rate": 74,
                "fasting_glucose": 112,
                "hba1c": 5.9,
                "total_cholesterol": 224,
                "hdl_cholesterol": 42,
                "ldl_cholesterol": 142,
                "triglycerides": 185,
                "hemoglobin": 14.5
            },
            "symptoms": {
                "chest_pain": False,
                "shortness_of_breath": False,
                "fatigue": True,
                "dizziness": False,
                "palpitations": False
            }
        },
        "model_name": "XGBoost (Optimized)"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_category"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert 0.0 <= data["risk_score"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert "top_contributing_features" in data
    assert "clinical_explanation" in data
    assert "pca_coordinates" in data
    assert "umap_coordinates" in data


def test_model_metrics_and_dim_reduction():
    # Metrics
    res_m = client.get("/api/model/metrics")
    assert res_m.status_code == 200
    metrics_data = res_m.json()
    assert "models" in metrics_data
    assert len(metrics_data["models"]) >= 7

    # PCA
    res_pca = client.get("/api/dim-reduction?method=PCA")
    assert res_pca.status_code == 200
    assert "points" in res_pca.json()

    # UMAP
    res_umap = client.get("/api/dim-reduction?method=UMAP")
    assert res_umap.status_code == 200
    assert "points" in res_umap.json()

    # EDA
    res_eda = client.get("/api/eda")
    assert res_eda.status_code == 200
    assert "summary_statistics" in res_eda.json()
