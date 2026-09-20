"""
Prediction and Multimodal Analysis API Endpoints
Coordinates early fusion, ML risk prediction, SHAP attribution, and multi-model consensus.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List

from backend.schemas.schemas import MultimodalPredictionRequest, PredictionResponse
from backend.services.fusion_service import fuse_multimodal_patient_data
from ml.predict import predict_patient_risk
from backend.models.db_models import history_db

router = APIRouter(prefix="/api", tags=["Prediction & Multimodal Analysis"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_risk(req: MultimodalPredictionRequest):
    """
    Executes multimodal risk prediction pipeline:
    1. Early fusion of structured data + verified document extractions
    2. Zero-leakage preprocessing & clinical index engineering
    3. Calibrated ML classification
    4. SHAP local feature attribution
    5. Dual-level narrative generation
    6. Historical logging
    """
    try:
        # 1. Early Fusion
        fused_patient = fuse_multimodal_patient_data(
            patient_profile=req.patient_data,
            extracted_items=req.extracted_items
        )

        # 2. Predict
        res = predict_patient_risk(
            patient_dict=fused_patient,
            model_override_name=req.model_name
        )

        # 3. Log to History
        history_record = {
            "id": res["prediction_id"],
            "timestamp": res["timestamp"],
            "patient_id": res["patient_id"],
            "age": req.patient_data.demographics.age,
            "sex": req.patient_data.demographics.sex,
            "risk_category": res["risk_category"],
            "risk_score": res["risk_score"],
            "model_used": res["model_used"],
            "abnormal_count": len(res["abnormal_findings"]),
            "summary": res["patient_friendly_summary"]
        }
        history_db.add_record(history_record)

        return PredictionResponse(**res)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


@router.post("/analyze")
async def analyze_multimodal_consensus(req: MultimodalPredictionRequest):
    """
    Runs patient data across all benchmarked model architectures simultaneously,
    evaluating inter-model consensus and prediction concordance.
    """
    candidate_models = [
        "XGBoost (Optimized)", "Random Forest", "Gradient Boosting",
        "Logistic Regression", "Support Vector Machine"
    ]

    fused_patient = fuse_multimodal_patient_data(
        patient_profile=req.patient_data,
        extracted_items=req.extracted_items
    )

    model_results = []
    category_votes = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}

    primary_res = None
    for m_name in candidate_models:
        try:
            pred = predict_patient_risk(fused_patient, model_override_name=m_name)
            cat = pred["risk_category"]
            category_votes[cat] = category_votes.get(cat, 0) + 1
            model_results.append({
                "model_name": m_name,
                "risk_category": cat,
                "risk_score": pred["risk_score"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"]
            })
            if primary_res is None:
                primary_res = pred
        except Exception:
            continue

    total_valid = len(model_results)
    consensus_class = max(category_votes, key=category_votes.get) if total_valid else "Moderate Risk"
    concordance_rate = (category_votes[consensus_class] / total_valid) if total_valid else 1.0

    return {
        "primary_prediction": primary_res,
        "model_consensus": {
            "consensus_category": consensus_class,
            "concordance_rate": round(concordance_rate, 2),
            "votes": category_votes,
            "models_evaluated": total_valid,
            "evaluations": model_results
        }
    }
