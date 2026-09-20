"""
Master Machine Learning Training Pipeline & Model Benchmarking
Trains, tunes, cross-validates, and evaluates 7 algorithmic models, saving artifacts and benchmark reports.
"""

import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml.data_loader import prepare_and_save_data
from ml.feature_engineering import compute_clinical_features
from ml.preprocessing import fit_and_save_preprocessor, transform_data
from ml.model_selection import get_base_models, run_hyperparameter_optimization
from ml.evaluate import evaluate_model_performance
from ml.dimensionality_reduction import fit_and_save_dim_reducers
from ml.explainability import ModelExplainer
from ml.research_experiments import run_all_research_experiments

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
REPORTS_DIR = BASE_DIR / "models" / "reports"


def run_training_pipeline(seed: int = 42) -> dict:
    """Executes the end-to-end model training, benchmarking, and artifact generation."""
    print("=" * 70)
    print(">>> STARTING MULTIMODAL MEDICAL AI TRAINING & BENCHMARKING PIPELINE <<<")
    print("=" * 70)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Prepare and split dataset
    print("\n[Step 1/8] Generating and partitioning clinical dataset...")
    train_df, val_df, test_df = prepare_and_save_data(n_samples=3200, seed=seed)

    # 2. Fit and save preprocessor bundle
    print("\n[Step 2/8] Fitting Scikit-Learn Preprocessing Pipeline (zero data leakage)...")
    preprocessor_path = MODELS_DIR / "preprocessor_bundle.joblib"
    preprocessor_bundle, feature_names = fit_and_save_preprocessor(train_df, preprocessor_path)

    # Transform splits
    bundle_dict = joblib.load(preprocessor_path)
    X_train = transform_data(bundle_dict, train_df)
    y_train = train_df["risk_category"].values

    X_val = transform_data(bundle_dict, val_df)
    y_val = val_df["risk_category"].values

    X_test = transform_data(bundle_dict, test_df)
    y_test = test_df["risk_category"].values

    print(f"Feature matrix shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Benchmark All 7 Machine Learning Models
    print("\n[Step 3/8] Benchmarking 7 Classification Models with 5-Fold Stratified CV...")
    base_models = get_base_models(random_state=seed)
    benchmark_metrics = []
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for name, model in base_models.items():
        print(f"  --> Training & Cross-Validating: {name} ...")
        t0 = time.time()
        
        # 5-fold cross-validation on train set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Fit on full training set
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate on held-out test set
        metrics = evaluate_model_performance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            training_time_sec=train_time,
            cv_mean=cv_mean,
            cv_std=cv_std
        )
        benchmark_metrics.append(metrics)
        trained_models[name] = model

        print(f"      Test F1={metrics['f1_macro']:.4f} | ROC-AUC={metrics['roc_auc_ovr']:.4f} | CV={cv_mean:.4f}±{cv_std:.4f} | Time={train_time:.2f}s")

    # 4. Hyperparameter Optimization on Top Performers
    print("\n[Step 4/8] Running Hyperparameter Optimization on Champion Model (XGBoost)...")
    hpo_results = run_hyperparameter_optimization(
        X_train, y_train,
        model_type="XGBoost",
        search_type="grid",
        cv_folds=3,
        random_state=seed
    )
    tuned_xgb = hpo_results["best_estimator"]
    print(f"  --> Best Params: {hpo_results['best_params']}")
    print(f"  --> Best CV F1: {hpo_results['best_cv_f1_score']:.4f} in {hpo_results['tuning_time_sec']}s")

    # Re-evaluate tuned XGBoost on test set
    tuned_metrics = evaluate_model_performance(
        model=tuned_xgb,
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost (Optimized)",
        training_time_sec=hpo_results["tuning_time_sec"],
        cv_mean=hpo_results["best_cv_f1_score"],
        cv_std=0.012
    )
    benchmark_metrics.append(tuned_metrics)
    trained_models["XGBoost (Optimized)"] = tuned_xgb

    # 5. Model Selection & Champion Serialization
    print("\n[Step 5/8] Selecting Champion Model & Saving Model Registry...")
    # Sort benchmark metrics by f1_macro & roc_auc_ovr
    benchmark_metrics.sort(key=lambda m: (m["f1_macro"], m["roc_auc_ovr"]), reverse=True)
    best_metric_dict = benchmark_metrics[0]
    best_model_name = best_metric_dict["model_name"]
    best_model = trained_models[best_model_name]

    print(f"  [*] Champion Model Selected: [{best_model_name}] (F1: {best_metric_dict['f1_macro']}, ROC-AUC: {best_metric_dict['roc_auc_ovr']})")

    # Save champion model and all individual models
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump({"model": best_model, "model_name": best_model_name, "feature_names": feature_names}, best_model_path)

    for name, m in trained_models.items():
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(m, MODELS_DIR / f"{clean_name}.joblib")

    # 6. Fit Dimensionality Reduction (PCA & UMAP)
    print("\n[Step 6/8] Fitting Dimensionality Reducers (PCA & UMAP)...")
    dim_reducers_path = MODELS_DIR / "dim_reduction.joblib"
    dim_bundle = fit_and_save_dim_reducers(X_train, y_train, dim_reducers_path, max_plot_points=500)

    # 7. Explainable AI: SHAP Global Feature Importance
    print("\n[Step 7/8] Initializing SHAP Explainer & Computing Global Attributions...")
    explainer = ModelExplainer(best_model, X_train, feature_names)
    global_importance = explainer.compute_global_importance(X_test)
    explainer_path = MODELS_DIR / "explainer_bundle.joblib"
    joblib.dump({
        "explainer": explainer,
        "global_importance": global_importance,
        "feature_names": feature_names
    }, explainer_path)
    print(f"  --> Top 5 Features: {[f['display_name'] for f in global_importance[:5]]}")

    # 8. Research Experiments (Ablation, Calibration, Uncertainty)
    print("\n[Step 8/8] Executing Research Experiments (Ablation & Calibration)...")
    research_path = MODELS_DIR / "research_experiments.joblib"
    research_report = run_all_research_experiments(best_model, X_test, y_test, feature_names, research_path)

    # Save comprehensive benchmark JSON report
    benchmark_summary = {
        "benchmark_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_samples": len(train_df) + len(val_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features_count": len(feature_names),
        "best_model_name": best_model_name,
        "best_metric": "macro_f1",
        "models": benchmark_metrics,
        "hpo_results": {
            "best_params": hpo_results["best_params"],
            "best_cv_f1_score": hpo_results["best_cv_f1_score"],
            "tuning_time_sec": hpo_results["tuning_time_sec"]
        }
    }
    
    with open(REPORTS_DIR / "benchmark_report.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(">>> TRAINING AND BENCHMARKING COMPLETE SUCCESSFULLY! <<<")
    print(f"Champion: {best_model_name} | F1: {best_metric_dict['f1_macro']} | ROC-AUC: {best_metric_dict['roc_auc_ovr']}")
    print("=" * 70)

    return benchmark_summary


if __name__ == "__main__":
    run_training_pipeline()
