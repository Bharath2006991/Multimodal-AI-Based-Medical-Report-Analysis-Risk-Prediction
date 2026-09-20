"""
Model Selection, Definitions, and Hyperparameter Optimization Module
Implements 7 ML classifiers and automated grid/randomized hyperparameter search.
"""

import time
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns initialized candidate classification models across algorithmic families."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            C=1.5,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="minkowski"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=400,
            alpha=0.001,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state
        )
    }


def run_hyperparameter_optimization(
    X_train, y_train,
    model_type: str = "XGBoost",
    search_type: str = "grid",
    cv_folds: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs hyperparameter search (GridSearchCV or RandomizedSearchCV)
    and returns parameters, scores, and best estimator.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    start_time = time.time()

    if model_type == "XGBoost":
        base_estimator = xgb.XGBClassifier(eval_metric="mlogloss", random_state=random_state)
        param_grid = {
            "n_estimators": [100, 150, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.08, 0.15],
            "subsample": [0.8, 1.0]
        }
    elif model_type == "Random Forest":
        base_estimator = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    else:  # Logistic Regression
        base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs", "saga"]
        }

    if search_type == "random":
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_grid,
            n_iter=8,
            scoring="f1_macro",
            cv=cv,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        # Use compact grid for fast, deterministic training
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1
        )

    search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    return {
        "model_type": model_type,
        "search_type": search_type,
        "best_params": search.best_params_,
        "best_cv_f1_score": round(float(search.best_score_), 4),
        "total_combinations": len(search.cv_results_["params"]),
        "tuning_time_sec": round(elapsed, 2),
        "best_estimator": search.best_estimator_
    }
