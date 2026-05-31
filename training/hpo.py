from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import mlflow

def run_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    n_trials: int = 5,
    random_state: int = 42,
) -> dict[str, any]:
    """Run Optuna hyperparameter optimization to find best XGBoost params."""
    print(f"\nStarting Optuna HPO ({n_trials} trials)...")
    
    # 1. Stratified train/val split for validation during HPO
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )

    def objective(trial: optuna.Trial) -> float:
        # Suggest parameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": random_state,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        
        # Suggest SMOTE ratio
        smote_ratio = trial.suggest_float("smote_ratio", 0.02, 0.2)
        
        # Apply SMOTE to the training fold only
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_res, y_tr_res)

        # Evaluate on validation fold using PR-AUC (Precision-Recall AUC)
        y_prob = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(recall, precision)

        # Log trial to MLflow under the parent run
        if mlflow.active_run():
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.log_params(params)
                mlflow.log_param("smote_ratio", smote_ratio)
                mlflow.log_metric("val_pr_auc", pr_auc)

        return pr_auc

    # Silence Optuna logger to keep output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("HPO Complete.")
    print(f"Best Trial PR-AUC: {study.best_value:.4f}")
    print("Best Params:", study.best_params)
    
    return study.best_params
